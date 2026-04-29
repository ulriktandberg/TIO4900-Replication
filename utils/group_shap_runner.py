"""Exact group-level Shapley values for the two macro ANN wrappers.

Why this module exists
----------------------
``utils.shap_runner`` computes per-feature DeepSHAP values and we then aggregate
them to groups post-hoc (sum or mean within a group). That is axiomatically
valid (linearity of ``+``) but has a well-known weakness: when features inside
a group are highly correlated, DeepSHAP spreads the shared "group information"
credit across the substitutes. Each correlated feature gets a small individual
value; summing recovers the group total, but within-group rankings become
noisy and cross-group comparisons are sensitive to how many features each
group happens to contain.

Grouped Shapley avoids that by redefining the cooperative game so the players
are *groups*, not features. A coalition ``S`` either has a group fully in or
fully out. With 9 players (forward + 8 FRED-MD groups), the game has 2**9 =
512 coalitions, which we enumerate exactly — no sampling approximation.

What "mask group g" means here
------------------------------
For both wrappers we use a single-reference Shapley game: the reference point
``r`` is the mean of a background sample drawn from the training window, and
``v(S)`` is the model's prediction when the inputs for groups outside ``S``
are replaced with the corresponding coordinates of ``r``. Concretely:

* ``GroupEnsembleANNWrapper``: each group has its own input tensor. Masking a
  group replaces that tensor with its background mean vector.
* ``MacroForwardANNWrapper``: the FRED-MD tower receives one pooled tensor.
  Masking a FRED-MD group replaces the columns of that tensor belonging to
  the group; masking the forward group replaces the forward tensor.

The reference ``r`` therefore plays the same role DeepSHAP implicitly uses as
its "expected value" — the two methods are measuring departures from the
same background, just aggregated over different player sets.

Output layout
-------------
Mirrors the DeepSHAP runner so downstream plotting can reuse conventions:

    <output_root>/<run_name>/<run_ts>/
        group_shap_mean.parquet            columns: date, maturity, group, mean_shap, abs_mean_shap, n_seeds
        group_shap_per_seed.parquet        columns: date, maturity, seed, group, shap_value
        group_base_values.parquet          columns: date, maturity, base_value, ensemble_pred, n_seeds, additivity_residual
        group_order.json                   canonical group player order used
        group_shap_meta.json               config + diagnostics

The "run_ts" segment reuses the *orchestrator* run timestamp, so a group-SHAP
directory sits next to the DeepSHAP directory for the same checkpoints and
both can be resumed independently.

**Two-player (spanning) mode.** Set ``GroupShapRunConfig(binary_macro=True)`` and
point ``output_root`` at e.g. ``artifacts/group_shap_binary``. Players are
``forward`` and ``macro`` only (four coalitions). Parquet schema is unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from math import factorial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .shap_adapters import get_adapter
from .shap_runner import _reconstruct_ckpt_path


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class GroupShapRunConfig:
    """Inputs that fully specify a group-SHAP run against an orchestrator run."""

    orchestrator_run_dir: str | Path
    dates: Any = "all"
    """Same semantics as ``ShapRunConfig.dates``."""

    maturities: Sequence[str] | None = None
    background_size: int = 128
    apply_y_scaling: bool = True
    y_center_default: bool = False
    device: str = "cpu"
    seed: int = 0
    output_root: str | Path = "artifacts/group_shap"
    overwrite: bool = False
    progress: bool = True

    save_per_seed: bool = True
    """Persist per-seed group-Shapley values. They're small (one row per
    (date, maturity, seed, group)) and unlock seed-stability diagnostics at
    the group level, so this defaults to True unlike the feature-level runner."""

    binary_macro: bool = False
    """If True, use a **two-player** game only: ``forward`` vs ``macro`` (all
    FRED / all macro towers enter or leave together). Exactly four coalitions
    per evaluation — suited to the spanning hypothesis. Output schema is
    unchanged (``group`` is ``forward`` or ``macro``). Point ``output_root``
    at e.g. ``artifacts/group_shap_binary`` so runs do not overwrite the
    nine-player artefacts."""


# --------------------------------------------------------------------------- #
# Date selector (same grammar as shap_runner)                                 #
# --------------------------------------------------------------------------- #


def _resolve_dates_selector(
    selector: Any, oos_dates: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    if isinstance(selector, str):
        if selector == "all":
            return oos_dates
        if selector.startswith("every_nth:"):
            n = int(selector.split(":", 1)[1])
            return oos_dates[::n]
        if selector.startswith("spaced:"):
            k = int(selector.split(":", 1)[1])
            if k <= 0:
                return pd.DatetimeIndex([])
            idx = np.unique(np.linspace(0, len(oos_dates) - 1, k, dtype=int))
            return oos_dates[idx]
        raise ValueError(f"Unknown string dates selector: {selector!r}")
    ts = pd.to_datetime(pd.Index(list(selector)))
    missing = [str(d) for d in ts if d not in oos_dates]
    if missing:
        raise ValueError(
            f"{len(missing)} requested date(s) are outside the OOS window; "
            f"first few: {missing[:3]}"
        )
    return ts.sort_values()


# --------------------------------------------------------------------------- #
# Shapley weights (exact enumeration)                                         #
# --------------------------------------------------------------------------- #


def _shapley_weight_matrix(n: int) -> np.ndarray:
    """Return ``W`` with ``W[i, S_bits] = [i not in S] * (|S|!(n-|S|-1)!/n!)``.

    Shapley for player ``i`` from pre-computed coalition values ``v`` is then
    ``phi_i = sum_{S: i not in S} W[i, S] * (v[S | {i}] - v[S])``. We build
    the matrix once per game size and reuse it.
    """
    W = np.zeros((n, 1 << n), dtype=np.float64)
    n_fac = float(factorial(n))
    size_factorials = np.array(
        [factorial(s) * factorial(n - s - 1) / n_fac for s in range(n)],
        dtype=np.float64,
    )
    for S in range(1 << n):
        s = int(bin(S).count("1"))
        if s == n:
            continue  # no "add i" move possible
        for i in range(n):
            if S & (1 << i):
                continue
            W[i, S] = size_factorials[s]
    return W


def _shapley_from_coalition_values(v: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Vectorised Shapley: ``v`` is (2**n, out_dim) or (2**n,); returns (n, out_dim).

    Uses the identity ``phi_i = sum_{S: i not in S} W[i,S] * (v[S|{i}] - v[S])``.
    Rather than building the delta tensor explicitly, we split the sum:

        phi_i = sum_{T: i in T} W[i, T - {i}] * v[T]
              - sum_{S: i not in S} W[i, S]        * v[S]

    Both sums are a linear combination of ``v`` by a sparse matrix keyed on
    bit patterns, so we precompute those coefficients once in ``W`` and one
    companion matrix ``W_pos``.
    """
    n, n_coal = W.shape
    if v.shape[0] != n_coal:
        raise ValueError(
            f"Coalition values have {v.shape[0]} rows but expected {n_coal}."
        )
    out = np.zeros((n,) + v.shape[1:], dtype=np.float64)
    for i in range(n):
        bit_i = 1 << i
        # S iterates over all subsets; for "add i" transitions S -> S|{i}:
        #   coefficient on v[S|{i}] is +W[i, S]
        #   coefficient on v[S]     is -W[i, S]
        # We mask S by "i not in S", then index v at S and at (S | bit_i).
        mask = np.array(
            [bool(W[i, S]) for S in range(n_coal)]
        )  # W is zero exactly when i in S or |S|==n
        S_idx = np.flatnonzero(mask)
        T_idx = S_idx | bit_i
        weights = W[i, S_idx]
        out[i] = weights @ v[T_idx] - weights @ v[S_idx]
    return out


# --------------------------------------------------------------------------- #
# Player-structure adapters                                                   #
# --------------------------------------------------------------------------- #


class _GroupStructure:
    """Wraps an orchestrator-run adapter with per-group masking helpers.

    ``players`` is the ordered list of group names (first entry always
    ``"forward"``). ``mask_input_batch`` builds a (2**n, ...) batched input
    whose i-th row has only the groups in coalition ``i`` at their real
    values; all other groups are replaced by their background means.
    """

    def __init__(self, *, wrapper_class: str, adapter, ckpt: dict, X: pd.DataFrame):
        self.wrapper_class = wrapper_class
        self.adapter = adapter
        self.ckpt = ckpt
        self.X = X
        self.players, self._meta = self._infer_players()

    @property
    def n_players(self) -> int:
        return len(self.players)

    # -- player discovery ------------------------------------------------ #

    def _infer_players(self) -> tuple[list[str], dict]:
        X = self.X
        if "forward" not in X.columns.get_level_values(0):
            raise ValueError("X must contain a top-level 'forward' block.")
        if "fred" not in X.columns.get_level_values(0):
            raise ValueError("X must contain a top-level 'fred' block.")

        fred_cols = X["fred"].columns
        if "group" not in fred_cols.names:
            raise ValueError(
                "X['fred'] must carry a 'group' level in its column MultiIndex."
            )

        if self.wrapper_class == "GroupEnsembleANNWrapper":
            # Canonical tensor order = state_dict insertion order.
            sd = self.ckpt["torch_state_dict"]
            import re
            seen: set[str] = set()
            order: list[str] = []
            pat = re.compile(r"^macro_towers\.(.+?)\.")
            for key in sd:
                m = pat.match(key)
                if m and m.group(1) not in seen:
                    seen.add(m.group(1))
                    order.append(m.group(1))
            players = ["forward"] + order
            meta = {"kind": "multi_tensor"}
            return players, meta

        if self.wrapper_class == "MacroForwardANNWrapper":
            # One tensor per tower (forward + pooled fred). We still want to
            # attribute group-level Shapley, so within the pooled fred tensor
            # we carve out column index lists per FRED-MD group.
            fred_groups = (
                X["fred"].columns.get_level_values("group").astype(str)
            )
            unique_groups: list[str] = []
            for g in fred_groups:
                if g not in unique_groups:
                    unique_groups.append(g)
            col_indices: dict[str, np.ndarray] = {
                g: np.flatnonzero(fred_groups == g) for g in unique_groups
            }
            players = ["forward"] + unique_groups
            meta = {"kind": "pooled_fred", "fred_col_indices": col_indices}
            return players, meta

        raise NotImplementedError(
            f"No group-SHAP player layout defined for {self.wrapper_class!r}."
        )

    # -- coalition masking ---------------------------------------------- #

    def background_means(
        self, X_train: pd.DataFrame, n: int, rng: np.random.Generator
    ) -> Any:
        """Average the adapter's prepared-input representation across a background sample.

        Returns the same shape/container the adapter yields for a single row
        (list/tuple of tensors), but averaged along the batch axis. Using the
        adapter handles scaling/PCA correctly for each wrapper.
        """
        bg = self.adapter.sample_background(
            X_train, self.ckpt, n, rng
        )
        # bg is either (Tensor, Tensor) or list[Tensor]; normalise to list.
        bg_list = list(bg) if isinstance(bg, (tuple, list)) else [bg]
        return [b.mean(dim=0, keepdim=False) for b in bg_list]

    def prepare_target(self, X_row: pd.DataFrame) -> list[torch.Tensor]:
        inp = self.adapter.prepare_inputs(X_row, self.ckpt)
        inp_list = list(inp) if isinstance(inp, (tuple, list)) else [inp]
        # Strip the leading batch dim so each entry is shape (n_i,).
        return [t.reshape(t.shape[-1]) if t.dim() == 2 else t for t in inp_list]

    def build_coalition_batch(
        self,
        target: list[torch.Tensor],
        bg_means: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Return a list of tensors, each shape (2**n, n_i), containing the
        coalition-masked inputs in bit-order (S = 0..2**n-1).

        Row ``S`` has the target value for every group whose bit is set in
        ``S`` and the background mean for the rest.
        """
        n = self.n_players
        n_coal = 1 << n
        out: list[torch.Tensor]

        if self._meta["kind"] == "multi_tensor":
            # One tensor per group (plus forward). Stack along batch.
            out = []
            for i, (tgt, mean) in enumerate(zip(target, bg_means)):
                feat_dim = tgt.shape[-1]
                mask = torch.tensor(
                    [(S >> i) & 1 for S in range(n_coal)],
                    dtype=torch.float32,
                ).unsqueeze(-1)  # (n_coal, 1)
                batch = mask * tgt.view(1, feat_dim) + (1.0 - mask) * mean.view(
                    1, feat_dim
                )
                out.append(batch)
            return out

        if self._meta["kind"] == "pooled_fred":
            tgt_fwd, tgt_fred = target
            mean_fwd, mean_fred = bg_means
            fwd_dim = tgt_fwd.shape[-1]
            fred_dim = tgt_fred.shape[-1]

            # Forward is player 0.
            mask_fwd = torch.tensor(
                [(S >> 0) & 1 for S in range(n_coal)],
                dtype=torch.float32,
            ).unsqueeze(-1)
            batch_fwd = mask_fwd * tgt_fwd.view(1, fwd_dim) + (
                1.0 - mask_fwd
            ) * mean_fwd.view(1, fwd_dim)

            # FRED groups are players 1..n-1.
            col_indices = self._meta["fred_col_indices"]
            batch_fred = mean_fred.view(1, fred_dim).repeat(n_coal, 1)
            for i_player, g in enumerate(self.players[1:], start=1):
                cols = col_indices[g]
                if len(cols) == 0:
                    continue
                mask = torch.tensor(
                    [(S >> i_player) & 1 for S in range(n_coal)],
                    dtype=torch.float32,
                ).unsqueeze(-1)
                tgt_slice = tgt_fred[cols].view(1, -1)        # (1, len(cols))
                mean_slice = mean_fred[cols].view(1, -1)      # (1, len(cols))
                batch_fred[:, cols] = mask * tgt_slice + (1.0 - mask) * mean_slice
            return [batch_fwd, batch_fred]

        raise NotImplementedError(self._meta)

    def forward(
        self, model: torch.nn.Module, coalition_batch: list[torch.Tensor], device: str
    ) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            xs = tuple(x.to(device) for x in coalition_batch)
            out = model(*xs)
        return out.detach().cpu().numpy()


class _BinaryGroupStructure:
    """Two-player Shapley game: ``forward`` vs pooled ``macro``.

    Coalitions use two bits: bit 0 = forward on/off, bit 1 = entire macro
    block on/off. For ``MacroForwardANNWrapper`` the macro player toggles the
    full FRED tensor. For ``GroupEnsembleANNWrapper`` it toggles all macro
    tower tensors together.
    """

    def __init__(self, *, wrapper_class: str, adapter, ckpt: dict, X: pd.DataFrame):
        self._inner = _GroupStructure(
            wrapper_class=wrapper_class,
            adapter=adapter,
            ckpt=ckpt,
            X=X,
        )
        self.wrapper_class = wrapper_class
        self.adapter = adapter
        self.ckpt = ckpt
        self.X = X
        self._meta = self._inner._meta

    @property
    def players(self) -> tuple[str, ...]:
        return ("forward", "macro")

    @property
    def n_players(self) -> int:
        return 2

    def background_means(
        self, X_train: pd.DataFrame, n: int, rng: np.random.Generator
    ) -> list[torch.Tensor]:
        return self._inner.background_means(X_train, n, rng)

    def prepare_target(self, X_row: pd.DataFrame) -> list[torch.Tensor]:
        return self._inner.prepare_target(X_row)

    def build_coalition_batch(
        self,
        target: list[torch.Tensor],
        bg_means: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        n_coal = 4
        if self._meta["kind"] == "multi_tensor":
            out: list[torch.Tensor] = []
            fwd_tgt, fwd_mean = target[0], bg_means[0]
            fwd_dim = int(fwd_tgt.shape[-1])
            mask_fwd = torch.tensor(
                [(S >> 0) & 1 for S in range(n_coal)],
                dtype=torch.float32,
            ).unsqueeze(-1)
            out.append(
                mask_fwd * fwd_tgt.view(1, fwd_dim)
                + (1.0 - mask_fwd) * fwd_mean.view(1, fwd_dim)
            )
            mask_macro = torch.tensor(
                [(S >> 1) & 1 for S in range(n_coal)],
                dtype=torch.float32,
            ).unsqueeze(-1)
            for i in range(1, len(target)):
                tgt, mean = target[i], bg_means[i]
                fd = int(tgt.shape[-1])
                out.append(
                    mask_macro * tgt.view(1, fd)
                    + (1.0 - mask_macro) * mean.view(1, fd)
                )
            return out

        if self._meta["kind"] == "pooled_fred":
            tgt_fwd, tgt_fred = target
            mean_fwd, mean_fred = bg_means
            fwd_dim = int(tgt_fwd.shape[-1])
            fred_dim = int(tgt_fred.shape[-1])
            mask_fwd = torch.tensor(
                [(S >> 0) & 1 for S in range(n_coal)],
                dtype=torch.float32,
            ).unsqueeze(-1)
            batch_fwd = mask_fwd * tgt_fwd.view(1, fwd_dim) + (
                1.0 - mask_fwd
            ) * mean_fwd.view(1, fwd_dim)
            mask_macro = torch.tensor(
                [(S >> 1) & 1 for S in range(n_coal)],
                dtype=torch.float32,
            ).unsqueeze(-1)
            batch_fred = mask_macro * tgt_fred.view(1, fred_dim) + (
                1.0 - mask_macro
            ) * mean_fred.view(1, fred_dim)
            return [batch_fwd, batch_fred]

        raise NotImplementedError(self._meta)

    def forward(
        self, model: torch.nn.Module, coalition_batch: list[torch.Tensor], device: str
    ) -> np.ndarray:
        return self._inner.forward(model, coalition_batch, device)


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #


def compute_group_shap_for_run(
    cfg: GroupShapRunConfig,
    X: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict:
    """Compute exact group-level Shapley values per (date, maturity)."""

    t0 = time.time()
    run_dir = Path(cfg.orchestrator_run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Orchestrator run directory not found: {run_dir}")

    with open(run_dir / "run_config.json") as fh:
        run_config = json.load(fh)

    manifest = pd.read_csv(run_dir / "checkpoint_manifest.csv")
    manifest["date"] = pd.to_datetime(manifest["date"])
    manifest = manifest.sort_values(["seed", "t_index"]).reset_index(drop=True)

    topk_indices = np.load(run_dir / "topk_indices.npy")
    ensemble_forecast = np.load(run_dir / "ensemble_forecast.npy")

    # -- adapter + sample ckpt ----------------------------------------- #
    sample_row = manifest.iloc[0]
    sample_path = Path(sample_row["checkpoint_path"])
    if not sample_path.is_absolute():
        sample_path = (run_dir / ".." / ".." / ".." / sample_path).resolve()
    if not sample_path.exists():
        sample_path = _reconstruct_ckpt_path(run_dir, sample_row)
    sample_ckpt = torch.load(sample_path, map_location=cfg.device, weights_only=False)
    wrapper_class = sample_ckpt["wrapper_class"]
    adapter = get_adapter(wrapper_class)

    # Build a reference player structure so we can allocate the Shapley weight
    # matrix once and validate that per-seed player order is identical (it is
    # expected to be; the safety check protects against ckpt drift).
    if cfg.binary_macro:
        structure = _BinaryGroupStructure(
            wrapper_class=wrapper_class,
            adapter=adapter,
            ckpt=sample_ckpt,
            X=X,
        )
        canonical_players = structure.players
    else:
        structure = _GroupStructure(
            wrapper_class=wrapper_class,
            adapter=adapter,
            ckpt=sample_ckpt,
            X=X,
        )
        canonical_players = tuple(structure.players)
    n_players = structure.n_players
    n_coal = 1 << n_players
    W = _shapley_weight_matrix(n_players)

    # -- maturities / dates -------------------------------------------- #
    all_mats = [str(m) for m in (run_config.get("maturities") or [])]
    mats = [str(m) for m in (cfg.maturities or all_mats)]
    missing = [m for m in mats if m not in all_mats]
    if missing:
        raise ValueError(
            f"Requested maturities {missing} not in run's maturities {all_mats}."
        )
    mat_idx = {m: all_mats.index(m) for m in mats}

    oos_start = pd.Timestamp(run_config["oos_start"], unit="ms")
    oos_dates_full = dates[dates >= oos_start]
    target_dates = _resolve_dates_selector(cfg.dates, oos_dates_full)
    if len(target_dates) == 0:
        raise ValueError("No target dates selected.")

    # -- output dir ----------------------------------------------------- #
    run_name = run_config.get("run_name", run_dir.parent.name)
    run_ts = run_dir.name
    out_root = Path(cfg.output_root).resolve()
    out_dir = out_root / run_name / run_ts
    per_date_dir = out_dir / "per_date"
    if cfg.overwrite and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    per_date_dir.mkdir(parents=True, exist_ok=True)

    # -- lookups -------------------------------------------------------- #
    ckpt_lookup: dict[tuple[int, int], Path] = {}
    for row in manifest.itertuples(index=False):
        p = Path(row.checkpoint_path)
        if not p.is_absolute():
            p = _reconstruct_ckpt_path(run_dir, row)
        ckpt_lookup[(int(row.seed), int(row.t_index))] = p

    date_to_t = pd.Series(
        data=np.arange(len(dates)), index=pd.DatetimeIndex(dates)
    )
    refit_indices_by_seed: dict[int, np.ndarray] = {
        int(seed): np.sort(
            manifest.loc[manifest["seed"] == seed, "t_index"].to_numpy()
        )
        for seed in manifest["seed"].unique()
    }

    def _find_refit_t(seed: int, t_index: int) -> int:
        arr = refit_indices_by_seed[seed]
        pos = int(np.searchsorted(arr, t_index, side="right") - 1)
        if pos < 0:
            raise ValueError(
                f"No refit <= t_index={t_index} for seed={seed}."
            )
        return int(arr[pos])

    # -- resume detection ---------------------------------------------- #
    done_by_date: dict[str, set[str]] = {}

    def _merge_done(tag: str, ms: set[str]) -> None:
        done_by_date.setdefault(tag, set()).update(ms)

    for p in per_date_dir.glob("*.parquet"):
        if p.stem.endswith(("__base", "__seeds")):
            continue
        try:
            sub = pd.read_parquet(p, columns=["maturity"])
        except Exception:
            continue
        _merge_done(p.stem, set(sub["maturity"].astype(str).unique()))

    merged_path = out_dir / "group_shap_mean.parquet"
    if merged_path.exists():
        prior = pd.read_parquet(merged_path, columns=["date", "maturity"]).drop_duplicates()
        prior["date"] = pd.to_datetime(prior["date"]).dt.strftime("%Y-%m-%d")
        for d, grp in prior.groupby("date"):
            _merge_done(d, set(grp["maturity"].astype(str)))

    # -- main loop ------------------------------------------------------ #
    rng = np.random.default_rng(cfg.seed)
    additivity_residuals: list[float] = []
    written_count = 0

    iterator = target_dates
    if cfg.progress:
        iterator = tqdm(target_dates, desc="Group SHAP dates", leave=False)

    for target_date in iterator:
        tag = target_date.strftime("%Y-%m-%d")
        already_have = done_by_date.get(tag, set())
        if not cfg.overwrite and set(mats).issubset(already_have):
            continue

        t_index = int(date_to_t.loc[target_date])
        X_row = X.iloc[[t_index]]

        per_mat_rows: list[dict] = []
        per_mat_base: list[dict] = []
        per_mat_seed_rows: list[dict] = []

        for m in mats:
            m_idx = mat_idx[m]
            seeds_for_date = [
                int(s) for s in topk_indices[t_index, m_idx, :] if s != -1
            ]
            if not seeds_for_date:
                continue

            per_seed_phi: list[np.ndarray] = []
            per_seed_pred: list[float] = []
            per_seed_base: list[float] = []
            per_seed_additivity: list[float] = []

            for seed in seeds_for_date:
                refit_t = _find_refit_t(seed, t_index)
                ckpt_path = ckpt_lookup[(seed, refit_t)]
                ckpt = torch.load(
                    ckpt_path, map_location=cfg.device, weights_only=False
                )
                model = adapter.rebuild_model(ckpt, run_config).to(cfg.device)

                gap = int(run_config.get("gap", 0))
                train_end = max(refit_t - gap, 0)
                X_train = X.iloc[:train_end]

                if cfg.binary_macro:
                    struct = _BinaryGroupStructure(
                        wrapper_class=wrapper_class,
                        adapter=adapter,
                        ckpt=ckpt,
                        X=X,
                    )
                else:
                    struct = _GroupStructure(
                        wrapper_class=wrapper_class,
                        adapter=adapter,
                        ckpt=ckpt,
                        X=X,
                    )
                if tuple(struct.players) != canonical_players:
                    raise RuntimeError(
                        "Per-seed player order disagrees with the reference "
                        f"order. reference={canonical_players}, "
                        f"seed={seed} t_index={refit_t}: {tuple(struct.players)}. "
                        "This indicates a checkpoint with a different group "
                        "layout than the sample checkpoint."
                    )
                bg_means = struct.background_means(X_train, cfg.background_size, rng)
                target = struct.prepare_target(X_row)
                coalition_batch = struct.build_coalition_batch(target, bg_means)
                preds = struct.forward(model, coalition_batch, cfg.device)
                # preds is (n_coal, n_outputs); select maturity m_idx.
                v = preds[:, m_idx] if preds.ndim == 2 else preds
                v = v.astype(np.float64)

                phi = _shapley_from_coalition_values(v, W)  # (n_players,)
                v_full = float(v[n_coal - 1])
                v_empty = float(v[0])

                scale, shift = adapter.y_scale_and_shift(ckpt, cfg.y_center_default)
                y_scale = float(scale[m_idx]) if scale.size > m_idx else float(scale[0])
                y_shift = float(shift[m_idx]) if shift.size > m_idx else float(shift[0])

                if cfg.apply_y_scaling:
                    phi_u = phi * y_scale
                    base_u = v_empty * y_scale + y_shift
                    pred_u = v_full * y_scale + y_shift
                else:
                    phi_u = phi
                    base_u = v_empty
                    pred_u = v_full

                additivity = abs(pred_u - (base_u + float(np.sum(phi_u))))
                additivity_residuals.append(additivity)

                per_seed_phi.append(phi_u)
                per_seed_pred.append(pred_u)
                per_seed_base.append(base_u)
                per_seed_additivity.append(additivity)

                if cfg.save_per_seed:
                    for g_idx, g_name in enumerate(canonical_players):
                        per_mat_seed_rows.append({
                            "date": target_date,
                            "maturity": m,
                            "seed": int(seed),
                            "group": g_name,
                            "shap_value": float(phi_u[g_idx]),
                        })

            stacked = np.stack(per_seed_phi, axis=0)   # (k, n_players)
            mean_phi = stacked.mean(axis=0)
            std_phi = stacked.std(axis=0, ddof=0) if stacked.shape[0] > 1 else np.zeros_like(mean_phi)

            for g_idx, g_name in enumerate(canonical_players):
                per_mat_rows.append({
                    "date": target_date,
                    "maturity": m,
                    "group": g_name,
                    "mean_shap": float(mean_phi[g_idx]),
                    "abs_mean_shap": float(np.abs(mean_phi[g_idx])),
                    "std_shap": float(std_phi[g_idx]),
                    "n_seeds": int(stacked.shape[0]),
                })

            per_mat_base.append({
                "date": target_date,
                "maturity": m,
                "base_value": float(np.mean(per_seed_base)),
                "ensemble_pred": float(np.mean(per_seed_pred)),
                "orchestrator_ensemble_pred": float(ensemble_forecast[t_index, m_idx]),
                "n_seeds": int(stacked.shape[0]),
                "additivity_residual": float(np.mean(per_seed_additivity)),
            })

        if not per_mat_rows:
            continue

        pd.DataFrame(per_mat_rows).to_parquet(
            per_date_dir / f"{tag}.parquet", index=False
        )
        pd.DataFrame(per_mat_base).to_parquet(
            per_date_dir / f"{tag}__base.parquet", index=False
        )
        if cfg.save_per_seed and per_mat_seed_rows:
            pd.DataFrame(per_mat_seed_rows).to_parquet(
                per_date_dir / f"{tag}__seeds.parquet", index=False
            )
        written_count += 1

    # -- merge per-date files ------------------------------------------ #
    all_files = sorted(per_date_dir.glob("*.parquet"))
    base_files = [p for p in all_files if p.stem.endswith("__base")]
    seed_files = [p for p in all_files if p.stem.endswith("__seeds")]
    main_files = [
        p for p in all_files if not p.stem.endswith(("__base", "__seeds"))
    ]

    if main_files:
        merged = pd.concat(
            [pd.read_parquet(p) for p in main_files], ignore_index=True
        ).sort_values(["date", "maturity", "group"]).reset_index(drop=True)
        merged.to_parquet(out_dir / "group_shap_mean.parquet", index=False)

    if base_files:
        merged_b = pd.concat(
            [pd.read_parquet(p) for p in base_files], ignore_index=True
        ).sort_values(["date", "maturity"]).reset_index(drop=True)
        merged_b.to_parquet(out_dir / "group_base_values.parquet", index=False)

    if seed_files:
        merged_s = pd.concat(
            [pd.read_parquet(p) for p in seed_files], ignore_index=True
        ).sort_values(["date", "maturity", "seed", "group"]).reset_index(drop=True)
        merged_s.to_parquet(out_dir / "group_shap_per_seed.parquet", index=False)

    with open(out_dir / "group_order.json", "w") as fh:
        json.dump(list(canonical_players), fh, indent=2)

    elapsed = time.time() - t0
    meta = {
        "orchestrator_run_dir": str(run_dir),
        "orchestrator_run_name": run_name,
        "orchestrator_run_timestamp": run_ts,
        "wrapper_class": wrapper_class,
        "players": list(canonical_players),
        "binary_macro": bool(cfg.binary_macro),
        "n_players": int(n_players),
        "n_coalitions": int(n_coal),
        "maturities": mats,
        "n_dates": int(len(target_dates)),
        "n_dates_written_this_call": int(written_count),
        "n_dates_already_present": int(len(done_by_date)),
        "background_size": int(cfg.background_size),
        "apply_y_scaling": bool(cfg.apply_y_scaling),
        "device": cfg.device,
        "seed": int(cfg.seed),
        "additivity_error_mean": float(np.mean(additivity_residuals)) if additivity_residuals else None,
        "additivity_error_max": float(np.max(additivity_residuals)) if additivity_residuals else None,
        "elapsed_s": round(elapsed, 2),
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": _serialisable_cfg(cfg),
    }
    with open(out_dir / "group_shap_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    logger.info("Done in %.1fs. Output: %s", elapsed, out_dir)

    return {
        "output_dir": str(out_dir),
        "n_dates_total": int(len(target_dates)),
        "n_dates_written_this_call": int(written_count),
        "n_dates_already_present": int(len(done_by_date)),
        "elapsed_s": round(elapsed, 2),
        "additivity_error_mean": meta["additivity_error_mean"],
        "additivity_error_max": meta["additivity_error_max"],
        "players": list(canonical_players),
    }


def _serialisable_cfg(cfg: GroupShapRunConfig) -> dict:
    d = asdict(cfg)
    d["orchestrator_run_dir"] = str(cfg.orchestrator_run_dir)
    d["output_root"] = str(cfg.output_root)
    if not isinstance(cfg.dates, (str, int, float)):
        d["dates"] = [str(x) for x in cfg.dates]
    return d
