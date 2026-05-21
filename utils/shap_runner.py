"""Runner for computing SHAP values from an orchestrator run's checkpoints.

Typical use from a notebook::

    from utils.shap_runner import ShapRunConfig, compute_shap_for_run

    cfg = ShapRunConfig(
        orchestrator_run_dir="artifacts/orchestrator_runs/fwd_ann_5&5_5runs_top2/20260401_170442",
        dates="every_nth:12",
        maturities=["120"],
        background_size=64,
    )
    summary = compute_shap_for_run(cfg, X=X, y_all=y_all, dates=dates)

Output layout: ``artifacts/shap/<run_name>/<timestamp>/`` when the orchestrator
uses legacy unsuffixed ``topk_indices.npy``; otherwise
``artifacts/shap/<run_name>/<timestamp>/<ensemble_metric>/`` (e.g. ``val_loss``,
``trailing_oos``) so different ensemble definitions never share files.

Results are written incrementally per (date, maturity) batch so a long run can be
resumed after interruption.

For ``macro_variant: realtime`` orchestrator runs (or ``ShapRunConfig(realtime=True)``),
macro rows are rebuilt like ``expanding_window(..., realtime=True)``: explanation
inputs use vintage as of the target date; background samples use the vintage as
of each seed's refit step. Match ``fred_md`` construction to training (no
``shift(1)`` for realtime).

**Training matrix parity:** Deep SHAP mirrors ``expanding_window`` refit
preprocessing via ``_preprocess_X_like_expanding_window_refit`` in this module:
columns still NaN anywhere on the training slice are dropped, then forward-fill
is applied. Without that, saved scalers can see a different feature count than
at train time (e.g. 9 vs 7 columns).

"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import shap  # type: ignore

from .shap_adapters import get_adapter
from .window_utils import (
    _apply_column_selection,
    _carry_forward_latest,
    _realtime_feature_frame,
    _select_train_available_columns,
)


logger = logging.getLogger(__name__)


def _effective_realtime_flag(cfg_explicit: bool | None, run_config: dict) -> bool:
    """Mirror ``expanding_window(..., realtime=True)`` runs.

    When ``cfg.realtime is None``, enable realtime if ``run_config`` carries
    ``macro_variant == "realtime"`` (orchestrator_ann / run_core_models JSON).
    """
    if cfg_explicit is not None:
        return bool(cfg_explicit)
    return str(run_config.get("macro_variant", "") or "").lower() == "realtime"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ShapRunConfig:
    """Inputs that fully specify a SHAP run against an orchestrator run."""

    orchestrator_run_dir: str | Path
    dates: Any = "all"
    """Date selector. Supported forms:
      - ``"all"``: every OOS date the run produced forecasts for.
      - ``"every_nth:<n>"``: every n-th OOS date.
      - ``"spaced:<k>"``: ``k`` approximately evenly-spaced OOS dates.
      - ``list[pd.Timestamp | str]``: explicit list (must fall inside OOS window).
    """

    maturities: Sequence[str] | None = None
    """If ``None``, use all maturities from the orchestrator run_config."""

    background_size: int = 64
    background_sampling: str = "train_window_random"
    explainer: str = "deep"  # "deep" or "gradient"
    check_additivity: bool = False
    apply_y_scaling: bool = True
    y_center_default: bool = False
    """Default for ``with_mean`` when the saved scaler state does not carry it.
    Current notebooks use ``y_center=False`` so ``False`` is the right default."""

    device: str = "cpu"
    seed: int = 0
    output_root: str | Path = "artifacts/shap"
    overwrite: bool = False
    keep_per_date: bool = True
    """Leaving per-date parquet files in place makes later ``overwrite=False``
    runs fully resumable (e.g. to add dates or maturities without redoing work).
    Set to ``False`` only when disk space matters more than resumability."""
    progress: bool = True

    save_per_seed: bool = False
    """Persist raw per-seed SHAP values to ``per_seed_shap.parquet``. Needed for
    seed-stability analyses (boxplots across seeds, rank correlations). Off by
    default because the file can be large for long full sweeps."""

    save_per_seed_meta: bool = True
    """Persist per-seed scalar diagnostics (prediction, base, sum_shap,
    additivity_residual) to ``per_seed_meta.parquet``. Tiny and useful — left on
    by default."""

    realtime: bool | None = None
    """If ``True``, rebuild macro features at each OOS date with
    ``ForecastVintageMacroStore`` + ``_realtime_feature_frame`` (same as
    ``expanding_window(..., realtime=True)``). If ``None``, use
    ``run_config["macro_variant"] == "realtime"`` when present, else ``False``."""

    realtime_store: Any | None = None
    """Optional ``ForecastVintageMacroStore`` instance. If ``None`` and realtime
    is active, a default store is constructed (same as ``expanding_window``)."""

    ensemble_metric: str | None = None
    """Which saved orchestrator ensemble to explain. Newer runs write
    ``topk_indices_<metric>.npy`` / ``ensemble_forecast_<metric>.npy`` (e.g.
    ``trailing_oos``, ``val_loss``) instead of legacy unsuffixed files.
    If ``None``, prefer legacy ``topk_indices.npy``, then ``trailing_oos``, then
    ``val_loss``, then any other ``topk_indices_*.npy`` pair on disk."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_ensemble_npy_paths(
    run_dir: Path, ensemble_metric: str | None = None
) -> tuple[Path, Path, str]:
    """Return ``(topk_indices_path, ensemble_forecast_path, metric_label)``."""

    run_dir = Path(run_dir)

    def _pair_exists(topk_name: str, ens_name: str) -> tuple[Path, Path] | None:
        pt = run_dir / topk_name
        pe = run_dir / ens_name
        if pt.is_file() and pe.is_file():
            return pt, pe
        return None

    if ensemble_metric is not None:
        m = str(ensemble_metric).strip()
        if not m:
            raise ValueError("ensemble_metric must be non-empty when set")
        got = _pair_exists(f"topk_indices_{m}.npy", f"ensemble_forecast_{m}.npy")
        if got:
            return (*got, m)
        raise FileNotFoundError(
            f"Orchestrator run dir has no ensemble arrays for metric={m!r}: "
            f"expected {run_dir / f'topk_indices_{m}.npy'} and "
            f"{run_dir / f'ensemble_forecast_{m}.npy'}"
        )

    for label, tk_name, ef_name in (
        ("legacy", "topk_indices.npy", "ensemble_forecast.npy"),
        ("trailing_oos", "topk_indices_trailing_oos.npy", "ensemble_forecast_trailing_oos.npy"),
        ("val_loss", "topk_indices_val_loss.npy", "ensemble_forecast_val_loss.npy"),
    ):
        got = _pair_exists(tk_name, ef_name)
        if got:
            return (*got, label)

    for tk in sorted(run_dir.glob("topk_indices_*.npy")):
        suffix = tk.stem.removeprefix("topk_indices_")
        if not suffix:
            continue
        ef = run_dir / f"ensemble_forecast_{suffix}.npy"
        got = _pair_exists(tk.name, ef.name)
        if got:
            return (*got, suffix)

    raise FileNotFoundError(
        f"No top-k / ensemble forecast .npy pair found under {run_dir}. "
        "Expected legacy topk_indices.npy + ensemble_forecast.npy, or "
        "topk_indices_<metric>.npy + ensemble_forecast_<metric>.npy "
        "(e.g. trailing_oos, val_loss)."
    )


def _shap_output_base_dir(
    output_root: Path, run_name: str, orchestrator_ts: str, ensemble_metric_used: str
) -> Path:
    """Directory for Deep SHAP parquet/metadata under ``output_root``.

    Legacy orchestrator outputs (unsuffixed ``topk_indices.npy``) keep the old
    flat layout ``.../<run_name>/<ts>/``. Any metric-specific ensemble arrays use
    ``.../<run_name>/<ts>/<metric>/`` so ``val_loss`` vs ``trailing_oos`` cannot mix.
    """

    base = Path(output_root).resolve() / run_name / orchestrator_ts
    if ensemble_metric_used != "legacy":
        base = base / ensemble_metric_used
    return base


def _resolve_dates_selector(selector: Any, oos_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
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
            idx = np.linspace(0, len(oos_dates) - 1, k, dtype=int)
            idx = np.unique(idx)
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


def _maturity_columns(run_config: dict) -> list[str]:
    mats = run_config.get("maturities") or []
    return [str(m) for m in mats]


_DEFAULT_YEARLY_MATURITIES = ("24", "36", "48", "60", "84", "120")


def _coerce_run_config_maturities(run_config: dict, ensemble_forecast: np.ndarray) -> list[str]:
    """Orchestrator ``run_config.json`` sometimes omits ``maturities``; infer from arrays."""

    mats = _maturity_columns(run_config)
    if mats:
        return mats
    n_out = int(np.asarray(ensemble_forecast).shape[1])
    defaults = list(_DEFAULT_YEARLY_MATURITIES)
    if n_out == len(defaults):
        logger.warning(
            "run_config.json has no 'maturities'; using default yearly tenors %s "
            "(%d outputs; same default as experiments/run_core_models.py).",
            defaults,
            n_out,
        )
        return defaults
    labels = [str(i) for i in range(n_out)]
    logger.warning(
        "run_config.json has no 'maturities'; using placeholder labels %s (%d outputs). "
        "Pass ShapRunConfig.maturities explicitly if order/names do not match training.",
        labels,
        n_out,
    )
    return labels


def expanding_window_train_end(refit_t: int, gap: int) -> int:
    """Exclusive end index for ``X.iloc[:end]`` — matches ``expanding_window`` at refit.

    In ``window_utils.expanding_window``, each refit at row ``t`` uses
    ``train_end = t - gap`` and fits on ``X.iloc[:train_end]`` (and ``y`` likewise).
    That implements the usual **non-overlapping** annual horizon: with ``gap=11``
    and monthly data, the last ``gap`` months before ``t`` are excluded so ``y`` at
    training rows does not overlap the label being predicted at ``t``.

    We clamp with ``max(..., 0)`` so ``iloc[:0]`` is an empty frame instead of a
    negative slice (invalid / ambiguous in pandas) if ``refit_t < gap`` ever appears.
    """

    return max(int(refit_t) - int(gap), 0)


def _preprocess_X_like_expanding_window_refit(
    X_model: pd.DataFrame,
    train_end: int,
    *,
    drop_unavailable_columns: bool = True,
    carry_forward_latest: bool = True,
) -> tuple[pd.DataFrame, pd.Index]:
    """Match ``expanding_window`` refit: drop still-NaN-on-train columns, then ffill.

    Local to Deep SHAP only — uses existing private helpers from
    ``window_utils`` without changing that module's public surface.
    """

    if train_end <= 0:
        raise ValueError(
            "_preprocess_X_like_expanding_window_refit requires train_end > 0 "
            f"(got {train_end})."
        )
    te = min(train_end, len(X_model))
    X_work = X_model
    if drop_unavailable_columns:
        selected_columns = _select_train_available_columns(X_work.iloc[:te])
        if len(selected_columns) == 0:
            raise ValueError(
                "Every column has NaN on the training slice "
                f"rows [:train_end={train_end}); cannot align with expanding_window."
            )
        X_work = _apply_column_selection(X_work, selected_columns)
    else:
        selected_columns = X_work.columns
    if carry_forward_latest:
        X_work = _carry_forward_latest(X_work)
    return X_work, selected_columns


def _expected_value(explainer, fallback_model, background, device) -> np.ndarray:
    """Return SHAP's additive base value. Prefer ``explainer.expected_value``
    (the ground truth for additivity) and fall back to ``mean(model(background))``
    if an older SHAP version does not expose it.
    """
    ev = getattr(explainer, "expected_value", None)
    if ev is not None:
        ev_arr = np.atleast_1d(np.asarray(ev, dtype=np.float64))
        return ev_arr

    fallback_model.eval()
    with torch.no_grad():
        if isinstance(background, (tuple, list)):
            bg = tuple(b.to(device) for b in background)
            preds = fallback_model(*bg)
        else:
            preds = fallback_model(background.to(device))
    return np.atleast_1d(preds.detach().cpu().numpy().mean(axis=0))


def _predict(
    model: torch.nn.Module, inputs: Any, device: str
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if isinstance(inputs, (tuple, list)):
            xs = tuple(x.to(device) for x in inputs)
            out = model(*xs)
        else:
            out = model(inputs.to(device))
    return out.detach().cpu().numpy()


def _to_shap_input(x: Any) -> Any:
    """SHAP's DeepExplainer expects multi-input as a *list* (it special-cases
    ``isinstance(data, list)``). Tuples get wrapped and then collapsed into a
    single argument, which breaks multi-input models. Convert tuples to lists
    here; leave single tensors alone.
    """
    if isinstance(x, tuple):
        return list(x)
    return x


def _make_explainer(name: str, model: torch.nn.Module, background: Any):
    name_l = name.lower()
    bg = _to_shap_input(background)
    if name_l == "deep":
        return shap.DeepExplainer(model, bg)
    if name_l == "gradient":
        return shap.GradientExplainer(model, bg)
    raise ValueError(f"Unknown explainer: {name!r}")


def _shap_values(explainer, inputs: Any, check_additivity: bool):
    """Call an explainer, tolerating signature differences between versions."""
    inp = _to_shap_input(inputs)
    try:
        return explainer.shap_values(inp, check_additivity=check_additivity)
    except TypeError:
        return explainer.shap_values(inp)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_shap_for_run(
    cfg: ShapRunConfig,
    X: pd.DataFrame,
    dates: pd.DatetimeIndex,
    y_all: np.ndarray | None = None,
) -> dict:
    """Compute top-k-ensemble-averaged SHAP values for every selected (date, maturity).

    Parameters
    ----------
    cfg : ShapRunConfig
    X, dates :
        The same objects passed to ``expanding_window`` when the checkpoints were
        saved. For **revised** macro runs, ``X`` is the shifted FRED MD panel; for
        **realtime** runs, ``fred`` in ``X`` uses the same unshifted / placeholder
        convention as training --- SHAP then replaces the ``fred`` block at each
        OOS date via ``realtime=True`` (see ``cfg.realtime`` and
        ``macro_variant`` auto-detection).

    y_all :
        Optional. Reserved for future sanity checks against realised excess
        returns; not currently read by the runner.
    """
    del y_all  # reserved for future use
    t0 = time.time()
    run_dir = Path(cfg.orchestrator_run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Orchestrator run directory not found: {run_dir}")

    # ------------------------------------------------------------------ #
    # Load run artifacts                                                  #
    # ------------------------------------------------------------------ #
    with open(run_dir / "run_config.json") as fh:
        run_config = json.load(fh)

    use_realtime = _effective_realtime_flag(cfg.realtime, run_config)
    realtime_panel_cache: dict[pd.Timestamp, Any] = {}
    rt_store: Any | None = None
    if use_realtime:
        from .forecast_vintages import ForecastVintageMacroStore

        rt_store = cfg.realtime_store or ForecastVintageMacroStore()

    manifest = pd.read_csv(run_dir / "checkpoint_manifest.csv")
    manifest["date"] = pd.to_datetime(manifest["date"])
    manifest = manifest.sort_values(["seed", "t_index"]).reset_index(drop=True)

    topk_path, ens_path, ensemble_metric_used = _resolve_ensemble_npy_paths(
        run_dir, cfg.ensemble_metric
    )
    logger.info(
        "Using ensemble arrays metric=%s topk=%s ensemble=%s",
        ensemble_metric_used,
        topk_path.name,
        ens_path.name,
    )
    topk_indices = np.load(topk_path)  # (T, n_outputs, k)
    ensemble_forecast = np.load(ens_path)  # (T, n_outputs)

    # Peek at one checkpoint to get wrapper_class
    sample_ckpt_path = Path(manifest.iloc[0]["checkpoint_path"])
    if not sample_ckpt_path.is_absolute():
        sample_ckpt_path = (run_dir / ".." / ".." / ".." / sample_ckpt_path).resolve()
    if not sample_ckpt_path.exists():
        # fallback: reconstruct from seed / step
        sample_ckpt_path = _reconstruct_ckpt_path(run_dir, manifest.iloc[0])
    sample_ckpt = torch.load(sample_ckpt_path, map_location=cfg.device, weights_only=False)
    wrapper_class = sample_ckpt["wrapper_class"]
    adapter = get_adapter(wrapper_class)

    # ------------------------------------------------------------------ #
    # Resolve maturities / dates                                          #
    # ------------------------------------------------------------------ #
    all_mats = _coerce_run_config_maturities(run_config, ensemble_forecast)
    mats = [str(m) for m in (cfg.maturities or all_mats)]
    missing = [m for m in mats if m not in all_mats]
    if missing:
        raise ValueError(
            f"Requested maturities {missing} not in run's maturities {all_mats}."
        )
    mat_idx = {m: all_mats.index(m) for m in mats}

    oos_start = pd.Timestamp(run_config["oos_start"], unit="ms")
    oos_mask = dates >= oos_start
    oos_dates_full = dates[oos_mask]

    target_dates = _resolve_dates_selector(cfg.dates, oos_dates_full)
    if len(target_dates) == 0:
        raise ValueError("No target dates selected.")

    # ------------------------------------------------------------------ #
    # Output directory                                                    #
    # ------------------------------------------------------------------ #
    run_name = run_config.get("run_name", run_dir.parent.name)
    run_ts = run_dir.name
    out_root = Path(cfg.output_root).resolve()
    out_dir = _shap_output_base_dir(out_root, run_name, run_ts, ensemble_metric_used)
    per_date_dir = out_dir / "per_date"

    if cfg.overwrite and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    per_date_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Configure a simple file handler for this run (idempotent).
    log_path = out_dir / "logs" / "run.log"
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    logger.info(
        "Starting SHAP run: run_dir=%s, output=%s, wrapper=%s, n_dates=%d, "
        "maturities=%s, realtime=%s, ensemble_metric=%s",
        run_dir,
        out_dir,
        wrapper_class,
        len(target_dates),
        mats,
        use_realtime,
        ensemble_metric_used,
    )

    feature_names: list[str] | None = None

    # ------------------------------------------------------------------ #
    # Build lookups                                                       #
    # ------------------------------------------------------------------ #
    # Map (seed, t_index) -> checkpoint_path for fast lookup.
    ckpt_lookup: dict[tuple[int, int], Path] = {}
    for row in manifest.itertuples(index=False):
        p = Path(row.checkpoint_path)
        if not p.is_absolute():
            # Manifest stores whatever path was resolved at write time; the
            # safer bet is to reconstruct from convention.
            p = _reconstruct_ckpt_path(run_dir, row)
        ckpt_lookup[(int(row.seed), int(row.t_index))] = p

    date_to_t = pd.Series(
        data=np.arange(len(dates)), index=pd.DatetimeIndex(dates)
    )

    # With refit_freq=1 the checkpoint-bearing dates are exactly the OOS dates
    # where this seed was refit. For refit_freq>1 the model used at date t is
    # the most recent refit <= t. We resolve this per seed using the manifest.
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

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(cfg.seed)

    # Gather already-processed (date, maturity) pairs so we can resume without
    # redoing work. We look in (a) the per-date parquet files and (b) the
    # merged ``shap_mean.parquet`` from a prior successful run.
    done_by_date: dict[str, set[str]] = {}

    def _merge_done(date_tag: str, mat_set: set[str]) -> None:
        done_by_date.setdefault(date_tag, set()).update(mat_set)

    for p in per_date_dir.glob("*.parquet"):
        if p.stem.endswith(("__base", "__seedmeta", "__seeds")):
            continue
        try:
            sub = pd.read_parquet(p, columns=["maturity"])
        except Exception:
            continue
        _merge_done(p.stem, set(sub["maturity"].astype(str).unique()))

    merged_path = out_dir / "shap_mean.parquet"
    if merged_path.exists():
        prior = pd.read_parquet(merged_path, columns=["date", "maturity"]).drop_duplicates()
        prior["date"] = pd.to_datetime(prior["date"]).dt.strftime("%Y-%m-%d")
        for d, grp in prior.groupby("date"):
            _merge_done(d, set(grp["maturity"].astype(str)))

    additivity_errors: list[float] = []
    written_count = 0

    iterator = target_dates
    if cfg.progress:
        iterator = tqdm(target_dates, desc="SHAP dates", leave=False)

    for target_date in iterator:
        tag = target_date.strftime("%Y-%m-%d")
        already_have = done_by_date.get(tag, set())
        if not cfg.overwrite and set(mats).issubset(already_have):
            continue

        t_index = int(date_to_t.loc[target_date])
        hist_start = pd.Timestamp(dates[0])
        panel_eval_cached = None
        if use_realtime:
            fc_eval_d = pd.Timestamp(dates[t_index])
            panel_eval_cached = realtime_panel_cache.get(fc_eval_d)
            if panel_eval_cached is None:
                panel_eval_cached = rt_store.panel_for_forecast_date(
                    fc_eval_d, start=hist_start, end=fc_eval_d
                )
                realtime_panel_cache[fc_eval_d] = panel_eval_cached

        cols_this_date: pd.Index | None = None

        per_mat_rows = []
        per_mat_base = []
        per_mat_seed_rows: list[dict] = []
        per_mat_seed_meta: list[dict] = []

        for m in mats:
            m_idx = mat_idx[m]
            seeds_for_date = [
                int(s) for s in topk_indices[t_index, m_idx, :] if s != -1
            ]
            if not seeds_for_date:
                logger.warning(
                    "No top-k seeds for date=%s maturity=%s; skipping.", tag, m
                )
                continue

            per_seed_shap: list[np.ndarray] = []
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
                train_end = expanding_window_train_end(refit_t, gap)

                if use_realtime:
                    fc_fit = pd.Timestamp(dates[refit_t])
                    panel_fit = realtime_panel_cache.get(fc_fit)
                    if panel_fit is None:
                        panel_fit = rt_store.panel_for_forecast_date(
                            fc_fit, start=hist_start, end=fc_fit
                        )
                        realtime_panel_cache[fc_fit] = panel_fit
                    X_fit_trunc = _realtime_feature_frame(
                        X, panel_fit.transformed, refit_t
                    )
                    X_fit_proc, cols = _preprocess_X_like_expanding_window_refit(
                        X_fit_trunc, train_end
                    )
                    X_train = X_fit_proc.iloc[:train_end]

                    assert panel_eval_cached is not None
                    X_eval_trunc = _realtime_feature_frame(
                        X, panel_eval_cached.transformed, t_index
                    )
                    X_eval_sel = _apply_column_selection(X_eval_trunc, cols)
                    X_eval_proc = _carry_forward_latest(X_eval_sel)
                    X_row = X_eval_proc.iloc[[t_index]]
                else:
                    X_proc, cols = _preprocess_X_like_expanding_window_refit(
                        X, train_end
                    )
                    X_train = X_proc.iloc[:train_end]
                    X_row = X_proc.iloc[[t_index]]

                if cols_this_date is None:
                    cols_this_date = cols
                    tmpl = X.iloc[0:1].loc[:, cols_this_date]
                    feature_names = adapter.feature_names(tmpl, ckpt=sample_ckpt)
                elif not cols_this_date.equals(cols):
                    raise ValueError(
                        "SHAP column selection differs across top-k seeds on "
                        f"{tag}; cannot merge attributions. "
                        "(Often refit_freq>1 with misaligned checkpoints.)"
                    )
                assert feature_names is not None

                background = adapter.sample_background(
                    X_train, ckpt, cfg.background_size, rng
                )
                inputs = adapter.prepare_inputs(X_row, ckpt)

                explainer = _make_explainer(cfg.explainer, model, background)
                shap_out = _shap_values(explainer, inputs, cfg.check_additivity)
                shap_flat = adapter.flatten_shap(
                    shap_out, feature_names, m_idx=m_idx
                )  # (1, F) — already sliced to output head m_idx

                scale, shift = adapter.y_scale_and_shift(ckpt, cfg.y_center_default)
                # Single output: use the m_idx'th component of scale/shift.
                y_scale = float(scale[m_idx]) if scale.size > m_idx else float(scale[0])
                y_shift = float(shift[m_idx]) if shift.size > m_idx else float(shift[0])

                base_scaled = _expected_value(explainer, model, background, cfg.device)
                if base_scaled.size > 1:
                    base_scaled_m = float(base_scaled[m_idx])
                else:
                    base_scaled_m = float(base_scaled[0])

                # flatten_shap already returned the m_idx-th output's slice,
                # shape (1, F). Take the single row.
                shap_seed = shap_flat[0]

                pred_scaled = _predict(model, inputs, cfg.device).flatten()
                if pred_scaled.size > 1:
                    pred_scaled_m = float(pred_scaled[m_idx])
                else:
                    pred_scaled_m = float(pred_scaled[0])

                if cfg.apply_y_scaling:
                    shap_seed = shap_seed * y_scale
                    base_seed = base_scaled_m * y_scale + y_shift
                    pred_seed = pred_scaled_m * y_scale + y_shift
                else:
                    base_seed = base_scaled_m
                    pred_seed = pred_scaled_m

                additivity = abs(pred_seed - (base_seed + float(np.sum(shap_seed))))
                additivity_errors.append(additivity)

                per_seed_shap.append(shap_seed)
                per_seed_pred.append(pred_seed)
                per_seed_base.append(base_seed)
                per_seed_additivity.append(additivity)

                if cfg.save_per_seed_meta:
                    per_mat_seed_meta.append({
                        "date": target_date,
                        "maturity": m,
                        "seed": int(seed),
                        "pred": float(pred_seed),
                        "base": float(base_seed),
                        "sum_shap": float(np.sum(shap_seed)),
                        "additivity_residual": float(additivity),
                    })

                if cfg.save_per_seed:
                    for f_idx, f_name in enumerate(feature_names):
                        per_mat_seed_rows.append({
                            "date": target_date,
                            "maturity": m,
                            "seed": int(seed),
                            "feature": f_name,
                            "shap_value": float(shap_seed[f_idx]),
                        })

            stacked = np.stack(per_seed_shap, axis=0)  # (k, F)
            mean_shap = stacked.mean(axis=0)
            std_shap = stacked.std(axis=0, ddof=0) if stacked.shape[0] > 1 else np.zeros_like(mean_shap)

            ensemble_pred_from_shap = float(np.mean(per_seed_pred))
            base_value_avg = float(np.mean(per_seed_base))

            for f_idx, f_name in enumerate(feature_names):
                per_mat_rows.append({
                    "date": target_date,
                    "maturity": m,
                    "feature": f_name,
                    "mean_shap": float(mean_shap[f_idx]),
                    "abs_mean_shap": float(np.abs(mean_shap[f_idx])),
                    "std_shap": float(std_shap[f_idx]),
                    "n_seeds": int(stacked.shape[0]),
                })

            per_mat_base.append({
                "date": target_date,
                "maturity": m,
                "base_value": base_value_avg,
                "ensemble_pred": ensemble_pred_from_shap,
                "orchestrator_ensemble_pred": float(ensemble_forecast[t_index, m_idx]),
                "n_seeds": int(stacked.shape[0]),
            })

        if not per_mat_rows:
            continue

        shap_df = pd.DataFrame(per_mat_rows)
        base_df = pd.DataFrame(per_mat_base)

        out_path = per_date_dir / f"{tag}.parquet"
        shap_df.to_parquet(out_path, index=False)
        base_df.to_parquet(per_date_dir / f"{tag}__base.parquet", index=False)

        if cfg.save_per_seed_meta and per_mat_seed_meta:
            pd.DataFrame(per_mat_seed_meta).to_parquet(
                per_date_dir / f"{tag}__seedmeta.parquet", index=False
            )
        if cfg.save_per_seed and per_mat_seed_rows:
            pd.DataFrame(per_mat_seed_rows).to_parquet(
                per_date_dir / f"{tag}__seeds.parquet", index=False
            )

        written_count += 1

    # ------------------------------------------------------------------ #
    # Merge per-date files                                                #
    # ------------------------------------------------------------------ #
    shap_files = sorted(per_date_dir.glob("*.parquet"))
    base_main = [p for p in shap_files if p.stem.endswith("__base")]
    seedmeta_main = [p for p in shap_files if p.stem.endswith("__seedmeta")]
    seeds_main = [p for p in shap_files if p.stem.endswith("__seeds")]
    aux_suffixes = ("__base", "__seedmeta", "__seeds")
    shap_main = [p for p in shap_files if not p.stem.endswith(aux_suffixes)]

    if shap_main:
        merged = pd.concat([pd.read_parquet(p) for p in shap_main], ignore_index=True)
        merged = merged.sort_values(["date", "maturity", "feature"]).reset_index(drop=True)
        merged[["mean_shap", "abs_mean_shap", "std_shap"]] = merged[[
            "mean_shap", "abs_mean_shap", "std_shap"
        ]].astype(float)

        merged_mean = merged[[
            "date", "maturity", "feature", "mean_shap", "abs_mean_shap", "n_seeds"
        ]]
        merged_std = merged[["date", "maturity", "feature", "std_shap"]]
        merged_mean.to_parquet(out_dir / "shap_mean.parquet", index=False)
        merged_std.to_parquet(out_dir / "shap_std.parquet", index=False)

    if base_main:
        merged_base = pd.concat([pd.read_parquet(p) for p in base_main], ignore_index=True)
        merged_base = merged_base.sort_values(["date", "maturity"]).reset_index(drop=True)
        merged_base.to_parquet(out_dir / "base_values.parquet", index=False)

    if seedmeta_main:
        merged_meta = pd.concat(
            [pd.read_parquet(p) for p in seedmeta_main], ignore_index=True
        )
        merged_meta = merged_meta.sort_values(
            ["date", "maturity", "seed"]
        ).reset_index(drop=True)
        merged_meta.to_parquet(out_dir / "per_seed_meta.parquet", index=False)

    if seeds_main:
        merged_seeds = pd.concat(
            [pd.read_parquet(p) for p in seeds_main], ignore_index=True
        )
        merged_seeds = merged_seeds.sort_values(
            ["date", "maturity", "seed", "feature"]
        ).reset_index(drop=True)
        merged_seeds.to_parquet(out_dir / "per_seed_shap.parquet", index=False)

    if feature_names is not None:
        with open(out_dir / "feature_names.json", "w") as fh:
            json.dump(feature_names, fh, indent=2)

    meta = {
        "orchestrator_run_dir": str(run_dir),
        "orchestrator_run_name": run_name,
        "orchestrator_run_timestamp": run_ts,
        "wrapper_class": wrapper_class,
        "maturities": mats,
        "n_dates": int(len(target_dates)),
        "n_dates_written_this_call": int(written_count),
        "explainer": cfg.explainer,
        "background_size": cfg.background_size,
        "background_sampling": cfg.background_sampling,
        "apply_y_scaling": cfg.apply_y_scaling,
        "y_center_default": cfg.y_center_default,
        "device": cfg.device,
        "seed": cfg.seed,
        "check_additivity": cfg.check_additivity,
        "additivity_error_mean": float(np.mean(additivity_errors)) if additivity_errors else None,
        "additivity_error_max": float(np.max(additivity_errors)) if additivity_errors else None,
        "macro_variant": run_config.get("macro_variant"),
        "realtime_shap": use_realtime,
        "ensemble_metric": ensemble_metric_used,
        "orchestrator_topk_indices_path": str(topk_path),
        "orchestrator_ensemble_forecast_path": str(ens_path),
        "config": asdict(_serialisable_cfg(cfg)),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "shap_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    if not cfg.keep_per_date:
        for p in per_date_dir.glob("*.parquet"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            per_date_dir.rmdir()
        except OSError:
            pass

    elapsed = time.time() - t0
    logger.info("Done in %.1fs. Output: %s", elapsed, out_dir)

    return {
        "output_dir": str(out_dir),
        "n_dates_total": int(len(target_dates)),
        "n_dates_written_this_call": int(written_count),
        "n_dates_already_present": int(len(done_by_date)),
        "elapsed_s": round(elapsed, 2),
        "additivity_error_mean": meta["additivity_error_mean"],
        "additivity_error_max": meta["additivity_error_max"],
        "feature_names_head": (feature_names[:5] if feature_names else []),
    }


# ---------------------------------------------------------------------------
# Small private helpers
# ---------------------------------------------------------------------------


def _serialisable_cfg(cfg: ShapRunConfig) -> ShapRunConfig:
    """Return a copy of the config with Path objects stringified for JSON."""
    d = asdict(cfg)
    d["realtime_store"] = None
    return ShapRunConfig(
        **{
            **d,
            "orchestrator_run_dir": str(cfg.orchestrator_run_dir),
            "output_root": str(cfg.output_root),
            "dates": cfg.dates
            if isinstance(cfg.dates, (str, int, float))
            else [str(x) for x in cfg.dates],
        }
    )


def _reconstruct_ckpt_path(run_dir: Path, row) -> Path:
    """Given a manifest row with seed / t_index / date, rebuild the on-disk path
    using the orchestrator's naming convention. Accepts either a namedtuple
    (``itertuples``) or a pandas Series.
    """
    def _get(key):
        return getattr(row, key) if hasattr(row, key) else row[key]

    seed = int(_get("seed"))
    t_index = int(_get("t_index"))
    date = pd.Timestamp(_get("date")).date()
    return (
        run_dir / "checkpoints" / f"seed_{seed:03d}"
        / f"step_{t_index:04d}_{date}.pt"
    )
