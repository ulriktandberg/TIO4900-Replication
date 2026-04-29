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

Output layout mirrors the orchestrator: ``artifacts/shap/<run_name>/<timestamp>/``.
Results are written incrementally per (date, maturity) batch so a long run can be
resumed after interruption.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import shap  # type: ignore

from .shap_adapters import get_adapter


logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        Exactly the objects passed to ``run_experiment`` when the checkpoints
        were produced. ``X`` must have the same columns and row index as then,
        otherwise the saved scaler/PCA state will either dim-mismatch or produce
        attributions on out-of-distribution inputs.
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

    manifest = pd.read_csv(run_dir / "checkpoint_manifest.csv")
    manifest["date"] = pd.to_datetime(manifest["date"])
    manifest = manifest.sort_values(["seed", "t_index"]).reset_index(drop=True)

    topk_indices = np.load(run_dir / "topk_indices.npy")  # (T, n_outputs, k)
    ensemble_forecast = np.load(run_dir / "ensemble_forecast.npy")  # (T, n_outputs)

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
    all_mats = _maturity_columns(run_config)
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
    out_dir = out_root / run_name / run_ts
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
        "Starting SHAP run: run_dir=%s, wrapper=%s, n_dates=%d, maturities=%s",
        run_dir, wrapper_class, len(target_dates), mats,
    )

    feature_names = adapter.feature_names(X, ckpt=sample_ckpt)

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
        X_row = X.iloc[[t_index]]

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

                # Training window used at that refit (gap=0 in the runs we have).
                gap = int(run_config.get("gap", 0))
                train_end = max(refit_t - gap, 0)
                X_train = X.iloc[:train_end]

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
        "feature_names_head": feature_names[:5],
    }


# ---------------------------------------------------------------------------
# Small private helpers
# ---------------------------------------------------------------------------


def _serialisable_cfg(cfg: ShapRunConfig) -> ShapRunConfig:
    """Return a copy of the config with Path objects stringified for JSON."""
    return ShapRunConfig(
        **{**asdict(cfg),
           "orchestrator_run_dir": str(cfg.orchestrator_run_dir),
           "output_root": str(cfg.output_root),
           "dates": cfg.dates if isinstance(cfg.dates, (str, int, float))
                   else [str(d) for d in cfg.dates]}
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
