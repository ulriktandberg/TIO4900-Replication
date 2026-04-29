#!/usr/bin/env python3
"""Solstrom HPC training CLI for core replication models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.base_utils as bu
import utils.window_utils as wu
from models.base import PCABaselineModel, PCABaselineModelPlusN
from models.classical import CochranePiazzesiModel
from models.gbt import LightGBMModel, XGBoostModel, lgb, xgb
from utils.macro_grouping import add_group_level, build_full_group_mapping
from utils.publication_lags import apply_fred_md_publication_lag


FAMILY_CHOICES = (
    "smoke",
    "baselines",
    "trees",
    "xgboost",
    "lightgbm",
    "fwd_ann",
    "macro_forward_ann",
    "group_ensemble_ann",
    "model_configs",
    "all",
)

MASTER_FIELDS = (
    "timestamp",
    "run_id",
    "family",
    "job_name",
    "status",
    "run_dir",
    "performance",
    "error",
)


@dataclass(frozen=True)
class PreparedData:
    X: pd.DataFrame
    y_all: np.ndarray
    dates: pd.DatetimeIndex
    maturities: list[str]


@dataclass(frozen=True)
class JobSpec:
    family: str
    job_name: str
    runner: Callable[[PreparedData | None, Path, argparse.Namespace], dict[str, Any]]
    uses_shared_data: bool = True


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_maturities(raw: str) -> list[str]:
    maturities = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        maturity = int(item)
        if maturity <= 12:
            raise argparse.ArgumentTypeError(
                "Excess-return target maturities must be greater than 12 months."
            )
        maturities.append(str(maturity))
    if not maturities:
        raise argparse.ArgumentTypeError("At least one maturity is required.")
    return maturities


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _performance(
    y_true: np.ndarray,
    forecast: np.ndarray,
    maturities: list[str],
    gap: int = 0,
) -> list[dict[str, float]]:
    r2s = wu.oos_r2(y_true, forecast, benchmark="hist_mean", gap=gap)
    if np.ndim(r2s) == 0:
        r2s = np.array([r2s])

    rows = []
    for idx, maturity in enumerate(maturities):
        try:
            pval = bu.RSZ_Signif(y_true[:, idx], forecast[:, idx], gap=gap)
        except Exception:
            pval = np.nan
        rows.append(
            {
                "maturity": int(maturity),
                "r2_oos": float(r2s[idx]) if not pd.isna(r2s[idx]) else np.nan,
                "rsz_pval": float(pval) if not pd.isna(pval) else np.nan,
            }
        )
    return rows


def _save_job_outputs(
    job_dir: Path,
    job_name: str,
    family: str,
    forecast: np.ndarray,
    data: PreparedData,
    args: argparse.Namespace,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    job_dir.mkdir(parents=True, exist_ok=True)
    perf = _performance(data.y_all, forecast, data.maturities, gap=args.gap)

    np.save(job_dir / "forecast.npy", forecast)
    np.save(job_dir / "y_all.npy", data.y_all)
    pd.DataFrame(forecast, index=data.dates, columns=data.maturities).to_csv(
        job_dir / "forecast.csv"
    )
    pd.Series(data.dates.astype(str), name="date").to_csv(job_dir / "dates.csv", index=False)
    (job_dir / "maturities.json").write_text(json.dumps(data.maturities, indent=2))
    pd.DataFrame(perf).to_csv(job_dir / "performance.csv", index=False)

    config = {
        "job_name": job_name,
        "family": family,
        "maturities": data.maturities,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "yield_type": args.yield_type,
        "oos_start": args.oos_start,
        "gap": args.gap,
    }
    if extra_config:
        config.update(extra_config)
    (job_dir / "job_config.json").write_text(json.dumps(_jsonable(config), indent=2))

    return {
        "run_dir": str(job_dir.resolve()),
        "performance": perf,
        "forecast_shape": forecast.shape,
    }


def _append_master_row(master_path: Path, row: dict[str, Any]) -> None:
    master_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = master_path.with_suffix(master_path.suffix + ".lock")

    import fcntl

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        exists = master_path.exists() and master_path.stat().st_size > 0
        with master_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=MASTER_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in MASTER_FIELDS})
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


_PREPARED_DATA_CACHE: dict[tuple[Any, ...], PreparedData] = {}


def _prepare_data(
    args: argparse.Namespace,
    *,
    yield_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    maturities: list[str] | None = None,
    macro_variant: str = "revised",
) -> PreparedData:
    yearly_maturities = [str(i) for i in range(12, 121, 12)]
    target_maturities = maturities or args.maturities
    yield_type = yield_type or args.yield_type
    start_date = start_date or args.start_date
    end_date = end_date or args.end_date
    cache_key = (yield_type, start_date, end_date, tuple(target_maturities), macro_variant)
    if cache_key in _PREPARED_DATA_CACHE:
        return _PREPARED_DATA_CACHE[cache_key]

    print(
        "Preparing yield, forward-rate, excess-return, and macro panels "
        f"({yield_type=}, {start_date=}, {end_date=}, {macro_variant=})..."
    )
    yields = bu.get_yields(
        type=yield_type,
        start=start_date,
        end=end_date,
        maturities=yearly_maturities,
    )
    forward = bu.get_forward_rates(yields)
    excess_returns = bu.get_excess_returns(yields, horizon=12).dropna()

    missing_targets = [m for m in target_maturities if m not in excess_returns.columns]
    if missing_targets:
        raise ValueError(
            f"Requested target maturities are not available as annual excess returns: {missing_targets}"
        )

    if macro_variant == "revised":
        fred_raw = bu.get_fred_data("data/2026-01-MD.csv", start=start_date, end=end_date)
        fred = apply_fred_md_publication_lag(fred_raw)
    elif macro_variant == "realtime":
        fred = bu.get_realtime_fred_data(start=start_date, end=end_date)
    else:
        raise ValueError(f"Unknown macro_variant: {macro_variant}")
    fred = bu.prepare_macro_panel_for_project(fred)
    fred = fred.ffill().bfill()

    final_index = excess_returns.index
    yields = yields.reindex(final_index)
    forward = forward.reindex(final_index)
    fred = fred.reindex(final_index).ffill().bfill()

    series_to_group = build_full_group_mapping(fred, forward, yields)
    X = pd.concat([fred, forward, yields], axis=1, keys=["fred", "forward", "yields"])
    X = add_group_level(X, series_to_group, level_name="group")
    X = X.sort_index(axis=1, level="group")

    y_all = excess_returns.loc[final_index, target_maturities].values
    print(
        "Prepared panel: "
        f"{len(final_index)} dates, {X.shape[1]} features, "
        f"{len(target_maturities)} target maturities."
    )
    prepared = PreparedData(X=X, y_all=y_all, dates=final_index, maturities=target_maturities)
    _PREPARED_DATA_CACHE[cache_key] = prepared
    return prepared


def _run_multioutput_window_job(
    family: str,
    job_name: str,
    model: Any,
    data: PreparedData,
    run_root: Path,
    args: argparse.Namespace,
    refit_freq: int = 1,
) -> dict[str, Any]:
    job_dir = run_root / "jobs" / job_name / _run_timestamp()
    forecast = wu.expanding_window(
        model,
        data.X,
        data.y_all,
        data.dates,
        pd.Timestamp(args.oos_start),
        gap=args.gap,
        refit_freq=refit_freq,
        progress=True,
        tqdm_desc=job_name,
    )
    return _save_job_outputs(
        job_dir,
        job_name,
        family,
        forecast,
        data,
        args,
        extra_config={"model_class": model.__class__.__name__, "refit_freq": refit_freq},
    )


def _run_single_target_window_job(
    family: str,
    job_name: str,
    model_builder: Callable[[int], Any],
    data: PreparedData,
    run_root: Path,
    args: argparse.Namespace,
    maturities: Iterable[str] | None = None,
    refit_freq: int = 1,
) -> dict[str, Any]:
    selected = list(maturities or data.maturities)
    selected_indices = [data.maturities.index(m) for m in selected]
    forecast = np.full_like(data.y_all, np.nan, dtype=float)

    for maturity, idx in zip(selected, selected_indices):
        sub_job = f"{job_name}_{maturity}m"
        print(f"  maturity {maturity}m")
        y = data.y_all[:, idx]
        pred = wu.expanding_window(
            model_builder(int(maturity)),
            data.X,
            y,
            data.dates,
            pd.Timestamp(args.oos_start),
            gap=args.gap,
            refit_freq=refit_freq,
            progress=True,
            tqdm_desc=sub_job,
        )
        forecast[:, idx] = pred

    job_dir = run_root / "jobs" / job_name / _run_timestamp()
    return _save_job_outputs(
        job_dir,
        job_name,
        family,
        forecast,
        data,
        args,
        extra_config={
            "single_target_maturities": selected,
            "refit_freq": refit_freq,
        },
    )


def _run_ann_job(
    family: str,
    job_name: str,
    model_builder: Callable[[int], Any],
    data: PreparedData,
    run_root: Path,
    args: argparse.Namespace,
    *,
    n_models: int | None = None,
    k_top: int | None = None,
    refit_freq: int = 1,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from utils.orchestrator_utils import RunConfig, run_experiment

    if data is None:
        raise ValueError("ANN jobs require prepared data.")
    actual_n_models = int(n_models if n_models is not None else args.n_models)
    actual_k_top = min(int(k_top if k_top is not None else args.k_top), actual_n_models)
    cfg = RunConfig(
        run_name=job_name,
        model_builder=model_builder,
        n_models=actual_n_models,
        k_top=actual_k_top,
        maturities=data.maturities,
        oos_start=pd.Timestamp(args.oos_start),
        gap=args.gap,
        refit_freq=refit_freq,
        benchmark="hist_mean",
        progress=True,
        save_checkpoints=bool(args.save_checkpoints),
        artifacts_root=run_root / "jobs",
        ensemble_metrics=("val_loss", "trailing_oos"),
        ensemble_selection_mode="per_maturity",
        trailing_lookback=args.trailing_lookback,
        trailing_min_history=args.trailing_min_history,
    )
    summary = run_experiment(cfg, data.X, data.y_all, data.dates)
    run_dir = Path(summary["run_dir"])
    np.save(run_dir / "y_all.npy", data.y_all)
    pd.Series(data.dates.astype(str), name="date").to_csv(run_dir / "dates.csv", index=False)
    (run_dir / "maturities.json").write_text(json.dumps(data.maturities, indent=2))
    if extra_config:
        (run_dir / "model_config.json").write_text(json.dumps(_jsonable(extra_config), indent=2))
    return summary


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _tree_n_jobs() -> int:
    return _env_int("TIO4900_TREE_N_JOBS", _env_int("OMP_NUM_THREADS", 1))


def _xgb_grid(tuning_level: str) -> list[dict[str, Any]]:
    light = [
        {
            "max_depth": 2,
            "n_estimators": 80,
            "learning_rate": 0.10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
    ]
    if tuning_level == "light":
        return light
    return light + [
        {
            "max_depth": 2,
            "n_estimators": 250,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
        {
            "max_depth": 3,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 2.0,
        },
        {
            "max_depth": 4,
            "n_estimators": 200,
            "learning_rate": 0.04,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.01,
            "reg_lambda": 3.0,
        },
    ]


def _lgbm_grid(tuning_level: str) -> list[dict[str, Any]]:
    light = [
        {
            "num_leaves": 15,
            "max_depth": 3,
            "n_estimators": 80,
            "learning_rate": 0.10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_data_in_leaf": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        }
    ]
    if tuning_level == "light":
        return light
    return light + [
        {
            "num_leaves": 15,
            "max_depth": 3,
            "n_estimators": 250,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_data_in_leaf": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        {
            "num_leaves": 31,
            "max_depth": -1,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "min_data_in_leaf": 30,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
        },
        {
            "num_leaves": 63,
            "max_depth": -1,
            "n_estimators": 200,
            "learning_rate": 0.04,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_data_in_leaf": 50,
            "reg_alpha": 0.01,
            "reg_lambda": 2.0,
        },
    ]


def _gbt_features() -> dict[str, dict[str, Any]]:
    return {
        "forward": {"method": "raw"},
        "fred": {"method": "pca", "n_components": 8},
    }


def _build_baseline_jobs() -> list[JobSpec]:
    return [
        JobSpec(
            family="baselines",
            job_name="cochrane_piazzesi",
            runner=lambda data, root, args: _run_multioutput_window_job(
                "baselines", "cochrane_piazzesi", CochranePiazzesiModel(), data, root, args
            ),
        ),
        JobSpec(
            family="baselines",
            job_name="pca_forward_3",
            runner=lambda data, root, args: _run_multioutput_window_job(
                "baselines",
                "pca_forward_3",
                PCABaselineModel(components=3, series="forward"),
                data,
                root,
                args,
            ),
        ),
        JobSpec(
            family="baselines",
            job_name="pca_forward_macro_pc1",
            runner=lambda data, root, args: _run_multioutput_window_job(
                "baselines",
                "pca_forward_macro_pc1",
                PCABaselineModelPlusN(components=3, series="forward", n_extra=1),
                data,
                root,
                args,
            ),
        ),
    ]


def _build_tree_jobs(only: str | None = None) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    if only in {None, "xgboost"} and xgb is not None:
        jobs.append(
            JobSpec(
                family="trees",
                job_name="xgboost_forward_fred",
                runner=lambda data, root, args: _run_single_target_window_job(
                    "trees",
                    "xgboost_forward_fred",
                    lambda seed: XGBoostModel(
                        features=_gbt_features(),
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=seed,
                        n_jobs=_tree_n_jobs(),
                        arch_grid=_xgb_grid(args.tuning_level),
                        tune_every=60,
                        impute_strategy="median",
                    ),
                    data,
                    root,
                    args,
                ),
            )
        )
    elif only in {None, "xgboost"}:
        print("Skipping XGBoost tree job: xgboost is not installed.")

    if only in {None, "lightgbm"} and lgb is not None:
        jobs.append(
            JobSpec(
                family="trees",
                job_name="lightgbm_forward_fred",
                runner=lambda data, root, args: _run_single_target_window_job(
                    "trees",
                    "lightgbm_forward_fred",
                    lambda seed: LightGBMModel(
                        features=_gbt_features(),
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        num_leaves=15,
                        random_state=seed,
                        n_jobs=_tree_n_jobs(),
                        force_row_wise=True,
                        verbose=-1,
                        arch_grid=_lgbm_grid(args.tuning_level),
                        tune_every=60,
                        impute_strategy="median",
                    ),
                    data,
                    root,
                    args,
                ),
            )
        )
    elif only in {None, "lightgbm"}:
        print("Skipping LightGBM tree job: lightgbm is not installed.")
    return jobs


def _fwd_ann_builder(epochs: int, patience: int, penalties: list[float]) -> Callable[[int], Any]:
    def build(seed: int) -> Any:
        from models.ann_vector_validation import PyTorchMLPWrapper

        return PyTorchMLPWrapper(
            archi=(3,),
            lr=0.01,
            epochs=epochs,
            tune_every=60,
            patience=patience,
            param_grid={"penalty": penalties},
            seed=seed,
            use_pca=False,
            n_components=None,
            y_center=True,
        )

    return build


def _macro_forward_ann_builder(
    epochs: int,
    patience: int,
    penalties: list[float],
    dropout_rates: list[float],
) -> Callable[[int], Any]:
    def build(seed: int) -> Any:
        from models.macro_forward_ann import MacroForwardANNWrapper

        return MacroForwardANNWrapper(
            archi_forward=(3,),
            archi_macro=(16, 8),
            lr=0.01,
            epochs=epochs,
            tune_every=60,
            patience=patience,
            param_grid={"penalty": penalties, "dropout_rate": dropout_rates},
            seed=seed,
            y_center=True,
        )

    return build


def _group_ensemble_ann_builder(
    epochs: int,
    patience: int,
    penalties: list[float],
    dropout_rates: list[float],
) -> Callable[[int], Any]:
    def build(seed: int) -> Any:
        from models.group_ensemble_ann import GroupEnsembleANNWrapper

        return GroupEnsembleANNWrapper(
            archi_forward=(3,),
            archi_macro=(16, 8),
            lr=0.01,
            epochs=epochs,
            tune_every=60,
            patience=patience,
            param_grid={"penalty": penalties, "dropout_rate": dropout_rates},
            seed=seed,
            y_center=True,
        )

    return build


def _build_ann_jobs() -> list[JobSpec]:
    return [
        JobSpec(
            family="fwd_ann",
            job_name="fwd_ann_3",
            runner=lambda data, root, args: _run_ann_job(
                "fwd_ann",
                "fwd_ann_3",
                _fwd_ann_builder(epochs=1000, patience=50, penalties=[0.01, 0.001, 0.0001]),
                data,
                root,
                args,
            ),
        ),
        JobSpec(
            family="macro_forward_ann",
            job_name="macro_forward_ann_fwd3_macro16_8",
            runner=lambda data, root, args: _run_ann_job(
                "macro_forward_ann",
                "macro_forward_ann_fwd3_macro16_8",
                _macro_forward_ann_builder(
                    epochs=1000,
                    patience=50,
                    penalties=[0.001, 0.0001],
                    dropout_rates=[0.0, 0.1, 0.3],
                ),
                data,
                root,
                args,
            ),
        ),
        JobSpec(
            family="group_ensemble_ann",
            job_name="group_ensemble_ann_fwd3_macro16_8",
            runner=lambda data, root, args: _run_ann_job(
                "group_ensemble_ann",
                "group_ensemble_ann_fwd3_macro16_8",
                _group_ensemble_ann_builder(
                    epochs=1000,
                    patience=50,
                    penalties=[0.001, 0.0001],
                    dropout_rates=[0.0, 0.1, 0.3],
                ),
                data,
                root,
                args,
            ),
        ),
    ]


def _load_model_configs() -> list[dict[str, Any]]:
    from models.model_configs.fred_and_realtime_models import models as configured_models

    return [dict(config) for config in configured_models]


def _config_model_builder(config: dict[str, Any]) -> Callable[[int], Any]:
    model_family = config["model_family"]

    def build(seed: int) -> Any:
        common = {
            "archi_forward": tuple(config.get("archi_forward", (3,))),
            "archi_macro": tuple(config.get("archi_macro", (16, 8))),
            "lr": config.get("lr", 0.01),
            "epochs": config.get("epochs", 1000),
            "tune_every": config.get("tune_every", 60),
            "patience": config.get("patience", 50),
            "param_grid": config.get("param_grid"),
            "seed": seed,
            "y_center": config.get("y_center", True),
            "activation": config.get("activation", "relu"),
        }

        if model_family == "macro_forward":
            from models.macro_forward_ann import MacroForwardANNWrapper

            return MacroForwardANNWrapper(**common)
        if model_family == "group_ensemble":
            from models.group_ensemble_ann import GroupEnsembleANNWrapper

            return GroupEnsembleANNWrapper(**common)
        raise ValueError(f"Unsupported configured model_family: {model_family}")

    return build


def _run_model_config_job(
    config: dict[str, Any],
    _data: PreparedData | None,
    run_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    job_args = argparse.Namespace(**vars(args))
    job_args.yield_type = config.get("yield_type", args.yield_type)
    job_args.start_date = config.get("start_date", args.start_date)
    job_args.end_date = config.get("end_date", args.end_date)
    job_args.oos_start = config.get("oos_start", args.oos_start)
    job_args.gap = int(config.get("gap", args.gap))

    data = _prepare_data(
        job_args,
        yield_type=job_args.yield_type,
        start_date=job_args.start_date,
        end_date=job_args.end_date,
        maturities=args.maturities,
        macro_variant=config.get("macro_variant", "revised"),
    )
    return _run_ann_job(
        "model_configs",
        config["run_name"],
        _config_model_builder(config),
        data,
        run_root,
        job_args,
        n_models=args.n_models,
        k_top=args.k_top,
        refit_freq=int(config.get("refit_freq", 1)),
        extra_config=config,
    )


def _parse_config_names(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _parse_config_shard(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    try:
        index_raw, total_raw = raw.split("/", 1)
        index = int(index_raw)
        total = int(total_raw)
    except ValueError as exc:
        raise ValueError("--config-shard must use INDEX/TOTAL format, for example 0/3.") from exc
    if total < 1 or index < 0 or index >= total:
        raise ValueError("--config-shard requires 0 <= INDEX < TOTAL.")
    return index, total


def _build_model_config_jobs(args: argparse.Namespace) -> list[JobSpec]:
    names = _parse_config_names(args.config_names)
    shard = _parse_config_shard(args.config_shard)
    configs = _load_model_configs()
    jobs = []

    for idx, config in enumerate(configs):
        run_name = config["run_name"]
        if names is not None and run_name not in names:
            continue
        if shard is not None and idx % shard[1] != shard[0]:
            continue
        jobs.append(
            JobSpec(
                family="model_configs",
                job_name=run_name,
                runner=lambda data, root, args, cfg=config: _run_model_config_job(
                    cfg,
                    data,
                    root,
                    args,
                ),
                uses_shared_data=False,
            )
        )
    return jobs


def _build_smoke_jobs() -> list[JobSpec]:
    return [
        JobSpec(
            family="smoke",
            job_name="smoke_cochrane_piazzesi",
            runner=lambda data, root, args: _run_multioutput_window_job(
                "smoke",
                "smoke_cochrane_piazzesi",
                CochranePiazzesiModel(),
                data,
                root,
                args,
                refit_freq=120,
            ),
        ),
        JobSpec(
            family="smoke",
            job_name="smoke_xgboost_first_maturity",
            runner=lambda data, root, args: _run_single_target_window_job(
                "smoke",
                "smoke_xgboost_first_maturity",
                lambda seed: XGBoostModel(
                    features={"forward": {"method": "raw"}},
                    n_estimators=25,
                    max_depth=2,
                    learning_rate=0.10,
                    random_state=seed,
                    arch_grid=[],
                    tune_every=10_000,
                    impute_strategy="median",
                ),
                data,
                root,
                args,
                maturities=data.maturities[:1],
                refit_freq=120,
            ),
        )
        if xgb is not None
        else JobSpec(
            family="smoke",
            job_name="smoke_pca_forward_3",
            runner=lambda data, root, args: _run_multioutput_window_job(
                "smoke",
                "smoke_pca_forward_3",
                PCABaselineModel(components=3, series="forward"),
                data,
                root,
                args,
                refit_freq=120,
            ),
        ),
        JobSpec(
            family="smoke",
            job_name="smoke_fwd_ann_3",
            runner=lambda data, root, args: _run_ann_job(
                "smoke",
                "smoke_fwd_ann_3",
                _fwd_ann_builder(epochs=20, patience=3, penalties=[0.001]),
                data,
                root,
                args,
                n_models=min(args.n_models, 2),
                k_top=min(args.k_top, min(args.n_models, 2)),
                refit_freq=120,
            ),
        ),
    ]


def _select_jobs(args: argparse.Namespace) -> list[JobSpec]:
    selected: list[JobSpec] = []
    expanded = []
    for family in args.families:
        if family == "all":
            expanded.extend(
                ["baselines", "trees", "fwd_ann", "macro_forward_ann", "group_ensemble_ann"]
            )
        else:
            expanded.append(family)

    for family in dict.fromkeys(expanded):
        if family == "smoke":
            selected.extend(_build_smoke_jobs())
        elif family == "baselines":
            selected.extend(_build_baseline_jobs())
        elif family == "trees":
            selected.extend(_build_tree_jobs())
        elif family == "xgboost":
            selected.extend(_build_tree_jobs("xgboost"))
        elif family == "lightgbm":
            selected.extend(_build_tree_jobs("lightgbm"))
        elif family in {"fwd_ann", "macro_forward_ann", "group_ensemble_ann"}:
            selected.extend(job for job in _build_ann_jobs() if job.family == family)
        elif family == "model_configs":
            selected.extend(_build_model_config_jobs(args))
        else:
            raise ValueError(f"Unknown family: {family}")
    return selected


def _write_run_manifest(run_root: Path, args: argparse.Namespace, jobs: list[JobSpec]) -> None:
    payload = {
        "run_id": args.run_id,
        "created_at": _timestamp(),
        "families": args.families,
        "jobs": [{"family": job.family, "job_name": job.job_name} for job in jobs],
        "options": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "yield_type": args.yield_type,
            "maturities": args.maturities,
            "oos_start": args.oos_start,
            "gap": args.gap,
            "n_models": args.n_models,
            "k_top": args.k_top,
            "tuning_level": args.tuning_level,
            "config_names": args.config_names,
            "config_shard": args.config_shard,
            "trailing_lookback": args.trailing_lookback,
            "trailing_min_history": args.trailing_min_history,
            "save_checkpoints": args.save_checkpoints,
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_manifest.json").write_text(json.dumps(_jsonable(payload), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run curated Solstrom HPC core model families."
    )
    parser.add_argument("families", nargs="+", choices=FAMILY_CHOICES)
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs only.")
    parser.add_argument("--n-models", type=int, default=100)
    parser.add_argument("--k-top", type=int, default=10)
    parser.add_argument("--maturities", type=_parse_maturities, default=_parse_maturities("24,36,48,60,84,120"))
    parser.add_argument("--artifacts-root", type=Path, default=REPO_ROOT / "artifacts" / "hpc_runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--yield-type", choices=["lw", "kr", "gsw"], default="lw")
    parser.add_argument("--start-date", default="1971-08-31")
    parser.add_argument("--end-date", default="2018-12-31")
    parser.add_argument("--oos-start", default="1990-01-31")
    parser.add_argument("--gap", type=int, default=11)
    parser.add_argument(
        "--tuning-level",
        choices=["light", "standard"],
        default="standard",
        help="Tuning grid size for non-baseline models. ANN grids are already bounded; this mainly controls tree grids.",
    )
    parser.add_argument(
        "--config-names",
        default=None,
        help="Comma-separated run_name filter for the model_configs family.",
    )
    parser.add_argument(
        "--config-shard",
        default=None,
        help="Run only a shard of model_configs using INDEX/TOTAL, for example 0/3.",
    )
    parser.add_argument("--trailing-lookback", type=int, default=120)
    parser.add_argument("--trailing-min-history", type=int, default=24)
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Persist ANN checkpoints. Disabled by default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.n_models < 1:
        parser.error("--n-models must be at least 1.")
    if args.k_top < 1:
        parser.error("--k-top must be at least 1.")

    args.artifacts_root = args.artifacts_root.resolve()
    args.run_id = args.run_id or _run_timestamp()
    run_root = args.artifacts_root / args.run_id
    master_path = args.artifacts_root / "master_summary.csv"
    jobs = _select_jobs(args)
    if not jobs:
        parser.error("No jobs selected. Check --config-names or --config-shard.")

    print(f"Run id: {args.run_id}")
    print(f"Families: {', '.join(args.families)}")
    print(f"Planned jobs ({len(jobs)}):")
    for idx, job in enumerate(jobs, start=1):
        print(f"  {idx:02d}. [{job.family}] {job.job_name}")

    if args.dry_run:
        print("Dry run: no artifacts or training outputs will be created.")
        return 0

    _write_run_manifest(run_root, args, jobs)
    shared_data = _prepare_data(args) if any(job.uses_shared_data for job in jobs) else None
    print(f"Artifacts run directory: {run_root}")
    print(f"Master summary: {master_path}")

    failures = 0
    for idx, job in enumerate(jobs, start=1):
        print(f"\n[{idx}/{len(jobs)}] Starting {job.family}/{job.job_name}")
        try:
            summary = job.runner(shared_data, run_root, args)
            performance = summary.get("performance", "")
            run_dir = summary.get("run_dir", str(run_root / "jobs" / job.job_name))
            _append_master_row(
                master_path,
                {
                    "timestamp": _timestamp(),
                    "run_id": args.run_id,
                    "family": job.family,
                    "job_name": job.job_name,
                    "status": "success",
                    "run_dir": run_dir,
                    "performance": json.dumps(_jsonable(performance), separators=(",", ":")),
                    "error": "",
                },
            )
            print(f"[success] {job.job_name} -> {run_dir}")
        except Exception as exc:
            failures += 1
            error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            _append_master_row(
                master_path,
                {
                    "timestamp": _timestamp(),
                    "run_id": args.run_id,
                    "family": job.family,
                    "job_name": job.job_name,
                    "status": "failure",
                    "run_dir": str((run_root / "jobs" / job.job_name).resolve()),
                    "performance": "",
                    "error": error_text,
                },
            )
            print(f"[failure] {job.job_name}: {error_text}")

    if failures:
        print(f"\nCompleted with {failures} failed job(s). See {master_path}.")
        return 1
    print("\nAll jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
