import pandas as pd
import os
import json
import joblib
from datetime import datetime
import numpy as np
import random
from utils import window_utils as wu
from tqdm import tqdm
import glob
from typing import Optional, Iterable, Iterator, Tuple, Dict, Any

class JoblibModelStore:
    """
    Saves one model per (seed, refit_step) with metadata.
    Folder layout:
      root/run_name/seed_0001/step_00042/model.joblib
      root/run_name/seed_0001/step_00042/meta.json
    """
    def __init__(self, root_dir="artifacts/models", run_name="default_run", compress=3):
        self.root_dir = root_dir
        self.run_name = run_name
        self.compress = compress

    def _step_dir(self, seed: int, refit_i: int):
        return os.path.join(
            self.root_dir,
            self.run_name,
            f"seed_{seed:04d}",
            f"step_{refit_i:05d}"
        )

    def save(self, model, seed: int, refit_i: int, t_index: int, date_value, extra_meta=None):
        step_dir = self._step_dir(seed, refit_i)
        os.makedirs(step_dir, exist_ok=True)

        model_path = os.path.join(step_dir, "model.joblib")
        meta_path = os.path.join(step_dir, "meta.json")

        joblib.dump(model, model_path, compress=self.compress)

        meta = {
            "saved_at_utc": datetime.utcnow().isoformat(),
            "seed": int(seed),
            "refit_i": int(refit_i),
            "t_index": int(t_index),
            "date": str(date_value),
            "model_path": model_path,
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

def run_expanding_multi_seed(
    model_factory,
    seeds,
    X, y, dates, oos_start,
    run_name="expanding_run",
    gap=0, refit_freq=1,
    benchmark="hist_mean",
):
    out = {}
    results_dir = os.path.join("artifacts", "results", run_name)
    os.makedirs(results_dir, exist_ok=True)

    for seed in tqdm(seeds, desc="seeds", position=0, leave=True, dynamic_ncols=True):
        set_global_seed(seed)
        model = model_factory(seed)
        store = JoblibModelStore(run_name=run_name)

        def _save_cb(model, refit_i, t_index, date_value):
            store.save(
                model=model,
                seed=seed,
                refit_i=refit_i,
                t_index=t_index,
                date_value=date_value,
                extra_meta={"val_loss": float(getattr(model, "_last_val_loss", np.nan))}
            )

        y_forecast = wu.expanding_window(
            model, X, y, dates, oos_start,
            gap=gap, refit_freq=refit_freq,
            save_callback=_save_cb,
            progress=True,
            tqdm_position=1,
            tqdm_desc=f"seed {seed}",
            tqdm_leave=False
        )
        r2 = wu.oos_r2(y, y_forecast, benchmark=benchmark, gap=gap)
        out[seed] = {"forecast": y_forecast, "r2": r2}

        # Persist forecast array so results can be reloaded without replaying
        joblib.dump(
            {"forecast": y_forecast, "r2": r2},
            os.path.join(results_dir, f"seed_{seed:04d}.joblib"),
            compress=3
        )

    return out


def load_results(run_name: str, seeds=None, root_dir: str = "artifacts/results") -> dict:
    """Load persisted per-seed forecast dicts, returning same structure as run_expanding_multi_seed."""
    results_dir = os.path.join(root_dir, run_name)
    pattern = os.path.join(results_dir, "seed_*.joblib")
    files = sorted(glob.glob(pattern))

    out = {}
    for path in files:
        seed = int(os.path.basename(path).replace("seed_", "").replace(".joblib", ""))
        if seeds is not None and seed not in set(seeds):
            continue
        out[seed] = joblib.load(path)
    return out


def build_snapshot_index(
    run_name: str,
    root_dir: str = "artifacts/models",
    require_model_file: bool = True
) -> pd.DataFrame:
    """
    Build a tidy index of all saved snapshots for a run.
    Returns columns:
      run_name, seed, refit_i, t_index, date, model_path, meta_path
    """
    run_dir = os.path.join(root_dir, run_name)
    meta_files = glob.glob(os.path.join(run_dir, "seed_*", "step_*", "meta.json"))
    rows = []

    for meta_path in meta_files:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        model_path = meta.get("model_path")
        if model_path is None:
            model_path = os.path.join(os.path.dirname(meta_path), "model.joblib")

        if require_model_file and (not os.path.exists(model_path)):
            continue

        rows.append({
            "run_name": run_name,
            "seed": int(meta.get("seed")),
            "refit_i": int(meta.get("refit_i")),
            "t_index": int(meta.get("t_index")),
            "date": pd.to_datetime(meta.get("date")),
            "model_path": model_path,
            "meta_path": meta_path,
            "val_loss": meta.get("val_loss", np.nan),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "run_name", "seed", "refit_i", "t_index", "date", "model_path", "meta_path", "val_loss"
        ])

    df = pd.DataFrame(rows).sort_values(["seed", "refit_i"]).reset_index(drop=True)
    return df


def load_snapshot_model(model_path: str):
    """Load one snapshot model from disk."""
    return joblib.load(model_path)


def iter_snapshot_models(
    run_name: str,
    root_dir: str = "artifacts/models",
    seeds: Optional[Iterable[int]] = None,
    min_refit_i: Optional[int] = None,
    max_refit_i: Optional[int] = None,
) -> Iterator[Tuple[Dict[str, Any], Any]]:
    """
    Stream (row_dict, model) snapshot-by-snapshot to keep memory usage low.
    """
    idx = build_snapshot_index(run_name=run_name, root_dir=root_dir)

    if seeds is not None:
        seeds = set(int(s) for s in seeds)
        idx = idx[idx["seed"].isin(seeds)]
    if min_refit_i is not None:
        idx = idx[idx["refit_i"] >= int(min_refit_i)]
    if max_refit_i is not None:
        idx = idx[idx["refit_i"] <= int(max_refit_i)]

    for _, row in idx.iterrows():
        rec = row.to_dict()
        model = load_snapshot_model(rec["model_path"])
        yield rec, model


def latest_snapshot_per_seed(run_name: str, root_dir: str = "artifacts/models") -> pd.DataFrame:
    """Return one row per seed (latest refit_i)."""
    idx = build_snapshot_index(run_name=run_name, root_dir=root_dir)
    if idx.empty:
        return idx
    return (
        idx.sort_values(["seed", "refit_i"])
           .groupby("seed", as_index=False)
           .tail(1)
           .reset_index(drop=True)
    )


def extract_linear_importance(model) -> Optional[np.ndarray]:
    """
    Generic helper for linear-style saved models.
    Returns abs(coef) if available, else None.
    """
    # sklearn-style wrappers in your repo often expose .model.coef_
    if hasattr(model, "model") and hasattr(model.model, "coef_"):
        return np.abs(np.asarray(model.model.coef_))
    # plain sklearn estimators
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_))
    return None

# ANNs:

def _is_ann_wrapper(model) -> bool:
    # Your ANN wrappers have these attributes after fit
    return hasattr(model, "_model") and hasattr(model, "_scalers") and hasattr(model, "predict")


def extract_ann_payload(model):
    """
    Returns ANN payload metadata for SHAP pipelines.
    """
    if not _is_ann_wrapper(model):
        return None, None

    scalers = getattr(model, "_scalers", None)
    feature_slices = {
        "group_names": getattr(model, "_group_names", None),
        "wrapper_class": model.__class__.__name__,
    }
    return scalers, feature_slices


def iter_ann_payloads(
    run_name: str,
    root_dir: str = "artifacts/models",
    seeds: Optional[Iterable[int]] = None,
    min_refit_i: Optional[int] = None,
    max_refit_i: Optional[int] = None,
):
    """
    Yields:
      (record, model, scalers, feature_slices)
    """
    for rec, model in iter_snapshot_models(
        run_name=run_name,
        root_dir=root_dir,
        seeds=seeds,
        min_refit_i=min_refit_i,
        max_refit_i=max_refit_i,
    ):
        if not _is_ann_wrapper(model):
            continue
        scalers, feature_slices = extract_ann_payload(model)
        yield rec, model, scalers, feature_slices
