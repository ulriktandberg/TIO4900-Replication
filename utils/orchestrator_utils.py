import pandas as pd
import utils.base_utils as bu
import utils.window_utils as wu
import numpy as np
import torch

def _nanmean_with_counts(arr, axis):
    """NaN-safe mean and valid-count helper for selection statistics."""
    valid_counts = np.sum(~np.isnan(arr), axis=axis)
    sums = np.nansum(arr, axis=axis)
    means = np.divide(
        sums,
        valid_counts,
        out=np.full(np.shape(sums), np.nan, dtype=float),
        where=valid_counts > 0,
    )
    return means, valid_counts


def compute_top_k_ensemble(
    forecasts_array: np.ndarray,
    val_losses_array: np.ndarray,
    k: int,
    selection_mode: str = "per_maturity",
    selection_metric: str = "val_loss",
    y_true: np.ndarray | None = None,
    lookback: int = 120,
    min_history: int = 24,
    realization_lag: int = 0,
):
    """Compute a top-k ensemble from seed forecasts.

    ``selection_metric='val_loss'`` reproduces the original validation-loss selector.
    ``selection_metric='trailing_oos'`` ranks seeds by trailing realized OOS MSE using
    only outcomes available at forecast time, then falls back to validation loss when
    realized history is too short.
    """
    if selection_mode not in {"per_maturity", "total"}:
        raise ValueError("selection_mode must be either 'per_maturity' or 'total'.")
    if selection_metric not in {"val_loss", "trailing_oos"}:
        raise ValueError("selection_metric must be either 'val_loss' or 'trailing_oos'.")

    T, n_seeds, n_outputs = forecasts_array.shape
    ensemble_forecast = np.full((T, n_outputs), np.nan)
    topk_indices = np.full((T, n_outputs, min(k, n_seeds)), -1, dtype=int)

    if selection_metric == "trailing_oos":
        if y_true is None:
            raise ValueError("y_true must be provided when selection_metric='trailing_oos'.")
        y_true = np.asarray(y_true)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_true.shape != (T, n_outputs):
            raise ValueError(f"y_true must have shape {(T, n_outputs)}; got {y_true.shape}.")

    def _select_total_from_val_losses(t):
        seed_losses, seed_valid_counts = _nanmean_with_counts(val_losses_array[t], axis=1)
        valid_idx = np.where((~np.isnan(seed_losses)) & (seed_valid_counts > 0))[0]
        return seed_losses, valid_idx

    def _select_per_maturity_from_val_losses(t, m):
        v_losses = val_losses_array[t, :, m]
        valid_idx = np.where(~np.isnan(v_losses))[0]
        return v_losses, valid_idx

    for t in range(T):
        if selection_mode == "total":
            use_val_fallback = True
            if selection_metric == "trailing_oos":
                hist_end = t - realization_lag
                hist_start = max(0, hist_end - lookback)
                if hist_end > hist_start:
                    trailing_err = (
                        forecasts_array[hist_start:hist_end]
                        - y_true[hist_start:hist_end, None, :]
                    ) ** 2
                    seed_losses, seed_valid_counts = _nanmean_with_counts(
                        trailing_err,
                        axis=(0, 2),
                    )
                    valid_idx = np.where(
                        (~np.isnan(seed_losses)) & (seed_valid_counts >= min_history)
                    )[0]
                    use_val_fallback = len(valid_idx) == 0

            if use_val_fallback:
                seed_losses, valid_idx = _select_total_from_val_losses(t)
            if len(valid_idx) == 0:
                continue
            actual_k = min(k, len(valid_idx))
            sorted_valid_idx = valid_idx[np.argsort(seed_losses[valid_idx])]
            selected = sorted_valid_idx[:actual_k]

            topk_indices[t, :, :actual_k] = selected[None, :]
            ensemble_forecast[t], _ = _nanmean_with_counts(forecasts_array[t, selected, :], axis=0)
            continue

        trailing_seed_losses = None
        trailing_seed_counts = None
        if selection_metric == "trailing_oos":
            hist_end = t - realization_lag
            hist_start = max(0, hist_end - lookback)
            if hist_end > hist_start:
                trailing_err = (
                    forecasts_array[hist_start:hist_end]
                    - y_true[hist_start:hist_end, None, :]
                ) ** 2
                trailing_seed_losses, trailing_seed_counts = _nanmean_with_counts(
                    trailing_err,
                    axis=0,
                )

        for m in range(n_outputs):
            use_val_fallback = True
            if selection_metric == "trailing_oos" and trailing_seed_losses is not None:
                v_losses = trailing_seed_losses[:, m]
                counts = trailing_seed_counts[:, m]
                valid_idx = np.where((~np.isnan(v_losses)) & (counts >= min_history))[0]
                use_val_fallback = len(valid_idx) == 0

            if use_val_fallback:
                v_losses, valid_idx = _select_per_maturity_from_val_losses(t, m)
            if len(valid_idx) == 0:
                continue

            actual_k = min(k, len(valid_idx))
            sorted_valid_idx = valid_idx[np.argsort(v_losses[valid_idx])]
            selected = sorted_valid_idx[:actual_k]

            topk_indices[t, m, :actual_k] = selected
            ensemble_forecast[t, m], _ = _nanmean_with_counts(forecasts_array[t, selected, m], axis=0)

    return ensemble_forecast, topk_indices


def _extract_scaler_state(scaler):
    if scaler is None:
        return None
    state = {}
    for attr in ['mean_', 'scale_', 'var_', 'n_samples_seen_', 'n_features_in_']:
        if hasattr(scaler, attr):
            val = getattr(scaler, attr)
            if isinstance(val, np.ndarray):
                state[attr] = val.copy()
            elif np.isscalar(val):
                state[attr] = val.item() if hasattr(val, 'item') else val
            else:
                state[attr] = val
    return state


def _extract_pca_state(pca):
    if pca is None:
        return None
    state = {}
    for attr in ['components_', 'mean_', 'explained_variance_', 'explained_variance_ratio_', 'n_components_']:
        if hasattr(pca, attr):
            val = getattr(pca, attr)
            state[attr] = val.copy() if isinstance(val, np.ndarray) else val
    return state


def _estimate_model_size_mb(wrapper_model: torch.nn.Module) -> float:
    n_params = sum(p.numel() for p in wrapper_model.parameters())
    return (n_params * 4) / (1024 ** 2)

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from tqdm import tqdm


@dataclass
class RunConfig:
    run_name: str
    model_builder: Callable[[int], object]
    n_models: int
    k_top: int
    maturities: list
    oos_start: pd.Timestamp
    gap: int = 0
    refit_freq: int = 1
    benchmark: str = 'hist_mean'
    rsz_maxlags: int = 12
    progress: bool = False
    save_checkpoints: bool = True
    artifacts_root: Path = Path('../artifacts/orchestrator_runs')
    ensemble_metrics: tuple[str, ...] = ("val_loss", "trailing_oos")
    ensemble_selection_mode: str = "per_maturity"
    trailing_lookback: int = 120
    trailing_min_history: int = 24
    trailing_realization_lag: int | None = None


def _save_checkpoint(wrapper, seed: int, t_index: int, date_value, run_dir: Path) -> Path:
    ckpt_dir = run_dir / 'checkpoints' / f'seed_{seed:03d}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'step_{t_index:04d}_{pd.Timestamp(date_value).date()}.pt'

    x_scalers_macro_state = None
    if hasattr(wrapper, 'x_scalers_macro') and isinstance(wrapper.x_scalers_macro, dict):
        x_scalers_macro_state = {k: _extract_scaler_state(v) for k, v in wrapper.x_scalers_macro.items()}

    checkpoint = {
        'wrapper_class': wrapper.__class__.__name__,
        'wrapper_module': wrapper.__class__.__module__,
        'torch_state_dict': wrapper.model.state_dict() if hasattr(wrapper, 'model') and wrapper.model is not None else None,
        'best_params_': getattr(wrapper, 'best_params_', None),
        'fit_calls': getattr(wrapper, '_fit_calls', None),
        'x_scaler': _extract_scaler_state(getattr(wrapper, 'x_scaler', None)),
        'x_scaler_forward': _extract_scaler_state(getattr(wrapper, 'x_scaler_forward', None)),
        'x_scaler_fred': _extract_scaler_state(getattr(wrapper, 'x_scaler_fred', None)),
        'x_scalers_macro': x_scalers_macro_state,
        'y_scaler': _extract_scaler_state(getattr(wrapper, 'y_scaler', None)),
        'pca': _extract_pca_state(getattr(wrapper, 'pca', None)),
    }

    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def run_experiment(cfg: RunConfig, X: pd.DataFrame, y_all: np.ndarray, dates: pd.DatetimeIndex):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = (cfg.artifacts_root / cfg.run_name / ts).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    T = len(dates)
    n_outputs = y_all.shape[1] if y_all.ndim > 1 else 1

    all_forecasts = []
    all_val_losses = []
    ckpt_manifest = []

    model_iter = range(cfg.n_models)
    if cfg.progress:
        model_iter = tqdm(model_iter, desc='Seeds')

    for seed in model_iter:
        model = cfg.model_builder(seed)
        val_losses_for_seed = np.full((T, n_outputs), np.nan)

        # This callback is triggered at each refit step by expanding_window.
        def save_cb(model, refit_i, t_index, date_value, **kwargs):
            if hasattr(model, 'val_loss_') and model.val_loss_ is not None:
                val_losses_for_seed[t_index] = model.val_loss_
            if cfg.save_checkpoints:
                ckpt_path = _save_checkpoint(model, seed, t_index, date_value, run_dir)
                ckpt_manifest.append({
                    'seed': seed,
                    'refit_i': refit_i,
                    't_index': int(t_index),
                    'date': str(pd.Timestamp(date_value).date()),
                    'checkpoint_path': str(ckpt_path),
                })

        y_forecast = wu.expanding_window(
            model, X, y_all, dates, cfg.oos_start,
            gap=cfg.gap,
            refit_freq=cfg.refit_freq,
            save_callback=save_cb,
            progress=False,
        )

        all_forecasts.append(y_forecast)
        all_val_losses.append(val_losses_for_seed)

    forecasts_arr = np.stack(all_forecasts, axis=1)
    losses_arr = np.stack(all_val_losses, axis=1)

    ensemble_outputs = {}
    performance_rows = []
    realization_lag = cfg.gap if cfg.trailing_realization_lag is None else cfg.trailing_realization_lag

    for metric in cfg.ensemble_metrics:
        ensemble_forecast, topk_indices = compute_top_k_ensemble(
            forecasts_arr,
            losses_arr,
            cfg.k_top,
            selection_mode=cfg.ensemble_selection_mode,
            selection_metric=metric,
            y_true=y_all,
            lookback=cfg.trailing_lookback,
            min_history=cfg.trailing_min_history,
            realization_lag=realization_lag,
        )
        ensemble_outputs[metric] = (ensemble_forecast, topk_indices)

        r2s = wu.oos_r2(y_all, ensemble_forecast, benchmark=cfg.benchmark, gap=cfg.gap)
        pvals = np.array([
            bu.RSZ_Signif(y_all[:, i], ensemble_forecast[:, i], gap=cfg.gap)
            for i in range(n_outputs)
        ])
        for maturity, r2, pval in zip(cfg.maturities, r2s.tolist(), pvals.tolist()):
            performance_rows.append({
                "ensemble_method": metric,
                "selection_mode": cfg.ensemble_selection_mode,
                "maturity": maturity,
                "r2_oos": r2,
                "rsz_pval": pval,
            })

    # Persist arrays and metadata
    np.save(run_dir / 'forecasts_arr.npy', forecasts_arr)
    np.save(run_dir / 'losses_arr.npy', losses_arr)
    for metric, (ensemble_forecast, topk_indices) in ensemble_outputs.items():
        suffix = "" if metric == "val_loss" else f"_{metric}"
        np.save(run_dir / f'ensemble_forecast{suffix}.npy', ensemble_forecast)
        np.save(run_dir / f'topk_indices{suffix}.npy', topk_indices)
        np.save(run_dir / f'ensemble_forecast_{metric}.npy', ensemble_forecast)
        np.save(run_dir / f'topk_indices_{metric}.npy', topk_indices)
    if cfg.save_checkpoints:
        pd.DataFrame(ckpt_manifest).to_csv(run_dir / 'checkpoint_manifest.csv', index=False)
    
    perf_df = pd.DataFrame(performance_rows)
    perf_df.to_csv(run_dir / 'performance.csv', index=False)

    serializable_cfg = asdict(cfg)
    serializable_cfg['model_builder'] = str(cfg.model_builder)
    serializable_cfg['artifacts_root'] = str(cfg.artifacts_root)
    pd.Series(serializable_cfg).to_json(run_dir / 'run_config.json', indent=2)

    # Storage summary
    if cfg.save_checkpoints:
        ckpt_paths = list((run_dir / 'checkpoints').rglob('*.pt'))
        total_ckpt_bytes = sum(p.stat().st_size for p in ckpt_paths)
        num_checkpoints = len(ckpt_paths)
        total_checkpoint_gb = total_ckpt_bytes / (1024 ** 3)
    else:
        num_checkpoints = 0
        total_checkpoint_gb = 0.0

    summary = {
        'run_dir': str(run_dir),
        'num_checkpoints': num_checkpoints,
        'total_checkpoint_gb': total_checkpoint_gb,
        'save_checkpoints': cfg.save_checkpoints,
        'performance': performance_rows,
        'forecasts_arr_shape': forecasts_arr.shape,
        'losses_arr_shape': losses_arr.shape,
    }

    return summary

def arch_to_name(arch):
    return 'direct' if len(arch) == 0 else '&'.join(str(x) for x in arch)

def base_run_config(run_name, model_builder, n_models=100, k_top=10, save_checkpoints=True):
    return RunConfig(
        run_name=run_name,
        model_builder=model_builder,
        n_models=n_models,
        k_top=k_top,
        maturities=maturities,
        oos_start=pd.Timestamp('1990-01-31'),
        gap=11,
        refit_freq=1,
        benchmark='hist_mean',
        progress=True,
        save_checkpoints=save_checkpoints,
        artifacts_root=experiment_artifacts_root,
    )

def make_fwd_ann_cfg(arch, n_models=100, k_top=10, save_checkpoints=True):
    return base_run_config(
        run_name=f"fwd_ann_{arch_to_name(arch)}_{n_models}runs_top{k_top}",
        model_builder=lambda seed, arch=arch: PyTorchMLPWrapper(
            archi=arch,
            lr=0.01,
            epochs=1000,
            tune_every=60,
            patience=50,
            param_grid={'penalty': [0.01, 0.001, 0.0001]},
            seed=seed,
            use_pca=False,
            n_components=None,
            y_center=True,
        ),
        n_models=n_models,
        k_top=k_top,
        save_checkpoints=save_checkpoints,
    )

def make_macro_forward_cfg(arch_macro, arch_forward=(3,), n_models=100, k_top=10, save_checkpoints=True):
    return base_run_config(
        run_name=(
            f"macro_fwd_ann_fwd{arch_to_name(arch_forward)}_"
            f"macro{arch_to_name(arch_macro)}_{n_models}runs_top{k_top}"
        ),
        model_builder=lambda seed, arch_macro=arch_macro, arch_forward=arch_forward: MacroForwardANNWrapper(
            archi_forward=arch_forward,
            archi_macro=arch_macro,
            lr=0.01,
            epochs=1000,
            tune_every=60,
            patience=50,
            param_grid={'penalty': [0.001, 0.0001], 'dropout_rate': [0.0, 0.1, 0.3]},
            seed=seed,
            y_center=True,
        ),
        n_models=n_models,
        k_top=k_top,
        save_checkpoints=save_checkpoints,
    )

def make_group_ensemble_cfg(arch_macro, arch_forward, n_models=100, k_top=10, save_checkpoints=True):
    return base_run_config(
        run_name=(
            f"group_ens_ann_fwd{arch_to_name(arch_forward)}_"
            f"grp{arch_to_name(arch_macro)}_{n_models}runs_top{k_top}"
        ),
        model_builder=lambda seed, arch_macro=arch_macro, arch_forward=arch_forward: GroupEnsembleANNWrapper(
            archi_forward=arch_forward,
            archi_macro=arch_macro,
            lr=0.01,
            epochs=1000,
            tune_every=60,
            patience=50,
            param_grid={'penalty': [0.001, 0.0001], 'dropout_rate': [0.0, 0.1, 0.3]},
            seed=seed,
            y_center=True,
        ),
        n_models=n_models,
        k_top=k_top,
        save_checkpoints=save_checkpoints,
    )
