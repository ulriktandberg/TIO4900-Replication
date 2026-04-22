import pandas as pd
import utils.base_utils as bu
import utils.window_utils as wu
import numpy as np
import torch

def compute_top_k_ensemble(forecasts_array: np.ndarray, val_losses_array: np.ndarray, k: int):
    # Same ensembling logic as existing notebook code: top-k per maturity and date by val loss.
    T, n_seeds, n_outputs = forecasts_array.shape
    ensemble_forecast = np.full((T, n_outputs), np.nan)
    topk_indices = np.full((T, n_outputs, min(k, n_seeds)), -1, dtype=int)

    for t in range(T):
        for m in range(n_outputs):
            v_losses = val_losses_array[t, :, m]
            valid_idx = np.where(~np.isnan(v_losses))[0]
            if len(valid_idx) == 0:
                continue

            actual_k = min(k, len(valid_idx))
            sorted_valid_idx = valid_idx[np.argsort(v_losses[valid_idx])]
            selected = sorted_valid_idx[:actual_k]

            topk_indices[t, m, :actual_k] = selected
            ensemble_forecast[t, m] = np.mean(forecasts_array[t, selected, m], axis=0)

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

    ensemble_forecast, topk_indices = compute_top_k_ensemble(forecasts_arr, losses_arr, cfg.k_top)

    r2s = wu.oos_r2(y_all, ensemble_forecast, benchmark=cfg.benchmark)
    pvals = np.array([bu.RSZ_Signif(y_all[:, i], ensemble_forecast[:, i])
                     for i in range(n_outputs)])

    performance_tuples = list(zip(cfg.maturities, r2s.tolist(), pvals.tolist()))

    # Persist arrays and metadata
    np.save(run_dir / 'forecasts_arr.npy', forecasts_arr)
    np.save(run_dir / 'losses_arr.npy', losses_arr)
    np.save(run_dir / 'ensemble_forecast.npy', ensemble_forecast)
    np.save(run_dir / 'topk_indices.npy', topk_indices)
    if cfg.save_checkpoints:
        pd.DataFrame(ckpt_manifest).to_csv(run_dir / 'checkpoint_manifest.csv', index=False)
    
    perf_df = pd.DataFrame(performance_tuples, columns=['maturity', 'r2_oos', 'rsz_pval'])
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
        'performance': performance_tuples,
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
