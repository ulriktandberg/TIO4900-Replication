import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from tqdm.auto import tqdm
from copy import deepcopy
from utils.persistence_utils import iter_ann_payloads

DEVICE = "cpu"

def _col_to_str(c):
    if isinstance(c, tuple):
        return "::".join([str(x) for x in c])
    return str(c)


def _as_shap_input(tensors):
    return tensors[0] if len(tensors) == 1 else tensors


def _disable_inplace_ops(module: nn.Module):
    for m in module.modules():
        if hasattr(m, "inplace"):
            try:
                m.inplace = False
            except Exception:
                pass
    return module


class ANNShapAdapter(nn.Module):
    """
    Adapts saved internal PyTorch model for SHAP:
    - optional unscaled output for WGNN
    - optional target selection for multi-output
    """
    def __init__(self, core_model: nn.Module, target_idx=None, use_unscaled_output=False):
        super().__init__()
        self.core_model = core_model
        self.target_idx = target_idx
        self.use_unscaled_output = use_unscaled_output

    def forward(self, *inputs):
        if self.use_unscaled_output and hasattr(self.core_model, "predict_unscaled"):
            out = self.core_model.predict_unscaled(*inputs)
        else:
            out = self.core_model(*inputs)

        if out.ndim == 1:
            out = out.unsqueeze(-1)

        if self.target_idx is not None:
            out = out[:, self.target_idx:self.target_idx + 1]
        return out


def _get_raw_inputs_and_names(wrapper, X_slice: pd.DataFrame):
    cls = wrapper.__class__.__name__

    # 1) Try wrapper-native selector
    if hasattr(wrapper, "_select_features"):
        raw_inputs = wrapper._select_features(X_slice)
    # 2) Fallback for ForwardRateANN (which currently has no _select_features)
    elif cls == "ForwardRateANN":
        series = getattr(wrapper, "series", "forward")
        raw_inputs = [X_slice[series].values]
    else:
        raw_inputs = None

    if raw_inputs is None:
        raise ValueError(f"Wrapper {cls} does not expose _select_features; cannot build SHAP inputs.")

    # Build feature names per input branch
    names = []
    branch_names = []

    if cls == "ForwardRateANN":
        series = getattr(wrapper, "series", "forward")
        cols = [_col_to_str(c) for c in X_slice[series].columns]
        names = [cols]
        branch_names = [series]

    elif cls == "HybridANN":
        fred_cols = [_col_to_str(c) for c in X_slice["fred"].columns]
        fwd_cols = [_col_to_str(c) for c in X_slice["forward"].columns]
        names = [fred_cols, fwd_cols]
        branch_names = ["fred", "forward"]

    elif cls in ("GroupEnsembleANN", "WeightedGroupEnsembleANN"):
        group_names = wrapper._group_names
        if group_names is None:
            group_names = X_slice["fred"].columns.get_level_values(0).unique().tolist()

        for gn in group_names:
            gcols = [_col_to_str(c) for c in X_slice["fred"][gn].columns]
            names.append([f"fred::{gn}::{c}" for c in gcols])
            branch_names.append(f"fred::{gn}")

        fwd_cols = [_col_to_str(c) for c in X_slice["forward"].columns]
        names.append([f"forward::{c}" for c in fwd_cols])
        branch_names.append("forward")

    else:
        # fallback
        for i, arr in enumerate(raw_inputs):
            names.append([f"input{i}_f{j}" for j in range(arr.shape[1])])
            branch_names.append(f"input{i}")

    return raw_inputs, names, branch_names


def _scale_to_tensors(wrapper, raw_inputs):
    if getattr(wrapper, "_scalers", None) is None:
        raise ValueError("Saved wrapper has no _scalers.")
    if len(wrapper._scalers) != len(raw_inputs):
        raise ValueError("Mismatch between number of scalers and input branches.")

    tensors = []
    for s, arr in zip(wrapper._scalers, raw_inputs):
        x_scaled = s.transform(arr)
        tensors.append(torch.tensor(x_scaled, dtype=torch.float32, device=DEVICE))
    return tensors


def _normalize_shap_values(shap_values, n_inputs):
    # Common cases from DeepExplainer:
    # - single input: ndarray
    # - multi input: list[ndarray]
    # - nested output/input: list[list[ndarray]] (we use first output because adapter can select target)
    if isinstance(shap_values, list):
        if len(shap_values) > 0 and isinstance(shap_values[0], list):
            shap_values = shap_values[0]
        vals = [np.asarray(v) for v in shap_values]
    else:
        vals = [np.asarray(shap_values)]

    if len(vals) != n_inputs:
        # fallback for single-input model passed as list
        if n_inputs == 1 and len(vals) == 1:
            return vals
        raise ValueError(f"Unexpected SHAP output structure: got {len(vals)} inputs, expected {n_inputs}.")

    out = []
    for v in vals:
        if v.ndim == 3 and v.shape[-1] == 1:
            v = v[..., 0]
        out.append(v)
    return out


def shap_importance_for_snapshot(
    wrapper,
    rec: dict,
    X: pd.DataFrame,
    gap: int = 0,
    background_size: int = 128,
    explain_size: int = 1,
    target_idx: int = None,
    use_unscaled_output: bool = True,
):
    """
    Returns long DataFrame with mean(|SHAP|) by feature for one saved snapshot.
    """
    if getattr(wrapper, "_model", None) is None:
        raise ValueError("Wrapper snapshot has no _model.")

    t_index = int(rec["t_index"])
    gap_i = int(gap)
    train_end = max(t_index - gap_i, 0)
    if train_end <= 1:
        return pd.DataFrame()

    bg_start = max(0, train_end - background_size)
    X_bg = X.iloc[bg_start:train_end]
    X_ex = X.iloc[t_index:min(len(X), t_index + explain_size)]
    if len(X_ex) == 0:
        X_ex = X.iloc[[t_index]]

    raw_bg, feat_names_by_input, branch_names = _get_raw_inputs_and_names(wrapper, X_bg)
    raw_ex, _, _ = _get_raw_inputs_and_names(wrapper, X_ex)

    bg_tensors = _scale_to_tensors(wrapper, raw_bg)
    ex_tensors = _scale_to_tensors(wrapper, raw_ex)

    core = deepcopy(wrapper._model).to(DEVICE).eval()
    core = _disable_inplace_ops(core)

    model_for_shap = ANNShapAdapter(
        core_model=core,
        target_idx=target_idx,
        use_unscaled_output=use_unscaled_output
    ).to(DEVICE).eval()

    try:
        explainer = shap.DeepExplainer(model_for_shap, _as_shap_input(bg_tensors))
        try:
            sv = explainer.shap_values(_as_shap_input(ex_tensors), check_additivity=False)
        except TypeError:
            sv = explainer.shap_values(_as_shap_input(ex_tensors))
    except RuntimeError as e:
        if "view and is being modified inplace" in str(e):
            explainer = shap.GradientExplainer(model_for_shap, _as_shap_input(bg_tensors))
            sv = explainer.shap_values(_as_shap_input(ex_tensors))
        else:
            print("Error computing SHAP values for snapshot:", rec)
            raise

    sv_list = _normalize_shap_values(sv, n_inputs=len(ex_tensors))

    rows = []
    for i, sv_i in enumerate(sv_list):
        # sv_i shape: [n_samples, n_features]
        imp = np.mean(np.abs(sv_i), axis=0)
        for j, val in enumerate(imp):
            rows.append({
                "seed": int(rec["seed"]),
                "refit_i": int(rec["refit_i"]),
                "t_index": int(rec["t_index"]),
                "date": pd.to_datetime(rec["date"]),
                "target_idx": -1 if target_idx is None else int(target_idx),
                "input_branch": branch_names[i],
                "feature_idx": j,
                "feature_name": feat_names_by_input[i][j],
                "mean_abs_shap": float(val),
            })

    return pd.DataFrame(rows)


def run_ann_shap_over_time(
    run_name: str,
    X: pd.DataFrame,
    seeds=None,
    min_refit_i=None,
    max_refit_i=None,
    every_n_refits: int = 1,
    gap: int = 0,
    background_size: int = 128,
    explain_size: int = 1,
    target_indices=None,   # e.g. [0,1,...,8] for y_all outputs
    use_unscaled_output: bool = True,
    progress: bool = True,
    tqdm_position: int = 0,
    tqdm_leave: bool = True,
):
    """
    Iterates ANN snapshots and returns one long DataFrame for plotting stability over time.
    """
    out = []

    snapshot_iter = iter_ann_payloads(
        run_name=run_name,
        seeds=seeds,
        min_refit_i=min_refit_i,
        max_refit_i=max_refit_i,
    )

    if progress:
        snapshot_iter = tqdm(
            snapshot_iter,
            desc=f"SHAP snapshots [{run_name}]",
            position=tqdm_position,
            leave=tqdm_leave,
            dynamic_ncols=True
        )

    for rec, wrapper, _, _ in snapshot_iter:
        if every_n_refits > 1 and (int(rec["refit_i"]) % every_n_refits != 0):
            continue

        if target_indices is None:
            dfs = [shap_importance_for_snapshot(
                wrapper=wrapper, rec=rec, X=X, gap=gap,
                background_size=background_size, explain_size=explain_size,
                target_idx=None, use_unscaled_output=use_unscaled_output
            )]
        else:
            dfs = []
            target_iter = target_indices
            if progress:
                target_iter = tqdm(
                    target_indices,
                    desc=f"targets seed={rec['seed']} step={rec['refit_i']}",
                    position=tqdm_position + 1,
                    leave=False,
                    dynamic_ncols=True
                )
            for tidx in target_iter:
                dfs.append(shap_importance_for_snapshot(
                    wrapper=wrapper, rec=rec, X=X, gap=gap,
                    background_size=background_size, explain_size=explain_size,
                    target_idx=int(tidx), use_unscaled_output=use_unscaled_output
                ))

        for d in dfs:
            if not d.empty:
                out.append(d)

    if not out:
        return pd.DataFrame(columns=[
            "seed", "refit_i", "t_index", "date", "target_idx",
            "input_branch", "feature_idx", "feature_name", "mean_abs_shap"
        ])
    return pd.concat(out, axis=0, ignore_index=True)
