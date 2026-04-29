"""
Base protocol and shared helpers for SHAP adapters.

An adapter bridges a trained wrapper (e.g. PyTorchMLPWrapper) and SHAP by knowing
how to:
  - rebuild the torch.nn.Module from a saved checkpoint
  - reproduce the exact preprocessing (scalers, PCA) used at training time
  - prepare test-point inputs and sample background inputs
  - map SHAP output back to named features
  - translate SHAP from scaled-y space to original-y space (opt-in)

One adapter per wrapper class. The runner picks the right adapter via the registry
in ``utils.shap_adapters.__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ShapAdapter(Protocol):
    """Interface every wrapper-specific SHAP adapter must implement."""

    wrapper_class: str

    def rebuild_model(self, ckpt: dict, run_config: dict) -> torch.nn.Module: ...

    def feature_names(
        self, X: pd.DataFrame, ckpt: dict | None = None
    ) -> list[str]:
        """Return flat feature names matching the layout returned by
        :meth:`flatten_shap`. ``ckpt`` is an optional hint: adapters whose
        ordering depends on the checkpoint (e.g. group-ensemble models with a
        ModuleDict of group towers) should use it to read the training-time
        insertion order; others can ignore it.
        """
        ...

    def prepare_inputs(
        self, X_rows: pd.DataFrame, ckpt: dict
    ) -> torch.Tensor | tuple[torch.Tensor, ...]: ...

    def sample_background(
        self,
        X_train: pd.DataFrame,
        ckpt: dict,
        n: int,
        rng: np.random.Generator,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]: ...

    def flatten_shap(
        self, shap_out: Any, feature_names: Sequence[str], m_idx: int = 0
    ) -> np.ndarray:
        """Flatten SHAP output to shape ``(n_rows, n_features)`` for a single
        output head. ``m_idx`` selects which output to return when the model is
        multi-output; ignored for single-output models.
        """
        ...

    def y_scale_and_shift(
        self, ckpt: dict, y_center_default: bool
    ) -> tuple[np.ndarray, np.ndarray]: ...


# ---------------------------------------------------------------------------
# Scaler / PCA reconstruction
# ---------------------------------------------------------------------------


def rebuild_standard_scaler(
    state: dict | None, with_mean_default: bool = True, with_std_default: bool = True
) -> StandardScaler | None:
    """Rebuild a fitted ``StandardScaler`` from the dict produced by
    ``orchestrator_runs._extract_scaler_state``.

    ``with_mean`` / ``with_std`` are constructor flags and are not stored in the
    checkpoint. The defaults can be overridden by the caller.
    """
    if state is None:
        return None

    with_mean = state.get("with_mean", with_mean_default)
    with_std = state.get("with_std", with_std_default)

    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

    if "mean_" in state and state["mean_"] is not None:
        scaler.mean_ = np.asarray(state["mean_"], dtype=np.float64)
    if "scale_" in state and state["scale_"] is not None:
        scaler.scale_ = np.asarray(state["scale_"], dtype=np.float64)
    if "var_" in state and state["var_"] is not None:
        scaler.var_ = np.asarray(state["var_"], dtype=np.float64)
    if "n_samples_seen_" in state and state["n_samples_seen_"] is not None:
        scaler.n_samples_seen_ = state["n_samples_seen_"]
    if "n_features_in_" in state and state["n_features_in_"] is not None:
        scaler.n_features_in_ = state["n_features_in_"]

    return scaler


def rebuild_pca(state: dict | None) -> PCA | None:
    if state is None:
        return None
    components = state.get("components_")
    if components is None:
        return None

    n_components = state.get("n_components_", components.shape[0])
    pca = PCA(n_components=n_components)
    pca.components_ = np.asarray(components, dtype=np.float64)
    if state.get("mean_") is not None:
        pca.mean_ = np.asarray(state["mean_"], dtype=np.float64)
    if state.get("explained_variance_") is not None:
        pca.explained_variance_ = np.asarray(state["explained_variance_"], dtype=np.float64)
    if state.get("explained_variance_ratio_") is not None:
        pca.explained_variance_ratio_ = np.asarray(
            state["explained_variance_ratio_"], dtype=np.float64
        )
    pca.n_components_ = int(n_components)
    pca.n_features_in_ = pca.components_.shape[1]
    return pca


def apply_scaler_and_pca(
    X_raw: np.ndarray, scaler: StandardScaler | None, pca: PCA | None
) -> np.ndarray:
    X = np.asarray(X_raw, dtype=np.float64)
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# Torch / state_dict helpers
# ---------------------------------------------------------------------------


def linear_shapes_in_order(
    state_dict: dict, prefix: str
) -> list[tuple[int, int]]:
    """Return list of (out_features, in_features) for Linear layers under ``prefix``,
    in order of their numeric index in a ``nn.Sequential``.

    E.g. for an ``_MLPNetwork`` with prefix ``"network."`` returns shapes of every
    ``network.<i>.weight`` that is 2D.
    """
    rows = []
    for key, val in state_dict.items():
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        inner = key[len(prefix) : -len(".weight")]
        if "." in inner:
            # nested module (e.g. BatchNorm has different suffix, but also only 1D)
            continue
        try:
            idx = int(inner)
        except ValueError:
            continue
        shape = tuple(val.shape) if hasattr(val, "shape") else ()
        if len(shape) != 2:
            continue
        rows.append((idx, int(shape[0]), int(shape[1])))
    rows.sort(key=lambda r: r[0])
    return [(r[1], r[2]) for r in rows]


def infer_mlp_archi(state_dict: dict, prefix: str = "network.") -> tuple[int, tuple[int, ...], int]:
    """Infer (input_dim, archi, output_dim) of an ``_MLPNetwork`` from its state_dict."""
    shapes = linear_shapes_in_order(state_dict, prefix=prefix)
    if not shapes:
        raise ValueError(f"No Linear weights found under prefix={prefix!r}.")
    input_dim = shapes[0][1]
    output_dim = shapes[-1][0]
    archi = tuple(s[0] for s in shapes[:-1])
    return input_dim, archi, output_dim


def infer_tower_archi(state_dict: dict, prefix: str) -> tuple[int, tuple[int, ...]]:
    """Infer (input_dim, archi) for a tower that ends right before the merge layer.

    Tower has only Linear + ReLU (+ BatchNorm1d/Dropout) blocks; no output Linear.
    So *all* Linear shapes are hidden dims, and the last one is the tower's output.
    """
    shapes = linear_shapes_in_order(state_dict, prefix=prefix)
    if not shapes:
        raise ValueError(f"No Linear weights found under prefix={prefix!r}.")
    input_dim = shapes[0][1]
    archi = tuple(s[0] for s in shapes)
    return input_dim, archi


# ---------------------------------------------------------------------------
# Feature name utilities
# ---------------------------------------------------------------------------


def flatten_columns(columns: pd.Index, sep: str = "::") -> list[str]:
    """Flatten a possibly multi-level column index to plain strings like
    ``"group::series"``.
    """
    if isinstance(columns, pd.MultiIndex):
        return [sep.join(str(x) for x in tup) for tup in columns]
    return [str(c) for c in columns]


def normalize_multiinput_shap(shap_out: Any, n_inputs: int, m_idx: int = 0) -> list:
    """Unwrap SHAP's multi-input return into a flat list of length ``n_inputs``,
    each element shape ``(batch, features_i)`` or ``(batch, features_i, n_outputs)``.

    SHAP's DeepExplainer produces different shapes across versions:

    * Multi-input, single-output:   ``list[n_inputs] of ndarray(b, f_i)``
    * Multi-input, multi-output:    ``list[n_inputs] of ndarray(b, f_i, n_out)``
                                    *or* ``list[n_out] of list[n_inputs] of ndarray(b, f_i)``

    This helper returns the per-input list regardless of which form was used,
    picking output ``m_idx`` when the outer list is per-output.
    """
    if not isinstance(shap_out, list):
        raise ValueError(
            f"Expected a list from multi-input DeepExplainer, got {type(shap_out)!r}."
        )
    if not shap_out:
        raise ValueError("Empty SHAP output.")

    # Case A: outer list is per-input (arrays or ndarrays directly).
    if len(shap_out) == n_inputs and all(
        not isinstance(x, list) for x in shap_out
    ):
        return shap_out

    # Case B: outer list is per-output; each element is a per-input list.
    if all(isinstance(x, list) and len(x) == n_inputs for x in shap_out):
        if not 0 <= m_idx < len(shap_out):
            raise ValueError(
                f"m_idx={m_idx} out of range for per-output list of length "
                f"{len(shap_out)}."
            )
        return shap_out[m_idx]

    raise ValueError(
        "Unrecognized multi-input SHAP output structure "
        f"(outer len={len(shap_out)}, n_inputs={n_inputs})."
    )


def select_output(arr: Any, m_idx: int = 0) -> np.ndarray:
    """Collapse an optional output axis from a SHAP per-input array.

    SHAP's DeepExplainer returns arrays of shape ``(batch, features)`` for
    single-output models and ``(batch, features, n_outputs)`` for multi-output
    models (per input tensor in the multi-input case). This helper returns the
    2D ``(batch, features)`` slice corresponding to ``m_idx``.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        n_outputs = arr.shape[-1]
        if not 0 <= m_idx < n_outputs:
            raise ValueError(
                f"m_idx={m_idx} out of range for SHAP output with "
                f"{n_outputs} output heads."
            )
        return arr[..., m_idx]
    raise ValueError(
        f"Unexpected SHAP array ndim={arr.ndim}; expected 2 or 3."
    )
