"""SHAP adapter for ``MacroForwardANNWrapper`` (two-tower ANN).

The wrapper feeds ``X['forward']`` and ``X['fred']`` through two separate towers
that merge before the output layer. DeepExplainer natively supports multi-input
models: we pass a tuple of tensors for both the background and the test point
and concatenate the returned SHAP arrays along the feature axis.

Feature ordering: ``forward::*`` first, then ``fred::*``. The concatenation
order in ``flatten_shap`` must match.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from models.macro_forward_ann import _TwoTowerMLPNetwork
from .base import (
    apply_scaler_and_pca,
    flatten_columns,
    infer_tower_archi,
    normalize_multiinput_shap,
    rebuild_standard_scaler,
    select_output,
)


class MacroForwardAnnShapAdapter:
    wrapper_class = "MacroForwardANNWrapper"

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Feature extraction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _blocks(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        top = X.columns.get_level_values(0)
        for name in ("forward", "fred"):
            if name not in top:
                raise ValueError(f"X must contain a top-level {name!r} column block.")
        return X["forward"], X["fred"]

    def feature_names(
        self, X: pd.DataFrame, ckpt: dict | None = None
    ) -> list[str]:
        del ckpt  # unused: two-tower order is fixed (forward, fred)
        fwd, fred = self._blocks(X)
        fwd_names = [f"forward::{n}" for n in flatten_columns(fwd.columns)]
        fred_names = [f"fred::{n}" for n in flatten_columns(fred.columns)]
        return fwd_names + fred_names

    # ------------------------------------------------------------------ #
    # Model rebuild                                                      #
    # ------------------------------------------------------------------ #

    def rebuild_model(self, ckpt: dict, run_config: dict) -> torch.nn.Module:
        sd = ckpt["torch_state_dict"]
        if sd is None:
            raise ValueError("Checkpoint has no torch_state_dict.")

        input_dim_fwd, archi_fwd = infer_tower_archi(sd, prefix="fwd_tower.")
        input_dim_fred, archi_fred = infer_tower_archi(sd, prefix="fred_tower.")

        out_weight = sd.get("output.weight")
        if out_weight is None:
            raise ValueError("Two-tower checkpoint is missing output.weight.")
        output_dim = int(out_weight.shape[0])

        dropout_rate = 0.0
        best_params = ckpt.get("best_params_") or {}
        if isinstance(best_params, dict) and "dropout_rate" in best_params:
            try:
                dropout_rate = float(best_params["dropout_rate"])
            except (TypeError, ValueError):
                dropout_rate = 0.0

        model = _TwoTowerMLPNetwork(
            input_dim_fwd=input_dim_fwd,
            input_dim_fred=input_dim_fred,
            archi_fwd=archi_fwd,
            archi_fred=archi_fred,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
        )
        model.load_state_dict(sd)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    # Preprocessing                                                      #
    # ------------------------------------------------------------------ #

    def _scalers(self, ckpt: dict):
        fwd_scaler = rebuild_standard_scaler(ckpt.get("x_scaler_forward"))
        fred_scaler = rebuild_standard_scaler(ckpt.get("x_scaler_fred"))
        return fwd_scaler, fred_scaler

    def _transform(
        self, X_fwd: np.ndarray, X_fred: np.ndarray, ckpt: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        fwd_scaler, fred_scaler = self._scalers(ckpt)
        X_fwd_p = apply_scaler_and_pca(X_fwd, fwd_scaler, None)
        X_fred_p = apply_scaler_and_pca(X_fred, fred_scaler, None)
        return X_fwd_p, X_fred_p

    def prepare_inputs(
        self, X_rows: pd.DataFrame, ckpt: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fwd, fred = self._blocks(X_rows)
        X_fwd, X_fred = self._transform(fwd.values, fred.values, ckpt)
        return (
            torch.as_tensor(X_fwd, dtype=torch.float32),
            torch.as_tensor(X_fred, dtype=torch.float32),
        )

    def sample_background(
        self,
        X_train: pd.DataFrame,
        ckpt: dict,
        n: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fwd, fred = self._blocks(X_train)
        n_avail = len(fwd)
        if n_avail == 0:
            raise ValueError("Training window is empty; cannot sample background.")
        n_take = min(n, n_avail)
        idx = rng.choice(n_avail, size=n_take, replace=False)
        idx.sort()
        X_fwd, X_fred = self._transform(
            fwd.iloc[idx].values, fred.iloc[idx].values, ckpt
        )
        return (
            torch.as_tensor(X_fwd, dtype=torch.float32),
            torch.as_tensor(X_fred, dtype=torch.float32),
        )

    # ------------------------------------------------------------------ #
    # SHAP post-processing                                                #
    # ------------------------------------------------------------------ #

    def flatten_shap(
        self, shap_out: Any, feature_names: Sequence[str], m_idx: int = 0
    ) -> np.ndarray:
        """DeepExplainer with multi-input returns a list of per-input arrays.
        We concatenate them in (forward, fred) order to match ``feature_names``
        and select output head ``m_idx`` from any 3D arrays.
        """
        per_input = normalize_multiinput_shap(shap_out, n_inputs=2, m_idx=m_idx)
        fwd = select_output(per_input[0], m_idx=m_idx)
        fred = select_output(per_input[1], m_idx=m_idx)
        combined = np.concatenate([fwd, fred], axis=1)
        if combined.shape[1] != len(feature_names):
            raise ValueError(
                f"SHAP feature count {combined.shape[1]} does not match "
                f"len(feature_names)={len(feature_names)}."
            )
        return combined

    # ------------------------------------------------------------------ #
    # Y-scaling                                                           #
    # ------------------------------------------------------------------ #

    def y_scale_and_shift(
        self, ckpt: dict, y_center_default: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        y_state = ckpt.get("y_scaler") or {}
        scale = np.asarray(y_state.get("scale_", [1.0]), dtype=np.float64)
        mean = np.asarray(y_state.get("mean_", np.zeros_like(scale)), dtype=np.float64)
        shift = mean if y_center_default else np.zeros_like(scale)
        return scale, shift
