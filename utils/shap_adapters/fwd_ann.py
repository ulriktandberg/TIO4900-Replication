"""SHAP adapter for ``PyTorchMLPWrapper`` (forward-rates-only ANN).

The wrapper in ``models.ann_vector_validation`` slices out ``X['forward']`` inside
``_extract_array``, so the network only ever sees forward rates. That means:
  - feature names = the (flattened) columns of ``X['forward']``
  - single-input DeepExplainer call

If ``use_pca=True`` was used at training, the checkpoint will contain ``pca`` state
and the adapter will apply it before feeding the network. In that case the SHAP
values live in PCA-component space, not original-feature space. For the current
notebook (``use_pca=False``) this is a no-op.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from models.ann_vector_validation import _MLPNetwork
from .base import (
    apply_scaler_and_pca,
    flatten_columns,
    infer_mlp_archi,
    rebuild_pca,
    rebuild_standard_scaler,
    select_output,
)


class FwdAnnShapAdapter:
    wrapper_class = "PyTorchMLPWrapper"

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Feature extraction                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _forward_block(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if "forward" not in X.columns.get_level_values(0):
            raise ValueError("X must contain a top-level 'forward' column block.")
        return X["forward"]

    def feature_names(
        self, X: pd.DataFrame, ckpt: dict | None = None
    ) -> list[str]:
        del ckpt  # unused: single-input model
        block = self._forward_block(X)
        return flatten_columns(block.columns)

    # ------------------------------------------------------------------ #
    # Model rebuild                                                      #
    # ------------------------------------------------------------------ #

    def rebuild_model(self, ckpt: dict, run_config: dict) -> torch.nn.Module:
        sd = ckpt["torch_state_dict"]
        if sd is None:
            raise ValueError("Checkpoint has no torch_state_dict.")
        input_dim, archi, output_dim = infer_mlp_archi(sd, prefix="network.")
        model = _MLPNetwork(input_dim=input_dim, archi=archi, output_dim=output_dim)
        model.load_state_dict(sd)
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    # Preprocessing                                                      #
    # ------------------------------------------------------------------ #

    def _scaler_pca(self, ckpt: dict):
        # y_center is not stored in the current checkpoint schema; for x_scaler
        # it doesn't matter (x uses default StandardScaler(with_mean=True,
        # with_std=True) inside the wrapper).
        x_scaler = rebuild_standard_scaler(ckpt.get("x_scaler"))
        pca = rebuild_pca(ckpt.get("pca"))
        return x_scaler, pca

    def _transform(self, X_block_values: np.ndarray, ckpt: dict) -> np.ndarray:
        scaler, pca = self._scaler_pca(ckpt)
        return apply_scaler_and_pca(X_block_values, scaler, pca)

    def prepare_inputs(self, X_rows: pd.DataFrame, ckpt: dict) -> torch.Tensor:
        block = self._forward_block(X_rows)
        X = self._transform(block.values, ckpt)
        return torch.as_tensor(X, dtype=torch.float32)

    def sample_background(
        self,
        X_train: pd.DataFrame,
        ckpt: dict,
        n: int,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        block = self._forward_block(X_train)
        n_avail = len(block)
        if n_avail == 0:
            raise ValueError("Training window is empty; cannot sample background.")
        n_take = min(n, n_avail)
        idx = rng.choice(n_avail, size=n_take, replace=False)
        idx.sort()
        X = self._transform(block.iloc[idx].values, ckpt)
        return torch.as_tensor(X, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # SHAP post-processing                                                #
    # ------------------------------------------------------------------ #

    def flatten_shap(
        self, shap_out: Any, feature_names: Sequence[str], m_idx: int = 0
    ) -> np.ndarray:
        """Flatten SHAP output to ``(n_rows, n_features)`` selecting output
        ``m_idx``. Accepts SHAP's per-version shapes:
          * ``ndarray(n, f)``            — single output
          * ``ndarray(n, f, n_outputs)`` — multi-output
          * ``list[ndarray(n, f)]``      — list-per-output (length n_outputs)
        """
        if isinstance(shap_out, list):
            if not shap_out:
                raise ValueError("Empty SHAP output.")
            if len(shap_out) == 1:
                arr = shap_out[0]
            else:
                # list-per-output: pick m_idx-th output
                if not 0 <= m_idx < len(shap_out):
                    raise ValueError(
                        f"m_idx={m_idx} out of range for list-per-output SHAP "
                        f"(len={len(shap_out)})."
                    )
                arr = shap_out[m_idx]
        else:
            arr = shap_out
        arr = select_output(arr, m_idx=m_idx)
        if arr.shape[1] != len(feature_names):
            raise ValueError(
                f"SHAP feature count {arr.shape[1]} does not match "
                f"len(feature_names)={len(feature_names)}."
            )
        return arr

    # ------------------------------------------------------------------ #
    # Y-scaling                                                           #
    # ------------------------------------------------------------------ #

    def y_scale_and_shift(
        self, ckpt: dict, y_center_default: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scale_, shift_) s.t. y_raw = y_scaled * scale_ + shift_.

        ``shift_`` is zero when the wrapper was configured with ``y_center=False``
        (default in the notebooks); otherwise it is ``y_scaler.mean_``. Since
        current checkpoints do not store ``with_mean``, we fall back to
        ``y_center_default`` from the runner config.
        """
        y_state = ckpt.get("y_scaler") or {}
        scale = np.asarray(y_state.get("scale_", [1.0]), dtype=np.float64)
        mean = np.asarray(y_state.get("mean_", np.zeros_like(scale)), dtype=np.float64)
        shift = mean if y_center_default else np.zeros_like(scale)
        return scale, shift
