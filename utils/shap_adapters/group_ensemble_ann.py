"""SHAP adapter for ``GroupEnsembleANNWrapper``.

Structure of the wrapped model (see ``models.group_ensemble_ann``):

    inputs:
        x_fwd                 : (batch, n_fwd)
        x_macro_dict          : dict mapping ``str(group_name) -> (batch, n_group_i)``

    towers:
        fwd_tower             : MLP over forward rates, ends with BatchNorm1d
        macro_towers[<grp>]   : one MLP per FRED group, each ends with BatchNorm1d
        merge_bn + output     : concat all tower outputs, BatchNorm1d, final Linear

DeepExplainer can only pass a flat sequence of tensors, not a dict. We therefore
wrap the inner model in :class:`_GroupEnsembleForSHAP` whose ``forward`` takes
positional tensors (``x_fwd, x_g1, x_g2, ...``) and reconstructs the dict.

**Canonical group order**: the *insertion order* of the macro-tower keys in
the checkpoint's ``state_dict``. PyTorch's ``OrderedDict`` state dicts
preserve the order in which ``ModuleDict`` entries were registered at train
time; because ``_GroupEnsembleMLPNetwork.forward`` concatenates tower outputs
in that iteration order, rebuilding with any other order would misalign the
subsequent ``merge_bn`` + ``output`` layers. We therefore recover and reuse
this order in rebuild / feature_names / prepare_inputs / flatten_shap, and
validate set-equality against X['fred'] groups to catch pipeline drift.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.group_ensemble_ann import _GroupEnsembleMLPNetwork
from .base import (
    apply_scaler_and_pca,
    flatten_columns,
    infer_tower_archi,
    normalize_multiinput_shap,
    rebuild_standard_scaler,
    select_output,
)


# --------------------------------------------------------------------------- #
# SHAP-compatible model wrapper                                               #
# --------------------------------------------------------------------------- #


class _GroupEnsembleForSHAP(nn.Module):
    """Adapts ``_GroupEnsembleMLPNetwork`` to a positional-arg ``forward``.

    SHAP's DeepExplainer calls ``model(*inputs)`` with a list of tensors;
    the inner network expects ``(x_fwd, dict)``. This wrapper rebuilds the dict
    from positional args using the fixed ``group_order``.
    """

    def __init__(self, inner: _GroupEnsembleMLPNetwork, group_order: Sequence[str]):
        super().__init__()
        self.inner = inner
        self._group_order = tuple(str(g) for g in group_order)

    @property
    def group_order(self) -> tuple[str, ...]:
        return self._group_order

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        if len(xs) != 1 + len(self._group_order):
            raise ValueError(
                f"Expected 1 + {len(self._group_order)} input tensors "
                f"(x_fwd + one per macro group), got {len(xs)}."
            )
        x_fwd = xs[0]
        macro_dict = {g: xs[i + 1] for i, g in enumerate(self._group_order)}
        return self.inner(x_fwd, macro_dict)


# --------------------------------------------------------------------------- #
# Adapter                                                                     #
# --------------------------------------------------------------------------- #


class GroupEnsembleAnnShapAdapter:
    wrapper_class = "GroupEnsembleANNWrapper"

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Block / group extraction                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_blocks(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        top = X.columns.get_level_values(0)
        for name in ("forward", "fred"):
            if name not in top:
                raise ValueError(f"X must contain a top-level {name!r} column block.")
        return X["forward"], X["fred"]

    @staticmethod
    def _groups_from_X(X_fred: pd.DataFrame) -> list[str]:
        if "group" not in X_fred.columns.names:
            raise ValueError(
                "X['fred'] must have a MultiIndex with a 'group' level."
            )
        return list(X_fred.columns.get_level_values("group").unique().astype(str))

    @staticmethod
    def _group_order_from_state_dict(sd: dict) -> list[str]:
        """Return the macro-tower group names in state_dict *insertion order*.

        PyTorch's state_dict is an OrderedDict. ``ModuleDict`` preserves
        insertion order, and ``state_dict()`` iterates submodules in that
        order. So the first occurrence of each ``macro_towers.<name>.*`` key
        reflects the training-time registration order.
        """
        ordered: list[str] = []
        seen: set[str] = set()
        pat = re.compile(r"^macro_towers\.(.+?)\.")
        for key in sd:
            m = pat.match(key)
            if m is None:
                continue
            grp = m.group(1)
            if grp not in seen:
                seen.add(grp)
                ordered.append(grp)
        return ordered

    @classmethod
    def _canonical_group_order(cls, sd: dict, X_fred: pd.DataFrame) -> list[str]:
        """State-dict insertion order, after validating X and state_dict agree."""
        x_groups = set(cls._groups_from_X(X_fred))
        sd_order = cls._group_order_from_state_dict(sd)
        if not sd_order:
            raise ValueError(
                "Checkpoint contains no macro_towers.* keys; is this really a "
                "GroupEnsembleANNWrapper checkpoint?"
            )
        sd_groups = set(sd_order)
        missing = sd_groups.difference(x_groups)
        extra = x_groups.difference(sd_groups)
        if missing or extra:
            raise ValueError(
                "FRED group mismatch between checkpoint and X['fred']. "
                f"In checkpoint only: {sorted(missing)}. "
                f"In X only: {sorted(extra)}."
            )
        return sd_order

    def feature_names(
        self, X: pd.DataFrame, ckpt: dict | None = None
    ) -> list[str]:
        """Feature names using the checkpoint's state-dict group order.

        ``ckpt`` is required for this adapter: group order at training time is
        recorded in the state_dict and any other order produces wrong results
        through the downstream ``merge_bn`` + ``output`` layers. The runner
        loads a representative checkpoint up front and threads it through.
        """
        fwd, fred = self._split_blocks(X)
        if ckpt is None or not ckpt.get("torch_state_dict"):
            raise ValueError(
                "GroupEnsembleAnnShapAdapter.feature_names requires a checkpoint "
                "hint (ckpt=sample_ckpt) because the group ordering is defined "
                "by the training-time state_dict. Pass the sample checkpoint "
                "from the runner."
            )
        group_order = self._canonical_group_order(ckpt["torch_state_dict"], fred)
        names = [f"forward::{n}" for n in flatten_columns(fwd.columns)]
        for g in group_order:
            sub = fred[g]
            for n in flatten_columns(sub.columns):
                names.append(f"{g}::{n}")
        return names

    # ------------------------------------------------------------------ #
    # Model rebuild                                                      #
    # ------------------------------------------------------------------ #

    def rebuild_model(self, ckpt: dict, run_config: dict) -> torch.nn.Module:
        sd = ckpt.get("torch_state_dict")
        if sd is None:
            raise ValueError("Checkpoint has no torch_state_dict.")

        input_dim_fwd, archi_fwd = infer_tower_archi(sd, prefix="fwd_tower.")

        group_order = self._group_order_from_state_dict(sd)
        if not group_order:
            raise ValueError(
                "Checkpoint contains no macro_towers.* keys; is this really a "
                "GroupEnsembleANNWrapper checkpoint?"
            )
        # Canonical order = state_dict insertion order. This must match the
        # ModuleDict insertion order at train time so ``forward`` concatenates
        # tower outputs in the layout expected by ``merge_bn`` + ``output``.
        macro_group_dims: dict[str, int] = {}
        archi_macro: tuple[int, ...] | None = None
        for grp in group_order:
            in_dim, archi_grp = infer_tower_archi(
                sd, prefix=f"macro_towers.{grp}."
            )
            macro_group_dims[grp] = in_dim
            if archi_macro is None:
                archi_macro = archi_grp
            elif archi_macro != archi_grp:
                raise ValueError(
                    f"Macro tower archi mismatch across groups: "
                    f"{archi_macro} vs {archi_grp} (group={grp!r})."
                )
        assert archi_macro is not None

        out_weight = sd.get("output.weight")
        if out_weight is None:
            raise ValueError("Group-ensemble checkpoint is missing output.weight.")
        output_dim = int(out_weight.shape[0])

        dropout_rate = 0.0
        best_params = ckpt.get("best_params_") or {}
        if isinstance(best_params, dict) and "dropout_rate" in best_params:
            try:
                dropout_rate = float(best_params["dropout_rate"])
            except (TypeError, ValueError):
                dropout_rate = 0.0

        inner = _GroupEnsembleMLPNetwork(
            input_dim_fwd=input_dim_fwd,
            macro_group_dims=macro_group_dims,
            archi_fwd=archi_fwd,
            archi_macro=archi_macro,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
        )
        inner.load_state_dict(sd)
        inner.eval()
        return _GroupEnsembleForSHAP(inner, group_order=group_order)

    # ------------------------------------------------------------------ #
    # Preprocessing                                                      #
    # ------------------------------------------------------------------ #

    def _scalers(self, ckpt: dict) -> tuple[Any, dict[str, Any]]:
        fwd_scaler = rebuild_standard_scaler(ckpt.get("x_scaler_forward"))
        macro_states = ckpt.get("x_scalers_macro") or {}
        macro_scalers: dict[str, Any] = {}
        for grp, state in macro_states.items():
            macro_scalers[str(grp)] = rebuild_standard_scaler(state)
        return fwd_scaler, macro_scalers

    def _prepare_arrays(
        self, X_rows: pd.DataFrame, ckpt: dict
    ) -> list[np.ndarray]:
        sd = ckpt["torch_state_dict"]
        fwd_block, fred_block = self._split_blocks(X_rows)
        group_order = self._canonical_group_order(sd, fred_block)

        fwd_scaler, macro_scalers = self._scalers(ckpt)
        pieces: list[np.ndarray] = [
            apply_scaler_and_pca(fwd_block.values, fwd_scaler, None)
        ]
        for grp in group_order:
            scaler = macro_scalers.get(grp)
            if scaler is None:
                raise ValueError(
                    f"Checkpoint is missing saved scaler for FRED group {grp!r}."
                )
            arr = fred_block[grp].values
            pieces.append(apply_scaler_and_pca(arr, scaler, None))
        return pieces

    def prepare_inputs(
        self, X_rows: pd.DataFrame, ckpt: dict
    ) -> list[torch.Tensor]:
        arrs = self._prepare_arrays(X_rows, ckpt)
        return [torch.as_tensor(a, dtype=torch.float32) for a in arrs]

    def sample_background(
        self,
        X_train: pd.DataFrame,
        ckpt: dict,
        n: int,
        rng: np.random.Generator,
    ) -> list[torch.Tensor]:
        n_avail = len(X_train)
        if n_avail == 0:
            raise ValueError("Training window is empty; cannot sample background.")
        n_take = min(n, n_avail)
        idx = rng.choice(n_avail, size=n_take, replace=False)
        idx.sort()
        return self.prepare_inputs(X_train.iloc[idx], ckpt)

    # ------------------------------------------------------------------ #
    # SHAP post-processing                                                #
    # ------------------------------------------------------------------ #

    def flatten_shap(
        self, shap_out: Any, feature_names: Sequence[str], m_idx: int = 0
    ) -> np.ndarray:
        """Concatenate per-input SHAP arrays into a single ``(n_rows, F)``
        matrix in (forward, group_1, group_2, ...) order, matching the
        order produced by :meth:`feature_names`.
        """
        if isinstance(shap_out, list) and shap_out and isinstance(
            shap_out[0], list
        ):
            n_inputs = len(shap_out[0])
        elif isinstance(shap_out, list):
            n_inputs = len(shap_out)
        else:
            raise ValueError(
                f"Expected a list from multi-input DeepExplainer, got "
                f"{type(shap_out)!r}."
            )

        per_input = normalize_multiinput_shap(
            shap_out, n_inputs=n_inputs, m_idx=m_idx
        )
        pieces = [select_output(arr, m_idx=m_idx) for arr in per_input]
        combined = np.concatenate(pieces, axis=1)
        if combined.shape[1] != len(feature_names):
            raise ValueError(
                f"SHAP feature count {combined.shape[1]} does not match "
                f"len(feature_names)={len(feature_names)}. Check that the "
                "group order in feature_names matches the checkpoint."
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
