"""Adapter registry. New wrapper classes register themselves here."""

from __future__ import annotations

from typing import Dict, Type

from .base import ShapAdapter  # noqa: F401
from .fwd_ann import FwdAnnShapAdapter
from .group_ensemble_ann import GroupEnsembleAnnShapAdapter
from .macro_forward_ann import MacroForwardAnnShapAdapter


_REGISTRY: Dict[str, Type] = {
    FwdAnnShapAdapter.wrapper_class: FwdAnnShapAdapter,
    GroupEnsembleAnnShapAdapter.wrapper_class: GroupEnsembleAnnShapAdapter,
    MacroForwardAnnShapAdapter.wrapper_class: MacroForwardAnnShapAdapter,
}


def get_adapter(wrapper_class: str):
    """Return an instantiated adapter for a given wrapper class name.

    Raises a clear error listing the registered adapters if unknown.
    """
    if wrapper_class not in _REGISTRY:
        raise ValueError(
            f"No SHAP adapter registered for wrapper_class={wrapper_class!r}. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[wrapper_class]()


def register_adapter(cls) -> None:
    """Register an adapter class (decorator-friendly)."""
    _REGISTRY[cls.wrapper_class] = cls
    return cls


__all__ = ["ShapAdapter", "get_adapter", "register_adapter"]
