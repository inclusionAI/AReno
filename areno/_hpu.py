"""HPU bootstrap shared by packaging and runtime; imports only the standard library."""

from __future__ import annotations

import os
import platform
import sys
from importlib.machinery import PathFinder
from importlib.util import find_spec


def has_hpu_bridge() -> bool:
    if platform.system() != "Linux":
        return False
    parent = find_spec("habana_frameworks")
    if parent is None or parent.submodule_search_locations is None:
        return False
    # find_spec("habana_frameworks.torch") would import the parent package.
    return PathFinder.find_spec("habana_frameworks.torch", parent.submodule_search_locations) is not None


def configure_hpu_environment() -> str:
    """Set defaults before torch can auto-load the bridge; retain explicit values."""
    if sys.modules.get("habana_frameworks.torch") is not None and any(
        name not in os.environ for name in ("PT_HPU_LAZY_MODE", "PT_ENABLE_INT64_SUPPORT")
    ):
        raise RuntimeError(
            "The Gaudi bridge was imported before AReno could set its environment defaults. "
            "Import areno before torch, or set PT_HPU_LAZY_MODE and PT_ENABLE_INT64_SUPPORT before starting Python."
        )
    mode = os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
    if mode not in {"0", "1"}:
        raise ValueError("PT_HPU_LAZY_MODE must be 0 (eager) or 1 (lazy)")
    # Native index kernels consume actual int64 storage, not emulated int32.
    if os.environ.setdefault("PT_ENABLE_INT64_SUPPORT", "1").lower() not in {"1", "true"}:
        raise RuntimeError("AReno HPU kernels require PT_ENABLE_INT64_SUPPORT=1 before importing the Gaudi bridge")
    return mode
