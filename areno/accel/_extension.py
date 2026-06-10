"""Lazy loader for the compiled ``areno.accel._areno_accel`` C++/CUDA extension.

The extension module is imported on first use rather than at package import
time so that ``import areno.accel`` succeeds in environments where only the
Python shims are needed (e.g. for type checking). Each shim calls
``extension()`` to obtain the compiled module and dispatch into the fused
kernel. There is no pure-Python fallback: if the extension was not built the
``importlib.import_module`` call below raises ``ModuleNotFoundError``.
"""
from __future__ import annotations

import importlib
from types import ModuleType

# Cached reference to the compiled extension; populated on first call.
_EXT: ModuleType | None = None


def extension() -> ModuleType:
    """Return the compiled C++/CUDA extension module, importing it lazily."""
    global _EXT
    if _EXT is None:
        _EXT = importlib.import_module("areno.accel._areno_accel")
    return _EXT
