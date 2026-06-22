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
import os
from types import ModuleType

# Cached reference to the compiled extension; populated on first call.
_EXT: ModuleType | None = None


def extension() -> ModuleType:
    """Return the compiled C++/CUDA extension module, importing it lazily."""
    global _EXT
    if _EXT is None:
        try:
            _EXT = importlib.import_module("areno.accel._areno_accel")
        except ModuleNotFoundError as exc:
            build_ext = os.environ.get("ARENO_BUILD_EXT")
            build_hint = (
                " The current environment has ARENO_BUILD_EXT=0, which intentionally skips compiling the extension."
                if build_ext is not None and build_ext.lower() in {"0", "false", "no", "off"}
                else ""
            )
            raise RuntimeError(
                "AReno runtime setup failed: the compiled `areno_accel` extension is not installed."
                f"{build_hint}\n"
                "Why: training and serving require AReno's CUDA extension at runtime.\n"
                "Next steps: reinstall with CUDA enabled, for example `pip install -e . --no-build-isolation`, "
                "then run `areno check`."
            ) from exc
    return _EXT
