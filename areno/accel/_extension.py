"""Lazy device selection for AReno's native C extensions.

The extension module is imported on first use rather than at package import
time so that ``import areno.accel`` succeeds in environments where only the
Python shims are needed (e.g. for type checking). Each shim calls
``extension(tensor.device)`` to obtain the compiled module and dispatch into the fused
kernel. A missing native extension raises an explicit error; there is no
pure-Python or cross-device fallback.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

# Cached reference to the compiled extension; populated on first call.
_EXT: ModuleType | None = None
_NPU_EXT: ModuleType | None = None


def extension(device="cuda") -> ModuleType:
    """Load native kernels for this tensor's device; no cross-device fallback."""
    global _EXT, _NPU_EXT
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type == "npu":
        if _NPU_EXT is None:
            importlib.import_module("torch_npu")
            try:
                _NPU_EXT = importlib.import_module("areno.accel._areno_accel_npu")
            except ModuleNotFoundError as exc:
                if exc.name not in {None, "areno.accel._areno_accel_npu"}:
                    raise
                raise RuntimeError(
                    "AReno native NPU kernels are not installed: areno.accel._areno_accel_npu. "
                    "Run `python -m pip install -e . --no-build-isolation` in the CANN/torch_npu environment."
                ) from exc
        return _NPU_EXT
    if device_type != "cuda":
        raise RuntimeError(f"AReno native kernels require CUDA or NPU tensors, got {device_type}")
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
