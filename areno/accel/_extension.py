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
import importlib.util
import os
from types import ModuleType

from areno._hpu import configure_hpu_environment

# Cached reference to the compiled extension; populated on first call.
_EXT: ModuleType | None = None
_HPU_EXT: ModuleType | None = None


def configure_hpu_kernel_library() -> None:
    """Expose AReno's TPC library before the Gaudi graph compiler initializes."""
    spec = importlib.util.find_spec("areno.accel._areno_hpu_kernels")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "AReno native HPU kernels are not installed: areno.accel._areno_hpu_kernels. "
            "Build them with `python areno/accel/csrc/hpu/setup.py build_ext --inplace` in a Gaudi SDK environment."
        )
    configure_hpu_environment()
    # Setting GC_KERNEL_PATH replaces the compiler's defaults. Retain the
    # standard Gaudi library when the environment has no explicit list.
    configured = os.environ.get("GC_KERNEL_PATH") or "/usr/lib/habanalabs/libtpc_kernels.so"
    paths = [path for path in configured.split(os.pathsep) if path]
    if spec.origin not in paths:
        os.environ["GC_KERNEL_PATH"] = os.pathsep.join([*paths, spec.origin])


def extension(device="cuda") -> ModuleType:
    """Load native kernels for this tensor's device; no cross-device fallback."""
    global _EXT, _HPU_EXT
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type == "hpu":
        if _HPU_EXT is None:
            configure_hpu_kernel_library()
            try:
                _HPU_EXT = importlib.import_module("areno.accel._areno_accel_hpu")
            except ModuleNotFoundError as exc:
                if exc.name not in {None, "areno.accel._areno_accel_hpu"}:
                    raise
                raise RuntimeError(
                    "AReno native HPU kernels are not installed: areno.accel._areno_accel_hpu. "
                    "Build them with `python areno/accel/csrc/hpu/setup.py build_ext --inplace` in a Gaudi SDK environment."
                ) from exc
        return _HPU_EXT
    if device_type != "cuda":
        raise RuntimeError(f"AReno native kernels require CUDA or HPU tensors, got {device_type}")
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
