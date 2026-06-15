from __future__ import annotations

import os
import sys
import warnings

from setuptools import setup

_METADATA_COMMANDS = {"egg_info", "dist_info", "sdist"}


def _metadata_only_command() -> bool:
    return any(arg in _METADATA_COMMANDS for arg in sys.argv[1:])


def _cuda_extensions():
    if _metadata_only_command():
        return [], {}
    mode = os.environ.get("ARENO_BUILD_EXT", "1").lower()
    if mode in {"0", "false", "no", "off"}:
        return [], {}
    try:
        import psutil  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "building areno.accel requires psutil to be installed before running "
            "`pip install ... --no-build-isolation`. PyTorch's CUDA extension "
            "builder imports psutil to size parallel compile jobs, and build "
            "isolation is disabled so pip will not install build-time dependencies "
            "for you. Fix: run `pip install psutil` in this environment, then "
            "retry the AReno install."
        ) from exc
    _set_default_cuda_arch_list()
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    if CUDA_HOME is None:
        raise RuntimeError(
            "building areno.accel requires CUDA_HOME; set ARENO_BUILD_EXT=0 to build docs/metadata without CUDA"
        )
    return [
        CUDAExtension(
            "areno.accel._areno_accel",
            sources=[
                "areno/accel/csrc/extension.cpp",
                "areno/accel/csrc/activation.cu",
                "areno/accel/csrc/conv.cu",
                "areno/accel/csrc/embedding.cu",
                "areno/accel/csrc/linear.cu",
                "areno/accel/csrc/moe_align_kernel.cu",
                "areno/accel/csrc/moe_permute.cu",
                "areno/accel/csrc/normalization.cu",
                "areno/accel/csrc/router.cu",
                "areno/accel/csrc/topk.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-Wno-deprecated-declarations"],
                "nvcc": ["-O3", "--use_fast_math", "-Xcompiler", "-Wno-deprecated-declarations"],
            },
        )
    ], {"build_ext": BuildExtension}


def _set_default_cuda_arch_list() -> None:
    """Default CUDA extension builds to the visible GPU architectures."""

    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    archs: set[str] = set()
    try:
        for idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(idx)
            archs.add(f"{major}.{minor}")
    except Exception:
        return
    if not archs:
        return
    arch_list = ";".join(sorted(archs))
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    warnings.warn(
        "TORCH_CUDA_ARCH_LIST is not set; building areno.accel only for "
        f"visible CUDA architecture(s): {arch_list}. Set TORCH_CUDA_ARCH_LIST "
        "explicitly to build for other GPUs.",
        RuntimeWarning,
        stacklevel=2,
    )


ext_modules, cmdclass = _cuda_extensions()


setup(ext_modules=ext_modules, cmdclass=cmdclass)
