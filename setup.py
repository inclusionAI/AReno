from __future__ import annotations

import os
import platform
import shutil
import sys
import warnings

from setuptools import setup

_METADATA_COMMANDS = {"egg_info", "dist_info", "sdist"}
_MIN_TORCH_VERSION = (2, 6)


def _metadata_only_command() -> bool:
    return any(arg in _METADATA_COMMANDS for arg in sys.argv[1:])


def _cuda_extensions():
    if _metadata_only_command():
        return [], {}
    mode = os.environ.get("ARENO_BUILD_EXT", "1").lower()
    if mode in {"0", "false", "no", "off"}:
        return [], {}
    _check_supported_build_platform()
    torch = _require_torch()
    _check_torch_version(torch)
    _check_cuda_torch(torch)
    try:
        import psutil  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "AReno build-time setup failed: missing dependency `psutil`.\n"
            "Why: AReno builds CUDA extensions with `--no-build-isolation`, and PyTorch's CUDA extension builder "
            "imports psutil while sizing parallel compile jobs.\n"
            "Next steps: run `pip install psutil`, then retry `pip install -e . --no-build-isolation`."
        ) from exc
    _set_default_cuda_arch_list()
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

    if CUDA_HOME is None:
        raise RuntimeError(
            "AReno build-time setup failed: CUDA_HOME is not set.\n"
            "Why: building areno.accel requires the CUDA toolkit, not just a PyTorch CUDA wheel.\n"
            "Next steps: install the CUDA toolkit and export CUDA_HOME=/usr/local/cuda, or set "
            "ARENO_BUILD_EXT=0 only for docs/metadata installs that will not train or serve."
        )
    nvcc = shutil.which("nvcc") or shutil.which(str(os.path.join(CUDA_HOME, "bin", "nvcc")))
    if nvcc is None:
        raise RuntimeError(
            "AReno build-time setup failed: nvcc was not found.\n"
            "Why: building areno.accel requires CUDA's compiler in PATH or under CUDA_HOME/bin.\n"
            "Next steps: add CUDA's bin directory to PATH (`export PATH=$CUDA_HOME/bin:$PATH`) and retry."
        )
    return [
        CUDAExtension(
            "areno.accel._areno_accel",
            sources=[
                "areno/accel/csrc/extension.cpp",
                "areno/accel/csrc/activation.cu",
                "areno/accel/csrc/attention.cu",
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


def _check_supported_build_platform() -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux" and machine in {"x86_64", "amd64", "aarch64", "arm64"}:
        return
    raise RuntimeError(
        f"AReno runtime install is not supported on this platform: {system} {platform.machine()}.\n"
        "Why: AReno training/serving requires Linux with NVIDIA CUDA.\n"
        "Next steps: use Linux x86_64/aarch64 with an NVIDIA GPU, Windows WSL2, or set ARENO_BUILD_EXT=0 "
        "for docs/metadata-only installs on unsupported platforms."
    )


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "AReno build-time setup failed: PyTorch is not installed.\n"
            "Why: areno.accel is compiled with PyTorch's CUDA extension tooling.\n"
            "Next steps: install CUDA-enabled PyTorch >= 2.6 first, then retry with `--no-build-isolation`."
        ) from exc
    return torch


def _check_torch_version(torch) -> None:
    version = getattr(torch, "__version__", "")
    if _version_at_least(version, _MIN_TORCH_VERSION):
        return
    raise RuntimeError(
        f"AReno build-time setup failed: unsupported PyTorch version {version or '<unknown>'}.\n"
        "Why: AReno requires PyTorch >= 2.6 for its local runtime and CUDA extension ABI.\n"
        "Next steps: upgrade to CUDA-enabled PyTorch >= 2.6, then retry the AReno install."
    )


def _check_cuda_torch(torch) -> None:
    cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda_build:
        return
    raise RuntimeError(
        "AReno build-time setup failed: this PyTorch install is CPU-only.\n"
        "Why: AReno training/serving requires a CUDA-enabled PyTorch build and NVIDIA CUDA extensions.\n"
        "Next steps: install a CUDA-enabled PyTorch wheel matching your CUDA toolkit, then retry."
    )


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


def _version_at_least(version: str | None, minimum: tuple[int, int]) -> bool:
    if not version:
        return False
    parts: list[int] = []
    for piece in version.split("+", 1)[0].split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    while len(parts) < len(minimum):
        parts.append(0)
    return tuple(parts[: len(minimum)]) >= minimum


ext_modules, cmdclass = _cuda_extensions()


setup(ext_modules=ext_modules, cmdclass=cmdclass)
