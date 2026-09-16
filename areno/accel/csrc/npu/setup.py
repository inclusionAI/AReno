"""Build the Ascend extension against the existing CANN-compatible PyTorch."""

from __future__ import annotations

import platform

from setuptools import setup


def build_extensions():
    if platform.system() != "Linux":
        raise RuntimeError("Build Ascend extensions on Linux with the existing PyTorch and torch_npu installation")
    import torch_npu  # noqa: F401
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    # ATen dispatch calls torch_npu's compiled CANN operators. This extension
    # contains no CUDA sources and does not replace the user's PyTorch build.
    return [
        CppExtension(
            "areno.accel._areno_accel_npu",
            sources=["areno/accel/csrc/npu/activation.cpp"],
            extra_compile_args=["-O2"],
        )
    ], {"build_ext": BuildExtension}


if __name__ == "__main__":
    extensions, commands = build_extensions()
    setup(ext_modules=extensions, cmdclass=commands)
