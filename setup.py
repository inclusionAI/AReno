from __future__ import annotations

import os

from setuptools import setup


def _cuda_extensions():
    mode = os.environ.get("ARENO_BUILD_EXT", "1").lower()
    if mode in {"0", "false", "no", "off"}:
        return [], {}
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    if CUDA_HOME is None:
        raise RuntimeError(
            "building areno.accel requires CUDA_HOME; "
            "set ARENO_BUILD_EXT=0 to build docs/metadata without CUDA"
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


ext_modules, cmdclass = _cuda_extensions()


setup(ext_modules=ext_modules, cmdclass=cmdclass)
