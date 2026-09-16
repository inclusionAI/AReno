"""Build the Ascend extension against the existing CANN-compatible PyTorch."""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import setup


def _cmake_dir(root: Path) -> Path:
    for relative in (
        "compiler/tikcpp/ascendc_kernel_cmake",
        "tools/tikcpp/ascendc_kernel_cmake",
        "ascendc_devkit/tikcpp/samples/cmake",
    ):
        directory = root / relative
        if (directory / "ascendc.cmake").is_file():
            return directory
    raise RuntimeError(f"Ascend C compiler CMake files are missing under {root}; install the CANN development toolkit")


def _cann_root() -> Path:
    for name in ("ASCEND_HOME_PATH", "ASCEND_CANN_PACKAGE_PATH"):
        if value := os.environ.get(name):
            root = Path(value).expanduser().resolve()
            _cmake_dir(root)
            return root
    for root in (Path("/usr/local/Ascend/cann"), Path("/usr/local/Ascend/ascend-toolkit/latest")):
        if root.is_dir():
            try:
                _cmake_dir(root)
            except RuntimeError:
                continue
            return root.resolve()
    raise RuntimeError("CANN development toolkit not found; source its set_env.sh before installing AReno")


def _soc_version(cann: Path) -> str:
    soc = os.environ.get("ARENO_NPU_SOC")
    if not soc:
        import torch

        # Initialize through TorchNPU, keeping CANN ownership with the framework.
        # Do not guess a B/C variant from the generic npu-smi display name.
        torch.npu.init()
        library = ctypes.CDLL(str(cann / "lib64/libascendcl.so"))
        library.aclrtGetSocName.argtypes = []
        library.aclrtGetSocName.restype = ctypes.c_char_p
        value = library.aclrtGetSocName()
        if not value:
            raise RuntimeError("CANN could not detect the SoC; set ARENO_NPU_SOC only when cross-compiling")
        soc = value.decode("ascii")
    # DataCopyPad and BF16 require the A2/A3 ISA. Original Ascend 910 hardware
    # must not accidentally select these binaries merely because it shares a name.
    if not re.fullmatch(r"Ascend910(?:B[1-4](?:C|-1)?|_93[0-9]{2})", soc, re.IGNORECASE):
        raise RuntimeError(f"Ascend C activations require an A2/A3 SoC, detected {soc!r}")
    return soc


def build_extensions():
    if platform.system() != "Linux":
        raise RuntimeError("Build Ascend extensions on Linux with the existing PyTorch and torch_npu installation")
    import torch_npu  # noqa: F401
    from torch.utils.cpp_extension import BuildExtension
    from torch_npu.utils.cpp_extension import NpuExtension

    cann = _cann_root()
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("Ascend C extension builds require CMake; install cmake and retry")
    soc = _soc_version(cann)
    source_dir = Path(__file__).resolve().parent

    class AscendBuildExtension(BuildExtension):
        def build_extensions(self):
            kernel_build = Path(self.build_temp).resolve() / "ascendc"
            subprocess.run(
                [
                    cmake,
                    "-S",
                    str(source_dir),
                    "-B",
                    str(kernel_build),
                    f"-DASCEND_CANN_PACKAGE_PATH={cann}",
                    f"-DARENO_ASCENDC_CMAKE_DIR={_cmake_dir(cann)}",
                    f"-DSOC_VERSION={soc}",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                check=True,
            )
            command = [cmake, "--build", str(kernel_build)]
            if jobs := os.environ.get("MAX_JOBS"):
                command += ["--parallel", str(int(jobs))]
            subprocess.run(command, check=True)
            archive = kernel_build / "libareno_npu_kernels.a"
            if not archive.is_file():
                raise RuntimeError(f"Ascend C build did not produce {archive}")
            for ext in self.extensions:
                ext.extra_objects = [*ext.extra_objects, str(archive)]
                ext.depends = [*ext.depends, str(archive)]
            super().build_extensions()

    return [
        NpuExtension(
            "areno.accel._areno_accel_npu",
            sources=[
                f"areno/accel/csrc/npu/{name}.cpp"
                for name in ("extension", "activation", "normalization", "optimizer", "embedding", "linear", "conv")
            ],
            depends=[
                "areno/accel/csrc/grouped_linear_common.h",
                *[
                    f"areno/accel/csrc/npu/{name}_launch.h"
                    for name in ("activation", "normalization", "optimizer", "embedding", "linear", "conv")
                ],
            ],
            include_dirs=[str(cann / "include")],
            library_dirs=[str(cann / "lib64")],
            libraries=["ascendcl", "opapi_nn", "nnopbase"],
            extra_compile_args=["-O2"],
        )
    ], {"build_ext": AscendBuildExtension}


if __name__ == "__main__":
    extensions, commands = build_extensions()
    setup(ext_modules=extensions, cmdclass=commands)
