"""Packaging and target detection checks; these do not emulate Ascend kernels."""

import ctypes
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from setuptools import Distribution, Extension
from torch.utils.cpp_extension import BuildExtension

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(monkeypatch):
    for name in ("ARENO_NPU_SOC", "ASCEND_HOME_PATH", "ASCEND_CANN_PACKAGE_PATH"):
        monkeypatch.delenv(name, raising=False)
    return runpy.run_path(str(ROOT / "areno/accel/csrc/npu/setup.py"))


@pytest.mark.parametrize("soc", ["Ascend910B2", "Ascend910B2C", "Ascend910B4-1", "Ascend910_9391"])
def test_exact_soc_is_queried_from_cann(builder, monkeypatch, tmp_path, soc):
    init = Mock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(init=init), raising=False)
    query = Mock(return_value=soc.encode())
    load = Mock(return_value=SimpleNamespace(aclrtGetSocName=query))
    monkeypatch.setattr(ctypes, "CDLL", load)
    assert builder["_soc_version"](tmp_path) == soc
    init.assert_called_once_with()
    query.assert_called_once_with()
    assert query.restype is ctypes.c_char_p
    load.assert_called_once_with(str(tmp_path / "lib64/libascendcl.so"))


def test_cross_build_soc_override_does_not_initialize_hardware(builder, monkeypatch, tmp_path):
    monkeypatch.setenv("ARENO_NPU_SOC", "Ascend910_9381")
    monkeypatch.setattr(ctypes, "CDLL", Mock(side_effect=AssertionError("unexpected device query")))
    assert builder["_soc_version"](tmp_path) == "Ascend910_9381"


@pytest.mark.parametrize("soc", ["Ascend910", "Ascend910B", "Ascend310P3", "Ascend950", "unknown"])
def test_incompatible_isa_is_rejected(builder, monkeypatch, tmp_path, soc):
    monkeypatch.setenv("ARENO_NPU_SOC", soc)
    with pytest.raises(RuntimeError, match="require an A2/A3 SoC"):
        builder["_soc_version"](tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "compiler/tikcpp/ascendc_kernel_cmake",
        "tools/tikcpp/ascendc_kernel_cmake",
        "ascendc_devkit/tikcpp/samples/cmake",
    ],
)
def test_cann_toolkit_layouts(builder, monkeypatch, tmp_path, relative):
    directory = tmp_path / relative
    directory.mkdir(parents=True)
    (directory / "ascendc.cmake").touch()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    assert builder["_cann_root"]() == tmp_path.resolve()
    assert builder["_cmake_dir"](tmp_path) == directory


def test_runtime_only_cann_install_is_rejected(builder, monkeypatch, tmp_path):
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="install the CANN development toolkit"):
        builder["_cann_root"]()


def test_kernel_archive_is_linked_and_triggers_extension_rebuild(builder, monkeypatch, tmp_path):
    # Mock compiler execution, not its output's correctness. Verify the build
    # driver cannot omit the device archive or keep a stale host .so after edits.
    monkeypatch.setattr(builder["platform"], "system", lambda: "Linux")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setenv("ARENO_NPU_SOC", "Ascend910_9391")
    monkeypatch.setenv("MAX_JOBS", "2")
    directory = tmp_path / "compiler/tikcpp/ascendc_kernel_cmake"
    directory.mkdir(parents=True)
    (directory / "ascendc.cmake").touch()
    monkeypatch.setattr(builder["shutil"], "which", lambda name: "/test/bin/cmake")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch_npu.utils", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch_npu.utils.cpp_extension", SimpleNamespace(NpuExtension=Extension))
    extensions, commands = builder["build_extensions"]()
    command = commands["build_ext"](Distribution({"ext_modules": extensions}))
    command.ensure_finalized()
    command.build_temp = str(tmp_path / "build")
    kernel_build = Path(command.build_temp) / "ascendc"
    archive = kernel_build / "libareno_npu_kernels.a"
    calls = []

    def mock_cmake(args, *, check):
        assert check
        calls.append(args)
        if "--build" in args:
            archive.parent.mkdir(parents=True)
            archive.touch()

    def mock_host_build(self):
        assert str(archive) in self.extensions[0].extra_objects
        assert str(archive) in self.extensions[0].depends
        assert "areno/accel/csrc/npu/activation_launch.h" in self.extensions[0].depends
        assert "areno/accel/csrc/grouped_linear_common.h" in self.extensions[0].depends
        assert "areno/accel/csrc/routing_common.h" in self.extensions[0].depends
        assert "areno/accel/csrc/npu/conv_launch.h" in self.extensions[0].depends
        assert "areno/accel/csrc/npu/conv.cpp" in self.extensions[0].sources
        assert "areno/accel/csrc/npu/routing.cpp" in self.extensions[0].sources
        assert {"opapi_nn", "nnopbase"}.issubset(self.extensions[0].libraries)
        calls.append("host")

    monkeypatch.setattr(builder["subprocess"], "run", mock_cmake)
    monkeypatch.setattr(BuildExtension, "build_extensions", mock_host_build)
    command.build_extensions()
    assert "-DSOC_VERSION=Ascend910_9391" in calls[0]
    assert calls[1] == ["/test/bin/cmake", "--build", str(kernel_build), "--parallel", "2"]
    assert calls[2] == "host"


def test_sdist_includes_native_build_inputs(tmp_path):
    for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"):
        shutil.copy2(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "requirements", tmp_path / "requirements")
    shutil.copytree(ROOT / "areno/accel/csrc/npu", tmp_path / "areno/accel/csrc/npu")
    shutil.copy2(ROOT / "areno/accel/csrc/grouped_linear_common.h", tmp_path / "areno/accel/csrc")
    shutil.copy2(ROOT / "areno/accel/csrc/routing_common.h", tmp_path / "areno/accel/csrc")
    (tmp_path / "areno/__init__.py").touch()
    result = subprocess.run(
        [sys.executable, "-c", "from setuptools.build_meta import build_sdist; build_sdist('dist')"],
        cwd=tmp_path,
        env={**os.environ, "ARENO_BUILD_EXT": "0"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    assert "areno/accel/csrc/grouped_linear_common.h" in names
    assert "areno/accel/csrc/routing_common.h" in names
    for name in (
        "setup.py",
        "CMakeLists.txt",
        "extension.cpp",
        "activation.cpp",
        "activation_kernel.cpp",
        "activation_launch.h",
        "normalization.cpp",
        "normalization_kernel.cpp",
        "normalization_launch.h",
        "optimizer.cpp",
        "optimizer_kernel.cpp",
        "optimizer_quantized_kernel.cpp",
        "optimizer_factored_kernel.cpp",
        "optimizer_launch.h",
        "embedding.cpp",
        "embedding_kernel.cpp",
        "embedding_launch.h",
        "linear.cpp",
        "linear_kernel.cpp",
        "linear_launch.h",
        "conv.cpp",
        "conv_kernel.cpp",
        "conv_launch.h",
        "routing.cpp",
        "routing_kernel.cpp",
        "routing_launch.h",
    ):
        assert f"areno/accel/csrc/npu/{name}" in names
