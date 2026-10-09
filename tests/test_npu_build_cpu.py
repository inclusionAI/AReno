"""Packaging and target detection checks; these do not emulate Ascend kernels."""

import ctypes
import os
import runpy
import shutil
import subprocess
import sys
import sysconfig
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


@pytest.mark.parametrize("soc", ["Ascend910B2", "Ascend910B2C", "Ascend910B4-1", "Ascend910_9382", "Ascend910_9391"])
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


@pytest.mark.parametrize("archive_relative", ["lib/libareno_npu_kernels.a", "libareno_npu_kernels.a", None])
def test_kernel_archive_is_linked_and_triggers_extension_rebuild(builder, monkeypatch, tmp_path, archive_relative):
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
    archive = kernel_build / (archive_relative or "libareno_npu_kernels.a")
    calls = []
    validated = []
    monkeypatch.setitem(builder["build_extensions"].__globals__, "_check_launcher_symbols", validated.append)
    imported = []
    monkeypatch.setitem(builder["build_extensions"].__globals__, "_check_extension_import", imported.append)

    def mock_cmake(args, *, check):
        assert check
        calls.append(args)
        if "--build" in args and archive_relative is not None:
            archive.parent.mkdir(parents=True)
            archive.touch()

    def mock_host_build(self):
        assert str(archive) in self.extensions[0].extra_objects
        assert str(archive) in self.extensions[0].depends
        assert "areno/accel/csrc/npu/activation_launch.h" in self.extensions[0].depends
        assert "areno/accel/csrc/grouped_linear_common.h" in self.extensions[0].depends
        assert "areno/accel/csrc/routing_common.h" in self.extensions[0].depends
        assert "areno/accel/csrc/moe_permute_common.h" in self.extensions[0].depends
        assert "areno/accel/csrc/npu/conv_launch.h" in self.extensions[0].depends
        assert "areno/accel/csrc/npu/conv.cpp" in self.extensions[0].sources
        assert "areno/accel/csrc/npu/attention.cpp" in self.extensions[0].sources
        assert "areno/accel/csrc/npu/attention_launch.h" in self.extensions[0].depends
        assert "areno/accel/csrc/npu/routing.cpp" in self.extensions[0].sources
        assert "areno/accel/csrc/npu/moe.cpp" in self.extensions[0].sources
        assert "areno/accel/csrc/npu/fused_experts.cpp" in self.extensions[0].sources
        assert "areno/accel/csrc/npu/fused_experts_launch.h" in self.extensions[0].depends
        assert "areno/accel/csrc/npu/moe.h" in self.extensions[0].depends
        assert "areno/accel/csrc/npu/tensor_format.h" in self.extensions[0].depends
        assert {"opapi_nn", "nnopbase", "tiling_api", "platform"}.issubset(self.extensions[0].libraries)
        calls.append("host")

    monkeypatch.setattr(builder["subprocess"], "run", mock_cmake)
    monkeypatch.setattr(BuildExtension, "build_extensions", mock_host_build)
    if archive_relative is None:
        with pytest.raises(RuntimeError, match="Ascend C build did not produce"):
            command.build_extensions()
        assert "host" not in calls
        assert not validated
        assert not imported
        return
    command.build_extensions()
    assert "-DSOC_VERSION=Ascend910_9391" in calls[0]
    assert calls[1] == ["/test/bin/cmake", "--build", str(kernel_build), "--parallel", "2"]
    assert calls[2] == "host"
    assert validated == [Path(command.get_ext_fullpath(extensions[0].name))]
    assert imported == validated


@pytest.mark.parametrize("defined", [False, True])
def test_unresolved_template_launchers_fail_during_build(builder, tmp_path, defined):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None or shutil.which("nm") is None:
        pytest.skip("C++ compiler and nm required")
    # Link an actual shared library. Like CANN's host wrapper, it references
    # a template specialization that may be absent from the generated stub.
    source = tmp_path / "launch.cpp"
    source.write_text(
        "template<typename T, unsigned Op> unsigned aclrtlaunch_activation_kernel();\n"
        + ("template<> unsigned aclrtlaunch_activation_kernel<float, 0>() { return 0; }\n" if defined else "")
        + 'extern "C" unsigned launch() { return aclrtlaunch_activation_kernel<float, 0>(); }\n'
    )
    library = tmp_path / "launch.so"
    flags = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
    subprocess.run([compiler, "-shared", "-fPIC", *flags, str(source), "-o", str(library)], check=True)
    if defined:
        builder["_check_launcher_symbols"](library)
    else:
        with pytest.raises(
            RuntimeError,
            match="unresolved CANN kernel launchers.*",
        ):
            builder["_check_launcher_symbols"](library)


@pytest.mark.parametrize("failure", [None, "missing_symbol", "module_init"])
def test_built_extension_is_loaded_with_frameworks_before_install(builder, monkeypatch, tmp_path, failure):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None or shutil.which("nm") is None:
        pytest.skip("C++ compiler and nm required")
    # Use a real Python extension and the dynamic loader, with only the two
    # framework imports stubbed. No Ascend hardware or TorchNPU wheel is needed.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "torch.py").write_text("")
    (tmp_path / "torch_npu.py").write_text("import sys\nassert 'torch' in sys.modules\n")
    (tmp_path / "_areno_accel_npu.py").write_text("raise AssertionError('loaded a stale extension')\n")
    source = tmp_path / "extension.cpp"
    source.write_text(
        "#include <Python.h>\n"
        'extern "C" int areno_test_dependency();\n'
        + ('extern "C" int areno_test_dependency() { return 1; }\n' if failure != "missing_symbol" else "")
        + "int (*dependency)() = areno_test_dependency;\n"
        'static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_areno_accel_npu", nullptr, -1, nullptr};\n'
        "PyMODINIT_FUNC PyInit__areno_accel_npu() {\n"
        '    if (!PyDict_GetItemString(PyImport_GetModuleDict(), "torch") ||\n'
        '        !PyDict_GetItemString(PyImport_GetModuleDict(), "torch_npu")) {\n'
        '        PyErr_SetString(PyExc_ImportError, "frameworks must be imported first"); return nullptr;\n'
        "    }\n"
        + (
            '    PyErr_SetString(PyExc_RuntimeError, "test module init failure"); return nullptr;\n'
            if failure == "module_init"
            else "    return PyModule_Create(&module);\n"
        )
        + "}\n"
    )
    library = tmp_path / "build output" / "_areno_accel_npu.so"
    library.parent.mkdir()
    flags = ["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []
    subprocess.run(
        [compiler, "-shared", "-fPIC", *flags, "-I", sysconfig.get_path("include"), str(source), "-o", str(library)],
        check=True,
    )
    # These errors are invisible to the CANN-only launcher check.
    builder["_check_launcher_symbols"](library)
    if failure is None:
        builder["_check_extension_import"](library)
    else:
        detail = "areno_test_dependency" if failure == "missing_symbol" else "test module init failure"
        with pytest.raises(RuntimeError, match=detail) as error:
            builder["_check_extension_import"](library)
        assert "cannot be imported after building" in str(error.value)
        assert str(library) in str(error.value)


def test_sdist_includes_native_build_inputs(tmp_path):
    for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE", "MANIFEST.in"):
        shutil.copy2(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "requirements", tmp_path / "requirements")
    shutil.copytree(ROOT / "areno/accel/csrc/npu", tmp_path / "areno/accel/csrc/npu")
    shutil.copy2(ROOT / "areno/accel/csrc/grouped_linear_common.h", tmp_path / "areno/accel/csrc")
    shutil.copy2(ROOT / "areno/accel/csrc/routing_common.h", tmp_path / "areno/accel/csrc")
    shutil.copy2(ROOT / "areno/accel/csrc/moe_permute_common.h", tmp_path / "areno/accel/csrc")
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
    assert "areno/accel/csrc/moe_permute_common.h" in names
    for name in (
        "setup.py",
        "CMakeLists.txt",
        "extension.cpp",
        "tensor_format.h",
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
        "attention.cpp",
        "attention_kernel.cpp",
        "attention_launch.h",
        "routing.cpp",
        "routing_kernel.cpp",
        "routing_launch.h",
        "moe.cpp",
        "moe_kernel.cpp",
        "moe_launch.h",
        "moe.h",
        "fused_experts.cpp",
        "fused_experts_kernel.cpp",
        "fused_experts_matmul_kernel.cpp",
        "fused_experts_launch.h",
    ):
        assert f"areno/accel/csrc/npu/{name}" in names
