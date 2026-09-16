"""HPU defaults and build target detection without importing the Gaudi bridge."""

import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from setuptools import Extension

from areno._hpu import configure_hpu_environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def hpu_build(monkeypatch):
    for name in ("ARENO_HPU_ARCH", "PT_HPU_LAZY_MODE", "PT_ENABLE_INT64_SUPPORT"):
        monkeypatch.delenv(name, raising=False)
    # Track defaults set by the build helper so they cannot leak to other tests.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    return runpy.run_path(str(ROOT / "areno/accel/csrc/hpu/setup.py"))


def test_environment_defaults(hpu_build):
    assert configure_hpu_environment() == "1"
    assert os.environ["PT_HPU_LAZY_MODE"] == "1"
    assert os.environ["PT_ENABLE_INT64_SUPPORT"] == "1"


def test_explicit_eager_mode_is_preserved(hpu_build, monkeypatch):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    monkeypatch.setenv("PT_ENABLE_INT64_SUPPORT", "true")
    assert configure_hpu_environment() == "0"
    assert os.environ["PT_ENABLE_INT64_SUPPORT"] == "true"


@pytest.mark.parametrize("variable,value", [("PT_HPU_LAZY_MODE", "2"), ("PT_ENABLE_INT64_SUPPORT", "0")])
def test_invalid_explicit_settings_are_not_overwritten(hpu_build, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises((ValueError, RuntimeError), match=variable):
        configure_hpu_environment()
    assert os.environ[variable] == value


def test_already_loaded_bridge_requires_early_configuration(hpu_build, monkeypatch):
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", SimpleNamespace())
    with pytest.raises(RuntimeError, match="Import areno before torch"):
        configure_hpu_environment()
    assert "PT_HPU_LAZY_MODE" not in os.environ
    assert "PT_ENABLE_INT64_SUPPORT" not in os.environ
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    monkeypatch.setenv("PT_ENABLE_INT64_SUPPORT", "1")
    assert configure_hpu_environment() == "0"


@pytest.mark.parametrize(
    "names,expected",
    [("HL-225\nHL-225\n", "gaudi2"), ("HL-325\nHL-338\n", "gaudi3"), ("Gaudi 2\n", "gaudi2"), ('"Gaudi3"\n', "gaudi3")],
)
def test_detect_arch_from_hl_smi(hpu_build, monkeypatch, names, expected):
    def query(command, **kwargs):
        assert command == ["hl-smi", "-Q", "name", "-f", "csv,noheader"]
        assert kwargs["timeout"] == 10
        return SimpleNamespace(stdout=names)

    monkeypatch.setattr(hpu_build["subprocess"], "run", query)
    assert hpu_build["detect_arch"]() == expected


@pytest.mark.parametrize("names", ["", "HL-225\nHL-325\n", "Gaudi\n", "unknown\n"])
def test_ambiguous_or_unsupported_arch_never_defaults_to_gaudi2(hpu_build, monkeypatch, names):
    monkeypatch.setattr(hpu_build["subprocess"], "run", lambda *args, **kwargs: SimpleNamespace(stdout=names))
    with pytest.raises(RuntimeError, match="ARENO_HPU_ARCH"):
        hpu_build["detect_arch"]()


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(),
        PermissionError(),
        subprocess.TimeoutExpired("hl-smi", 10),
        subprocess.CalledProcessError(1, "hl-smi"),
    ],
)
def test_detection_failure_explains_offline_build(hpu_build, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(hpu_build["subprocess"], "run", fail)
    with pytest.raises(RuntimeError, match="offline build"):
        hpu_build["detect_arch"]()


@pytest.mark.parametrize("arch", ["gaudi2", "gaudi3", "gaudi1", ""])
def test_explicit_arch_does_not_query_hardware(hpu_build, monkeypatch, arch):
    monkeypatch.setenv("ARENO_HPU_ARCH", arch)

    def unexpected(*args, **kwargs):
        pytest.fail("Explicit architecture should bypass hardware detection")

    monkeypatch.setattr(hpu_build["subprocess"], "run", unexpected)
    if arch in {"gaudi2", "gaudi3"}:
        assert hpu_build["detect_arch"]() == arch
    else:
        with pytest.raises(ValueError, match="ARENO_HPU_ARCH"):
            hpu_build["detect_arch"]()


@pytest.mark.parametrize("mode,plugin", [(None, "habana_pytorch_plugin"), ("0", "habana_pytorch2_plugin")])
def test_builder_uses_defaults_before_bridge_and_selects_target(hpu_build, monkeypatch, tmp_path, mode, plugin):
    if mode is not None:
        monkeypatch.setenv("PT_HPU_LAZY_MODE", mode)
    monkeypatch.setattr(hpu_build["platform"], "system", lambda: "Linux")
    monkeypatch.setattr(hpu_build["shutil"], "which", lambda name: "tpc-clang")
    monkeypatch.setattr(hpu_build["subprocess"], "run", lambda *a, **k: SimpleNamespace(stdout="HL-325\n"))
    monkeypatch.setenv("TPC_INCLUDE_DIR", str(tmp_path))
    for name in ("gc_interface.h", "tpc_kernel_lib_interface.h"):
        (tmp_path / name).touch()

    def sdk_directory():
        assert os.environ["PT_HPU_LAZY_MODE"] == (mode or "1")
        assert os.environ["PT_ENABLE_INT64_SUPPORT"] == "1"
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "habana_frameworks.torch.utils.lib_utils",
        SimpleNamespace(get_include_dir=sdk_directory, get_lib_dir=sdk_directory),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch.utils.cpp_extension",
        SimpleNamespace(BuildExtension=type("BuildExtension", (), {}), CppExtension=Extension),
    )
    extensions, _ = hpu_build["build_extensions"]()
    assert ("ARENO_TPC_DEVICE", "tpc_lib_api::DEVICE_ID_GAUDI3") in extensions[0].define_macros
    assert ("ARENO_HPU_LAZY_MODE", mode or "1") in extensions[1].define_macros
    assert extensions[1].libraries == [plugin]
    assert all(not Path(source).is_absolute() for extension in extensions for source in extension.sources)


@pytest.mark.parametrize("bridge,mode", [(False, None), (True, None), (True, "0")])
def test_import_sets_defaults_without_loading_torch_or_bridge(tmp_path, bridge, mode):
    parent = tmp_path / "habana_frameworks"
    parent.mkdir()
    (parent / "__init__.py").write_text("raise RuntimeError('parent must not be imported')\n")
    if bridge:
        (parent / "torch.py").write_text("raise RuntimeError('bridge must not be imported')\n")
    env = {
        key: value for key, value in os.environ.items() if key not in {"PT_HPU_LAZY_MODE", "PT_ENABLE_INT64_SUPPORT"}
    }
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(ROOT)))
    if mode is not None:
        env["PT_HPU_LAZY_MODE"] = mode
    script = (
        "import os, platform, sys; platform.system = lambda: 'Linux'; import areno; "
        "assert 'torch' not in sys.modules; assert 'habana_frameworks' not in sys.modules; "
        f"assert os.environ.get('PT_HPU_LAZY_MODE') == {((mode or '1') if bridge else None)!r}; "
        f"assert os.environ.get('PT_ENABLE_INT64_SUPPORT') == {('1' if bridge else None)!r}"
    )
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
