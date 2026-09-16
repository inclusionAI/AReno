from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from email.parser import Parser
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile


def _load_setup_module() -> dict:
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    with (
        patch.object(sys, "argv", ["setup.py", "egg_info"]),
        patch("setuptools.setup"),
    ):
        return runpy.run_path(str(setup_path))


class SetupGuardrailsTest(unittest.TestCase):
    def test_hpu_detection_does_not_import_torch(self):
        setup_mod = _load_setup_module()
        detect = setup_mod["_using_hpu"]
        with (
            patch.object(setup_mod["platform"], "system", return_value="Linux"),
            patch.dict(sys.modules, {"torch": None, "habana_frameworks.torch": None}),
            patch.dict(detect.__globals__, {"find_spec": Mock(return_value=object())}),
        ):
            self.assertTrue(detect())

    def test_missing_bridge_preserves_default_build(self):
        setup_mod = _load_setup_module()
        detect = setup_mod["_using_hpu"]
        for result in (Mock(return_value=None), Mock(side_effect=ModuleNotFoundError)):
            with (
                patch.object(setup_mod["platform"], "system", return_value="Linux"),
                patch.dict(detect.__globals__, {"find_spec": result}),
            ):
                self.assertFalse(detect())
        cuda_build = Mock(return_value=(["cuda"], {}))
        with patch.dict(setup_mod["_extensions"].__globals__, {"_cuda_extensions": cuda_build}):
            self.assertEqual(setup_mod["_extensions"](False), (["cuda"], {}))
        cuda_build.assert_called_once_with()

    def test_macos_does_not_probe_hpu(self):
        setup_mod = _load_setup_module()
        detect = setup_mod["_using_hpu"]
        with (
            patch.object(setup_mod["platform"], "system", return_value="Darwin"),
            patch.dict(detect.__globals__, {"find_spec": Mock(side_effect=AssertionError("unexpected probe"))}),
        ):
            self.assertFalse(detect())

    def test_hpu_build_uses_native_builder(self):
        setup_mod = _load_setup_module()
        build = Mock(return_value=(["hpu"], {"build_ext": "hpu_builder"}))
        with (
            patch.object(sys, "argv", ["setup.py", "editable_wheel"]),
            patch.dict(os.environ, {"ARENO_BUILD_EXT": "auto"}),
            patch.object(setup_mod["runpy"], "run_path", return_value={"build_extensions": build}) as load,
        ):
            self.assertEqual(setup_mod["_extensions"](True), (["hpu"], {"build_ext": "hpu_builder"}))
        build.assert_called_once_with()
        self.assertTrue(load.call_args.args[0].endswith("areno/accel/csrc/hpu/setup.py"))

    def test_hpu_metadata_and_disabled_build_need_no_sdk(self):
        setup_mod = _load_setup_module()
        for command, mode in (("egg_info", "auto"), ("dist_info", "auto"), ("sdist", "auto"), ("editable_wheel", "0")):
            with (
                self.subTest(command=command, mode=mode),
                patch.object(sys, "argv", ["setup.py", command]),
                patch.dict(os.environ, {"ARENO_BUILD_EXT": mode}),
                patch.object(setup_mod["runpy"], "run_path", side_effect=AssertionError("unexpected SDK import")),
            ):
                self.assertEqual(setup_mod["_extensions"](True), ([], {}))

    def test_hpu_dependencies_preserve_bridge_torch(self):
        from packaging.requirements import Requirement

        setup_mod = _load_setup_module()
        hpu = {Requirement(value).name for value in setup_mod["_runtime_dependencies"](True)}
        default = {Requirement(value).name for value in setup_mod["_runtime_dependencies"](False)}
        self.assertTrue({"transformers", "safetensors", "datasets", "fastapi"} <= hpu)
        self.assertFalse({"torch", "torchvision", "flash-linear-attention", "mlx", "mlx-lm", "mlx-vlm"} & hpu)
        self.assertTrue({"torch", "torchvision", "flash-linear-attention", "mlx", "mlx-lm", "mlx-vlm"} <= default)

    def test_editable_build_selects_dependencies_without_importing_bridge(self):
        from packaging.requirements import Requirement

        root = Path(__file__).resolve().parents[1]
        for hpu in (False, True):
            with self.subTest(hpu=hpu), tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
                    shutil.copy2(root / name, project / name)
                shutil.copytree(root / "requirements", project / "requirements")
                (project / "areno").mkdir()
                (project / "areno/__init__.py").touch()
                # A discoverable bridge is enough; importing it or torch during
                # metadata generation would fail in this isolated subprocess.
                (project / "torch.py").write_text("raise RuntimeError('torch must not be imported')\n")
                bridge = project / "habana_frameworks"
                bridge.mkdir()
                (bridge / "__init__.py").touch()
                if hpu:
                    (bridge / "torch.py").write_text("raise RuntimeError('bridge must not be imported')\n")
                (project / "metadata").mkdir()
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import platform; platform.system = lambda: 'Linux'; "
                        "from setuptools.build_meta import prepare_metadata_for_build_editable, build_editable; "
                        "prepare_metadata_for_build_editable('metadata'); build_editable('metadata')",
                    ],
                    cwd=project,
                    env={**os.environ, "ARENO_BUILD_EXT": "0"},
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                metadata = next((project / "metadata").glob("*.dist-info/METADATA")).read_text()
                actual = [Requirement(value) for value in Parser().parsestr(metadata).get_all("Requires-Dist", [])]
                actual = [value for value in actual if not value.marker or "extra" not in str(value.marker)]
                expected = [Requirement(value) for value in _load_setup_module()["_runtime_dependencies"](hpu)]
                self.assertEqual(actual, expected)
                with ZipFile(next((project / "metadata").glob("*.whl"))) as wheel:
                    name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
                    self.assertEqual(wheel.read(name).decode(), metadata)

    def test_missing_torch_error_is_actionable(self):
        setup_mod = _load_setup_module()

        with (
            patch.dict(sys.modules, {"torch": None}),
            self.assertRaises(RuntimeError) as exc,
        ):
            setup_mod["_require_torch"]()
        message = str(exc.exception)
        self.assertIn("PyTorch is not installed", message)
        self.assertIn("CUDA-enabled PyTorch", message)

    def test_cpu_only_torch_error_is_actionable(self):
        setup_mod = _load_setup_module()
        fake_torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda=None))

        with self.assertRaises(RuntimeError) as exc:
            setup_mod["_check_cuda_torch"](fake_torch)
        message = str(exc.exception)
        self.assertIn("CPU-only", message)
        self.assertIn("CUDA-enabled PyTorch", message)

    def test_unsupported_platform_error_mentions_metadata_install(self):
        setup_mod = _load_setup_module()

        with (
            patch.object(setup_mod["platform"], "system", return_value="Darwin"),
            patch.object(setup_mod["platform"], "machine", return_value="arm64"),
            self.assertRaises(RuntimeError) as exc,
        ):
            setup_mod["_check_supported_build_platform"]()
        message = str(exc.exception)
        self.assertIn("not supported", message)
        self.assertIn("ARENO_BUILD_EXT=0", message)


if __name__ == "__main__":
    unittest.main()
