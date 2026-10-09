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
    def test_macos_does_not_probe_npu(self):
        setup_mod = _load_setup_module()
        detect = setup_mod["_using_npu"]
        with (
            patch.object(setup_mod["platform"], "system", return_value="Darwin"),
            patch.dict(detect.__globals__, {"find_spec": Mock(side_effect=AssertionError("unexpected probe"))}),
        ):
            self.assertFalse(detect())

    def test_npu_build_uses_native_builder(self):
        setup_mod = _load_setup_module()
        build = Mock(return_value=(["npu"], {"build_ext": "npu_builder"}))
        with (
            patch.object(sys, "argv", ["setup.py", "editable_wheel"]),
            patch.dict(os.environ, {"ARENO_BUILD_EXT": "auto"}),
            patch.object(setup_mod["runpy"], "run_path", return_value={"build_extensions": build}) as load,
        ):
            self.assertEqual(setup_mod["_extensions"](True), (["npu"], {"build_ext": "npu_builder"}))
        build.assert_called_once_with()
        self.assertTrue(load.call_args.args[0].endswith("areno/accel/csrc/npu/setup.py"))

    def test_npu_metadata_and_disabled_build_need_no_sdk(self):
        setup_mod = _load_setup_module()
        for command, mode in (("egg_info", "auto"), ("dist_info", "auto"), ("sdist", "auto"), ("editable_wheel", "0")):
            with (
                self.subTest(command=command, mode=mode),
                patch.object(sys, "argv", ["setup.py", command]),
                patch.dict(os.environ, {"ARENO_BUILD_EXT": mode}),
                patch.object(setup_mod["runpy"], "run_path", side_effect=AssertionError("unexpected SDK import")),
            ):
                self.assertEqual(setup_mod["_extensions"](True), ([], {}))

    def test_npu_dependencies_preserve_bridge_torch(self):
        from packaging.requirements import Requirement

        setup_mod = _load_setup_module()
        npu = {Requirement(value).name for value in setup_mod["_runtime_dependencies"](True)}
        default = {Requirement(value).name for value in setup_mod["_runtime_dependencies"](False)}
        self.assertTrue({"transformers", "safetensors", "datasets", "fastapi"} <= npu)
        self.assertTrue({"flash-linear-attention", "flash-attn-npu"} <= npu)
        self.assertFalse({"torch", "torchvision", "triton", "mlx", "mlx-lm", "mlx-vlm"} & npu)
        fla = next(
            Requirement(value) for value in setup_mod["_runtime_dependencies"](True) if value.startswith("flash-linear")
        )
        self.assertFalse(fla.extras)
        self.assertTrue(fla.url.endswith("e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2"))
        self.assertTrue({"torch", "torchvision", "flash-linear-attention", "mlx", "mlx-lm", "mlx-vlm"} <= default)

    def test_editable_build_selects_dependencies_without_importing_bridge(self):
        from packaging.requirements import Requirement

        root = Path(__file__).resolve().parents[1]
        for npu in (False, True):
            with self.subTest(npu=npu), tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
                    shutil.copy2(root / name, project / name)
                shutil.copytree(root / "requirements", project / "requirements")
                (project / "areno").mkdir()
                (project / "areno/__init__.py").touch()
                # A discoverable bridge is enough; importing it or torch during
                # metadata generation would fail in this isolated subprocess.
                (project / "torch.py").write_text("raise RuntimeError('torch must not be imported')\n")
                if npu:
                    (project / "torch_npu.py").write_text("raise RuntimeError('bridge must not be imported')\n")
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
                expected = [Requirement(value) for value in _load_setup_module()["_runtime_dependencies"](npu)]
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
