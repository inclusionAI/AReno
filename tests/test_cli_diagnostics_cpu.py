from __future__ import annotations

import json
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from areno.accel import _extension
from areno.cli import diagnostics
from areno.cli.main import main


def _ready_report(tmp_path: str) -> dict:
    return {
        "areno": {"version": "0.1.0"},
        "python": {"version": "3.11.0", "executable": "/python"},
        "platform": {"system": "Linux", "release": "6.0", "machine": "x86_64", "platform": "Linux"},
        "torch": {
            "imported": True,
            "error": None,
            "version": "2.6.0",
            "cuda_build": "12.4",
            "cuda_runtime": "12.4.0",
            "cuda_runtime_error": None,
            "cuda_available": True,
            "device_count": 1,
            "gpus": [{"index": 0, "name": "NVIDIA H100", "capability": "9.0"}],
        },
        "cuda": {
            "cuda_home": "/usr/local/cuda",
            "inferred_cuda_home": "/usr/local/cuda",
            "nvcc": {"path": "/usr/local/cuda/bin/nvcc", "version": "release 12.4"},
            "driver": {"path": "/usr/bin/nvidia-smi", "driver_version": "550.0", "cuda_version": "12.4", "error": None},
        },
        "gpus": [{"index": 0, "name": "NVIDIA H100", "capability": "9.0"}],
        "dependencies": {
            "flash_attn": {
                "distribution": "flash-attn",
                "module": "flash_attn",
                "version": "2.7.0",
                "imported": True,
                "error": None,
            },
            "flash_linear_attention": {
                "distribution": "flash-linear-attention",
                "module": "fla",
                "version": "0.2.0",
                "imported": True,
                "error": None,
            },
            "areno_accel": {
                "distribution": None,
                "module": "areno.accel._areno_accel",
                "version": None,
                "imported": True,
                "error": None,
            },
        },
        "install": {"build_ext_disabled": False},
        "env": {"CUDA_HOME": "/usr/local/cuda", "MAX_JOBS": "8"},
        "paths": {"metrics_log_dir": tmp_path, "hf_cache": tmp_path},
    }


def _mlx_report(tmp_path: str) -> dict:
    report = _ready_report(tmp_path)
    report["platform"].update(system="Darwin", machine="arm64", platform="macOS")
    for key in ("torch", "cuda", "gpus"):
        del report[key]
    report["dependencies"] = {
        name: {"distribution": distribution, "version": "1.0", "imported": True, "error": None}
        for name, distribution in (("mlx", "mlx"), ("mlx_lm", "mlx-lm"), ("mlx_vlm", "mlx-vlm"))
    }
    report["metal"] = {"available": True, "error": None}
    report["install"]["build_ext_disabled"] = True
    return report


class CliDiagnosticsTest(unittest.TestCase):
    def test_mlx_ready_without_torch_or_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            for machine in ("arm64", "aarch64"):
                with self.subTest(machine=machine):
                    report = _mlx_report(tmp)
                    report["platform"]["machine"] = machine
                    with patch.object(diagnostics, "collect_env", return_value=report):
                        result = CliRunner().invoke(diagnostics.check_command)
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertIn("AReno check: ready", result.output)
                    self.assertIn("OK   MLX Metal availability", result.output)
                    for text in ("CUDA", "PyTorch", "NVIDIA", "nvcc", "areno_accel", "flash", "ARENO_BUILD_EXT"):
                        self.assertNotIn(text, result.output)

    def test_mlx_missing_required_dependencies_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("mlx", "mlx_lm"):
                with self.subTest(name=name):
                    report = _mlx_report(tmp)
                    report["dependencies"][name].update(imported=False, error="ImportError: broken dependency")
                    with patch.object(diagnostics, "collect_env", return_value=report):
                        result = CliRunner().invoke(diagnostics.check_command)
                    self.assertEqual(result.exit_code, 1)
                    self.assertIn(f"FAIL {name} import", result.output)
                    self.assertIn("ImportError: broken dependency", result.output)
                    self.assertIn("python -m pip install -e .", result.output)
                    self.assertNotIn("CUDA", result.output)

    def test_mlx_vlm_missing_warns_for_multimodal_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _mlx_report(tmp)
            report["dependencies"]["mlx_vlm"].update(imported=False, error="ImportError: missing mlx_vlm")
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = CliRunner().invoke(diagnostics.check_command)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("WARN mlx_vlm import", result.output)
        self.assertIn("multimodal", result.output)

    def test_mlx_metal_unavailable_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _mlx_report(tmp)
            report["metal"]["available"] = False
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = CliRunner().invoke(diagnostics.check_command)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("FAIL MLX Metal availability", result.output)

    def test_intel_mac_is_not_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["platform"].update(system="Darwin", machine="x86_64")
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = CliRunner().invoke(diagnostics.check_command)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("FAIL Supported platform", result.output)
        self.assertIn("native arm64 Python", result.output)

    def test_collect_mlx_env_never_probes_cuda(self):
        mx = types.SimpleNamespace(metal=types.SimpleNamespace(is_available=lambda: True))
        modules = {"mlx.core": mx, "mlx_lm": object(), "mlx_vlm": object()}
        with (
            patch.object(diagnostics.platform, "system", return_value="Darwin"),
            patch.object(diagnostics.platform, "machine", return_value="arm64"),
            patch.object(diagnostics, "import_module", side_effect=modules.__getitem__) as imports,
            patch.object(diagnostics, "_package_version", return_value="1.0"),
            patch.object(diagnostics, "_torch_info") as torch_info,
            patch.object(diagnostics, "_nvidia_smi_driver_info") as nvidia_info,
            patch.object(diagnostics, "_nvcc_info") as nvcc_info,
        ):
            report = diagnostics.collect_env()
        torch_info.assert_not_called()
        nvidia_info.assert_not_called()
        nvcc_info.assert_not_called()
        self.assertEqual({call.args[0] for call in imports.call_args_list}, set(modules))
        self.assertTrue(report["metal"]["available"])
        self.assertTrue(all(dep["imported"] for dep in report["dependencies"].values()))

    def test_mlx_env_text_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _mlx_report(tmp)
            with patch.object(diagnostics, "collect_env", return_value=report):
                text = CliRunner().invoke(diagnostics.env_command)
                encoded = CliRunner().invoke(diagnostics.env_command, ["--json"])
        self.assertEqual(text.exit_code, 0, text.output)
        self.assertIn("Metal available: True", text.output)
        self.assertIn("mlx_lm: ok", text.output)
        self.assertNotIn("PyTorch CUDA build", text.output)
        self.assertEqual(encoded.exit_code, 0)
        self.assertTrue(json.loads(encoded.output)["metal"]["available"])

    def test_failed_mlx_core_import_is_not_retried_by_dependents(self):
        with (
            patch.object(diagnostics.platform, "system", return_value="Darwin"),
            patch.object(diagnostics.platform, "machine", return_value="arm64"),
            patch.object(diagnostics, "import_module", side_effect=ImportError("Metal device unavailable")) as imports,
            patch.object(diagnostics, "_package_version", return_value="1.0"),
        ):
            result = CliRunner().invoke(diagnostics.check_command)
        imports.assert_called_once_with("mlx.core")
        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("ImportError: Metal device unavailable", result.output)
        self.assertIn("Import skipped because mlx.core failed to import", result.output)

    def test_metal_probe_reports_errors(self):
        mx = types.SimpleNamespace(metal=types.SimpleNamespace())
        with patch.object(diagnostics, "import_module", return_value=mx):
            info = diagnostics._metal_info({"imported": True})
        self.assertFalse(info["available"])
        self.assertIn("AttributeError", info["error"])

        with patch.object(diagnostics, "import_module") as imports:
            info = diagnostics._metal_info({"imported": False, "error": "ImportError: missing mlx"})
        imports.assert_not_called()
        self.assertFalse(info["available"])
        self.assertEqual(info["error"], "ImportError: missing mlx")

    def test_collect_linux_env_keeps_cuda_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _ready_report(tmp)
            with (
                patch.object(diagnostics.platform, "system", return_value="Linux"),
                patch.object(diagnostics.platform, "machine", return_value="x86_64"),
                patch.object(diagnostics, "_torch_info", return_value=expected["torch"]),
                patch.object(diagnostics, "_cuda_home", return_value="/usr/local/cuda"),
                patch.object(diagnostics, "_nvcc_info", return_value=expected["cuda"]["nvcc"]),
                patch.object(diagnostics, "_nvidia_smi_driver_info", return_value=expected["cuda"]["driver"]),
                patch.object(
                    diagnostics, "_dependency_info", return_value=expected["dependencies"]["flash_attn"]
                ) as deps,
                patch.object(diagnostics, "_metal_info") as metal,
            ):
                report = diagnostics.collect_env()
        metal.assert_not_called()
        self.assertEqual(report["torch"], expected["torch"])
        self.assertEqual(report["gpus"], expected["gpus"])
        self.assertEqual(
            {call.args[1] for call in deps.call_args_list},
            {"flash_attn", "fla", "areno.accel._areno_accel"},
        )
        with patch.object(diagnostics, "collect_env", return_value=report):
            result = CliRunner().invoke(diagnostics.env_command)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("NVIDIA H100", result.output)
        self.assertIn("PyTorch CUDA build: 12.4", result.output)

    def test_top_level_cli_lists_env_and_check(self):
        result = CliRunner().invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("env", result.output)
        self.assertIn("check", result.output)

    def test_env_json_emits_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = CliRunner().invoke(diagnostics.env_command, ["--json"])

        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["areno"]["version"], "0.1.0")
        self.assertEqual(parsed["gpus"][0]["name"], "NVIDIA H100")

    def test_check_reports_failures_with_next_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["torch"]["cuda_available"] = False
            report["torch"]["device_count"] = 0
            report["torch"]["gpus"] = []
            report["gpus"] = []
            report["cuda"]["cuda_home"] = None
            report["cuda"]["nvcc"] = {"path": None, "version": None}
            report["dependencies"]["areno_accel"]["imported"] = False
            report["dependencies"]["areno_accel"]["error"] = "ModuleNotFoundError: missing extension"
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = CliRunner().invoke(diagnostics.check_command)

        self.assertEqual(result.exit_code, 1)
        self.assertIn("AReno check: not ready", result.output)
        self.assertIn("WARN CUDA_HOME", result.output)
        self.assertIn("WARN nvcc", result.output)
        self.assertIn("export CUDA_HOME=/usr/local/cuda", result.output)
        self.assertIn("FAIL areno_accel import", result.output)

    def test_cuda_toolkit_is_optional_when_runtime_extension_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["cuda"]["cuda_home"] = None
            report["cuda"]["nvcc"] = {"path": None, "version": None}
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = CliRunner().invoke(diagnostics.check_command)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("AReno check: ready", result.output)
        self.assertIn("OK   CUDA_HOME", result.output)
        self.assertIn("not required for runtime", result.output)
        self.assertIn("OK   nvcc", result.output)

    def test_check_reports_build_ext_disabled_runtime_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["install"]["build_ext_disabled"] = True
            report["dependencies"]["areno_accel"]["imported"] = False
            report["dependencies"]["areno_accel"]["error"] = "ModuleNotFoundError: missing extension"
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = CliRunner().invoke(diagnostics.check_command)

        self.assertEqual(result.exit_code, 1)
        self.assertIn("FAIL ARENO_BUILD_EXT", result.output)
        self.assertIn("ARENO_BUILD_EXT=0 skipped the runtime CUDA extension", result.output)
        self.assertIn("Reinstall without ARENO_BUILD_EXT=0", result.output)

    def test_writable_path_check_warns_for_existing_file(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            result = diagnostics._writable_path_check("cache", tmp_file.name)

        self.assertEqual(result.status, "WARN")
        self.assertIn("exists but is a file", result.detail)

    def test_version_check_pads_short_versions(self):
        self.assertTrue(diagnostics._version_at_least("3", (2, 6)))
        self.assertFalse(diagnostics._version_at_least("2", (2, 6)))

    def test_torch_info_handles_missing_cuda_build_attr(self):
        fake_torch = types.SimpleNamespace(
            __version__="2.6.0",
            cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
            version=types.SimpleNamespace(),
        )
        with patch.object(diagnostics, "import_module", return_value=fake_torch):
            info = diagnostics._torch_info()

        self.assertTrue(info["imported"])
        self.assertIsNone(info["cuda_build"])

    def test_nvidia_smi_empty_output_is_reported(self):
        completed = subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout="", stderr="")
        with (
            patch.object(diagnostics.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(diagnostics.subprocess, "run", return_value=completed),
        ):
            info = diagnostics._nvidia_smi_driver_info()

        self.assertEqual(info["error"], "nvidia-smi returned empty output")

    def test_runtime_extension_missing_error_is_actionable(self):
        _extension._EXT = None
        with (
            patch.dict("os.environ", {"ARENO_BUILD_EXT": "0"}),
            patch.object(_extension.importlib, "import_module", side_effect=ModuleNotFoundError("missing")),
            self.assertRaises(RuntimeError) as exc,
        ):
            _extension.extension()
        message = str(exc.exception)
        self.assertIn("ARENO_BUILD_EXT=0", message)
        self.assertIn("pip install -e . --no-build-isolation", message)


if __name__ == "__main__":
    unittest.main()
