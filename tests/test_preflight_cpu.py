"""CPU tests for pre-flight environment checks and resolved command output.

These tests mock ``collect_env()`` so they run on any machine without requiring
a GPU or CUDA.  The ``_ready_report`` helper builds a fully-passing environment
report; individual tests mutate specific fields to simulate failures.

Test coverage:
- PreflightCheck / PreflightResult dataclass behavior
- CRITICAL failures (no GPU, no PyTorch) block unconditionally
- ERROR failures (wrong version, missing accel, bad paths) block by default
- WARNING (missing flash_attn) never blocks
- Resolved command filtering per algorithm (SFT vs GSPO vs DPO)
- train_command integration: preflight + resolved command output, --skip-check
"""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from areno.api.trainer_config import PolicyTrainerConfig, TrainerConfig
from areno.cli import diagnostics
from areno.cli import train as train_cli
from areno.cli.diagnostics import (
    PreflightCheck,
    PreflightResult,
    run_preflight_checks,
)
from click.testing import CliRunner
from click import unstyle


def _ready_report(tmp_path: str) -> dict:
    """A fully-passing environment report, mirroring test_cli_diagnostics_cpu."""

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
            "device_count": 2,
            "gpus": [
                {"index": 0, "name": "NVIDIA T4", "capability": "7.5"},
                {"index": 1, "name": "NVIDIA T4", "capability": "7.5"},
            ],
        },
        "cuda": {
            "cuda_home": "/usr/local/cuda",
            "inferred_cuda_home": "/usr/local/cuda",
            "nvcc": {"path": "/usr/local/cuda/bin/nvcc", "version": "release 12.4"},
            "driver": {"path": "/usr/bin/nvidia-smi", "driver_version": "550.0", "cuda_version": "12.4", "error": None},
        },
        "gpus": [
            {"index": 0, "name": "NVIDIA T4", "capability": "7.5"},
            {"index": 1, "name": "NVIDIA T4", "capability": "7.5"},
        ],
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


def _ready_config(world_size: int = 2, algo: str = "gspo", **kwargs) -> TrainerConfig:
    """Build a minimal passing config for preflight tests."""

    defaults = dict(
        algo=algo,
        ckpt="Qwen/Qwen3-0.6B",
        dataset_path="gsm8k:main",
        model_hub="hf",
        tp_size=1,
        world_size=world_size,
        batch_size=2,
        mini_bs=1,
        save_path=None,
        save_interval=100,
        epochs=1,
        metrics_log_dir=None,
    )
    defaults.update(kwargs)
    if algo in {"gspo", "grpo"}:
        defaults.setdefault("reward_fn_path", "examples/math/math_verify_reward.py")
        defaults.setdefault("n_samples", 2)
        return PolicyTrainerConfig(**defaults)
    return TrainerConfig(**defaults)


class PreflightCheckTest(unittest.TestCase):
    """Test run_preflight_checks with various environment conditions."""

    def test_all_pass_returns_no_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            config = _ready_config()
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = run_preflight_checks(config)

        self.assertEqual(result.critical_failures, [])
        self.assertEqual(result.errors, [])
        self.assertGreater(len(result.passed), 0)

    def test_critical_failure_when_pytorch_not_importable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["torch"]["imported"] = False
            report["torch"]["version"] = None
            report["torch"]["error"] = "ModuleNotFoundError: No module named 'torch'"
            report["torch"]["cuda_available"] = False
            report["torch"]["device_count"] = 0
            report["gpus"] = []
            config = _ready_config()
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = run_preflight_checks(config)

        crit_names = [c.name for c in result.critical_failures]
        self.assertIn("PyTorch import", crit_names)
        self.assertIn("torch.cuda.is_available()", crit_names)

    def test_critical_failure_when_no_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["torch"]["cuda_available"] = False
            report["torch"]["device_count"] = 0
            report["torch"]["gpus"] = []
            report["gpus"] = []
            config = _ready_config()
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = run_preflight_checks(config)

        crit_names = [c.name for c in result.critical_failures]
        self.assertIn("torch.cuda.is_available()", crit_names)
        self.assertIn("NVIDIA GPU visibility", crit_names)

    def test_error_when_gpu_count_less_than_world_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["torch"]["device_count"] = 2
            report["gpus"] = [
                {"index": 0, "name": "NVIDIA T4", "capability": "7.5"},
                {"index": 1, "name": "NVIDIA T4", "capability": "7.5"},
            ]
            config = _ready_config(world_size=8)
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = run_preflight_checks(config)

        err_names = [c.name for c in result.errors]
        self.assertIn("GPU count >= world_size", err_names)
        self.assertEqual(result.critical_failures, [])

    def test_error_when_areno_accel_not_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["dependencies"]["areno_accel"]["imported"] = False
            report["dependencies"]["areno_accel"]["error"] = "ModuleNotFoundError"
            config = _ready_config()
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = run_preflight_checks(config)

        err_names = [c.name for c in result.errors]
        self.assertIn("areno_accel import", err_names)

    def test_warning_for_missing_flash_attn(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["dependencies"]["flash_attn"]["imported"] = False
            report["dependencies"]["flash_attn"]["error"] = "ModuleNotFoundError"
            config = _ready_config()
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = run_preflight_checks(config)

        warn_names = [c.name for c in result.warnings]
        self.assertIn("flash_attn import", warn_names)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.critical_failures, [])

    def test_error_for_missing_local_dataset_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            config = _ready_config(dataset_path="/nonexistent/file.json")
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = run_preflight_checks(config)

        err_names = [c.name for c in result.errors]
        self.assertIn("Dataset path", err_names)

    def test_pass_for_remote_dataset_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            config = _ready_config(dataset_path="AI-MO/NuminaMath-CoT")
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = run_preflight_checks(config)

        dataset_check = next(c for c in result.checks if c.name == "Dataset path")
        self.assertEqual(dataset_check.level, "PASS")
        self.assertIn("remote", dataset_check.detail)

    def test_error_for_missing_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            config = _ready_config(ckpt="/nonexistent/model.ckpt")
            with patch.object(diagnostics, "collect_env", return_value=report):
                result = run_preflight_checks(config)

        err_names = [c.name for c in result.errors]
        self.assertIn("Checkpoint path", err_names)

    def test_kaggle_detection_via_env_var(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            config = _ready_config()
            with (
                patch.object(diagnostics, "collect_env", return_value=report),
                patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "interactive"}),
            ):
                result = run_preflight_checks(config)

        self.assertTrue(result.kaggle_detected)


class PreflightResultDataclassTest(unittest.TestCase):
    """Test the PreflightResult dataclass properties."""

    def test_result_properties_partition_checks_correctly(self):
        checks = [
            PreflightCheck("PASS", "ok1"),
            PreflightCheck("WARN", "warn1"),
            PreflightCheck("ERROR", "err1"),
            PreflightCheck("CRITICAL", "crit1"),
            PreflightCheck("PASS", "ok2"),
        ]
        result = PreflightResult(checks=checks)

        self.assertEqual(len(result.passed), 2)
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(len(result.critical_failures), 1)


class ResolvedCommandTest(unittest.TestCase):
    """Test _format_resolved_command output."""

    def test_sft_command_omits_rollout_params(self):
        config = _ready_config(algo="sft", dataset_loader_fn="examples/sft/alpaca/dataset_loader.py")
        cmd = train_cli._format_resolved_command(config)

        self.assertIn("--algo sft", cmd)
        self.assertIn("--ckpt", cmd)
        self.assertIn("--dataset-loader-fn", cmd)
        # Rollout-only params should not appear
        self.assertNotIn("--reward-fn-path", cmd)
        self.assertNotIn("--agent-fn", cmd)
        self.assertNotIn("--n-samples", cmd)
        self.assertNotIn("--max-running-prompts", cmd)
        self.assertNotIn("--gspo-clip-eps", cmd)

    def test_gspo_command_includes_rollout_params(self):
        config = _ready_config(algo="gspo")
        cmd = train_cli._format_resolved_command(config)

        self.assertIn("--algo gspo", cmd)
        self.assertIn("--reward-fn-path", cmd)
        self.assertIn("--n-samples", cmd)

    def test_command_includes_model_hub(self):
        config = _ready_config(algo="sft", model_hub="hf", dataset_loader_fn="loader.py")
        cmd = train_cli._format_resolved_command(config)

        self.assertIn("--model-hub hf", cmd)

    def test_command_is_copy_pastable(self):
        config = _ready_config(algo="sft", dataset_loader_fn="loader.py")
        cmd = train_cli._format_resolved_command(config)

        # Should start with "areno train"
        self.assertTrue(cmd.startswith("areno train"))
        # Lines should use backslash continuation except the last
        lines = cmd.split("\n")
        for line in lines[:-1]:
            self.assertTrue(line.rstrip().endswith("\\") or line.startswith("areno train"),
                          f"Line missing backslash: {line}")

    def test_dpo_command_omits_rollout_and_ppo_params(self):
        from areno.api.trainer_config import DPOTrainerConfig

        config = DPOTrainerConfig(
            algo="dpo",
            ckpt="actor",
            dataset_path="dataset",
            model_hub="hf",
            dataset_loader_fn="loader.py",
            tp_size=1,
            world_size=2,
            batch_size=2,
            mini_bs=1,
            save_path=None,
            save_interval=100,
            epochs=1,
            ref_ckpt="ref",
            dpo_beta=0.1,
            metrics_log_dir=None,
        )
        cmd = train_cli._format_resolved_command(config)

        self.assertIn("--algo dpo", cmd)
        self.assertIn("--dpo-beta", cmd)
        # Should not include rollout or PPO params
        self.assertNotIn("--reward-fn-path", cmd)
        self.assertNotIn("--n-samples", cmd)
        self.assertNotIn("--critic-lr", cmd)
        self.assertNotIn("--clip-eps", cmd)


class TrainCommandPreflightIntegrationTest(unittest.TestCase):
    """Test that train_command runs preflight and resolved command output.

    These tests mock ``run_preflight_checks`` directly (instead of mocking
    ``collect_env``) because the integration test's purpose is to verify that
    train_command *calls* preflight and *prints* the result — not to test
    the check logic itself (that's covered by PreflightCheckTest above).
    Mocking at the run_preflight_checks level also avoids environment-dependent
    failures: a test using --ckpt actor would fail on a real preflight check
    because "actor" doesn't exist as a local file.
    """

    def test_train_command_shows_preflight_and_resolved_command(self):
        events = []

        def fake_run(config):
            events.append(("run", config.algo))

        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            with (
                patch.object(train_cli, "run", fake_run),
                patch.object(diagnostics, "collect_env", return_value=report),
                patch.object(diagnostics, "run_preflight_checks",
                             lambda config, **kw: PreflightResult(checks=[], kaggle_detected=False)),
                patch.object(
                    train_cli, "resolve_model_refs_for_config", lambda config: config
                ),
                patch.object(
                    train_cli, "_model_config_for_summary", lambda config: None
                ),
                patch("areno.cli.dashboard_registry.register_dashboard_job"),
            ):
                result = CliRunner().invoke(
                    train_cli.train_command,
                    [
                        "--algo", "sft",
                        "--ckpt", "actor",
                        "--dataset-path", "dataset",
                        "--dataset-loader-fn", "examples/sft/alpaca/dataset_loader.py",
                        "--world-size", "2",
                        "--tp-size", "1",
                        "--save-path", "out",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        output = unstyle(result.output)
        self.assertIn("Pre-flight environment check", output)
        self.assertIn("Resolved command", output)
        self.assertIn("areno train", output)
        self.assertEqual(events, [("run", "sft")])

    def test_train_command_blocks_on_critical_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["torch"]["imported"] = False
            report["torch"]["version"] = None
            report["torch"]["error"] = "ModuleNotFoundError"
            report["torch"]["cuda_available"] = False
            report["torch"]["device_count"] = 0
            report["gpus"] = []
            with (
                patch.object(diagnostics, "collect_env", return_value=report),
            ):
                result = CliRunner().invoke(
                    train_cli.train_command,
                    [
                        "--algo", "sft",
                        "--ckpt", "actor",
                        "--dataset-path", "dataset",
                        "--dataset-loader-fn", "examples/sft/alpaca/dataset_loader.py",
                        "--world-size", "2",
                        "--tp-size", "1",
                    ],
                )

        self.assertNotEqual(result.exit_code, 0)
        output = unstyle(result.output)
        self.assertIn("CRITICAL", output)
        self.assertIn("Pre-flight check failed", output)

    def test_train_command_blocks_on_error_without_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["dependencies"]["areno_accel"]["imported"] = False
            report["dependencies"]["areno_accel"]["error"] = "ModuleNotFoundError"
            with (
                patch.object(diagnostics, "collect_env", return_value=report),
            ):
                result = CliRunner().invoke(
                    train_cli.train_command,
                    [
                        "--algo", "sft",
                        "--ckpt", "actor",
                        "--dataset-path", "dataset",
                        "--dataset-loader-fn", "examples/sft/alpaca/dataset_loader.py",
                        "--world-size", "2",
                        "--tp-size", "1",
                    ],
                )

        self.assertNotEqual(result.exit_code, 0)
        output = unstyle(result.output)
        self.assertIn("Pre-flight check failed", output)
        self.assertIn("areno_accel", output)

    def test_skip_check_bypasses_errors_but_not_critical(self):
        events = []

        def fake_run(config):
            events.append(("run", config.algo))

        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["dependencies"]["areno_accel"]["imported"] = False
            report["dependencies"]["areno_accel"]["error"] = "ModuleNotFoundError"
            with (
                patch.object(train_cli, "run", fake_run),
                patch.object(diagnostics, "collect_env", return_value=report),
                patch.object(
                    train_cli, "resolve_model_refs_for_config", lambda config: config
                ),
                patch.object(
                    train_cli, "_model_config_for_summary", lambda config: None
                ),
                patch("areno.cli.dashboard_registry.register_dashboard_job"),
            ):
                result = CliRunner().invoke(
                    train_cli.train_command,
                    [
                        "--algo", "sft",
                        "--ckpt", "actor",
                        "--dataset-path", "dataset",
                        "--dataset-loader-fn", "examples/sft/alpaca/dataset_loader.py",
                        "--world-size", "2",
                        "--tp-size", "1",
                        "--skip-check",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        output = unstyle(result.output)
        self.assertIn("Pre-flight checks skipped", output)
        self.assertEqual(events, [("run", "sft")])

    def test_skip_check_does_not_bypass_critical(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = _ready_report(tmp)
            report["torch"]["imported"] = False
            report["torch"]["version"] = None
            report["torch"]["error"] = "ModuleNotFoundError"
            report["torch"]["cuda_available"] = False
            report["torch"]["device_count"] = 0
            report["gpus"] = []
            with (
                patch.object(diagnostics, "collect_env", return_value=report),
            ):
                result = CliRunner().invoke(
                    train_cli.train_command,
                    [
                        "--algo", "sft",
                        "--ckpt", "actor",
                        "--dataset-path", "dataset",
                        "--dataset-loader-fn", "examples/sft/alpaca/dataset_loader.py",
                        "--world-size", "2",
                        "--tp-size", "1",
                        "--skip-check",
                    ],
                )

        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()