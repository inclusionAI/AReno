"""CPU tests for disk space budget estimation and runtime monitoring (#232).

All tests use mocked ``shutil.disk_usage`` to simulate healthy, warn, and
stop transitions — no real filesystem writes, no GPU, no torch.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from areno.cli import disk_guard
from areno.cli import diagnostics
from areno.cli.disk_guard import (
    DiskBudget,
    DiskMonitor,
    DiskMonitorConfig,
    build_disk_monitor_from_config,
    disk_budget_to_json,
    estimate_disk_usage,
    format_disk_budget_text,
)


def _disk_usage_mock(total: int, free: int):
    """Return a mock shutil.disk_usage result."""

    return SimpleNamespace(total=total, free=free, used=total - free)


class EstimateDiskUsageTest(unittest.TestCase):
    """Tests for the estimation formula."""

    def test_estimate_basic(self):
        usage = estimate_disk_usage(total_steps=100, save_interval=100)
        expected = 100 * 12 * 1024  # 100 steps × 12 KB/step
        self.assertEqual(usage, expected)

    def test_estimate_excludes_checkpoints_by_default(self):
        usage = estimate_disk_usage(
            total_steps=1000, save_interval=100, checkpoint_size_bytes=1_000_000_000
        )
        # Should NOT include checkpoint size.
        expected = 1000 * 12 * 1024
        self.assertEqual(usage, expected)

    def test_estimate_includes_checkpoints_when_requested(self):
        usage = estimate_disk_usage(
            total_steps=1000,
            save_interval=100,
            checkpoint_size_bytes=1_000_000_000,
            include_checkpoints=True,
        )
        # 1000 steps / 100 interval = 10 saves × 1GB
        expected = 1000 * 12 * 1024 + 10 * 1_000_000_000
        self.assertEqual(usage, expected)

    def test_estimate_zero_steps(self):
        usage = estimate_disk_usage(total_steps=0, save_interval=100)
        self.assertEqual(usage, 0)

    def test_estimate_zero_save_interval(self):
        """save_interval=0 should not crash; checkpoint portion is skipped."""
        usage = estimate_disk_usage(
            total_steps=100, save_interval=0, checkpoint_size_bytes=1000, include_checkpoints=True
        )
        self.assertEqual(usage, 100 * 12 * 1024)


class DiskMonitorThresholdTest(unittest.TestCase):
    """Tests for threshold computation (percent + absolute override)."""

    def test_thresholds_percent_default(self):
        config = DiskMonitorConfig()
        monitor = DiskMonitor(
            config=config, paths=["/tmp"], total_steps=100,
        )
        with patch.object(disk_guard, "_min_total_bytes", return_value=100 * 1_000_000_000):
            warn, stop = monitor._compute_thresholds()
        # 5% of 100GB = 5GB, 1% = 1GB
        self.assertAlmostEqual(warn, 5 * 1_000_000_000, delta=1)
        self.assertAlmostEqual(stop, 1 * 1_000_000_000, delta=1)

    def test_thresholds_absolute_override_stricter(self):
        config = DiskMonitorConfig(warn_gb=20, stop_gb=5)
        monitor = DiskMonitor(config=config, paths=["/tmp"], total_steps=100)
        with patch.object(disk_guard, "_min_total_bytes", return_value=100 * 1_000_000_000):
            warn, stop = monitor._compute_thresholds()
        # max(5% of 100GB=5GB, 20GB) = 20GB
        self.assertEqual(warn, int(20 * 1e9))
        self.assertEqual(stop, int(5 * 1e9))

    def test_thresholds_percent_stricter_than_absolute(self):
        """When percent gives a larger value, it wins."""
        config = DiskMonitorConfig(warn_gb=1, stop_gb=0.5)
        monitor = DiskMonitor(config=config, paths=["/tmp"], total_steps=100)
        with patch.object(disk_guard, "_min_total_bytes", return_value=1000 * 1_000_000_000):
            warn, stop = monitor._compute_thresholds()
        # 5% of 1000GB = 50GB > 1GB
        self.assertEqual(warn, int(50 * 1_000_000_000))


class DiskMonitorCheckTest(unittest.TestCase):
    """Tests for the runtime check() method — healthy/warn/stop transitions."""

    def _make_monitor(self, total_bytes=100 * 1_000_000_000):
        config = DiskMonitorConfig(check_interval_steps=1)
        monitor = DiskMonitor(config=config, paths=["/fake"], total_steps=100)
        monitor._warn_bytes = int(total_bytes * 0.05)
        monitor._stop_bytes = int(total_bytes * 0.01)
        return monitor

    def test_disk_healthy(self):
        monitor = self._make_monitor()
        with patch.object(disk_guard, "_min_free_bytes", return_value=80 * 1_000_000_000):
            self.assertEqual(monitor.check(0), "ok")

    def test_disk_warn_transition(self):
        monitor = self._make_monitor()
        with patch.object(disk_guard, "_min_free_bytes", return_value=3 * 1_000_000_000):
            self.assertEqual(monitor.check(0), "warn")

    def test_disk_stop_transition(self):
        monitor = self._make_monitor()
        with patch.object(disk_guard, "_min_free_bytes", return_value=500 * 1_000_000):
            self.assertEqual(monitor.check(0), "stop")

    def test_warn_emitted_only_once(self):
        monitor = self._make_monitor()
        with patch.object(disk_guard, "_min_free_bytes", return_value=3 * 1_000_000_000):
            self.assertEqual(monitor.check(0), "warn")
            self.assertTrue(monitor._already_warned)
            # Second check at same level should still return "warn" but not re-log.
            self.assertEqual(monitor.check(1), "warn")

    def test_check_interval_throttling(self):
        """Steps between check_interval_steps should not probe the filesystem."""
        config = DiskMonitorConfig(check_interval_steps=10)
        monitor = DiskMonitor(config=config, paths=["/fake"], total_steps=100)
        monitor._warn_bytes = 5_000_000_000
        monitor._stop_bytes = 1_000_000_000
        call_count = 0
        original_min_free = disk_guard._min_free_bytes

        def counting_min_free(paths):
            nonlocal call_count
            call_count += 1
            return original_min_free(paths)

        with patch.object(disk_guard, "_min_free_bytes", side_effect=counting_min_free):
            monitor.check(0)   # Should probe (step 0)
            monitor.check(1)   # Should NOT probe (1 - 0 < 10)
            monitor.check(5)   # Should NOT probe
            monitor.check(10)  # Should probe (10 - 0 >= 10)
        self.assertEqual(call_count, 2)

    def test_zero_free_space_boundary(self):
        monitor = self._make_monitor()
        with patch.object(disk_guard, "_min_free_bytes", return_value=0):
            self.assertEqual(monitor.check(0), "stop")


class DiskBudgetEstimateTest(unittest.TestCase):
    """Tests for the DiskMonitor.estimate_budget() method."""

    def test_budget_sufficient(self):
        config = DiskMonitorConfig()
        monitor = DiskMonitor(config=config, paths=["/fake"], total_steps=100, save_interval=100)
        with (
            patch.object(disk_guard, "_min_free_bytes", return_value=100 * 1_000_000_000),
            patch.object(disk_guard, "_min_total_bytes", return_value=200 * 1_000_000_000),
        ):
            budget = monitor.estimate_budget()
        self.assertTrue(budget.sufficient)
        self.assertEqual(budget.status if hasattr(budget, "status") else None, None)

    def test_budget_insufficient(self):
        config = DiskMonitorConfig()
        monitor = DiskMonitor(config=config, paths=["/fake"], total_steps=100000, save_interval=1)
        with (
            patch.object(disk_guard, "_min_free_bytes", return_value=1 * 1_000_000),
            patch.object(disk_guard, "_min_total_bytes", return_value=10 * 1_000_000_000),
        ):
            budget = monitor.estimate_budget()
        self.assertFalse(budget.sufficient)
        self.assertTrue(len(budget.next_step) > 0)


class BuildDiskMonitorTest(unittest.TestCase):
    """Tests for build_disk_monitor_from_config()."""

    def test_returns_none_when_config_is_none(self):
        config = SimpleNamespace(save_path="/tmp/out", metrics_log_dir="/tmp/metrics", max_steps=100, save_interval=10)
        result = build_disk_monitor_from_config(config, disk_monitor_config=None)
        self.assertIsNone(result)

    def test_returns_monitor_with_valid_config(self):
        config = SimpleNamespace(
            save_path="/tmp/out",
            metrics_log_dir="/tmp/metrics",
            max_steps=100,
            save_interval=10,
            epochs=10,
            ckpt=None,
        )
        monitor_config = DiskMonitorConfig()
        result = build_disk_monitor_from_config(config, disk_monitor_config=monitor_config)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DiskMonitor)

    def test_returns_none_when_no_paths(self):
        config = SimpleNamespace(
            save_path=None, metrics_log_dir=None, max_steps=100, save_interval=10, epochs=10, ckpt=None
        )
        monitor_config = DiskMonitorConfig()
        result = build_disk_monitor_from_config(config, disk_monitor_config=monitor_config)
        self.assertIsNone(result)


class FormattingTest(unittest.TestCase):
    """Tests for output formatting."""

    def test_format_budget_text_sufficient(self):
        budget = DiskBudget(
            paths=["/tmp"],
            free_bytes=100 * 1_000_000_000,
            total_bytes=200 * 1_000_000_000,
            estimated_usage_bytes=1_000_000,
            sufficient=True,
            detail="estimated 1.0 MB usage, 100.0 GB free — sufficient",
        )
        text = format_disk_budget_text(budget)
        self.assertIn("OK", text)
        self.assertNotIn("Next", text)

    def test_format_budget_text_insufficient(self):
        budget = DiskBudget(
            paths=["/tmp"],
            free_bytes=1_000_000,
            total_bytes=10 * 1_000_000_000,
            estimated_usage_bytes=1_000_000_000,
            sufficient=False,
            detail="estimated 1000.0 MB usage, 0.0 GB free — insufficient",
            next_step="Free up disk space",
        )
        text = format_disk_budget_text(budget)
        self.assertIn("FAIL", text)
        self.assertIn("Free up disk space", text)

    def test_budget_to_json(self):
        budget = DiskBudget(
            paths=["/tmp"],
            free_bytes=100,
            total_bytes=200,
            estimated_usage_bytes=10,
            sufficient=True,
            detail="ok",
        )
        json_str = disk_budget_to_json(budget)
        parsed = json.loads(json_str)
        self.assertTrue(parsed["sufficient"])
        self.assertEqual(parsed["free_bytes"], 100)


class CheckCommandDiskBudgetTest(unittest.TestCase):
    """Integration tests for `areno check --disk-budget`."""

    def test_check_disk_budget_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = CliRunner()
            with patch.object(disk_guard, "_min_free_bytes", return_value=100 * 1_000_000_000):
                with patch.object(disk_guard, "_min_total_bytes", return_value=200 * 1_000_000_000):
                    result = runner.invoke(
                        diagnostics.check_command,
                        ["--disk-budget", "--save-path", tmp, "--max-steps", "10"],
                    )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("OK", result.output)

    def test_check_disk_budget_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = CliRunner()
            with patch.object(disk_guard, "_min_free_bytes", return_value=100):
                with patch.object(disk_guard, "_min_total_bytes", return_value=1000):
                    result = runner.invoke(
                        diagnostics.check_command,
                        ["--disk-budget", "--save-path", tmp, "--max-steps", "100000"],
                    )
            self.assertEqual(result.exit_code, 1)
            self.assertIn("FAIL", result.output)

    def test_check_disk_budget_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = CliRunner()
            with patch.object(disk_guard, "_min_free_bytes", return_value=100 * 1_000_000_000):
                with patch.object(disk_guard, "_min_total_bytes", return_value=200 * 1_000_000_000):
                    result = runner.invoke(
                        diagnostics.check_command,
                        ["--disk-budget", "--save-path", tmp, "--max-steps", "10", "--disk-json"],
                    )
            self.assertEqual(result.exit_code, 0)
            parsed = json.loads(result.output)
            self.assertTrue(parsed["sufficient"])

    def test_check_without_disk_budget_unchanged(self):
        """Existing `areno check` behavior is preserved."""
        runner = CliRunner()
        with patch.object(diagnostics, "collect_env", return_value={
            "platform": {"system": "Linux", "release": "6.0", "machine": "x86_64", "platform": "Linux"},
            "torch": {"imported": False, "error": "no torch", "version": None, "cuda_build": None,
                       "cuda_runtime": None, "cuda_runtime_error": None, "cuda_available": False,
                       "device_count": 0, "gpus": []},
            "cuda": {"cuda_home": None, "inferred_cuda_home": None,
                      "nvcc": {"path": None, "version": None},
                      "driver": {"path": None, "driver_version": None, "cuda_version": None, "error": "no smi"}},
            "gpus": [],
            "dependencies": {
                "flash_attn": {"distribution": "flash-attn", "module": "flash_attn", "version": None,
                               "imported": False, "error": "missing"},
                "flash_linear_attention": {"distribution": "flash-linear-attention", "module": "fla",
                                           "version": None, "imported": False, "error": "missing"},
                "areno_accel": {"distribution": None, "module": "areno.accel._areno_accel",
                                "version": None, "imported": False, "error": "missing"},
            },
            "install": {"build_ext_disabled": False},
            "env": {},
            "paths": {"metrics_log_dir": tmp, "hf_cache": tmp} if False else {"metrics_log_dir": "/tmp", "hf_cache": "/tmp"},
        }):
            with patch.object(diagnostics, "run_checks", return_value=[]):
                result = runner.invoke(diagnostics.check_command, [])
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
