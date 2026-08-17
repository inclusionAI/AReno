"""CPU-only tests for the pre-launch GPU occupancy check.

All nvidia-smi subprocess calls are mocked so these tests run without a GPU.
"""

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from areno.engine.runtime.gpu_check import (
    GpuStatus,
    GpuWarning,
    check_gpu_occupancy,
    format_gpu_warnings,
    query_gpu_status,
)


def _make_proc_result(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class QueryGpuStatusTest(unittest.TestCase):
    """Tests for query_gpu_status."""

    @patch("areno.engine.runtime.gpu_check.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("areno.engine.runtime.gpu_check.subprocess.run")
    def test_parses_nvidia_smi_output(self, mock_run, mock_which):
        gpu_csv = "81920,40960,40960,50,NVIDIA H100\n"
        proc_csv = "1234,python,2048\n5678,python,1024\n"
        mock_run.side_effect = [
            _make_proc_result(gpu_csv),
            _make_proc_result(proc_csv),
        ]
        statuses = query_gpu_status([0])
        self.assertEqual(len(statuses), 1)
        s = statuses[0]
        self.assertEqual(s.index, 0)
        self.assertEqual(s.name, "NVIDIA H100")
        self.assertEqual(s.total_mem_mb, 81920)
        self.assertEqual(s.free_mem_mb, 40960)
        self.assertEqual(s.used_mem_mb, 40960)
        self.assertEqual(s.utilization_pct, 50)
        self.assertEqual(len(s.processes), 2)
        self.assertEqual(s.processes[0]["pid"], 1234)
        self.assertEqual(s.processes[0]["name"], "python")
        self.assertEqual(s.processes[0]["used_mem_mb"], 2048)

    @patch("areno.engine.runtime.gpu_check.shutil.which", return_value=None)
    def test_no_nvidia_smi_returns_empty(self, mock_which):
        statuses = query_gpu_status([0, 1])
        self.assertEqual(statuses, [])

    @patch("areno.engine.runtime.gpu_check.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("areno.engine.runtime.gpu_check.subprocess.run")
    def test_smi_error_returns_empty(self, mock_run, mock_which):
        mock_run.return_value = _make_proc_result("", returncode=1)
        statuses = query_gpu_status([0])
        self.assertEqual(statuses, [])

    @patch("areno.engine.runtime.gpu_check.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("areno.engine.runtime.gpu_check.subprocess.run")
    def test_subprocess_exception_returns_empty(self, mock_run, mock_which):
        mock_run.side_effect = FileNotFoundError("nvidia-smi gone")
        statuses = query_gpu_status([0])
        self.assertEqual(statuses, [])

    @patch("areno.engine.runtime.gpu_check.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("areno.engine.runtime.gpu_check.subprocess.run")
    def test_empty_devices_returns_empty(self, mock_run, mock_which):
        statuses = query_gpu_status([])
        self.assertEqual(statuses, [])
        mock_run.assert_not_called()

    @patch("areno.engine.runtime.gpu_check.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("areno.engine.runtime.gpu_check.subprocess.run")
    def test_multiple_devices(self, mock_run, mock_which):
        gpu_csv = "81920,80000,1920,5,NVIDIA H100\n81920,1000,80920,95,NVIDIA H100\n"
        proc_csv = ""
        mock_run.side_effect = [
            _make_proc_result(gpu_csv),
            _make_proc_result(proc_csv),
        ]
        statuses = query_gpu_status([0, 1])
        self.assertEqual(len(statuses), 2)
        self.assertEqual(statuses[0].index, 0)
        self.assertEqual(statuses[0].free_mem_mb, 80000)
        self.assertEqual(statuses[1].index, 1)
        self.assertEqual(statuses[1].utilization_pct, 95)

    @patch("areno.engine.runtime.gpu_check.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("areno.engine.runtime.gpu_check.subprocess.run")
    def test_smi_command_construction_no_prefix_duplication(self, mock_run, mock_which):
        """Verify the nvidia-smi command is constructed without prefix duplication.

        Regression test: previously _run_smi hardcoded --query-gpu= prefix,
        causing --query-compute-apps calls to become --query-gpu=--query-compute-apps=...
        """
        gpu_csv = "81920,80000,1920,5,NVIDIA H100\n"
        proc_csv = ""
        mock_run.side_effect = [
            _make_proc_result(gpu_csv),
            _make_proc_result(proc_csv),
        ]
        query_gpu_status([0])

        # First call: GPU info query — must use --query-gpu, not --query-gpu=--query-gpu
        first_call_args = mock_run.call_args_list[0]
        cmd = first_call_args[0][0]  # positional args: the command list
        self.assertEqual(cmd[0], "/usr/bin/nvidia-smi")
        self.assertTrue(cmd[1].startswith("--query-gpu=memory.total"))
        self.assertFalse("--query-gpu=--query-gpu" in cmd[1])

        # Second call: process query — must use --query-compute-apps, not --query-gpu
        second_call_args = mock_run.call_args_list[1]
        cmd2 = second_call_args[0][0]
        self.assertTrue(cmd2[1].startswith("--query-compute-apps=pid"))
        self.assertFalse("--query-gpu" in cmd2[1])


class CheckGpuOccupancyTest(unittest.TestCase):
    """Tests for check_gpu_occupancy."""

    def test_low_memory_warning(self):
        status = GpuStatus(
            index=0,
            name="H100",
            total_mem_mb=81920,
            free_mem_mb=4000,
            used_mem_mb=77920,
            utilization_pct=30,
            processes=[{"pid": 123, "name": "python", "used_mem_mb": 77920}],
        )
        warnings = check_gpu_occupancy([status], mem_free_warn_pct=10, util_warn_pct=90)
        low_mem = [w for w in warnings if w.kind == "low_memory"]
        self.assertEqual(len(low_mem), 1)
        self.assertEqual(low_mem[0].device_index, 0)
        self.assertIn("4000 MB", low_mem[0].message)
        self.assertIn("python", low_mem[0].message)

    def test_high_utilization_warning(self):
        status = GpuStatus(
            index=1,
            name="H100",
            total_mem_mb=81920,
            free_mem_mb=70000,
            used_mem_mb=11920,
            utilization_pct=95,
            processes=[{"pid": 456, "name": "trainer", "used_mem_mb": 11920}],
        )
        warnings = check_gpu_occupancy([status], mem_free_warn_pct=10, util_warn_pct=90)
        high_util = [w for w in warnings if w.kind == "high_utilization"]
        self.assertEqual(len(high_util), 1)
        self.assertEqual(high_util[0].device_index, 1)
        self.assertIn("95%", high_util[0].message)
        self.assertIn("trainer", high_util[0].message)

    def test_no_warnings_when_healthy(self):
        status = GpuStatus(
            index=0,
            name="H100",
            total_mem_mb=81920,
            free_mem_mb=70000,
            used_mem_mb=11920,
            utilization_pct=30,
            processes=[],
        )
        warnings = check_gpu_occupancy([status], mem_free_warn_pct=10, util_warn_pct=90)
        self.assertEqual(warnings, [])

    def test_both_warnings_on_same_gpu(self):
        status = GpuStatus(
            index=0,
            name="H100",
            total_mem_mb=81920,
            free_mem_mb=2000,
            used_mem_mb=79920,
            utilization_pct=99,
            processes=[{"pid": 1, "name": "python", "used_mem_mb": 79920}],
        )
        warnings = check_gpu_occupancy([status], mem_free_warn_pct=10, util_warn_pct=90)
        self.assertEqual(len(warnings), 2)
        kinds = {w.kind for w in warnings}
        self.assertEqual(kinds, {"low_memory", "high_utilization"})

    def test_zero_total_mem_skipped(self):
        status = GpuStatus(
            index=0,
            name="unknown",
            total_mem_mb=0,
            free_mem_mb=0,
            used_mem_mb=0,
            utilization_pct=0,
            processes=[],
        )
        warnings = check_gpu_occupancy([status], mem_free_warn_pct=10, util_warn_pct=90)
        self.assertEqual(warnings, [])

    def test_custom_thresholds(self):
        status = GpuStatus(
            index=0,
            name="H100",
            total_mem_mb=81920,
            free_mem_mb=60000,
            used_mem_mb=21920,
            utilization_pct=50,
            processes=[],
        )
        # Default thresholds would not warn; stricter ones should.
        warnings = check_gpu_occupancy([status], mem_free_warn_pct=80, util_warn_pct=40)
        self.assertEqual(len(warnings), 2)

    def test_warning_includes_process_info(self):
        status = GpuStatus(
            index=0,
            name="H100",
            total_mem_mb=81920,
            free_mem_mb=1000,
            used_mem_mb=80920,
            utilization_pct=10,
            processes=[{"pid": 42, "name": "big_model", "used_mem_mb": 80000}],
        )
        warnings = check_gpu_occupancy([status], mem_free_warn_pct=10, util_warn_pct=90)
        self.assertEqual(len(warnings), 1)
        self.assertIn("big_model", warnings[0].message)
        self.assertIn("pid=42", warnings[0].message)

    def test_no_processes_shows_none(self):
        status = GpuStatus(
            index=0,
            name="H100",
            total_mem_mb=81920,
            free_mem_mb=1000,
            used_mem_mb=80920,
            utilization_pct=10,
            processes=[],
        )
        warnings = check_gpu_occupancy([status], mem_free_warn_pct=10, util_warn_pct=90)
        self.assertEqual(len(warnings), 1)
        self.assertIn("none", warnings[0].message)


class FormatGpuWarningsTest(unittest.TestCase):
    """Tests for format_gpu_warnings."""

    def test_empty_warnings_returns_empty_string(self):
        self.assertEqual(format_gpu_warnings([], []), "")

    def test_format_includes_gpu_index_and_key_info(self):
        status = GpuStatus(
            index=0,
            name="NVIDIA H100",
            total_mem_mb=81920,
            free_mem_mb=4000,
            used_mem_mb=77920,
            utilization_pct=50,
            processes=[{"pid": 1, "name": "python", "used_mem_mb": 77920}],
        )
        warning = GpuWarning(
            device_index=0,
            kind="low_memory",
            message="GPU 0 has low memory",
        )
        text = format_gpu_warnings([warning], [status])
        self.assertIn("GPU occupancy warnings:", text)
        self.assertIn("[WARN]", text)
        self.assertIn("GPU 0", text)
        self.assertIn("NVIDIA H100", text)
        self.assertIn("4000/81920 MB", text)
        self.assertIn("1 process(es)", text)

    def test_format_includes_overview_for_all_gpus(self):
        statuses = [
            GpuStatus(
                index=0,
                name="H100",
                total_mem_mb=81920,
                free_mem_mb=40000,
                used_mem_mb=41920,
                utilization_pct=30,
                processes=[],
            ),
            GpuStatus(
                index=1,
                name="H100",
                total_mem_mb=81920,
                free_mem_mb=2000,
                used_mem_mb=79920,
                utilization_pct=95,
                processes=[{"pid": 1, "name": "python", "used_mem_mb": 79920}],
            ),
        ]
        warnings = [GpuWarning(device_index=1, kind="low_memory", message="GPU 1 low mem")]
        text = format_gpu_warnings(warnings, statuses)
        # Overview should include both GPUs even though only GPU 1 has a warning.
        self.assertIn("GPU 0", text)
        self.assertIn("GPU 1", text)


if __name__ == "__main__":
    unittest.main()
