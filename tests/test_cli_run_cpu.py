"""CPU-safe tests for the 'areno run' CLI."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from areno.cli.main import main
from areno.cli.run import (
    RunSummary,
    _format_timestamp,
    _resolve_status,
    collect_metric_summaries,
    format_run_info,
    format_run_list,
)


class RunCliTest(unittest.TestCase):
    def test_top_level_cli_lists_run_command(self):
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("run", result.output)

    def test_run_help_shows_list_and_info(self):
        result = CliRunner().invoke(main, ["run", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)
        self.assertIn("info", result.output)

    def test_run_list_empty_registry(self):
        with patch("areno.cli.run.registered_job_items", return_value=[]):
            result = CliRunner().invoke(main, ["run", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No runs found.", result.output)

    def test_run_list_with_entries(self):
        entries = [
            {
                "id": "abc123",
                "kind": "train",
                "name": "train gspo Qwen",
                "pid": None,
                "created_at": 1700000000.0,
                "updated_at": 1700000000.0,
                "metrics_dir": None,
                "config": {},
                "command": [],
            }
        ]
        with patch("areno.cli.run.registered_job_items", return_value=entries):
            result = CliRunner().invoke(main, ["run", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("abc123", result.output)
        self.assertIn("train gspo Qwen", result.output)

    def test_run_info_not_found(self):
        with patch("areno.cli.run.registered_job_items", return_value=[]):
            result = CliRunner().invoke(main, ["run", "info", "nonexistent"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Run not found", result.output)

    def test_run_info_with_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = os.getpid()
            metrics_dir = tmp
            # Create dashboard_state file
            state_file = os.path.join(tmp, f"dashboard_state.{pid}.json")
            with open(state_file, "w") as f:
                json.dump({"pid": pid, "stage": "train_end", "status": "running", "step": 10}, f)
            # Create run config files
            config_txt = os.path.join(tmp, f"areno_run_config.{pid}.txt")
            with open(config_txt, "w") as f:
                f.write("algo: gspo\nckpt: Qwen/Qwen3-0.6B")
            config_json = os.path.join(tmp, f"areno_run_config.{pid}.json")
            with open(config_json, "w") as f:
                json.dump({"settings": {"algo": "gspo", "ckpt": "Qwen/Qwen3-0.6B"}}, f)

            entries = [
                {
                    "id": "test123",
                    "kind": "train",
                    "name": "test run",
                    "pid": pid,
                    "created_at": 1700000000.0,
                    "updated_at": 1700000000.0,
                    "metrics_dir": metrics_dir,
                    "config": {},
                    "command": [],
                }
            ]
            with patch("areno.cli.run.registered_job_items", return_value=entries):
                result = CliRunner().invoke(main, ["run", "info", "test123"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Run: test123", result.output)
        self.assertIn("Configuration:", result.output)
        self.assertIn("gspo", result.output)
        self.assertIn("train_end", result.output)

    def test_format_run_info_contains_all_sections(self):
        run = RunSummary(
            run_id="test",
            kind="train",
            name="test run",
            status="succeeded",
            step=10,
            config={"algo": "gspo"},
            config_text="algo: gspo\nckpt: Qwen/Qwen3-0.6B",
            metrics=[
                {"name": "loss", "value": 0.5, "step": 1},
                {"name": "loss", "value": 0.3, "step": 2},
            ],
            timeperf=[
                {"segments": [{"name": "rollout", "seconds": 1.0}, {"name": "train", "seconds": 2.0}]},
                {"segments": [{"name": "rollout", "seconds": 1.5}, {"name": "train", "seconds": 2.5}]},
            ],
            samples=[{"step": 1, "reward": 0.8, "response_len": 100}],
        )
        output = format_run_info(run)
        self.assertIn("Run: test", output)
        self.assertIn("Configuration:", output)
        self.assertIn("Metrics Summary:", output)
        self.assertIn("Time Breakdown", output)
        self.assertIn("Recent Rollout Samples", output)

    def test_format_run_list_empty(self):
        self.assertEqual(format_run_list([]), "No runs found.")

    def test_format_run_list_with_entries(self):
        runs = [RunSummary(run_id="abc", kind="train", name="test", status="running", step=5)]
        output = format_run_list(runs)
        self.assertIn("abc", output)
        self.assertIn("train", output)

    def test_collect_metric_summaries_groups_by_name(self):
        metrics = [
            {"name": "loss", "value": 0.5, "step": 1},
            {"name": "loss", "value": 0.3, "step": 2},
            {"name": "reward", "value": 1.0, "step": 2},
        ]
        result = collect_metric_summaries(metrics)
        self.assertEqual(len(result), 2)
        loss = next(m for m in result if m["name"] == "loss")
        self.assertEqual(loss["count"], 2)
        self.assertEqual(loss["latest_step"], 2)
        self.assertEqual(loss["latest_value"], 0.3)

    def test_resolve_status_running(self):
        with patch("areno.cli.run.pid_is_running", return_value=True):
            self.assertEqual(_resolve_status({"pid": 123}), "running")

    def test_resolve_status_exited(self):
        self.assertEqual(_resolve_status({"pid": None}), "exited")

    def test_resolve_status_succeeded(self):
        self.assertEqual(_resolve_status({"pid": None, "returncode": 0}), "succeeded")

    def test_resolve_status_failed(self):
        self.assertEqual(_resolve_status({"pid": None, "returncode": 1}), "failed")

    def test_format_timestamp_converts_epoch(self):
        result = _format_timestamp(0)
        self.assertTrue(result.startswith("1970"))

    def test_format_timestamp_empty(self):
        self.assertEqual(_format_timestamp(None), "")
        self.assertEqual(_format_timestamp(""), "")

    def test_returncode_populated_from_registry(self):
        """returncode is read from the registry entry and displayed."""
        entries = [
            {
                "id": "rc1",
                "kind": "train",
                "name": "failed run",
                "pid": None,
                "returncode": 1,
                "created_at": 1700000000.0,
                "updated_at": 1700000000.0,
                "metrics_dir": None,
                "config": {},
                "command": [],
            }
        ]
        with patch("areno.cli.run.registered_job_items", return_value=entries):
            result = CliRunner().invoke(main, ["run", "info", "rc1"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("returncode=1", result.output)

    def test_returncode_none_not_displayed(self):
        """returncode is None when not in registry, and not shown in output."""
        run = RunSummary(run_id="x", status="exited")
        output = format_run_info(run)
        self.assertNotIn("returncode", output)

    def test_summary_from_item_dedup(self):
        """list_runs and get_run produce the same base summary for the same entry."""
        from areno.cli.run import _summary_from_item

        item = {
            "id": "dup1",
            "kind": "train",
            "name": "test",
            "pid": None,
            "returncode": 0,
            "created_at": 1700000000.0,
            "updated_at": 1700000000.0,
            "metrics_dir": None,
            "config": {"algo": "gspo"},
            "command": ["areno", "train"],
        }
        s1 = _summary_from_item(item)
        s2 = _summary_from_item(item)
        self.assertEqual(s1, s2)
        self.assertEqual(s1.returncode, 0)
        self.assertEqual(s1.config, {"algo": "gspo"})


if __name__ == "__main__":
    unittest.main()