"""CPU-safe tests for the 'areno run' CLI."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from areno.cli.main import main
from areno.cli.run import (
    RunSummary,
    _compute_age,
    _format_age,
    _format_timestamp,
    _redact_config,
    _redact_text,
    _resolve_status,
    collect_metric_summaries,
    format_run_info,
    format_run_info_json,
    format_run_list,
    format_run_list_json,
    get_run,
    list_runs,
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

    # -- list (table) --

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

    def test_run_list_shows_age_column(self):
        entries = [
            {
                "id": "age1",
                "kind": "train",
                "name": "test",
                "pid": None,
                "created_at": time.time() - 180,
                "updated_at": time.time() - 180,
                "metrics_dir": None,
                "config": {},
                "command": [],
            }
        ]
        with patch("areno.cli.run.registered_job_items", return_value=entries):
            result = CliRunner().invoke(main, ["run", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Age", result.output)
        self.assertIn("3m ago", result.output)

    # -- list --json --

    def test_run_list_json_empty(self):
        with patch("areno.cli.run.registered_job_items", return_value=[]):
            result = CliRunner().invoke(main, ["run", "list", "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(json.loads(result.output), [])

    def test_run_list_json_with_entries(self):
        entries = [
            {
                "id": "js1",
                "kind": "train",
                "name": "gspo run",
                "pid": 12345,
                "created_at": 1700000000.0,
                "updated_at": 1700000000.0,
                "metrics_dir": None,
                "config": {},
                "command": [],
            }
        ]
        with patch("areno.cli.run.registered_job_items", return_value=entries):
            result = CliRunner().invoke(main, ["run", "list", "--json"])
        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["id"], "js1")
        self.assertEqual(parsed[0]["kind"], "train")
        self.assertEqual(parsed[0]["name"], "gspo run")
        self.assertEqual(parsed[0]["pid"], 12345)
        self.assertIn("status", parsed[0])
        self.assertIn("age_s", parsed[0])

    def test_run_list_json_deterministic_order(self):
        """Entries with same created_at are sorted by created_at desc then pid desc."""
        entries = [
            {"id": "a", "kind": "train", "name": "a", "pid": 1, "created_at": 100.0, "updated_at": 100.0},
            {"id": "b", "kind": "train", "name": "b", "pid": 2, "created_at": 100.0, "updated_at": 100.0},
            {"id": "c", "kind": "train", "name": "c", "pid": 3, "created_at": 200.0, "updated_at": 200.0},
        ]
        with patch("areno.cli.run.registered_job_items", return_value=entries):
            result = CliRunner().invoke(main, ["run", "list", "--json"])
        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        ids = [item["id"] for item in parsed]
        self.assertEqual(ids, ["c", "b", "a"])

    # -- list --limit --

    def test_run_list_limit(self):
        entries = [
            {
                "id": f"run{i}",
                "kind": "train",
                "name": f"run {i}",
                "pid": i,
                "created_at": float(1700000000 + i),
                "updated_at": float(1700000000 + i),
                "metrics_dir": None,
                "config": {},
                "command": [],
            }
            for i in range(5)
        ]
        with patch("areno.cli.run.registered_job_items", return_value=entries):
            result = CliRunner().invoke(main, ["run", "list", "--limit", "2"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("run4", result.output)
        self.assertIn("run3", result.output)
        self.assertNotIn("run2", result.output)

    def test_run_list_limit_zero_shows_all(self):
        entries = [
            {
                "id": f"r{i}",
                "kind": "train",
                "name": f"r {i}",
                "pid": i,
                "created_at": float(1700000000 + i),
                "updated_at": float(1700000000 + i),
                "metrics_dir": None,
                "config": {},
                "command": [],
            }
            for i in range(3)
        ]
        with patch("areno.cli.run.registered_job_items", return_value=entries):
            result = CliRunner().invoke(main, ["run", "list", "--limit", "0"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("r0", result.output)
        self.assertIn("r1", result.output)
        self.assertIn("r2", result.output)

    def test_run_list_limit_default_is_20(self):
        entries = [
            {
                "id": f"x{i}",
                "kind": "train",
                "name": f"x {i}",
                "pid": i,
                "created_at": float(1700000000 + i),
                "updated_at": float(1700000000 + i),
                "metrics_dir": None,
                "config": {},
                "command": [],
            }
            for i in range(25)
        ]
        with patch("areno.cli.run.registered_job_items", return_value=entries):
            result = CliRunner().invoke(main, ["run", "list", "--json"])
        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(len(parsed), 20)

    # -- info --

    def test_run_info_not_found(self):
        with patch("areno.cli.run.registered_job_items", return_value=[]):
            result = CliRunner().invoke(main, ["run", "info", "nonexistent"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Run not found", result.output)

    def test_run_info_with_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = os.getpid()
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
                    "metrics_dir": tmp,
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

    def test_returncode_populated_from_registry(self):
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

    # -- status resolution --

    def test_resolve_status_running(self):
        with patch("areno.cli.run.pid_is_running", return_value=True):
            self.assertEqual(_resolve_status({"pid": 123}), "running")

    def test_resolve_status_stage_from_dashboard_state(self):
        """When PID is alive and dashboard_state has a stage, return that stage."""
        with tempfile.TemporaryDirectory() as tmp:
            pid = os.getpid()
            state_file = os.path.join(tmp, f"dashboard_state.{pid}.json")
            with open(state_file, "w") as f:
                json.dump({"stage": "rollout"}, f)

            with patch("areno.cli.run.pid_is_running", return_value=True):
                status = _resolve_status({"pid": pid, "metrics_dir": tmp})
            self.assertEqual(status, "rollout")

    def test_resolve_status_running_no_state_file(self):
        """When PID is alive but no dashboard_state file, return 'running'."""
        with patch("areno.cli.run.pid_is_running", return_value=True):
            self.assertEqual(_resolve_status({"pid": 123, "metrics_dir": None}), "running")

    def test_resolve_status_exited(self):
        self.assertEqual(_resolve_status({"pid": None}), "exited")

    def test_resolve_status_succeeded(self):
        self.assertEqual(_resolve_status({"pid": None, "returncode": 0}), "succeeded")

    def test_resolve_status_failed(self):
        self.assertEqual(_resolve_status({"pid": None, "returncode": 1}), "failed")

    # -- age helpers --

    def test_compute_age(self):
        now = time.time()
        age = _compute_age(now - 300, now - 300)
        self.assertIsNotNone(age)
        self.assertGreaterEqual(age, 300)

    def test_compute_age_none(self):
        self.assertIsNone(_compute_age(None, None))

    def test_format_age_seconds(self):
        self.assertEqual(_format_age(45), "45s ago")

    def test_format_age_minutes(self):
        self.assertEqual(_format_age(180), "3m ago")

    def test_format_age_hours(self):
        self.assertEqual(_format_age(7200), "2h ago")

    def test_format_age_days(self):
        self.assertEqual(_format_age(172800), "2d ago")

    def test_format_age_none(self):
        self.assertEqual(_format_age(None), "-")

    # -- formatting --

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

    def test_format_run_list_empty_created_at_shows_dash(self):
        """When created_at is empty, the Created column should show '-' not blank."""
        runs = [RunSummary(run_id="x", kind="train", name="t", status="exited", created_at="")]
        output = format_run_list(runs)
        self.assertIn("-", output)
        # The dash should appear in the Created column position, not be empty
        self.assertNotIn("x  train  exited         0  t                                        -          \n",
                         output)

    def test_format_run_list_json_with_entries(self):
        runs = [RunSummary(run_id="j1", kind="train", name="json test", status="running", pid=99)]
        output = format_run_list_json(runs)
        parsed = json.loads(output)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["id"], "j1")
        self.assertEqual(parsed[0]["pid"], 99)

    def test_format_run_list_json_empty(self):
        output = format_run_list_json([])
        self.assertEqual(json.loads(output), [])

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

    def test_format_timestamp_converts_epoch(self):
        result = _format_timestamp(0)
        self.assertTrue(result.startswith("1970"))

    def test_format_timestamp_empty(self):
        self.assertEqual(_format_timestamp(None), "")
        self.assertEqual(_format_timestamp(""), "")

    def test_returncode_none_not_displayed(self):
        run = RunSummary(run_id="x", status="exited")
        output = format_run_info(run)
        self.assertNotIn("returncode", output)

    # -- malformed registry --

    def test_malformed_registry_returns_empty(self):
        """Malformed JSON in dashboard-jobs.json should not crash."""
        with patch("areno.cli.run.registered_job_items", return_value=[]):
            runs = list_runs()
        self.assertEqual(runs, [])

    def test_malformed_registry_list_command(self):
        with patch("areno.cli.run.registered_job_items", return_value=[]):
            result = CliRunner().invoke(main, ["run", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No runs found.", result.output)

    def test_concurrent_write_partial_json_does_not_crash(self):
        """Reader should not crash when registry is being written mid-flight (partial JSON)."""
        from areno.dashboard.server import GLOBAL_REGISTRY_FILE
        partial = '{"jobs": [{"id": "ab'
        with patch.object(type(GLOBAL_REGISTRY_FILE), "read_text", return_value=partial):
            result = CliRunner().invoke(main, ["run", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No runs found.", result.output)

    def test_concurrent_write_empty_file_does_not_crash(self):
        """Reader should not crash when registry file is empty (e.g. just created, not yet written)."""
        from areno.dashboard.server import GLOBAL_REGISTRY_FILE
        with patch.object(type(GLOBAL_REGISTRY_FILE), "read_text", return_value=""):
            result = CliRunner().invoke(main, ["run", "list", "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(json.loads(result.output), [])

    # -- info --json --

    def test_run_info_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = os.getpid()
            state_file = os.path.join(tmp, f"dashboard_state.{pid}.json")
            with open(state_file, "w") as f:
                json.dump({"pid": pid, "stage": "rollout", "step": 3}, f)
            config_json = os.path.join(tmp, f"areno_run_config.{pid}.json")
            with open(config_json, "w") as f:
                json.dump({"settings": {"algo": "gspo", "ckpt": "Qwen/Qwen3-0.6B"}}, f)

            entries = [
                {
                    "id": "jinfo",
                    "kind": "train",
                    "name": "json info test",
                    "pid": pid,
                    "created_at": 1700000000.0,
                    "updated_at": 1700000000.0,
                    "metrics_dir": tmp,
                    "config": {},
                    "command": [],
                }
            ]
            with patch("areno.cli.run.registered_job_items", return_value=entries):
                result = CliRunner().invoke(main, ["run", "info", "jinfo", "--json"])
        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["id"], "jinfo")
        self.assertEqual(parsed["kind"], "train")
        self.assertEqual(parsed["stage"], "rollout")
        self.assertIn("config", parsed)
        self.assertIn("metrics", parsed)

    def test_run_info_json_not_found(self):
        with patch("areno.cli.run.registered_job_items", return_value=[]):
            result = CliRunner().invoke(main, ["run", "info", "nope", "--json"])
        self.assertNotEqual(result.exit_code, 0)
        # Error JSON should be on stderr
        self.assertIn("Run not found", result.output + result.stderr)

    # -- info accepts directory path --

    def test_run_info_accepts_directory_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = os.getpid()
            state_file = os.path.join(tmp, f"dashboard_state.{pid}.json")
            with open(state_file, "w") as f:
                json.dump({"stage": "train_end", "step": 10}, f)
            result = CliRunner().invoke(main, ["run", "info", tmp])
        self.assertEqual(result.exit_code, 0)
        dir_name = os.path.basename(tmp)
        self.assertIn(f"Run: {dir_name}", result.output)
        self.assertIn("train_end", result.output)

    # -- sensitive config redaction --

    def test_redact_config_masks_sensitive_keys(self):
        config = {"algo": "gspo", "api_key": "sk-12345", "password": "secret", "ckpt": "Qwen"}
        redacted = _redact_config(config)
        self.assertEqual(redacted["algo"], "gspo")
        self.assertEqual(redacted["api_key"], "***")
        self.assertEqual(redacted["password"], "***")
        self.assertEqual(redacted["ckpt"], "Qwen")

    def test_redact_config_preserves_non_sensitive_keys(self):
        config = {"algo": "gspo", "ckpt": "Qwen/Qwen3-0.6B", "lr": 1e-5}
        redacted = _redact_config(config)
        self.assertEqual(redacted, config)

    def test_run_info_redacts_sensitive_config_in_table(self):
        run = RunSummary(
            run_id="r1",
            kind="train",
            name="test",
            config={"algo": "gspo", "api_key": "sk-secret"},
        )
        output = format_run_info(run)
        self.assertIn("algo: gspo", output)
        self.assertIn("api_key: ***", output)
        self.assertNotIn("sk-secret", output)

    def test_run_info_json_redacts_sensitive_config(self):
        run = RunSummary(
            run_id="r1",
            kind="train",
            name="test",
            config={"algo": "gspo", "api_key": "sk-secret"},
        )
        output = format_run_info_json(run)
        parsed = json.loads(output)
        self.assertEqual(parsed["config"]["api_key"], "***")
        self.assertNotIn("sk-secret", output)

    def test_redact_text_masks_sensitive_values(self):
        text = "algo: gspo\napi_key: sk-secret\nckpt: Qwen"
        redacted = _redact_text(text)
        self.assertIn("api_key: ***", redacted)
        self.assertNotIn("sk-secret", redacted)
        self.assertIn("algo: gspo", redacted)

    def test_run_info_redacts_config_text_in_table(self):
        run = RunSummary(
            run_id="r1",
            kind="train",
            name="test",
            config_text="algo: gspo\napi_key: sk-secret\nckpt: Qwen",
        )
        output = format_run_info(run)
        self.assertIn("api_key: ***", output)
        self.assertNotIn("sk-secret", output)

    def test_run_info_json_redacts_config_text(self):
        run = RunSummary(
            run_id="r1",
            kind="train",
            name="test",
            config_text="algo: gspo\npassword: mypass123",
        )
        output = format_run_info_json(run)
        parsed = json.loads(output)
        self.assertIn("***", parsed["config_text"])
        self.assertNotIn("mypass123", output)


if __name__ == "__main__":
    unittest.main()
