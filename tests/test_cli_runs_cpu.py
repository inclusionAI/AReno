"""CPU tests for ``areno runs``."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from areno.cli.main import main
from areno.cli.dashboard_registry import compute_age, compute_duration, derive_status, format_table, read_registry

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

NOW = 1753700000.0
PIDS = [12345, 12346, 12347, 12348, 12349]


def _make_registry_entry(
    idx: int,
    *,
    kind: str = "train",
    name: str = "gspo Qwen/Qwen3-0.6B",
    pid: int | None = None,
    created_at: float | None = None,
    updated_at: float | None = None,
    metrics_dir: str = "/tmp/areno/tfevent",
) -> dict:
    return {
        "id": f"job{idx:04d}",
        "kind": kind,
        "name": name,
        "pid": pid if pid is not None else PIDS[idx],
        "command": ["areno", "train", "--ckpt", "model"],
        "config": {},
        "metrics_dir": metrics_dir,
        "cwd": "/tmp",
        "created_at": created_at if created_at is not None else NOW - idx * 3600,
        "updated_at": updated_at if updated_at is not None else NOW - idx * 3600 + 600,
    }


def _make_dashboard_state(
    *,
    stage: str = "",
    status: str = "running",
    step: int | None = None,
) -> dict:
    payload: dict = {
        "pid": os.getpid(),
        "stage": stage,
        "status": status,
        "updated_at": NOW,
    }
    if step is not None:
        payload["step"] = step
    return payload


def _write_registry(path: Path, jobs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")


def _write_dashboard_state(metrics_dir: Path, pid: int, state: dict) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    state_path = metrics_dir / f"dashboard_state.{pid}.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit tests – run_utils helpers
# ---------------------------------------------------------------------------


class RunUtilsTest(unittest.TestCase):
    def test_compute_age_seconds(self):
        age = compute_age(NOW - 30, now=NOW)
        self.assertEqual(age, "30s ago")

    def test_compute_age_minutes(self):
        age = compute_age(NOW - 180, now=NOW)
        self.assertEqual(age, "3m 0s ago")

    def test_compute_age_hours(self):
        age = compute_age(NOW - 7200, now=NOW)
        self.assertEqual(age, "2h 0m ago")

    def test_compute_age_days(self):
        age = compute_age(NOW - 172800, now=NOW)
        self.assertEqual(age, "2d 0h ago")

    def test_compute_duration(self):
        dur = compute_duration(NOW - 125, updated_at=NOW, now=NOW)
        self.assertEqual(dur, "2m 5s")

    def test_compute_duration_running(self):
        dur = compute_duration(NOW - 500, updated_at=None, now=NOW)
        self.assertEqual(dur, "8m 20s")

    def test_format_table_empty(self):
        self.assertEqual(format_table(["A", "B"], []), "")

    def test_format_table_simple(self):
        output = format_table(["COL1", "COL2"], [["a", "b"], ["longer", "d"]])
        lines = output.split("\n")
        self.assertEqual(len(lines), 3)
        self.assertIn("COL1", lines[0])
        self.assertIn("COL2", lines[0])

    def test_read_registry_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text("{}", encoding="utf-8")
            jobs = read_registry(path)
            self.assertEqual(jobs, [])

    def test_read_registry_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json", encoding="utf-8")
            jobs = read_registry(path)
            self.assertEqual(jobs, [])

    def test_read_registry_deduplicates_by_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            jobs = [_make_registry_entry(0, pid=100), _make_registry_entry(1, pid=100, name="later")]
            _write_registry(path, jobs)
            result = read_registry(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["name"], "later")

    def test_read_registry_missing_file_returns_empty(self):
        jobs = read_registry(Path("/nonexistent/path/registry.json"))
        self.assertEqual(jobs, [])

    def test_derive_status_running_with_stage(self):
        entry = _make_registry_entry(0, metrics_dir="/tmp/fake")
        fake_state = _make_dashboard_state(stage="rollout")
        with (
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=True),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
        ):
            self.assertEqual(derive_status(entry), "rollout")

    def test_derive_status_running_no_stage(self):
        entry = _make_registry_entry(0, metrics_dir="/tmp/fake")
        fake_state = _make_dashboard_state(stage="")
        with (
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=True),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
        ):
            self.assertEqual(derive_status(entry), "running")

    def test_derive_status_succeeded(self):
        entry = _make_registry_entry(0, metrics_dir="/tmp/fake")
        fake_state = _make_dashboard_state(status="succeeded")
        with (
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=False),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
        ):
            self.assertEqual(derive_status(entry), "succeeded")

    def test_derive_status_failed(self):
        entry = _make_registry_entry(0, metrics_dir="/tmp/fake")
        fake_state = _make_dashboard_state(status="failed")
        with (
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=False),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
        ):
            self.assertEqual(derive_status(entry), "failed")

    def test_derive_status_stopped(self):
        entry = _make_registry_entry(0, metrics_dir="/tmp/fake")
        fake_state = _make_dashboard_state(status="stopped")
        with (
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=False),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
        ):
            self.assertEqual(derive_status(entry), "stopped")

    def test_derive_status_exited_no_state(self):
        entry = _make_registry_entry(0, metrics_dir="/tmp/fake")
        with (
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=False),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=None),
        ):
            self.assertEqual(derive_status(entry), "exited")

    def test_derive_status_unknown_no_pid(self):
        entry = _make_registry_entry(0, pid=12345, metrics_dir="/tmp/fake")
        del entry["pid"]
        self.assertEqual(derive_status(entry), "unknown")

    def test_derive_status_unknown_bad_pid_type(self):
        entry = _make_registry_entry(0, pid=12345, metrics_dir="/tmp/fake")
        entry["pid"] = "not-an-int"
        self.assertEqual(derive_status(entry), "unknown")


# ---------------------------------------------------------------------------
# Integration tests – CLI via CliRunner
# ---------------------------------------------------------------------------


class CliRunsTest(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def _invoke(self, *args: str):
        return self.runner.invoke(main, ["runs", *args])

    def test_no_registry_shows_message(self):
        with patch("areno.cli.runs.read_registry", return_value=[]):
            result = self._invoke()
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No registered", result.output)

    def test_no_registry_json_empty_array(self):
        with patch("areno.cli.runs.read_registry", return_value=[]):
            result = self._invoke("--json")
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data, [])

    def test_runs_listed_as_help_subcommand(self):
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("runs", result.output)

    def test_json_output_is_valid(self):
        entries = [
            _make_registry_entry(0, kind="train", name="job-zero"),
            _make_registry_entry(1, kind="serve", name="job-one"),
        ]
        fake_state = _make_dashboard_state(stage="rollout")
        with (
            patch("areno.cli.runs.read_registry", return_value=entries),
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=True),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
            patch("areno.cli.runs.time.time", return_value=NOW),
        ):
            result = self._invoke("--json")
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        for item in data:
            self.assertIn("id", item)
            self.assertIn("kind", item)
            self.assertIn("name", item)
            self.assertIn("pid", item)
            self.assertIn("status", item)
            self.assertIn("age", item)
            self.assertIn("duration", item)

    def test_deterministic_sort_most_recent_first(self):
        entries = [
            _make_registry_entry(0, created_at=NOW - 7200, pid=100),
            _make_registry_entry(1, created_at=NOW - 3600, pid=200),
            _make_registry_entry(2, created_at=NOW, pid=300),
        ]
        fake_state = _make_dashboard_state(stage="train")
        with (
            patch("areno.cli.runs.read_registry", return_value=entries),
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=True),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
            patch("areno.cli.runs.time.time", return_value=NOW),
        ):
            result = self._invoke("--json")
        data = json.loads(result.output)
        pids = [item["pid"] for item in data]
        self.assertEqual(pids, [300, 200, 100])

    def test_limit_respected(self):
        entries = [_make_registry_entry(i) for i in range(5)]
        fake_state = _make_dashboard_state(stage="rollout")
        with (
            patch("areno.cli.runs.read_registry", return_value=entries),
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=True),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
            patch("areno.cli.runs.time.time", return_value=NOW),
        ):
            result = self._invoke("--json", "--limit", "3")
        data = json.loads(result.output)
        self.assertEqual(len(data), 3)

    def test_all_shows_all(self):
        entries = [_make_registry_entry(i) for i in range(5)]
        fake_state = _make_dashboard_state(stage="rollout")
        with (
            patch("areno.cli.runs.read_registry", return_value=entries),
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=True),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
            patch("areno.cli.runs.time.time", return_value=NOW),
        ):
            result = self._invoke("--json", "--all")
        data = json.loads(result.output)
        self.assertEqual(len(data), 5)

    def test_table_output_contains_status_and_kind(self):
        entries = [_make_registry_entry(0, kind="train", name="my-train-job")]
        fake_state = _make_dashboard_state(stage="rollout")
        with (
            patch("areno.cli.runs.read_registry", return_value=entries),
            patch("areno.cli.dashboard_registry.pid_is_running", return_value=True),
            patch("areno.cli.dashboard_registry.read_dashboard_state", return_value=fake_state),
            patch("areno.cli.runs.time.time", return_value=NOW),
        ):
            result = self._invoke()
        self.assertEqual(result.exit_code, 0)
        self.assertIn("STATUS", result.output)
        self.assertIn("KIND", result.output)
        self.assertIn("my-train-job", result.output)
        self.assertIn("train", result.output)

    def test_runs_help(self):
        result = self._invoke("--help")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--json", result.output)
        self.assertIn("--limit", result.output)
        self.assertIn("--all", result.output)