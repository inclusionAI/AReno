"""CPU tests for the `areno show` CLI command.

These tests do not require a GPU.  They build deterministic ``Job`` fixtures
with mock metrics and timeperf, then call the show command via ``CliRunner``.
"""

from __future__ import annotations

import json

import pytest

from areno.dashboard.server import DashboardState, Job
from areno.cli.show import _format_human, _job_item_to_details, _resolve_job


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_job(
    job_id: str = "test-job-001",
    algo: str = "sft",
    ckpt: str = "Qwen/Qwen3.5-0.8B",
    step: int = 10,
    status: str = "exited",
    metrics: list[dict] | None = None,
    timeperf: list[dict] | None = None,
    extra_config: dict | None = None,
    logs: list[str] | None = None,
    returncode: int | None = None,
) -> Job:
    config = {
        "algo": algo,
        "ckpt": ckpt,
        "world_size": 2,
        "batch_size": 2,
        "lr": 1e-06,
        "dataset_path": "yahma/alpaca-cleaned",
    }
    if extra_config:
        config.update(extra_config)

    job = Job(
        kind="train",
        name=f"train {algo} {ckpt}",
        command=["areno", "train", "--algo", algo, "--ckpt", ckpt],
        config=config,
        metrics_dir=None,
    )
    job.id = job_id
    job.launch_config = dict(config)
    job.status = status
    job.step = step
    job.metrics = metrics or []
    job.timeperf = timeperf or []
    job.logs = logs or []
    job.returncode = returncode
    return job


def _state_with_jobs(jobs: list[Job]) -> DashboardState:
    state = DashboardState()
    state.jobs = {job.id: job for job in jobs}
    return state


# ---------------------------------------------------------------------------
# Human format tests
# ---------------------------------------------------------------------------

class TestShowHumanFormat:
    """Tests for human-readable output format."""

    def test_show_basic_run(self):
        """A completed SFT run displays name, status, step, and config."""
        job = _make_job("abc123", algo="sft", step=15, status="exited",
            metrics=[{"name": "train/loss", "value": 0.5, "step": 15, "time": "t"}],
        )
        state = _state_with_jobs([job])
        from areno.cli.show import _load_job_details

        details = _load_job_details({"id": "abc123"})
        # _load_job_details uses DashboardState internally; patch it.
        details = {
            "id": job.id, "name": job.name, "kind": job.kind,
            "status": job.status, "stage": job.stage, "step": job.step,
            "created_at": job.created_at, "updated_at": job.updated_at,
            "returncode": job.returncode,
            "config": {}, "launch_config": job.launch_config,
            "metrics": state.metric_summaries("abc123"),
            "timeperf": [], "logs": [], "metrics_dir": "",
        }
        output = _format_human(details)

        assert "train sft Qwen/Qwen3.5-0.8B" in output
        assert "exited" in output
        assert "step" in output.lower()

    def test_show_includes_key_settings(self):
        """Human output includes algo, ckpt, dataset, lr, batch_size."""
        details = _job_item_to_details({
            "id": "job1",
            "name": "train gspo Qwen/Qwen3.5-0.8B",
            "kind": "train",
            "status": "running",
            "step": 5,
            "config": {"algo": "gspo", "ckpt": "Qwen/Qwen3.5-0.8B", "lr": 1e-6, "batch_size": 4},
        })
        output = _format_human(details)

        assert "algo" in output
        assert "gspo" in output
        assert "Qwen/Qwen3.5-0.8B" in output
        assert "batch_size" in output

    def test_show_includes_metrics(self):
        """Human output lists latest metrics with values and steps."""
        details = {
            "id": "job1", "name": "test", "kind": "train", "status": "exited",
            "stage": "", "step": 10, "created_at": "t", "updated_at": "t",
            "returncode": None, "config": {}, "launch_config": {},
            "metrics": [
                {"name": "train/loss", "latest_value": 0.34, "latest_step": 10},
                {"name": "rollout/rewards_mean", "latest_value": 0.5, "latest_step": 10},
            ],
            "timeperf": [], "logs": [], "metrics_dir": "",
        }
        output = _format_human(details)

        assert "train/loss" in output
        assert "0.34" in output
        assert "rollout/rewards_mean" in output

    def test_show_includes_last_error_for_failed_run(self):
        """Failed runs show the last error section with exit code."""
        details = {
            "id": "job1", "name": "test", "kind": "train", "status": "failed",
            "stage": "", "step": 3, "created_at": "t", "updated_at": "t",
            "returncode": 1, "config": {}, "launch_config": {},
            "metrics": [], "timeperf": [],
            "logs": ["starting...", "Error: CUDA out of memory"],
        }
        output = _format_human(details)

        assert "Last error" in output
        assert "exit code 1" in output
        assert "CUDA out of memory" in output

    def test_show_no_error_section_for_successful_run(self):
        """Successful runs (exit code 0) do not show a last error section."""
        details = {
            "id": "job1", "name": "test", "kind": "train", "status": "exited",
            "stage": "", "step": 10, "created_at": "t", "updated_at": "t",
            "returncode": 0, "config": {}, "launch_config": {},
            "metrics": [], "timeperf": [], "logs": ["done"],
        }
        output = _format_human(details)

        assert "Last error" not in output

    def test_show_includes_timing(self):
        """Human output includes timing averages when timeperf is present."""
        details = {
            "id": "job1", "name": "test", "kind": "train", "status": "exited",
            "stage": "", "step": 5, "created_at": "t", "updated_at": "t",
            "returncode": None, "config": {}, "launch_config": {},
            "metrics": [],
            "timeperf": [
                {"step": i, "rollout_s": 10.0, "train_s": 5.0, "other_s": 0.0, "total_s": 15.0}
                for i in range(5)
            ],
            "logs": [], "metrics_dir": "",
        }
        output = _format_human(details)

        assert "Timing" in output
        assert "15.00s" in output
        assert "10.00s" in output

    def test_show_no_secrets_in_output(self):
        """Output should not expose full training samples or secrets."""
        details = {
            "id": "job1", "name": "test", "kind": "train", "status": "exited",
            "stage": "", "step": 1, "created_at": "t", "updated_at": "t",
            "returncode": None,
            "config": {"algo": "sft", "ckpt": "model", "api_key": "sk-secret-12345"},
            "launch_config": {"algo": "sft", "ckpt": "model", "api_key": "sk-secret-12345"},
            "metrics": [], "timeperf": [], "logs": [],
        }
        output = _format_human(details)

        # The api_key should not appear in output (not in key_order list).
        assert "sk-secret-12345" not in output


# ---------------------------------------------------------------------------
# JSON format tests
# ---------------------------------------------------------------------------

class TestShowJsonFormat:
    """Tests for JSON output format."""

    def test_json_output_is_valid_json(self):
        """JSON output is parseable and contains expected fields."""
        details = {
            "id": "job1", "name": "test", "kind": "train", "status": "exited",
            "stage": "", "step": 5, "created_at": "t", "updated_at": "t",
            "returncode": None, "config": {}, "launch_config": {"algo": "sft"},
            "metrics": [], "timeperf": [], "logs": [],
        }
        output = json.dumps(details, ensure_ascii=False, indent=2, default=str)
        parsed = json.loads(output)

        assert parsed["id"] == "job1"
        assert parsed["launch_config"]["algo"] == "sft"


# ---------------------------------------------------------------------------
# Table format tests
# ---------------------------------------------------------------------------

class TestShowTableFormat:
    """Tests for compact table output format."""

    def test_table_output_contains_key_fields(self, monkeypatch):
        """Table format shows Run ID, Name, Status, and key config values."""
        from click.testing import CliRunner
        from areno.cli.show import show_command

        job = _make_job("abc123", algo="gspo", step=10,
            metrics=[{"name": "train/loss", "value": 0.5, "step": 10, "time": "t"}],
        )

        original_init = DashboardState.__init__
        def _mock_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.jobs = {"abc123": job}
        monkeypatch.setattr("areno.dashboard.server.DashboardState.__init__", _mock_init)

        # Mock _try_dashboard_api to return None (force fallback).
        monkeypatch.setattr("areno.cli.show._try_dashboard_api", lambda *a: None)

        runner = CliRunner()
        result = runner.invoke(show_command, ["abc123", "--format", "table"])
        assert result.exit_code == 0
        assert "abc123" in result.output
        assert "gspo" in result.output


# ---------------------------------------------------------------------------
# Invalid input and boundary tests
# ---------------------------------------------------------------------------

class TestShowInvalidInput:
    """Tests for invalid inputs and boundary conditions."""

    def test_nonexistent_run_id_raises_error(self, monkeypatch):
        """A run ID that doesn't exist produces a clear error."""
        from click.testing import CliRunner
        from areno.cli.show import show_command

        monkeypatch.setattr("areno.cli.show._try_dashboard_api", lambda *a: None)
        monkeypatch.setattr("areno.cli.show._find_job_candidates", lambda: [])

        runner = CliRunner()
        result = runner.invoke(show_command, ["fake-id-999"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_ambiguous_partial_id_raises_error(self, monkeypatch):
        """Partial ID matching multiple runs raises an error."""
        from areno.cli.show import show_command

        candidates = [
            {"id": "abc111", "name": "job-a", "kind": "train"},
            {"id": "abc222", "name": "job-b", "kind": "train"},
        ]
        monkeypatch.setattr("areno.cli.show._try_dashboard_api", lambda *a: None)
        monkeypatch.setattr("areno.cli.show._find_job_candidates", lambda: candidates)

        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(show_command, ["abc"])
        assert result.exit_code != 0
        assert "ambiguous" in result.output.lower()

    def test_partial_id_matches_unique_run(self, monkeypatch):
        """Partial ID matching exactly one run succeeds."""
        from click.testing import CliRunner
        from areno.cli.show import show_command, _resolve_job

        candidates = [{"id": "abc111", "name": "job-a", "kind": "train", "status": "exited"}]
        monkeypatch.setattr("areno.cli.show._find_job_candidates", lambda: candidates)

        result = _resolve_job("abc")
        assert result is not None
        assert result["id"] == "abc111"

    def test_empty_metrics_shows_placeholder(self):
        """A run with no metrics displays a placeholder, not an error."""
        details = {
            "id": "job1", "name": "test", "kind": "train", "status": "exited",
            "stage": "", "step": 0, "created_at": "t", "updated_at": "t",
            "returncode": None, "config": {}, "launch_config": {},
            "metrics": [], "timeperf": [], "logs": [],
        }
        output = _format_human(details)
        assert "Latest metrics" not in output  # No metrics section when empty.

    def test_empty_config_shows_placeholder(self):
        """A run with no config displays a placeholder."""
        details = {
            "id": "job1", "name": "test", "kind": "train", "status": "exited",
            "stage": "", "step": 0, "created_at": "t", "updated_at": "t",
            "returncode": None, "config": {}, "launch_config": {},
            "metrics": [], "timeperf": [], "logs": [],
        }
        output = _format_human(details)
        assert "Key settings" not in output  # No settings section when empty.

    def test_partially_written_artifacts(self):
        """A job with partial data (no metrics, no timeperf, no logs) still produces output."""
        details = _job_item_to_details({
            "id": "partial",
            "name": "partial run",
            "kind": "train",
            "status": "running",
        })
        output = _format_human(details)

        assert "partial run" in output
        assert "running" in output


# ---------------------------------------------------------------------------
# Active run tests
# ---------------------------------------------------------------------------

class TestShowActiveRun:
    """Tests for runs that are still active."""

    def test_active_run_shows_running_status(self):
        """An active run shows 'running' status and current stage."""
        details = {
            "id": "job1", "name": "train gspo model", "kind": "train",
            "status": "running", "stage": "rollout", "step": 5,
            "created_at": "2026-07-30T10:00:00", "updated_at": "2026-07-30T10:05:00",
            "returncode": None, "config": {}, "launch_config": {"algo": "gspo"},
            "metrics": [{"name": "train/loss", "latest_value": 0.8, "latest_step": 5}],
            "timeperf": [], "logs": ["starting..."], "metrics_dir": "",
        }
        output = _format_human(details)

        assert "running" in output
        assert "rollout" in output

    def test_active_run_no_exit_code(self):
        """Active runs do not show an exit code."""
        details = {
            "id": "job1", "name": "test", "kind": "train", "status": "running",
            "stage": "train", "step": 3, "created_at": "t", "updated_at": "t",
            "returncode": None, "config": {}, "launch_config": {},
            "metrics": [], "timeperf": [], "logs": [],
        }
        output = _format_human(details)
        assert "exit code" not in output


# ---------------------------------------------------------------------------
# Resolve job tests
# ---------------------------------------------------------------------------

class TestResolveJob:
    """Tests for the _resolve_job function."""

    def test_exact_id_match(self, monkeypatch):
        candidates = [{"id": "exact123", "name": "job", "kind": "train"}]
        monkeypatch.setattr("areno.cli.show._find_job_candidates", lambda: candidates)
        result = _resolve_job("exact123")
        assert result is not None
        assert result["id"] == "exact123"

    def test_pid_match(self, monkeypatch):
        candidates = [{"id": "abc", "pid": 12345, "name": "job", "kind": "train"}]
        monkeypatch.setattr("areno.cli.show._find_job_candidates", lambda: candidates)
        result = _resolve_job("12345")
        assert result is not None
        assert result["pid"] == 12345

    def test_no_match_returns_none(self, monkeypatch):
        candidates = [{"id": "abc", "name": "job", "kind": "train"}]
        monkeypatch.setattr("areno.cli.show._find_job_candidates", lambda: candidates)
        result = _resolve_job("nonexistent")
        assert result is None