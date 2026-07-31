"""CPU tests for the dashboard compare-two-runs feature.

These tests do not require a GPU.  They build deterministic ``Job`` fixtures
with mock metrics and timeperf, then call :meth:`DashboardState.compare_jobs`
directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from areno.dashboard.server import DashboardState, Job


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_job(
    job_id: str,
    algo: str = "gspo",
    ckpt: str = "Qwen/Qwen3.5-0.8B",
    step: int = 0,
    metrics: list[dict] | None = None,
    timeperf: list[dict] | None = None,
    extra_config: dict | None = None,
) -> Job:
    """Create a deterministic Job for testing."""
    config = {
        "algo": algo,
        "ckpt": ckpt,
        "world_size": 2,
        "tp_size": 2,
        "batch_size": 2,
        "n_samples": 4,
        "lr": 1e-06,
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
    job.config = dict(config)
    job.status = "exited"
    job.step = step
    job.metrics = metrics or []
    job.timeperf = timeperf or []
    return job


def _state_with_jobs(jobs: list[Job]) -> DashboardState:
    """Create a DashboardState with pre-populated jobs (no file I/O)."""
    state = DashboardState()
    state.jobs = {job.id: job for job in jobs}
    return state


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------

class TestCompareSuccess:
    """Tests for the happy path with two comparable jobs."""

    def test_compare_two_normal_jobs(self):
        """Two GSPO jobs with different config and metrics produce a correct diff."""
        job_a = _make_job(
            "job-a", algo="gspo", step=13,
            metrics=[
                {"name": "loss", "value": 0.65, "step": 13, "time": "t"},
                {"name": "reward_mean", "value": 0.324, "step": 13, "time": "t"},
            ],
            timeperf=[
                {"step": i, "rollout_s": 37.0, "train_s": 14.5, "other_s": 0.0, "total_s": 51.5, "time": "t"}
                for i in range(14)
            ],
            extra_config={"n_samples": 4, "max_new_tokens": 1024},
        )
        job_b = _make_job(
            "job-b", algo="gspo", step=11,
            metrics=[
                {"name": "loss", "value": 0.80, "step": 11, "time": "t"},
                {"name": "reward_mean", "value": 0.45, "step": 11, "time": "t"},
            ],
            timeperf=[
                {"step": i, "rollout_s": 36.6, "train_s": 13.7, "other_s": 0.1, "total_s": 50.4, "time": "t"}
                for i in range(12)
            ],
            extra_config={"n_samples": 4, "max_new_tokens": 512},
        )
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("job-a", "job-b")

        assert result["comparable"] is True
        assert result["job_a"]["id"] == "job-a"
        assert result["job_b"]["id"] == "job-b"

        # Config: max_new_tokens differs.
        diff_keys = {item["key"]: item for item in result["config"]["different"]}
        assert "max_new_tokens" in diff_keys
        assert diff_keys["max_new_tokens"]["value_a"] == 1024
        assert diff_keys["max_new_tokens"]["value_b"] == 512

        # Config: ckpt is identical.
        identical_keys = {item["key"] for item in result["config"]["identical"]}
        assert "ckpt" in identical_keys
        assert "algo" in identical_keys

        # Metrics: loss diff and reward_mean diff.
        metrics_by_name = {m["name"]: m for m in result["metrics"]}
        assert "loss" in metrics_by_name
        assert metrics_by_name["loss"]["comparable"] is True
        assert metrics_by_name["loss"]["diff"] == round(0.65 - 0.80, 6)
        assert metrics_by_name["reward_mean"]["comparable"] is True

        # Timing.
        assert result["timing"]["job_a"]["total_steps"] == 14
        assert result["timing"]["job_b"]["total_steps"] == 12
        assert result["timing"]["comparison"]["steps_diff"] == 2

    def test_compare_gspo_vs_sft(self):
        """GSPO vs SFT: RL-only fields get notes, no reward metrics on SFT side."""
        job_a = _make_job(
            "gspo-job", algo="gspo", step=5,
            metrics=[{"name": "reward_mean", "value": 0.5, "step": 5, "time": "t"}],
            timeperf=[
                {"step": i, "rollout_s": 10.0, "train_s": 5.0, "other_s": 0.0, "total_s": 15.0, "time": "t"}
                for i in range(6)
            ],
        )
        job_b = _make_job(
            "sft-job", algo="sft", step=5,
            metrics=[{"name": "loss", "value": 1.2, "step": 5, "time": "t"}],
            timeperf=[
                {"step": i, "rollout_s": 0.0, "train_s": 20.0, "other_s": 0.0, "total_s": 20.0, "time": "t"}
                for i in range(6)
            ],
            extra_config={"n_samples": None, "reward_fn_path": None},
        )
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("gspo-job", "sft-job")

        # Config: algo differs, RL-only fields have notes.
        diff_keys = {item["key"]: item for item in result["config"]["different"]}
        assert diff_keys["algo"]["value_a"] == "gspo"
        assert diff_keys["algo"]["value_b"] == "sft"

        # n_samples should be in different with a note for SFT.
        if "n_samples" in diff_keys:
            assert diff_keys["n_samples"]["note"] is not None
            assert "sft" in diff_keys["n_samples"]["note"].lower()

        # Metrics: reward_mean only in gspo.
        metrics_by_name = {m["name"]: m for m in result["metrics"]}
        assert "reward_mean" in metrics_by_name
        assert metrics_by_name["reward_mean"]["comparable"] is False
        assert metrics_by_name["reward_mean"]["note"] is not None

        # Timing: rollout for SFT is null.
        assert result["timing"]["job_b"]["avg_rollout_s"] is None

    def test_compare_identical_config(self):
        """Two jobs with the same config: all compared fields in identical."""
        job_a = _make_job("a", algo="sft", step=5)
        job_b = _make_job("b", algo="sft", step=10)
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert len(result["config"]["different"]) == 0
        assert len(result["config"]["identical"]) > 0


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

class TestCompareInvalid:
    """Tests for invalid inputs that should raise ValueError."""

    def test_compare_missing_job_a(self):
        state = _state_with_jobs([])
        with pytest.raises(ValueError, match="job_a and job_b are required"):
            state.compare_jobs(None, "some-id")

    def test_compare_missing_job_b(self):
        state = _state_with_jobs([])
        with pytest.raises(ValueError, match="job_a and job_b are required"):
            state.compare_jobs("some-id", None)

    def test_compare_nonexistent_job(self):
        job = _make_job("real-job")
        state = _state_with_jobs([job])
        with pytest.raises(ValueError, match="not found"):
            state.compare_jobs("real-job", "fake-job")


# ---------------------------------------------------------------------------
# Boundary / edge cases
# ---------------------------------------------------------------------------

class TestCompareBoundary:
    """Tests for boundary conditions."""

    def test_compare_same_job(self):
        """Comparing a job with itself returns comparable=False."""
        job = _make_job("solo", step=5)
        state = _state_with_jobs([job])
        result = state.compare_jobs("solo", "solo")

        assert result["comparable"] is False
        assert result["reason"] == "same job"

    def test_compare_no_metrics(self):
        """Both jobs have zero metrics: empty metrics list, not error."""
        job_a = _make_job("a", step=0)
        job_b = _make_job("b", step=0)
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert result["metrics"] == []

    def test_compare_unequal_steps_with_note(self):
        """Jobs with very different step counts get a reliability note."""
        job_a = _make_job(
            "a", step=10,
            timeperf=[{"step": i, "rollout_s": 10.0, "train_s": 5.0, "other_s": 0.0, "total_s": 15.0, "time": "t"} for i in range(11)],
        )
        job_b = _make_job(
            "b", step=3,
            timeperf=[{"step": i, "rollout_s": 8.0, "train_s": 4.0, "other_s": 0.0, "total_s": 12.0, "time": "t"} for i in range(4)],
        )
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert result["timing"]["comparison"]["steps_diff"] == 7
        assert result["timing"]["comparison"]["note"] is not None
        assert "less reliable" in result["timing"]["comparison"]["note"].lower()

    def test_compare_no_timeperf(self):
        """Job with no timeperf: timing shows nulls with note."""
        job_a = _make_job("a", step=5, timeperf=[])
        job_b = _make_job("b", step=10, timeperf=[])
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert result["timing"]["job_a"]["total_steps"] == 0
        assert result["timing"]["job_a"]["avg_total_s"] is None
        assert result["timing"]["job_a"]["note"] == "no timing data available"

    def test_compare_empty_config(self):
        """Job with empty config: config section handles missing fields."""
        job_a = Job(kind="train", name="empty-a", command=[], config={}, metrics_dir=None)
        job_a.id = "empty-a"
        job_a.status = "exited"
        job_a.launch_config = {}

        job_b = Job(kind="train", name="empty-b", command=[], config={}, metrics_dir=None)
        job_b.id = "empty-b"
        job_b.status = "exited"
        job_b.launch_config = {}

        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("empty-a", "empty-b")

        assert result["comparable"] is True
        assert result["config"]["identical"] == []
        assert result["config"]["different"] == []


# ---------------------------------------------------------------------------
# Active-run and backward-compatibility tests
# ---------------------------------------------------------------------------

class TestCompareActiveAndCompat:
    """Tests for active (running) jobs and backward compatibility."""

    def test_compare_active_running_jobs(self):
        """Comparing two running jobs should work (active writes scenario)."""
        job_a = _make_job("a", algo="gspo", step=3,
            metrics=[{"name": "loss", "value": 1.2, "step": 3, "time": "t"}],
            timeperf=[{"step": i, "rollout_s": 10.0, "train_s": 5.0, "other_s": 0.0, "total_s": 15.0, "time": "t"} for i in range(4)],
        )
        job_a.status = "running"
        job_a.stage = "rollout"

        job_b = _make_job("b", algo="gspo", step=2,
            metrics=[{"name": "loss", "value": 1.5, "step": 2, "time": "t"}],
            timeperf=[{"step": i, "rollout_s": 12.0, "train_s": 6.0, "other_s": 0.0, "total_s": 18.0, "time": "t"} for i in range(3)],
        )
        job_b.status = "running"
        job_b.stage = "train"

        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert result["comparable"] is True
        assert result["job_a"]["status"] == "running"
        assert result["job_b"]["status"] == "running"
        # Metrics should still be compared even when jobs are active.
        metrics_by_name = {m["name"]: m for m in result["metrics"]}
        assert "loss" in metrics_by_name
        assert metrics_by_name["loss"]["comparable"] is True

    def test_compare_duration_calculated(self):
        """Duration is computed from created_at and updated_at."""
        job_a = _make_job("a", step=5,
            timeperf=[{"step": 0, "rollout_s": 10.0, "train_s": 5.0, "other_s": 0.0, "total_s": 15.0, "time": "t"}],
        )
        job_a.created_at = "2026-07-28T10:00:00+00:00"
        job_a.updated_at = "2026-07-28T10:05:00+00:00"

        job_b = _make_job("b", step=5,
            timeperf=[{"step": 0, "rollout_s": 12.0, "train_s": 6.0, "other_s": 0.0, "total_s": 18.0, "time": "t"}],
        )
        job_b.created_at = "2026-07-28T10:00:00+00:00"
        job_b.updated_at = "2026-07-28T10:10:00+00:00"

        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert result["timing"]["job_a"]["duration_s"] == 300.0
        assert result["timing"]["job_b"]["duration_s"] == 600.0

    def test_existing_behavior_unchanged(self):
        """When compare is not invoked, existing methods work identically."""
        job_a = _make_job("a", step=5,
            metrics=[{"name": "loss", "value": 0.5, "step": 5, "time": "t"}],
            timeperf=[{"step": i, "rollout_s": 10.0, "train_s": 5.0, "other_s": 0.0, "total_s": 15.0, "time": "t"} for i in range(6)],
        )
        job_b = _make_job("b", step=3)
        state = _state_with_jobs([job_a, job_b])

        # Existing API methods should produce the same results as before.
        summaries = state.metric_summaries("a")
        assert len(summaries) == 1
        assert summaries[0]["name"] == "loss"

        series = state.metric_series("a", "loss")
        assert len(series) == 1

        job_list = state.list_jobs()
        assert len(job_list) == 2

        job = state.get_job("a")
        assert job is not None
        assert job.id == "a"

        # get_job for nonexistent returns None (not an exception).
        assert state.get_job("nonexistent") is None


# ---------------------------------------------------------------------------
# CLI compare command tests
# ---------------------------------------------------------------------------

class TestCompareCLI:
    """Tests for the `areno compare` CLI command."""

    def test_cli_compare_human_format(self, monkeypatch):
        """CLI produces human-readable output by default."""
        from click.testing import CliRunner
        from areno.cli.compare import compare_command

        job_a = _make_job("a", algo="gspo", step=5,
            metrics=[{"name": "loss", "value": 0.65, "step": 5, "time": "t"}],
        )
        job_b = _make_job("b", algo="sft", step=5,
            metrics=[{"name": "loss", "value": 0.80, "step": 5, "time": "t"}],
        )

        original_init = DashboardState.__init__
        def _mock_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.jobs = {"a": job_a, "b": job_b}
        monkeypatch.setattr("areno.dashboard.server.DashboardState.__init__", _mock_init)
        monkeypatch.setattr("areno.cli.compare._try_dashboard_api", lambda *a: None)

        runner = CliRunner()
        result = runner.invoke(compare_command, ["--job-a", "a", "--job-b", "b"])
        assert result.exit_code == 0
        assert "Job A" in result.output
        assert "Job B" in result.output
        assert "loss" in result.output

    def test_cli_compare_json_format(self, monkeypatch):
        """CLI produces structured JSON with --format json."""
        from click.testing import CliRunner
        from areno.cli.compare import compare_command

        job_a = _make_job("a", algo="gspo", step=3)
        job_b = _make_job("b", algo="sft", step=3)

        original_init = DashboardState.__init__
        def _mock_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.jobs = {"a": job_a, "b": job_b}
        monkeypatch.setattr("areno.dashboard.server.DashboardState.__init__", _mock_init)
        monkeypatch.setattr("areno.cli.compare._try_dashboard_api", lambda *a: None)

        runner = CliRunner()
        result = runner.invoke(compare_command, ["--job-a", "a", "--job-b", "b", "--format", "json"])
        assert result.exit_code == 0
        import json as _json
        # Strip any stderr lines that CliRunner may mix into output.
        output_lines = [line for line in result.output.strip().splitlines() if not line.startswith("dashboard")]
        parsed = _json.loads("\n".join(output_lines))
        assert parsed["comparable"] is True
        assert parsed["job_a"]["id"] == "a"

    def test_cli_compare_invalid_job(self, monkeypatch):
        """CLI exits with error for non-existent job."""
        from click.testing import CliRunner
        from areno.cli.compare import compare_command

        original_init = DashboardState.__init__
        def _mock_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.jobs = {}
        monkeypatch.setattr("areno.dashboard.server.DashboardState.__init__", _mock_init)
        monkeypatch.setattr("areno.cli.compare._try_dashboard_api", lambda *a: None)

        runner = CliRunner()
        result = runner.invoke(compare_command, ["--job-a", "x", "--job-b", "y"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Tests for new features: metric_charts, diff_summary, throughput
# ---------------------------------------------------------------------------

class TestCompareNewFeatures:
    """Tests for metric_charts, diff_summary, and throughput fields."""

    def test_metric_charts_returns_time_series(self):
        """metric_charts should return time-series points for each metric."""
        job_a = _make_job("a", step=3, metrics=[
            {"name": "loss", "value": 1.5, "step": 1, "time": "t"},
            {"name": "loss", "value": 1.0, "step": 2, "time": "t"},
            {"name": "loss", "value": 0.8, "step": 3, "time": "t"},
        ])
        job_b = _make_job("b", step=2, metrics=[
            {"name": "loss", "value": 2.0, "step": 1, "time": "t"},
            {"name": "loss", "value": 1.5, "step": 2, "time": "t"},
        ])
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert "metric_charts" in result
        assert "loss" in result["metric_charts"]
        assert len(result["metric_charts"]["loss"]["points_a"]) == 3
        assert len(result["metric_charts"]["loss"]["points_b"]) == 2

    def test_metric_charts_empty_when_no_metrics(self):
        """metric_charts should be empty dict when jobs have no metrics."""
        job_a = _make_job("a", step=0)
        job_b = _make_job("b", step=0)
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert result["metric_charts"] == {}

    def test_metric_charts_handles_metric_only_in_one_job(self):
        """metric_charts should include metrics present in only one job."""
        job_a = _make_job("a", step=1, metrics=[
            {"name": "unique_metric", "value": 0.5, "step": 1, "time": "t"},
        ])
        job_b = _make_job("b", step=1, metrics=[
            {"name": "loss", "value": 1.0, "step": 1, "time": "t"},
        ])
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert "unique_metric" in result["metric_charts"]
        assert len(result["metric_charts"]["unique_metric"]["points_a"]) == 1
        assert len(result["metric_charts"]["unique_metric"]["points_b"]) == 0

    def test_diff_summary_generated_for_different_configs(self):
        """diff_summary should list changed config items with arrows."""
        job_a = _make_job("a", algo="gspo", step=1, extra_config={"lr": 1e-6, "batch_size": 4})
        job_b = _make_job("b", algo="gspo", step=1, extra_config={"lr": 1e-5, "batch_size": 4})
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert "diff_summary" in result
        assert len(result["diff_summary"]) > 0
        lr_line = [s for s in result["diff_summary"] if "lr" in s]
        assert len(lr_line) == 1
        assert "1e-06" in lr_line[0] or "0.000001" in lr_line[0]
        assert "1e-05" in lr_line[0] or "0.00001" in lr_line[0]

    def test_diff_summary_shows_ratio_for_large_changes(self):
        """diff_summary should show (Nx) ratio when value changes by >=2x."""
        job_a = _make_job("a", step=1, extra_config={"batch_size": 4, "lr": 1e-6})
        job_b = _make_job("b", step=1, extra_config={"batch_size": 16, "lr": 1e-6})
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        batch_line = [s for s in result["diff_summary"] if "batch_size" in s]
        assert len(batch_line) == 1
        assert "4.0x" in batch_line[0]

    def test_diff_summary_empty_when_configs_identical(self):
        """diff_summary should be empty when configs are identical."""
        job_a = _make_job("a", algo="sft", step=1)
        job_b = _make_job("b", algo="sft", step=1)
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert result["diff_summary"] == []

    def test_throughput_calculated_from_duration_and_steps(self):
        """throughput should be steps/duration when both are available."""
        job_a = _make_job("a", step=10, timeperf=[
            {"step": i, "rollout_s": 5.0, "train_s": 5.0, "other_s": 0.0, "total_s": 10.0, "time": "t"}
            for i in range(10)
        ])
        job_a.created_at = "2026-07-28T10:00:00+00:00"
        job_a.updated_at = "2026-07-28T10:01:40+00:00"  # 100 seconds

        job_b = _make_job("b", step=5, timeperf=[
            {"step": i, "rollout_s": 5.0, "train_s": 5.0, "other_s": 0.0, "total_s": 10.0, "time": "t"}
            for i in range(5)
        ])
        job_b.created_at = "2026-07-28T10:00:00+00:00"
        job_b.updated_at = "2026-07-28T10:02:00+00:00"  # 120 seconds

        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert result["throughput_a"] == 0.1  # 10 steps / 100s
        assert result["throughput_b"] is not None

    def test_throughput_none_when_no_duration(self):
        """throughput should be None when duration cannot be computed."""
        job_a = _make_job("a", step=5, timeperf=[])
        job_b = _make_job("b", step=5, timeperf=[])
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        assert result["throughput_a"] is None
        assert result["throughput_b"] is None

    def test_non_numeric_metric_diff_is_none(self):
        """Metric with non-numeric values should have diff=None, not crash."""
        job_a = _make_job("a", step=1, metrics=[
            {"name": "text_metric", "value": "hello", "step": 1, "time": "t"},
        ])
        job_b = _make_job("b", step=1, metrics=[
            {"name": "text_metric", "value": "world", "step": 1, "time": "t"},
        ])
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("a", "b")

        metrics_by_name = {m["name"]: m for m in result["metrics"]}
        assert "text_metric" in metrics_by_name
        assert metrics_by_name["text_metric"]["comparable"] is True
        assert metrics_by_name["text_metric"]["diff"] is None  # non-numeric, can't subtract

    def test_same_job_returns_all_new_fields(self):
        """Comparing same job should return empty new fields, not KeyError."""
        job = _make_job("solo", step=1)
        state = _state_with_jobs([job])
        result = state.compare_jobs("solo", "solo")

        assert result["comparable"] is False
        # New fields should exist even for non-comparable results.
        assert "metric_charts" not in result  # early return, not populated
        assert "diff_summary" not in result
        # This is acceptable: early return for same-job skips these fields.


class TestCompareSectionsConfig:
    """Test that compare_jobs handles CLI-style sections config (Issue: hyperparameters table empty)."""

    @staticmethod
    def _make_sections_job(job_id: str, settings: dict) -> Job:
        """Simulate a CLI-started job whose config is in sections format."""
        items = [{"key": k, "value": v} for k, v in settings.items()]
        sections_config = {"sections": [{"title": "Basic", "items": items}]}
        job = Job(
            kind="train",
            name=f"train {settings.get('algo', '?')} {settings.get('ckpt', '?')}",
            command=["areno", "train", "--algo", settings.get("algo", "gspo")],
            config=sections_config,
            metrics_dir=None,
        )
        job.id = job_id
        job.launch_config = dict(sections_config)
        job.config = dict(sections_config)
        job.status = "exited"
        job.step = 10
        return job

    def test_sections_config_flattened_for_compare(self):
        """CLI jobs store config as {'sections': [...]}, compare should flatten and find differences."""
        job_a = self._make_sections_job("cli-a", {
            "algo": "gspo", "ckpt": "Qwen/Qwen3-0.6B",
            "optimizer_lr": 1e-06, "batch_size": 8,
        })
        job_b = self._make_sections_job("cli-b", {
            "algo": "gspo", "ckpt": "Qwen/Qwen3-0.6B",
            "optimizer_lr": 5e-06, "batch_size": 16,
        })
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("cli-a", "cli-b")

        config = result["config"]
        # optimizer_lr should be mapped to lr and show up as different
        diff_keys = [d["key"] for d in config["different"]]
        assert "lr" in diff_keys
        assert "batch_size" in diff_keys
        # algo and ckpt are identical
        identical_keys = [d["key"] for d in config["identical"]]
        assert "algo" in identical_keys
        assert "ckpt" in identical_keys

    def test_sections_config_all_identical_shows_in_identical_list(self):
        """When two CLI jobs have identical configs, identical list should be populated."""
        settings = {"algo": "sft", "ckpt": "Qwen/Qwen3-0.6B", "batch_size": 8}
        job_a = self._make_sections_job("cli-c", settings)
        job_b = self._make_sections_job("cli-d", settings)
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("cli-c", "cli-d")

        config = result["config"]
        assert len(config["different"]) == 0
        assert len(config["identical"]) > 0
        identical_keys = [d["key"] for d in config["identical"]]
        assert "algo" in identical_keys
        assert "batch_size" in identical_keys

    def test_flat_and_sections_config_mix(self):
        """A dashboard-started job (flat config) vs CLI-started job (sections) should compare correctly."""
        job_a = _make_job("dash-a", algo="gspo", ckpt="Qwen/Qwen3-0.6B", extra_config={"lr": 1e-06})
        job_b = self._make_sections_job("cli-e", {
            "algo": "gspo", "ckpt": "Qwen/Qwen2-0.5B",
            "optimizer_lr": 1e-06,
        })
        state = _state_with_jobs([job_a, job_b])
        result = state.compare_jobs("dash-a", "cli-e")

        config = result["config"]
        diff_keys = [d["key"] for d in config["different"]]
        assert "ckpt" in diff_keys
        identical_keys = [d["key"] for d in config["identical"]]
        assert "algo" in identical_keys
        assert "lr" in identical_keys  # both have lr=1e-06 after alias mapping