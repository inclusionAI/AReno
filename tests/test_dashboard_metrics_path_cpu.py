from pathlib import Path
from unittest.mock import patch

from areno.dashboard.server import DashboardState, Job, agent_language_instruction


def test_relative_metrics_dir_uses_registered_job_cwd(tmp_path: Path):
    metrics_path = tmp_path / "run" / "metrics"
    metrics_path.mkdir(parents=True)
    job = Job(
        kind="train",
        name="registered train",
        command=["areno", "train"],
        config={},
        metrics_dir="run/metrics",
        cwd=str(tmp_path),
    )
    state = DashboardState()

    with (
        patch.object(state, "_load_dashboard_state") as load_state,
        patch.object(state, "_load_tensorboard_scalars") as load_scalars,
        patch.object(state, "_load_rollout_samples") as load_samples,
        patch.object(state, "_load_run_config") as load_config,
    ):
        state._load_metric_files(job)

    expected = metrics_path.resolve()
    load_state.assert_called_once_with(job, expected)
    load_scalars.assert_called_once_with(job, expected)
    load_samples.assert_called_once_with(job, expected)
    load_config.assert_called_once_with(job, expected)


def test_job_cwd_round_trips_through_dashboard_state(tmp_path: Path):
    job = Job(
        kind="train",
        name="registered train",
        command=["areno", "train"],
        config={},
        metrics_dir="metrics",
        cwd=str(tmp_path),
    )

    restored = Job.from_json(job.to_json())

    assert restored.cwd == str(tmp_path)


def test_metric_updates_latest_job_perf_signal():
    job = Job(
        kind="train",
        name="metric train",
        command=["areno", "train"],
        config={"algo": "gspo"},
        metrics_dir=None,
    )
    state = DashboardState()

    state._add_metric(job, "rollout/rewards_mean", 0.25, 1)
    state._add_metric(job, "rollout/rewards_mean", 0.75, 2)

    assert job.perf["rollout/rewards_mean"] == 0.75
    assert job.step == 2


def test_agent_language_instruction_follows_dashboard_language():
    assert "Simplified Chinese" in agent_language_instruction({"language": "zh"})
    assert "commands" in agent_language_instruction({"language": "zh"})
    assert "English" in agent_language_instruction({"language": "en"})
