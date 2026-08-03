from pathlib import Path
from unittest.mock import patch

from areno.dashboard.server import DashboardState, Job


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
