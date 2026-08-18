import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from areno.dashboard.server import (
    DashboardState,
    Job,
    agent_language_instruction,
    repair_action_for_check,
    start_runtime_repair,
)


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


@pytest.mark.parametrize(
    ("name", "package", "command_tail"),
    [
        ("flash_attn import", "flash-attn", ["flash-attn", "--no-build-isolation"]),
        ("flash_linear_attention import", "flash-linear-attention", ["flash-linear-attention"]),
    ],
)
def test_missing_runtime_package_has_executable_fix(name: str, package: str, command_tail: list[str]):
    action = repair_action_for_check({"name": name, "status": "WARN", "next_step": f"Install {package}."})

    assert action["label"] == "Fix"
    assert action["kind"] == "install_package"
    assert action["package"] == package
    assert action["command"][:4] == [sys.executable, "-m", "pip", "install"]
    assert action["command"][4:] == command_tail
    assert action["safe_to_run_automatically"] is True


def test_non_package_runtime_warning_has_no_executable_fix():
    action = repair_action_for_check(
        {"name": "NVIDIA GPU visibility", "status": "WARN", "next_step": "Make a GPU visible."}
    )

    assert action["label"] is None
    assert action["kind"] == "guidance"
    assert action["command"] is None
    assert action["safe_to_run_automatically"] is False


def test_runtime_repair_starts_allowlisted_package_job():
    action = repair_action_for_check({"name": "flash_attn import", "status": "WARN"})

    with patch("areno.dashboard.server.STATE.start", side_effect=lambda job: job) as start:
        job = start_runtime_repair(action)

    start.assert_called_once()
    assert job.kind == "runtime-repair"
    assert job.name == "Install flash-attn"
    assert job.command == [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"]


def test_runtime_repair_rejects_untrusted_command():
    action = repair_action_for_check({"name": "flash_attn import", "status": "WARN"})
    action["command"] = ["sh", "-c", "echo unsafe"]

    with pytest.raises(ValueError, match="not allowed"):
        start_runtime_repair(action)
