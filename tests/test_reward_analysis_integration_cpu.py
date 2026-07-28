"""Integration test: artifact -> loader -> analyzer -> dashboard route data.

Crosses ``areno/api/dashboard.py`` and the dashboard server's
``DashboardState.reward_component_summary`` using a fake job over a tiny local
fixture, no GPU, no network, no real process.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from areno.dashboard.server import DashboardState, Job


def _make_state_with_job(metrics_dir: str) -> DashboardState:
    state = DashboardState()
    job = Job(
        kind="train",
        name="reward-analysis fixture",
        command=["areno", "train", "--ckpt", "unused"],
        config={},
        metrics_dir=metrics_dir,
    )
    state.jobs[job.id] = job
    return state, job.id


def test_reward_component_summary_flows_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "reward_components.0.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"step": 0, "name": "correctness", "value": 0.8}),
                    json.dumps({"step": 0, "name": "format", "value": 0.2}),
                    json.dumps({"step": 1, "name": "correctness", "value": 0.0}),
                    json.dumps({"step": 1, "name": "format", "value": 0.2}),
                    json.dumps({"step": 2, "name": "correctness", "value": 1.0}),
                    json.dumps({"step": 2, "name": "format", "value": None}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        state, job_id = _make_state_with_job(str(d))
        summary = state.reward_component_summary(job_id)

    assert summary["available"] is True
    by_name = {c["name"]: c for c in summary["components"]}
    assert set(by_name) == {"correctness", "format"}
    # format had an explicit null at step 2 -> missing, not zero.
    assert by_name["format"]["missing_count"] >= 1
    # Drill-down is bounded and ordered.
    assert [s["step"] for s in summary["steps"]] == [0, 1, 2]
    # No sample/prompt text is carried into the route payload.
    assert summary["errors"] == []
    assert "prompt" not in json.dumps(summary).lower()


def test_reward_component_summary_missing_metrics_dir_returns_unavailable():
    state, job_id = _make_state_with_job("")
    summary = state.reward_component_summary(job_id)
    assert summary["available"] is False
    assert summary["components"] == []
    assert summary["steps"] == []


def test_reward_component_summary_no_artifact_file_returns_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        state, job_id = _make_state_with_job(str(Path(tmp)))
        summary = state.reward_component_summary(job_id)
    assert summary["available"] is False
    assert summary["components"] == []


def test_reward_component_summary_does_not_mutate_existing_job_metrics():
    """When the feature has no artifact, existing job.metrics are untouched.

    This guards the default/unenabled path: the read-only route must not alter
    data the existing metrics view depends on.
    """

    with tempfile.TemporaryDirectory() as tmp:
        state, job_id = _make_state_with_job(str(Path(tmp)))
        job = state.jobs[job_id]
        job.metrics = [{"name": "rollout/rewards_mean", "value": 0.5, "step": 0}]
        before = list(job.metrics)
        summary = state.reward_component_summary(job_id)

    assert summary["available"] is False
    assert job.metrics == before, "existing metrics must not be mutated"
