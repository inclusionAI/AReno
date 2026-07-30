"""Equivalence guard: dashboard read-side output before vs after the metric_reader extract (issue #254).

Locks the current output of ``DashboardState._load_tensorboard_scalars``
(``job.metrics`` + ``job.timeperf``) against a real
``events.out.tfevents.*`` fixture written via ``SummaryWriter``. After the
extract switches ``_load_tensorboard_scalars`` to call
``areno.api.metric_reader``, this test must keep passing unchanged -- it is
the regression guard that proves the refactor is behavior-preserving.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from areno.dashboard.server import DashboardState, Job, ROOT


def _write_tfevents_fixture(metrics_dir: Path) -> None:
    """Write a small set of real TensorBoard event files.

    Tags cover every branch of ``_load_tensorboard_scalars``: normal scalars,
    NaN-skip, single-point series, and the timeperf ``time/`` +
    ``train/step_*_time_s`` tags that drive the by_step aggregation. Values
    use powers-of-two fractions to survive TensorBoard's float32 round-trip.
    """
    from torch.utils.tensorboard import SummaryWriter

    metrics_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(metrics_dir))
    try:
        for step in range(5):
            writer.add_scalar("rollout/rewards_mean", 0.5 + 0.125 * step, step)
        writer.add_scalar("rollout/loss_with_nan", float("nan"), 0)
        writer.add_scalar("rollout/loss_with_nan", 1.5, 1)
        writer.add_scalar("rollout/loss_with_nan", float("nan"), 2)
        writer.add_scalar("rollout/loss_with_nan", 2.5, 3)
        writer.add_scalar("rollout/single_point", 9.0, 7)
        for step in range(3):
            writer.add_scalar("time/rollout", 2.0 + step, step)
            writer.add_scalar("time/train", 1.0 + step, step)
            writer.add_scalar("train/step_e2e_time_s", 3.0 + step, step)
    finally:
        writer.close()


def _load_job_metrics(metrics_dir: str) -> Job:
    """Drive the dashboard read path exactly as the server does."""
    state = DashboardState()
    job = Job(kind="train", name="probe", command=[], config={}, metrics_dir=metrics_dir)
    state.jobs[job.id] = job
    path = (ROOT / metrics_dir).resolve()
    state._load_tensorboard_scalars(job, path)
    return job


def _strip_nondeterministic(points: list[dict]) -> list[dict]:
    """Drop the ``time`` field, which is stamped with wall-clock at read time."""
    return [
        {"name": p.get("name"), "value": p.get("value"), "step": int(p.get("step") or 0)}
        for p in points
    ]


def _strip_timeperf_nondeterministic(rows: list[dict]) -> list[dict]:
    """Drop ``time`` from timeperf rows; keep the deterministic aggregation."""
    return [{k: v for k, v in dict(row).items() if k != "time"} for row in rows]


class MetricReaderEquivalenceTest(unittest.TestCase):
    def test_locks_metric_points_and_timeperf(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics_dir = Path(tmp)
            _write_tfevents_fixture(metrics_dir)
            job = _load_job_metrics(str(metrics_dir))

            metrics = _strip_nondeterministic(job.metrics)
            timeperf = _strip_timeperf_nondeterministic(job.timeperf)

            self.assertEqual(
                metrics,
                [
                    {"name": "rollout/rewards_mean", "value": 0.5, "step": 0},
                    {"name": "rollout/rewards_mean", "value": 0.625, "step": 1},
                    {"name": "rollout/rewards_mean", "value": 0.75, "step": 2},
                    {"name": "rollout/rewards_mean", "value": 0.875, "step": 3},
                    {"name": "rollout/rewards_mean", "value": 1.0, "step": 4},
                    {"name": "rollout/loss_with_nan", "value": 1.5, "step": 1},
                    {"name": "rollout/loss_with_nan", "value": 2.5, "step": 3},
                    {"name": "rollout/single_point", "value": 9.0, "step": 7},
                    {"name": "time/rollout", "value": 2.0, "step": 0},
                    {"name": "time/rollout", "value": 3.0, "step": 1},
                    {"name": "time/rollout", "value": 4.0, "step": 2},
                    {"name": "time/train", "value": 1.0, "step": 0},
                    {"name": "time/train", "value": 2.0, "step": 1},
                    {"name": "time/train", "value": 3.0, "step": 2},
                    {"name": "train/step_e2e_time_s", "value": 3.0, "step": 0},
                    {"name": "train/step_e2e_time_s", "value": 4.0, "step": 1},
                    {"name": "train/step_e2e_time_s", "value": 5.0, "step": 2},
                ],
            )

            self.assertEqual(
                timeperf,
                [
                    {
                        "step": 0,
                        "segments": [
                            {"name": "rollout", "seconds": 2.0},
                            {"name": "train", "seconds": 1.0},
                        ],
                        "rollout_s": 2.0,
                        "train_s": 1.0,
                        "other_s": 0.0,
                        "total_s": 3.0,
                        "source": "metrics",
                    },
                    {
                        "step": 1,
                        "segments": [
                            {"name": "rollout", "seconds": 3.0},
                            {"name": "train", "seconds": 2.0},
                        ],
                        "rollout_s": 3.0,
                        "train_s": 2.0,
                        "other_s": 0.0,
                        "total_s": 4.0,
                        "source": "metrics",
                    },
                    {
                        "step": 2,
                        "segments": [
                            {"name": "rollout", "seconds": 4.0},
                            {"name": "train", "seconds": 3.0},
                        ],
                        "rollout_s": 4.0,
                        "train_s": 3.0,
                        "other_s": 0.0,
                        "total_s": 5.0,
                        "source": "metrics",
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()