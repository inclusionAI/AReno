"""CPU coverage for graceful-shutdown state exposed by the dashboard."""

from __future__ import annotations

import json
import signal
import tempfile
import threading
import unittest
from pathlib import Path

from areno.dashboard.server import DashboardState, Job, parse_shutdown_event


def _job(*, command: list[str] | None = None, pid: int | None = None) -> Job:
    return Job(
        kind="train",
        name="shutdown-demo",
        command=command or ["areno", "train"],
        config={},
        metrics_dir="metrics",
        pid=pid,
    )


def _state() -> DashboardState:
    state = DashboardState.__new__(DashboardState)
    state.jobs = {}
    state.lock = threading.RLock()
    state._save_state = lambda: None
    return state


class FakeProcess:
    def __init__(self, *, returncode: int | None = None):
        self.returncode = returncode
        self.terminate_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1

    def wait(self):
        return self.returncode


class DashboardShutdownTest(unittest.TestCase):
    def test_job_round_trip_preserves_shutdown(self):
        job = _job()
        job.status = "stopping"
        job.shutdown = {
            "event": "shutdown_requested",
            "state": "shutdown_requested",
            "signal_number": signal.SIGTERM,
            "stage": "training",
            "reason": "Graceful shutdown requested",
            "deadline": 200.0,
        }

        restored = Job.from_json(job.to_json())

        self.assertEqual(restored.shutdown["state"], "shutdown_requested")
        self.assertEqual(restored.shutdown["stage"], "training")
        self.assertIn("deadline_remaining_s", restored.to_json()["shutdown"])

    def test_load_dashboard_state_exposes_shutdown_payload(self):
        state = _state()
        job = _job(pid=123)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "dashboard_state.123.json").write_text(
                json.dumps(
                    {
                        "pid": 123,
                        "stage": "shutdown",
                        "status": "stopping",
                        "shutdown": {
                            "state": "shutdown_requested",
                            "signal_number": signal.SIGINT,
                            "stage": "rollout",
                            "reason": "Graceful shutdown requested",
                            "first_signal": True,
                            "deadline": 200.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            state._load_dashboard_state(job, path)

        self.assertEqual(job.status, "stopping")
        self.assertEqual(job.shutdown["stage"], "rollout")
        self.assertEqual(job.shutdown["event"], "shutdown_requested")

    def test_structured_shutdown_log_updates_job(self):
        state = _state()
        job = _job()
        line = (
            'INFO shutdown_event={"event":"shutdown_forced","state":"forced",'
            '"signal_number":2,"stage":"training","reason":"initial reason",'
            '"timestamp":10.0,"deadline":20.0,"first_signal":false}'
        )

        state._append_log(job, line)

        self.assertEqual(job.shutdown["state"], "forced")
        self.assertEqual(job.shutdown["reason"], "initial reason")
        self.assertEqual(job.status, "stopped")

    def test_malformed_shutdown_log_is_ignored(self):
        self.assertIsNone(parse_shutdown_event("shutdown_event=not-json"))
        self.assertIsNone(parse_shutdown_event('shutdown_event={"state": []}'))

    def test_dashboard_stop_stays_available_for_second_signal(self):
        state = _state()
        job = _job(command=["areno", "train", "--graceful-shutdown"])
        job.status = "running"
        job.process = FakeProcess()
        state.jobs[job.id] = job

        self.assertTrue(state.stop(job.id))

        self.assertEqual(job.status, "stopping")
        self.assertEqual(job.process.terminate_calls, 1)

        self.assertTrue(state.stop(job.id))
        self.assertEqual(job.status, "stopping")
        self.assertEqual(job.process.terminate_calls, 2)

    def test_graceful_process_exit_finishes_as_stopped(self):
        state = _state()
        job = _job(command=["areno", "train", "--graceful-shutdown"])
        job.status = "stopping"
        job.shutdown = {"state": "shutdown_requested"}
        job.process = FakeProcess(returncode=143)
        state._load_metric_files = lambda _job: None

        state._watch(job)

        self.assertEqual(job.status, "stopped")
        self.assertEqual(job.returncode, 143)


if __name__ == "__main__":
    unittest.main()
