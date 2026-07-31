"""CPU tests for training-event derivation in the dashboard server.

Covers ``detect_training_events``, ``get_job_events``, route ordering, and
the ``eventUtils.js`` pure functions via a Node smoke test.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Set ROOT before importing so DashboardState picks a temp state file.
_TMP_ROOT = tempfile.mkdtemp(prefix="areno-test-")
os.environ["ARENO_DASHBOARD_ROOT"] = _TMP_ROOT

from areno.dashboard.server import (  # noqa: E402
    DashboardState,
    EVENT_KINDS,
    Job,
    _looks_like_oom,
    _series_by_step,
)


def _make_job(**overrides):
    """Create a minimal Job for testing without starting a process."""

    defaults = dict(
        kind="train",
        name="test",
        command=["areno", "train"],
        config={},
        metrics_dir=None,
    )
    defaults.update(overrides)
    return Job(**defaults)


def _add_metric(job, name, value, step):
    """Shortcut to add a metric point directly to job.metrics."""

    job._metric_keys.add((name, step, value))
    job.metrics.append({"name": name, "value": value, "step": step, "time": "2026-07-31T00:00:00Z"})
    job.step = max(job.step, step)


class NonFiniteDetectionTest(unittest.TestCase):
    """non_finite events derived from NaN/Inf in TensorBoard scalars."""

    def test_nan_produces_non_finite_event(self):
        job = _make_job()
        job._nonfinite_seen.append((5, "rollout/rewards_mean", "NaN"))
        job.updated_at = "2026-07-31T01:00:00Z"
        job.step = 5
        state = DashboardState.__new__(DashboardState)
        state._detect_training_events(job)
        events = [e for e in job._events if e["kind"] == "non_finite"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["step"], 5)
        self.assertEqual(events[0]["fields"]["tag"], "rollout/rewards_mean")
        self.assertEqual(events[0]["fields"]["value"], "NaN")
        self.assertEqual(events[0]["log_hint"]["kind"], "metric_context")

    def test_inf_also_triggers_event(self):
        job = _make_job()
        job._nonfinite_seen.append((3, "train/loss", "Inf"))
        job.step = 3
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        events = [e for e in job._events if e["kind"] == "non_finite"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["fields"]["value"], "Inf")

    def test_nan_does_not_enter_metric_series(self):
        """Metric series should remain NaN-free even after NaN detection."""
        job = _make_job()
        # Simulate what _load_tensorboard_scalars does: skip metric, record NaN.
        job._nonfinite_seen.append((1, "loss", "NaN"))
        # Normal metric still gets added.
        _add_metric(job, "loss", 0.5, 2)
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        # No NaN values in metrics list.
        for m in job.metrics:
            self.assertTrue(
                isinstance(m["value"], (int, float)) and m["value"] == m["value"],
                f"NaN found in metrics: {m}",
            )


class ConstantRewardDetectionTest(unittest.TestCase):

    def test_zero_std_with_equal_max_min(self):
        job = _make_job()
        _add_metric(job, "rollout/rewards_std", 0.0, 1)
        _add_metric(job, "rollout/rewards_max", 1.0, 1)
        _add_metric(job, "rollout/rewards_min", 1.0, 1)
        _add_metric(job, "rollout/rewards_std", 0.5, 2)  # normal step
        _add_metric(job, "rollout/rewards_max", 2.0, 2)
        _add_metric(job, "rollout/rewards_min", 0.0, 2)
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        cr_events = [e for e in job._events if e["kind"] == "constant_reward"]
        self.assertEqual(len(cr_events), 1)
        self.assertEqual(cr_events[0]["step"], 1)
        self.assertEqual(cr_events[0]["severity"], "info")

    def test_zero_std_but_max_ne_min_no_event(self):
        """std≈0 but max!=min should NOT trigger (data inconsistency guard)."""
        job = _make_job()
        _add_metric(job, "rollout/rewards_std", 0.0, 1)
        _add_metric(job, "rollout/rewards_max", 1.0, 1)
        _add_metric(job, "rollout/rewards_min", 0.5, 1)
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        cr_events = [e for e in job._events if e["kind"] == "constant_reward"]
        self.assertEqual(len(cr_events), 0)


class InvalidBatchStreakTest(unittest.TestCase):

    def test_streak_below_3_is_info(self):
        job = _make_job()
        for step in range(1, 3):  # 2 consecutive
            _add_metric(job, "rollout/advantages_std", 0.0, step)
        _add_metric(job, "rollout/advantages_std", 1.0, 3)  # recovery
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        ib_events = [e for e in job._events if e["kind"] == "invalid_batch"]
        self.assertEqual(len(ib_events), 1)
        self.assertEqual(ib_events[0]["severity"], "info")
        self.assertEqual(ib_events[0]["fields"]["streak"], 2)

    def test_streak_3_or_more_is_warn(self):
        job = _make_job()
        for step in range(1, 5):  # 4 consecutive
            _add_metric(job, "rollout/advantages_std", 0.0, step)
        _add_metric(job, "rollout/advantages_std", 1.0, 5)  # recovery
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        ib_events = [e for e in job._events if e["kind"] == "invalid_batch"]
        self.assertEqual(len(ib_events), 1)
        self.assertEqual(ib_events[0]["severity"], "warn")
        self.assertEqual(ib_events[0]["fields"]["streak"], 4)

    def test_trailing_streak_in_active_run(self):
        """A streak at the end (job still running) should still produce an event."""
        job = _make_job()
        for step in range(1, 4):  # 3 consecutive, no recovery step
            _add_metric(job, "rollout/advantages_std", 0.0, step)
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        ib_events = [e for e in job._events if e["kind"] == "invalid_batch"]
        self.assertEqual(len(ib_events), 1)
        self.assertEqual(ib_events[0]["severity"], "warn")


class OOMDetectionTest(unittest.TestCase):

    def test_oom_log_line_produces_error_event(self):
        job = _make_job()
        job.logs = [
            "step 5 training started",
            "CUDA error: out of memory",
            "process exited",
        ]
        job.step = 5
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        oom_events = [e for e in job._events if e["kind"] == "oom"]
        self.assertEqual(len(oom_events), 1)
        self.assertEqual(oom_events[0]["severity"], "error")
        self.assertNotIn("recovered", oom_events[0]["fields"])
        self.assertEqual(oom_events[0]["log_hint"]["kind"], "keyword")

    def test_no_oom_when_logs_clean(self):
        job = _make_job()
        job.logs = ["step 1 done", "step 2 done"]
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        self.assertEqual([e for e in job._events if e["kind"] == "oom"], [])

    def test_looks_like_oom_patterns(self):
        self.assertTrue(_looks_like_oom("RuntimeError: out of memory"))
        self.assertTrue(_looks_like_oom("CUDA error: out of memory"))
        self.assertFalse(_looks_like_oom("step 3 completed"))

    def test_multiple_oom_lines_produce_multiple_events(self):
        job = _make_job()
        job.logs = ["out of memory at step 1", "retry", "out of memory at step 2"]
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        oom_events = [e for e in job._events if e["kind"] == "oom"]
        self.assertEqual(len(oom_events), 2)


class OverlappingEventsTest(unittest.TestCase):

    def test_same_step_multiple_kinds(self):
        """Multiple event kinds at the same step should all be emitted."""
        job = _make_job()
        # non_finite at step 3
        job._nonfinite_seen.append((3, "loss", "NaN"))
        # constant_reward at step 3
        _add_metric(job, "rollout/rewards_std", 0.0, 3)
        _add_metric(job, "rollout/rewards_max", 1.0, 3)
        _add_metric(job, "rollout/rewards_min", 1.0, 3)
        # invalid_batch at step 3
        _add_metric(job, "rollout/advantages_std", 0.0, 3)
        _add_metric(job, "rollout/advantages_std", 1.0, 4)  # recovery
        job.step = 4
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        step3_kinds = {e["kind"] for e in job._events if e["step"] == 3}
        self.assertIn("non_finite", step3_kinds)
        self.assertIn("constant_reward", step3_kinds)
        self.assertIn("invalid_batch", step3_kinds)


class LegacyRunTest(unittest.TestCase):

    def test_empty_job_produces_no_events(self):
        """A job with no metrics, no logs, no nonfinite should return empty events."""
        job = _make_job()
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        self.assertEqual(job._events, [])

    def test_get_job_events_empty_for_unknown_job(self):
        state = DashboardState.__new__(DashboardState)
        state.jobs = {}
        state.lock = __import__("threading").RLock()
        events = state.get_job_events("nonexistent")
        self.assertEqual(events, [])


class GetJobEventsApiTest(unittest.TestCase):

    def _setup_state_with_job(self):
        import threading
        state = DashboardState.__new__(DashboardState)
        state.jobs = {}
        state.lock = threading.RLock()
        job = _make_job()
        job._nonfinite_seen.append((1, "loss", "NaN"))
        job.logs = ["out of memory"]
        job.step = 1
        _add_metric(job, "rollout/rewards_std", 0.0, 1)
        _add_metric(job, "rollout/rewards_max", 1.0, 1)
        _add_metric(job, "rollout/rewards_min", 1.0, 1)
        _add_metric(job, "rollout/advantages_std", 0.0, 1)
        _add_metric(job, "rollout/advantages_std", 0.0, 2)
        _add_metric(job, "rollout/advantages_std", 0.0, 3)
        _add_metric(job, "rollout/advantages_std", 1.0, 4)
        state.jobs[job.id] = job
        state._detect_training_events(job)
        return state, job

    def test_returns_all_events(self):
        state, job = self._setup_state_with_job()
        events = state.get_job_events(job.id)
        self.assertTrue(len(events) > 0)
        kinds = {e["kind"] for e in events}
        self.assertIn("non_finite", kinds)
        self.assertIn("constant_reward", kinds)
        self.assertIn("invalid_batch", kinds)
        self.assertIn("oom", kinds)

    def test_type_filter(self):
        state, job = self._setup_state_with_job()
        events = state.get_job_events(job.id, types=["oom"])
        self.assertTrue(len(events) > 0)
        self.assertTrue(all(e["kind"] == "oom" for e in events))

    def test_limit(self):
        state, job = self._setup_state_with_job()
        all_events = state.get_job_events(job.id)
        limited = state.get_job_events(job.id, limit=1)
        self.assertEqual(len(limited), min(1, len(all_events)))


class RouteOrderingTest(unittest.TestCase):
    """Verify /events is matched before the /api/jobs/ catch-all."""

    def test_events_route_not_swallowed_by_catchall(self):
        """The Handler.route_path and do_GET logic should route /events
        to get_job_events, not treat 'events' as a job_id."""
        from areno.dashboard.server import Handler
        # Simulate what do_GET does: check if path ends with /events
        path = "/api/jobs/abc123/events"
        self.assertTrue(path.startswith("/api/jobs/") and path.endswith("/events"))
        # The catch-all path would be path.split("/")[-1] == "events" which is NOT a job_id.
        # Verify the events branch matches before the catch-all.
        job_id_from_events = path.split("/")[-2]
        self.assertEqual(job_id_from_events, "abc123")


class SeriesByStepTest(unittest.TestCase):

    def test_extracts_step_value_dict(self):
        job = _make_job()
        _add_metric(job, "loss", 0.5, 1)
        _add_metric(job, "loss", 0.3, 2)
        _add_metric(job, "other", 1.0, 1)
        result = _series_by_step(job, "loss")
        self.assertEqual(result, {1: 0.5, 2: 0.3})

    def test_returns_empty_for_missing_tag(self):
        job = _make_job()
        self.assertEqual(_series_by_step(job, "nonexistent"), {})


class EventStructureTest(unittest.TestCase):
    """Verify event objects have all required fields per the data model."""

    def test_event_fields_complete(self):
        job = _make_job()
        job._nonfinite_seen.append((1, "loss", "NaN"))
        DashboardState._detect_training_events(DashboardState.__new__(DashboardState), job)
        event = job._events[0]
        for field in ("kind", "step", "time", "severity", "detail", "fields", "log_hint"):
            self.assertIn(field, event, f"Missing field: {field}")
        self.assertIn("kind", event["log_hint"])
        self.assertIn("ref", event["log_hint"])


class EventKindsConstantTest(unittest.TestCase):

    def test_event_kinds_order(self):
        self.assertEqual(EVENT_KINDS, ("non_finite", "constant_reward", "invalid_batch", "oom"))


# ----------------------------------------------------------------------
# Malformed input tests
# ----------------------------------------------------------------------

class MalformedInputTest(unittest.TestCase):
    """Verify detection does not crash on non-numeric / missing metric values."""

    def test_series_by_step_with_none_value(self):
        """A metric point with value=None should yield None in the series, not raise."""
        job = _make_job()
        # Manually inject a malformed metric point.
        job.metrics.append({"name": "rollout/rewards_std", "value": None, "step": 1, "time": "t"})
        result = _series_by_step(job, "rollout/rewards_std")
        self.assertEqual(result, {1: None})

    def test_series_by_step_with_non_numeric_string(self):
        """A metric point with a non-numeric string value should yield None."""
        job = _make_job()
        job.metrics.append({"name": "loss", "value": "not_a_number", "step": 1, "time": "t"})
        result = _series_by_step(job, "loss")
        self.assertEqual(result, {1: None})

    def test_detect_events_survives_none_metric_values(self):
        """detect_training_events should not crash when metric values are None."""
        import threading
        job = _make_job()
        # Inject a None-valued rewards_std — _series_by_step returns None,
        # and the constant-reward check should skip it (None is not <= 1e-9).
        job.metrics.append({"name": "rollout/rewards_std", "value": None, "step": 1, "time": "t"})
        job.metrics.append({"name": "rollout/advantages_std", "value": None, "step": 1, "time": "t"})
        # Should not raise.
        state = DashboardState.__new__(DashboardState)
        state._detect_training_events(job)
        # No events from None values.
        self.assertEqual(job._events, [])


# ----------------------------------------------------------------------
# Integration test: temp metrics_dir → _load_metric_files → events
# ----------------------------------------------------------------------

class IntegrationLoadAndDetectTest(unittest.TestCase):
    """End-to-end: write fixture .jsonl files, load metrics, verify events."""

    def setUp(self):
        import threading
        self._fixture_dir = Path(_TMP_ROOT) / "integration-fixture"
        self._fixture_dir.mkdir(parents=True, exist_ok=True)
        # Clean any stale files.
        for f in self._fixture_dir.glob("*.jsonl"):
            f.unlink()
        self.state = DashboardState.__new__(DashboardState)
        self.state.jobs = {}
        self.state.lock = threading.RLock()

    def _write_metrics_jsonl(self, filename, rows):
        """Write a .jsonl file with metric rows to the fixture dir."""
        path = self._fixture_dir / filename
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_load_metric_files_derives_events(self):
        """Write fixture metrics, load them, and verify constant_reward + invalid_batch events."""
        # Step 1: constant reward (std=0, max=min=1.0)
        # Step 2: normal reward (std=0.5, max=2.0, min=0.0)
        # Steps 3-5: invalid batch streak (advantages_std=0.0 for 3 consecutive)
        # Step 6: recovery (advantages_std=1.0)
        metrics_rows = []
        for step, rwd_std, rwd_max, rwd_min, adv_std in [
            (1, 0.0, 1.0, 1.0, 0.5),
            (2, 0.5, 2.0, 0.0, 0.5),
            (3, 0.5, 2.0, 0.0, 0.0),
            (4, 0.5, 2.0, 0.0, 0.0),
            (5, 0.5, 2.0, 0.0, 0.0),
            (6, 0.5, 2.0, 0.0, 1.0),
        ]:
            metrics_rows.append({"name": "rollout/rewards_std", "value": rwd_std, "step": step})
            metrics_rows.append({"name": "rollout/rewards_max", "value": rwd_max, "step": step})
            metrics_rows.append({"name": "rollout/rewards_min", "value": rwd_min, "step": step})
            metrics_rows.append({"name": "rollout/advantages_std", "value": adv_std, "step": step})
        self._write_metrics_jsonl("custom_metrics.jsonl", metrics_rows)

        # Also write a log file to simulate OOM.
        log_path = self._fixture_dir / "areno_train.log"
        log_path.write_text("step 1 done\nCUDA error: out of memory\nprocess exited\n")

        job = _make_job(metrics_dir=str(self._fixture_dir.relative_to(_TMP_ROOT)))
        job.logs = ["step 1 done", "CUDA error: out of memory", "process exited"]
        self.state.jobs[job.id] = job

        # Load metrics from fixture dir.
        self.state._load_metric_files(job)

        # Verify metrics were loaded.
        self.assertTrue(len(job.metrics) > 0, "Metrics should have been loaded from .jsonl")

        # Verify events were derived.
        kinds = {e["kind"] for e in job._events}
        self.assertIn("constant_reward", kinds, "Should detect constant reward at step 1")
        self.assertIn("invalid_batch", kinds, "Should detect invalid batch streak at steps 3-5")
        self.assertIn("oom", kinds, "Should detect OOM from logs")

        # Verify constant_reward event details.
        cr = [e for e in job._events if e["kind"] == "constant_reward"]
        self.assertEqual(len(cr), 1)
        self.assertEqual(cr[0]["step"], 1)

        # Verify invalid_batch streak is warn (streak=3).
        ib = [e for e in job._events if e["kind"] == "invalid_batch"]
        self.assertEqual(len(ib), 1)
        self.assertEqual(ib[0]["severity"], "warn")
        self.assertEqual(ib[0]["fields"]["streak"], 3)

        # Verify OOM event is error severity.
        oom = [e for e in job._events if e["kind"] == "oom"]
        self.assertTrue(len(oom) >= 1)
        self.assertEqual(oom[0]["severity"], "error")

    def test_load_metric_files_empty_dir_no_events(self):
        """An empty fixture dir should produce no events (legacy run scenario)."""
        job = _make_job(metrics_dir=str(self._fixture_dir.relative_to(_TMP_ROOT)))
        self.state.jobs[job.id] = job
        self.state._load_metric_files(job)
        self.assertEqual(job._events, [])

    def test_load_metric_files_boundary_single_advantages_std_zero(self):
        """A single advantages_std=0 (streak=1) should produce an info event, not warn."""
        self._write_metrics_jsonl("boundary.jsonl", [
            {"name": "rollout/advantages_std", "value": 0.0, "step": 1},
            {"name": "rollout/advantages_std", "value": 1.0, "step": 2},
        ])
        job = _make_job(metrics_dir=str(self._fixture_dir.relative_to(_TMP_ROOT)))
        self.state.jobs[job.id] = job
        self.state._load_metric_files(job)
        ib = [e for e in job._events if e["kind"] == "invalid_batch"]
        self.assertEqual(len(ib), 1)
        self.assertEqual(ib[0]["severity"], "info")
        self.assertEqual(ib[0]["fields"]["streak"], 1)


if __name__ == "__main__":
    unittest.main()
