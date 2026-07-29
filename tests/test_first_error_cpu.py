"""CPU tests for the ``preserve_first_error`` feature (Issue #245).

These tests exercise the error-aggregation logic in ``protocol.py`` without
spawning real worker processes.  They follow the same ``object.__new__``
+ ``FakeQueue``/``FakeProcess`` pattern established by
``test_protocol_cpu.py``.

Coverage matrix:
  * Default mode (preserve=False): immediate terminate, dead worker, async
  * Preserve mode: all-fail, mixed, single-rank, all-success, dead-as-first,
    dead-as-secondary, all-dead, multiple-dead-simultaneous, async compat
  * Output format: basic, no-secondary, truncation boundaries (5/6), empty
  * _extract_error_summary: normal, empty, whitespace-only
  * Exception semantics: catch as RuntimeError, attribute access
  * Timestamp ordering
  * Config: RuntimeConfig, TrainerConfig passthrough
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Load protocol module in isolation (same technique as test_protocol_cpu.py).
# ---------------------------------------------------------------------------


def _load_protocol_module():
    """Load protocol.py without importing areno.engine package side effects."""

    path = Path(__file__).resolve().parents[1] / "areno" / "engine" / "protocol.py"
    spec = importlib.util.spec_from_file_location("_areno_protocol_first_error_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol_module()
TPCluster = protocol.TPCluster
Op = protocol.Op
WorkerResult = protocol.WorkerResult
_WorkerError = protocol._WorkerError
FirstWorkerError = protocol.FirstWorkerError
_extract_error_summary = protocol._extract_error_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cluster(tp_size: int = 1, dp_size: int = 4, *, preserve: bool = False):
    """Create a TPCluster bypassing ``__init__`` (no real processes)."""

    cluster = object.__new__(TPCluster)
    cluster.config = SimpleNamespace(tp_size=tp_size, dp_size=dp_size)
    cluster.started = True
    cluster._preserve_first_error = preserve
    cluster._pending_lock = threading.Lock()
    cluster._send_lock = threading.Lock()
    cluster._pending_calls = {}
    cluster.cmd_queues = []
    cluster.processes = []
    return cluster


class _FakeProcess:
    """Minimal process double with a settable exitcode."""

    def __init__(self, exitcode=None, pid=9999):
        self.exitcode = exitcode
        self.pid = pid
        self._join_calls = []

    def join(self, timeout=None):
        self._join_calls.append(timeout)


_TRACEBACK_TEMPLATE = (
    "Traceback (most recent call last):\n"
    '  File "/fake/worker.py", line {line}, in handle\n'
    '    raise RuntimeError("{msg}")\n'
    "RuntimeError: {msg}"
)


def _fake_traceback(rank: int, msg: str = "business error") -> str:
    return _TRACEBACK_TEMPLATE.format(line=100 + rank, msg=msg)


# ---------------------------------------------------------------------------
# Default mode tests (preserve_first_error=False)
# ---------------------------------------------------------------------------


class DefaultBehaviorTest(unittest.TestCase):
    """When preserve_first_error is False (default), behavior is unchanged."""

    def test_default_first_error_immediate_terminate(self):
        """First failure immediately terminates the pending call (legacy mode)."""

        cluster = _make_cluster(dp_size=4, preserve=False)
        pending = cluster._submit_call(Op.TRAIN, request_id=1)
        self.assertEqual(len(pending.pending), 4)

        cluster._apply_result(1, 1, WorkerResult(ok=False, error=_fake_traceback(1)), pending)

        self.assertTrue(pending.event.is_set())
        self.assertIsInstance(pending.error, RuntimeError)
        self.assertNotIsInstance(pending.error, FirstWorkerError)
        self.assertIsNone(pending.first_error)

    def test_default_error_message_contains_rank_and_traceback(self):
        """Legacy RuntimeError message includes rank, op, and traceback."""

        cluster = _make_cluster(dp_size=2, preserve=False)
        pending = cluster._submit_call(Op.TRAIN, request_id=2)
        tb = _fake_traceback(0, "OOM")
        cluster._apply_result(2, 0, WorkerResult(ok=False, error=tb), pending)

        msg = str(pending.error)
        self.assertIn("rank 0", msg)
        self.assertIn("Op.TRAIN", msg)
        self.assertIn("OOM", msg)

    def test_default_dead_worker_uses_runtime_error(self):
        """Dead worker in default mode raises plain RuntimeError, not FirstWorkerError."""

        cluster = _make_cluster(dp_size=2, preserve=False)
        cluster.processes = [_FakeProcess(exitcode=None), _FakeProcess(exitcode=None)]
        pending = cluster._submit_call(Op.TRAIN, request_id=3)

        cluster.processes[0].exitcode = 1
        cluster._fail_dead_pending_calls()

        self.assertTrue(pending.event.is_set())
        self.assertIsInstance(pending.error, RuntimeError)
        self.assertNotIsInstance(pending.error, FirstWorkerError)

    def test_default_all_success_no_error(self):
        """All ranks succeed in default mode: no error, results returned."""

        cluster = _make_cluster(dp_size=2, preserve=False)
        pending = cluster._submit_call(Op.TRAIN, request_id=4)

        cluster._apply_result(4, 0, WorkerResult(ok=True, payload="r0"), pending)
        self.assertFalse(pending.event.is_set())
        cluster._apply_result(4, 1, WorkerResult(ok=True, payload="r1"), pending)

        self.assertTrue(pending.event.is_set())
        self.assertIsNone(pending.error)
        self.assertEqual(pending.results[0], "r0")
        self.assertEqual(pending.results[1], "r1")

    def test_default_async_failure_propagates(self):
        """Async call with a failing rank in default mode propagates error immediately."""

        cluster = _make_cluster(tp_size=2, dp_size=2, preserve=False)
        pending = cluster._submit_call(Op.INFER_ROLLOUT, request_id=5, result_ranks={0, 2})

        cluster._apply_result(5, 0, WorkerResult(ok=False, error=_fake_traceback(0)), pending)

        self.assertTrue(pending.event.is_set())
        self.assertIsInstance(pending.error, RuntimeError)
        self.assertNotIsInstance(pending.error, FirstWorkerError)


# ---------------------------------------------------------------------------
# Preserve mode tests — core logic
# ---------------------------------------------------------------------------


class PreserveFirstErrorTest(unittest.TestCase):
    """When preserve_first_error is True, errors are collected before raising."""

    def test_preserve_all_failures_collected(self):
        """All rank failures are collected; first error is preserved as root cause."""

        cluster = _make_cluster(dp_size=4, preserve=True)
        pending = cluster._submit_call(Op.TRAIN, request_id=10)

        cluster._apply_result(10, 1, WorkerResult(ok=False, error=_fake_traceback(1, "OOM")), pending)
        self.assertFalse(pending.event.is_set())
        self.assertIsNotNone(pending.first_error)
        self.assertEqual(pending.first_error.rank, 1)

        cluster._apply_result(10, 0, WorkerResult(ok=False, error=_fake_traceback(0, "NCCL timeout")), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(10, 2, WorkerResult(ok=False, error=_fake_traceback(2, "NCCL timeout")), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(10, 3, WorkerResult(ok=False, error=_fake_traceback(3, "watchdog")), pending)

        self.assertTrue(pending.event.is_set())
        self.assertIsInstance(pending.error, FirstWorkerError)
        self.assertEqual(pending.error.first_error.rank, 1)
        self.assertEqual(len(pending.error.secondary_errors), 3)
        secondary_ranks = {e.rank for e in pending.error.secondary_errors}
        self.assertEqual(secondary_ranks, {0, 2, 3})

    def test_preserve_mixed_success_and_failure(self):
        """Some ranks succeed while one fails; successful payloads are preserved."""

        cluster = _make_cluster(dp_size=4, preserve=True)
        pending = cluster._submit_call(Op.TRAIN, request_id=20)

        cluster._apply_result(20, 0, WorkerResult(ok=True, payload={"loss": 0.5}), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(20, 1, WorkerResult(ok=False, error=_fake_traceback(1)), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(20, 2, WorkerResult(ok=True, payload={"loss": 0.3}), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(20, 3, WorkerResult(ok=True, payload={"loss": 0.4}), pending)
        self.assertTrue(pending.event.is_set())

        self.assertIsInstance(pending.error, FirstWorkerError)
        self.assertEqual(pending.error.first_error.rank, 1)
        self.assertEqual(len(pending.error.secondary_errors), 0)
        self.assertEqual(pending.results[0], {"loss": 0.5})
        self.assertEqual(pending.results[2], {"loss": 0.3})
        self.assertEqual(pending.results[3], {"loss": 0.4})

    def test_preserve_single_rank_cluster(self):
        """Single-rank cluster (world_size=1) failure completes immediately."""

        cluster = _make_cluster(dp_size=1, preserve=True)
        pending = cluster._submit_call(Op.TRAIN, request_id=25)

        cluster._apply_result(25, 0, WorkerResult(ok=False, error=_fake_traceback(0, "OOM")), pending)

        self.assertTrue(pending.event.is_set())
        self.assertIsInstance(pending.error, FirstWorkerError)
        self.assertEqual(pending.error.first_error.rank, 0)
        self.assertEqual(len(pending.error.secondary_errors), 0)

    def test_preserve_all_success_no_error(self):
        """All ranks succeed in preserve mode: no error, no first_error."""

        cluster = _make_cluster(dp_size=3, preserve=True)
        pending = cluster._submit_call(Op.TRAIN, request_id=26)

        for rank in range(3):
            cluster._apply_result(26, rank, WorkerResult(ok=True, payload=f"r{rank}"), pending)

        self.assertTrue(pending.event.is_set())
        self.assertIsNone(pending.error)
        self.assertIsNone(pending.first_error)

    def test_preserve_includes_timestamp_and_stage(self):
        """_WorkerError records rank, op (stage), and a float timestamp."""

        cluster = _make_cluster(dp_size=2, preserve=True)
        pending = cluster._submit_call(Op.SCORE_LOGPROBS, request_id=30)

        before = time.monotonic()
        cluster._apply_result(30, 0, WorkerResult(ok=False, error=_fake_traceback(0)), pending)
        after = time.monotonic()

        self.assertIsNotNone(pending.first_error)
        self.assertEqual(pending.first_error.rank, 0)
        self.assertEqual(pending.first_error.op, Op.SCORE_LOGPROBS)
        self.assertGreaterEqual(pending.first_error.timestamp, before)
        self.assertLessEqual(pending.first_error.timestamp, after)

    def test_preserve_timestamp_ordering(self):
        """First error timestamp is strictly earlier than secondary."""

        cluster = _make_cluster(dp_size=2, preserve=True)
        pending = cluster._submit_call(Op.TRAIN, request_id=31)

        cluster._apply_result(31, 0, WorkerResult(ok=False, error=_fake_traceback(0, "first")), pending)
        time.sleep(0.001)  # Ensure monotonic clock advances.
        cluster._apply_result(31, 1, WorkerResult(ok=False, error=_fake_traceback(1, "second")), pending)

        self.assertTrue(pending.event.is_set())
        self.assertLess(pending.error.first_error.timestamp, pending.error.secondary_errors[0].timestamp)

    def test_preserve_first_error_content_preserved(self):
        """The first error's traceback string is preserved verbatim in the exception."""

        cluster = _make_cluster(dp_size=2, preserve=True)
        pending = cluster._submit_call(Op.TRAIN, request_id=32)
        tb = _fake_traceback(0, "unique OOM message")
        cluster._apply_result(32, 0, WorkerResult(ok=False, error=tb), pending)
        cluster._apply_result(32, 1, WorkerResult(ok=False, error=_fake_traceback(1, "NCCL")), pending)

        self.assertEqual(pending.error.first_error.error, tb)
        self.assertIn("unique OOM message", str(pending.error))


# ---------------------------------------------------------------------------
# Preserve mode — dead worker scenarios
# ---------------------------------------------------------------------------


class PreserveDeadWorkerTest(unittest.TestCase):
    """Dead worker (process exit) interaction with preserve mode."""

    def test_dead_worker_as_secondary(self):
        """Dead workers after a first error are secondary errors."""

        cluster = _make_cluster(dp_size=3, preserve=True)
        cluster.processes = [_FakeProcess(exitcode=None) for _ in range(3)]
        pending = cluster._submit_call(Op.TRAIN, request_id=40)

        cluster._apply_result(40, 0, WorkerResult(ok=False, error=_fake_traceback(0, "OOM")), pending)
        self.assertFalse(pending.event.is_set())

        cluster.processes[1].exitcode = 1
        cluster._fail_dead_pending_calls()
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(40, 2, WorkerResult(ok=False, error=_fake_traceback(2, "NCCL abort")), pending)
        self.assertTrue(pending.event.is_set())

        self.assertIsInstance(pending.error, FirstWorkerError)
        self.assertEqual(pending.error.first_error.rank, 0)
        secondary_ranks = {e.rank for e in pending.error.secondary_errors}
        self.assertIn(1, secondary_ranks)
        self.assertIn(2, secondary_ranks)

    def test_dead_worker_as_first_error(self):
        """When no worker has reported yet and a worker dies, it becomes the first error."""

        cluster = _make_cluster(dp_size=3, preserve=True)
        cluster.processes = [_FakeProcess(exitcode=None) for _ in range(3)]
        pending = cluster._submit_call(Op.TRAIN, request_id=45)

        cluster.processes[1].exitcode = 1
        cluster._fail_dead_pending_calls()

        self.assertIsNotNone(pending.first_error)
        self.assertEqual(pending.first_error.rank, 1)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(45, 0, WorkerResult(ok=False, error=_fake_traceback(0, "NCCL abort")), pending)
        cluster._apply_result(45, 2, WorkerResult(ok=False, error=_fake_traceback(2, "NCCL abort")), pending)
        self.assertTrue(pending.event.is_set())

        self.assertIsInstance(pending.error, FirstWorkerError)
        self.assertEqual(pending.error.first_error.rank, 1)
        self.assertEqual(len(pending.error.secondary_errors), 2)

    def test_all_workers_dead_completes_call(self):
        """All workers dying completes the pending call with FirstWorkerError."""

        cluster = _make_cluster(dp_size=3, preserve=True)
        cluster.processes = [_FakeProcess(exitcode=None) for _ in range(3)]
        pending = cluster._submit_call(Op.TRAIN, request_id=50)

        for i in range(3):
            cluster.processes[i].exitcode = 137
        cluster._fail_dead_pending_calls()

        self.assertTrue(pending.event.is_set())
        self.assertIsInstance(pending.error, FirstWorkerError)
        self.assertIsNotNone(pending.error.first_error)
        self.assertEqual(len(pending.error.secondary_errors), 2)

    def test_multiple_dead_workers_simultaneously(self):
        """Multiple workers dying at once: first becomes first_error, rest secondary."""

        cluster = _make_cluster(dp_size=4, preserve=True)
        cluster.processes = [_FakeProcess(exitcode=None) for _ in range(4)]
        pending = cluster._submit_call(Op.TRAIN, request_id=55)

        # Ranks 2 and 3 die simultaneously.
        cluster.processes[2].exitcode = 1
        cluster.processes[3].exitcode = 1
        cluster._fail_dead_pending_calls()

        self.assertFalse(pending.event.is_set())  # Ranks 0 and 1 still alive.
        self.assertIsNotNone(pending.first_error)
        # First dead rank should be first_error (rank 2, iterated before 3).
        self.assertEqual(pending.first_error.rank, 2)
        self.assertEqual(len(pending.secondary_errors), 1)
        self.assertEqual(pending.secondary_errors[0].rank, 3)

    def test_dead_worker_then_reported_error(self):
        """Worker dies, then another worker reports error: dead is first, report is secondary."""

        cluster = _make_cluster(dp_size=2, preserve=True)
        cluster.processes = [_FakeProcess(exitcode=None), _FakeProcess(exitcode=None)]
        pending = cluster._submit_call(Op.TRAIN, request_id=60)

        cluster.processes[0].exitcode = 1
        cluster._fail_dead_pending_calls()

        self.assertFalse(pending.event.is_set())
        self.assertEqual(pending.first_error.rank, 0)

        cluster._apply_result(60, 1, WorkerResult(ok=False, error=_fake_traceback(1, "NCCL")), pending)
        self.assertTrue(pending.event.is_set())

        self.assertEqual(pending.error.first_error.rank, 0)
        self.assertEqual(len(pending.error.secondary_errors), 1)


# ---------------------------------------------------------------------------
# Preserve mode — async compatibility
# ---------------------------------------------------------------------------


class PreserveAsyncTest(unittest.TestCase):
    """Async call paths work correctly with preserve mode."""

    def test_async_result_ranks_all_success(self):
        """Async call with result_ranks subset: all succeed, no error."""

        cluster = _make_cluster(tp_size=2, dp_size=2, preserve=True)
        pending = cluster._submit_call(Op.INFER_ROLLOUT, request_id=70, result_ranks={0, 2})

        cluster._apply_result(70, 1, WorkerResult(ok=True, payload="tp-sibling"), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(70, 0, WorkerResult(ok=True, payload="dp0"), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(70, 2, WorkerResult(ok=True, payload="dp1"), pending)
        self.assertTrue(pending.event.is_set())
        self.assertIsNone(pending.error)

    def test_async_result_ranks_failure_collected(self):
        """Async call with result_ranks subset: failures in waited ranks are collected."""

        cluster = _make_cluster(tp_size=2, dp_size=2, preserve=True)
        pending = cluster._submit_call(Op.INFER_ROLLOUT, request_id=71, result_ranks={0, 2})

        # TP sibling succeeds (not in result_ranks, doesn't affect completion).
        cluster._apply_result(71, 1, WorkerResult(ok=True, payload="tp-sibling"), pending)
        self.assertFalse(pending.event.is_set())

        # First waited rank fails.
        cluster._apply_result(71, 0, WorkerResult(ok=False, error=_fake_traceback(0)), pending)
        self.assertFalse(pending.event.is_set())

        # Second waited rank fails.
        cluster._apply_result(71, 2, WorkerResult(ok=False, error=_fake_traceback(2)), pending)
        self.assertTrue(pending.event.is_set())

        self.assertIsInstance(pending.error, FirstWorkerError)
        self.assertEqual(pending.error.first_error.rank, 0)
        self.assertEqual(len(pending.error.secondary_errors), 1)


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------


class OutputFormatTest(unittest.TestCase):
    """FirstWorkerError.__str__ output format."""

    def test_format_with_secondary_errors(self):
        """Output shows root cause prominently and secondary as summary."""

        first = _WorkerError(rank=2, op=Op.TRAIN, error=_fake_traceback(2, "CUDA OOM"), timestamp=1234567.89)
        secondary = [
            _WorkerError(rank=0, op=Op.TRAIN, error=_fake_traceback(0, "NCCL abort"), timestamp=1234570.0),
            _WorkerError(rank=1, op=Op.TRAIN, error=_fake_traceback(1, "NCCL abort"), timestamp=1234570.5),
        ]
        err = FirstWorkerError(first, secondary)
        text = str(err)

        self.assertIn("=== First Worker Failure (root cause) ===", text)
        self.assertIn("rank=2", text)
        self.assertIn("stage=TRAIN", text)
        self.assertIn("timestamp=1234567.890000", text)
        self.assertIn("CUDA OOM", text)
        self.assertIn("=== Secondary Errors (2 ranks, shown as summary) ===", text)
        self.assertIn("[rank=0", text)
        self.assertIn("[rank=1", text)
        self.assertEqual(text.count("Traceback"), 1)

    def test_format_without_secondary_errors(self):
        """Output with no secondary errors shows only the first failure section."""

        first = _WorkerError(rank=0, op=Op.SCORE_VALUES, error="RuntimeError: boom", timestamp=42.0)
        err = FirstWorkerError(first, [])
        text = str(err)

        self.assertIn("=== First Worker Failure (root cause) ===", text)
        self.assertIn("rank=0", text)
        self.assertIn("stage=SCORE_VALUES", text)
        self.assertNotIn("Secondary Errors", text)

    def test_format_secondary_truncated_at_five(self):
        """Exactly 5 secondary errors: no truncation message."""

        first = _WorkerError(rank=0, op=Op.TRAIN, error="RuntimeError: boom", timestamp=1.0)
        secondary = [
            _WorkerError(rank=i + 1, op=Op.TRAIN, error=f"RuntimeError: err {i}", timestamp=float(i)) for i in range(5)
        ]
        err = FirstWorkerError(first, secondary)
        text = str(err)

        self.assertIn("=== Secondary Errors (5 ranks, shown as summary) ===", text)
        self.assertNotIn("omitted", text)

    def test_format_secondary_truncated_at_six(self):
        """Exactly 6 secondary errors: truncation message shows 1 omitted."""

        first = _WorkerError(rank=0, op=Op.TRAIN, error="RuntimeError: boom", timestamp=1.0)
        secondary = [
            _WorkerError(rank=i + 1, op=Op.TRAIN, error=f"RuntimeError: err {i}", timestamp=float(i)) for i in range(6)
        ]
        err = FirstWorkerError(first, secondary)
        text = str(err)

        self.assertIn("=== Secondary Errors (6 ranks, shown as summary) ===", text)
        self.assertIn("1 more secondary error(s) omitted", text)
        # Rank 6 should NOT appear in the shown lines.
        self.assertNotIn("rank=6", text)

    def test_format_secondary_truncated_at_eight(self):
        """8 secondary errors: show first 5, note 3 omitted."""

        first = _WorkerError(rank=0, op=Op.TRAIN, error="RuntimeError: boom", timestamp=1.0)
        secondary = [
            _WorkerError(rank=i + 1, op=Op.TRAIN, error=f"RuntimeError: err {i}", timestamp=float(i)) for i in range(8)
        ]
        err = FirstWorkerError(first, secondary)
        text = str(err)

        self.assertIn("=== Secondary Errors (8 ranks, shown as summary) ===", text)
        for i in range(5):
            self.assertIn(f"rank={i + 1}", text)
        self.assertIn("3 more secondary error(s) omitted", text)


# ---------------------------------------------------------------------------
# _extract_error_summary tests
# ---------------------------------------------------------------------------


class ExtractErrorSummaryTest(unittest.TestCase):
    """_extract_error_summary extracts the last meaningful line."""

    def test_normal_traceback(self):
        """Returns the last line (exception class + message) of a normal traceback."""

        tb = "Traceback (most recent call last):\n  File ...\nRuntimeError: OOM"
        self.assertEqual(_extract_error_summary(tb), "RuntimeError: OOM")

    def test_empty_string(self):
        """Empty string returns the fallback message."""

        self.assertEqual(_extract_error_summary(""), "<empty traceback>")

    def test_whitespace_only(self):
        """Whitespace-only string returns the fallback message."""

        self.assertEqual(_extract_error_summary("  \n  \t  \n"), "<empty traceback>")

    def test_single_line(self):
        """Single-line error returns that line."""

        self.assertEqual(_extract_error_summary("ValueError: bad input"), "ValueError: bad input")

    def test_trailing_newlines(self):
        """Trailing newlines are handled correctly."""

        self.assertEqual(_extract_error_summary("RuntimeError: boom\n\n\n"), "RuntimeError: boom")


# ---------------------------------------------------------------------------
# Exception semantics tests
# ---------------------------------------------------------------------------


class ExceptionSemanticsTest(unittest.TestCase):
    """FirstWorkerError behaves correctly as a RuntimeError subclass."""

    def test_catchable_as_runtime_error(self):
        """FirstWorkerError can be caught by 'except RuntimeError'."""

        first = _WorkerError(rank=0, op=Op.TRAIN, error="boom", timestamp=1.0)
        err = FirstWorkerError(first, [])

        try:
            raise err
        except RuntimeError as caught:
            self.assertIs(caught, err)

    def test_attributes_accessible(self):
        """first_error and secondary_errors attributes are accessible after catch."""

        first = _WorkerError(rank=2, op=Op.TRAIN, error="OOM", timestamp=1.0)
        secondary = [_WorkerError(rank=0, op=Op.TRAIN, error="NCCL", timestamp=2.0)]
        err = FirstWorkerError(first, secondary)

        try:
            raise err
        except FirstWorkerError as caught:
            self.assertEqual(caught.first_error.rank, 2)
            self.assertEqual(len(caught.secondary_errors), 1)
            self.assertEqual(caught.secondary_errors[0].rank, 0)

    def test_str_matches_args(self):
        """super().__init__(str(self)) makes args[0] match __str__."""

        first = _WorkerError(rank=0, op=Op.TRAIN, error="boom", timestamp=1.0)
        err = FirstWorkerError(first, [])
        self.assertEqual(str(err), err.args[0])


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class RuntimeConfigTest(unittest.TestCase):
    """RuntimeConfig accepts the new field."""

    def test_default_is_false(self):
        from areno.engine.config import RuntimeConfig

        cfg = RuntimeConfig()
        self.assertFalse(cfg.preserve_first_error)

    def test_explicit_true(self):
        from areno.engine.config import RuntimeConfig

        cfg = RuntimeConfig(preserve_first_error=True)
        self.assertTrue(cfg.preserve_first_error)

    def test_does_not_break_existing_validation(self):
        """Existing RuntimeConfig validation still works with the new field."""

        from areno.engine.config import RuntimeConfig

        # Valid config with preserve_first_error.
        cfg = RuntimeConfig(preserve_first_error=True, attn_backend="native")
        self.assertTrue(cfg.preserve_first_error)
        self.assertEqual(cfg.attn_backend, "native")

        # Invalid attn_backend still raises.
        with self.assertRaises(ValueError):
            RuntimeConfig(attn_backend="invalid")


class TrainerConfigTest(unittest.TestCase):
    """TrainerConfig passes preserve_first_error through to areno_config()."""

    def test_trainer_config_default_false(self):
        from areno.api.trainer_config import TrainerConfig

        cfg = TrainerConfig(algo="sft", ckpt="/fake", dataset_path="/fake")
        self.assertFalse(cfg.preserve_first_error)

    def test_trainer_config_explicit_true(self):
        from areno.api.trainer_config import TrainerConfig

        cfg = TrainerConfig(algo="sft", ckpt="/fake", dataset_path="/fake", preserve_first_error=True)
        self.assertTrue(cfg.preserve_first_error)

    def test_trainer_config_areno_config_passthrough(self):
        """TrainerConfig.areno_config() includes preserve_first_error in runtime dict."""

        from areno.api.trainer_config import TrainerConfig

        cfg = TrainerConfig(algo="sft", ckpt="/fake", dataset_path="/fake", preserve_first_error=True)
        areno_cfg = cfg.areno_config()
        self.assertTrue(areno_cfg.runtime["preserve_first_error"])

    def test_rollout_trainer_config_areno_config_passthrough(self):
        """RolloutTrainerConfig.areno_config() includes preserve_first_error."""

        from areno.api.trainer_config import RolloutTrainerConfig

        cfg = RolloutTrainerConfig(algo="grpo", ckpt="/fake", dataset_path="/fake", preserve_first_error=True)
        areno_cfg = cfg.areno_config()
        self.assertTrue(areno_cfg.runtime["preserve_first_error"])

    def test_trainer_config_default_passthrough_false(self):
        """Default TrainerConfig passes False through to areno_config()."""

        from areno.api.trainer_config import TrainerConfig

        cfg = TrainerConfig(algo="sft", ckpt="/fake", dataset_path="/fake")
        areno_cfg = cfg.areno_config()
        self.assertFalse(areno_cfg.runtime["preserve_first_error"])


# ---------------------------------------------------------------------------
# ArenoConfig integration test
# ---------------------------------------------------------------------------


class ArenoConfigIntegrationTest(unittest.TestCase):
    """ArenoConfig(runtime=...) flows preserve_first_error to RuntimeConfig."""

    def test_areno_config_runtime_dict_to_engine_config(self):
        """ArenoConfig.runtime dict is accepted by RuntimeConfig(**runtime)."""

        from areno.api.config import ArenoConfig
        from areno.engine.config import RuntimeConfig

        areno_cfg = ArenoConfig(runtime={"preserve_first_error": True, "attn_backend": "native"})
        runtime_cfg = RuntimeConfig(**areno_cfg.runtime)
        self.assertTrue(runtime_cfg.preserve_first_error)
        self.assertEqual(runtime_cfg.attn_backend, "native")


if __name__ == "__main__":
    unittest.main()
