from __future__ import annotations

import asyncio
import importlib.util
import itertools
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


def _load_protocol_module():
    """Load protocol.py without importing areno.engine package side effects."""

    path = Path(__file__).resolve().parents[1] / "areno" / "engine" / "protocol.py"
    spec = importlib.util.spec_from_file_location("_areno_protocol_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol_module()
TPCluster = protocol.TPCluster
Op = protocol.Op
WorkerResult = protocol.WorkerResult


class FakeQueue:
    """Small queue double that records close/join_thread calls."""

    def __init__(self):
        self.closed = False
        self.joined = False
        self.items = []

    def put(self, item):
        self.items.append(item)

    def close(self):
        self.closed = True

    def join_thread(self):
        self.joined = True


class FakeProcess:
    """Small process double for TPCluster.close resource cleanup tests."""

    def __init__(self, alive: bool):
        self._alive = alive
        self.join_calls = []
        self.terminated = False

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False


class TPClusterResourceTest(unittest.TestCase):
    """Protocol resource tests avoid spawning real multiprocessing workers."""

    def test_close_closes_command_and_result_queues(self):
        """TPCluster.close should release queue semaphores after worker shutdown."""
        cluster = object.__new__(TPCluster)
        cluster.config = SimpleNamespace(tp_size=1, dp_size=2)
        cluster.started = True
        cluster.cmd_queues = [FakeQueue(), FakeQueue()]
        cluster.result_queue = FakeQueue()
        cluster.processes = [FakeProcess(alive=False), FakeProcess(alive=True)]

        cluster.close()

        self.assertFalse(cluster.started)
        self.assertFalse(cluster.processes[1].is_alive())
        self.assertTrue(cluster.processes[1].terminated)
        self.assertEqual(cluster.processes[0].join_calls, [5, 0])
        self.assertEqual(cluster.processes[1].join_calls, [5, 0])
        for queue in [*cluster.cmd_queues, cluster.result_queue]:
            self.assertTrue(queue.closed)
            self.assertTrue(queue.joined)

    def test_async_call_can_wait_for_user_visible_rollout_ranks_only(self):
        """Async rollout futures should not wait for TP sibling acks before returning."""

        cluster = object.__new__(TPCluster)
        cluster.config = SimpleNamespace(tp_size=2, dp_size=2)
        cluster.started = True
        cluster.cmd_queues = [FakeQueue() for _ in range(4)]
        cluster._pending_lock = threading.Lock()
        cluster._send_lock = threading.Lock()
        cluster._pending_calls = {}

        pending = cluster._submit_call(Op.INFER_ROLLOUT, request_id=7, result_ranks={0, 2})

        cluster._apply_result(7, 1, WorkerResult(ok=True, payload="tp-sibling"), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(7, 0, WorkerResult(ok=True, payload="dp0"), pending)
        self.assertFalse(pending.event.is_set())

        cluster._apply_result(7, 2, WorkerResult(ok=True, payload="dp1"), pending)
        self.assertTrue(pending.event.is_set())
        self.assertEqual(pending.results[0], "dp0")
        self.assertEqual(pending.results[2], "dp1")

    def test_call_async_stream_yields_steps_and_propagates_rank_errors(self):
        """Streaming calls route intermediate steps and raise on rank failure."""

        cluster = object.__new__(TPCluster)
        cluster.config = SimpleNamespace(tp_size=1, dp_size=1)
        cluster.started = True
        cluster.cmd_queues = [FakeQueue()]
        cluster._pending_lock = threading.Lock()
        cluster._send_lock = threading.Lock()
        cluster._pending_calls = {}
        cluster._request_ids = itertools.count(1)
        cluster._pump_stop = threading.Event()
        cluster._pump_thread = None

        async def scenario():
            steps = []
            stream = cluster.call_async_stream(Op.INFER_ROLLOUT, None, result_ranks={0}, timeout=0.01)
            try:
                first_step = asyncio.ensure_future(stream.__anext__())
                await asyncio.sleep(0.05)
                (request_id, pending), = cluster._pending_calls.items()

                # Intermediate stream step: routed into the stream queue
                # without completing the call.
                cluster._apply_result(
                    request_id,
                    0,
                    WorkerResult(ok=True, payload=SimpleNamespace(prompt_idx=2, token_id=7), request_id=request_id),
                    pending,
                )
                steps.append(await first_step)
                self.assertFalse(pending.event.is_set())

                # Rank failure: the generator must raise instead of ending
                # the stream silently.
                second_step = asyncio.ensure_future(stream.__anext__())
                await asyncio.sleep(0.05)
                cluster._apply_result(request_id, 0, WorkerResult(ok=False, error="boom"), pending)
                with self.assertRaises(RuntimeError):
                    await second_step
                return steps
            finally:
                await stream.aclose()

        steps = asyncio.run(scenario())

        self.assertEqual([(step.prompt_idx, step.token_id) for step in steps], [(2, 7)])
        self.assertEqual(cluster._pending_calls, {})


if __name__ == "__main__":
    unittest.main()
