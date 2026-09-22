from __future__ import annotations

import importlib.util
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


class TPClusterStreamEofTest(unittest.TestCase):
    """The streaming read loop must terminate once the worker closes its pipe end.

    EOF on a ``multiprocessing.Pipe`` only fires when *every* send-end reference
    is closed. The coordinator creates the pipe, ships the send end to the worker
    through the command queue, and must close its own copy; otherwise the recv
    loop blocks forever after the last token, leaving the SSE response without a
    finish chunk or ``[DONE]``.
    """

    def _run_stream_call(self):
        import asyncio

        from areno.engine.data.batch import StreamTokenStep

        cluster = object.__new__(TPCluster)
        cluster.config = SimpleNamespace(tp_size=1, dp_size=1)
        cluster.started = True
        cluster._request_ids = iter([1])
        cluster._pending_lock = threading.Lock()
        cluster._pending_calls = {}

        seen: list[tuple[int, int, str | None]] = []
        loop = asyncio.new_event_loop()

        def submit(op, payload=None, **kwargs):
            # Play the worker with a genuinely separate fd, mirroring how the
            # command queue hands the child an independent duplicate. A plain
            # pickle round-trip would share the same fd in-process and mask the
            # bug, so duplicate the descriptor explicitly.
            import os

            from multiprocessing.connection import Connection

            child_fd = os.dup(payload.stream_conn.fileno())
            dup_send = Connection(child_fd)
            kwargs["future"].get_loop().call_soon_threadsafe(kwargs["future"].set_result, None)
            dup_send.send(StreamTokenStep(prompt_idx=0, token_id=11))
            dup_send.send(StreamTokenStep(prompt_idx=0, token_id=12, finish_reason="stop"))
            dup_send.close()
            return None

        cluster._submit_call = submit

        async def run() -> None:
            async for step in cluster.stream_call_async(SimpleNamespace()):
                seen.append((step.prompt_idx, step.token_id, step.finish_reason))

        try:
            asyncio.set_event_loop(loop)
            # A hang (the pre-fix behaviour) surfaces as a timeout, not a stall.
            loop.run_until_complete(asyncio.wait_for(run(), timeout=10))
        finally:
            loop.close()

        self.assertEqual(seen, [(0, 11, None), (0, 12, "stop")])

    def test_stream_call_async_ends_when_worker_closes_pipe(self):
        """Regression: the parent must drop its send-end copy or recv hangs."""

        self._run_stream_call()


if __name__ == "__main__":
    unittest.main()
