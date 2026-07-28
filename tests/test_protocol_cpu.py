from __future__ import annotations

import importlib.util
import socket as _socket
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
_close_queue = protocol._close_queue
find_free_port = protocol.find_free_port


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


class RaisingFakeQueue(FakeQueue):
    """Queue double whose close()/join_thread() raise on second call."""

    def __init__(self):
        super().__init__()
        self._close_count = 0
        self._join_count = 0

    def close(self):
        self._close_count += 1
        if self._close_count > 1:
            raise ValueError("queue already closed")
        super().close()

    def join_thread(self):
        self._join_count += 1
        if self._join_count > 1:
            raise ValueError("queue already joined")
        super().join_thread()


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


def _make_cluster(started=True, num_ranks=2):
    """Create a TPCluster bypassing __init__, with fake queues and processes."""
    cluster = object.__new__(TPCluster)
    cluster.config = SimpleNamespace(tp_size=1, dp_size=num_ranks)
    cluster.started = started
    cluster.cmd_queues = [FakeQueue() for _ in range(num_ranks)]
    cluster.result_queue = FakeQueue()
    cluster.processes = [FakeProcess(alive=False) for _ in range(num_ranks)]
    cluster._pump_stop = threading.Event()
    cluster._pump_thread = None
    cluster._closed = False
    cluster._close_lock = threading.Lock()
    cluster._port_socket = None
    return cluster


class TPClusterResourceTest(unittest.TestCase):
    """Protocol resource tests avoid spawning real multiprocessing workers."""

    def test_close_closes_command_and_result_queues(self):
        """TPCluster.close should release queue semaphores after worker shutdown."""
        cluster = _make_cluster(started=True)
        cluster.processes[1] = FakeProcess(alive=True)

        # Save references before close() clears the lists.
        cmd_queues = list(cluster.cmd_queues)
        result_queue = cluster.result_queue
        processes = list(cluster.processes)

        cluster.close()

        self.assertFalse(cluster.started)
        self.assertFalse(processes[1].is_alive())
        self.assertTrue(processes[1].terminated)
        self.assertEqual(processes[0].join_calls, [5, 0])
        self.assertEqual(processes[1].join_calls, [5, 0])
        for queue in [*cmd_queues, result_queue]:
            self.assertTrue(queue.closed)
            self.assertTrue(queue.joined)

    def test_close_is_idempotent(self):
        """Repeated close() calls are safe and do not raise."""
        cluster = _make_cluster(started=True)
        cluster.close()
        cluster.close()  # second call should be a no-op
        cluster.close()  # third call should also be a no-op
        self.assertTrue(cluster._closed)

    def test_close_clears_process_and_queue_lists(self):
        """After close(), cmd_queues and processes are emptied."""
        cluster = _make_cluster(started=True)
        cluster.close()
        self.assertEqual(cluster.cmd_queues, [])
        self.assertEqual(cluster.processes, [])

    def test_close_concurrent_is_safe(self):
        """Concurrent close() calls from multiple threads do not crash."""
        cluster = _make_cluster(started=True)
        errors = []

        def _close():
            try:
                cluster.close()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_close) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertTrue(cluster._closed)

    def test_close_after_failed_start_clears_resources(self):
        """If start() fails, close() should still be safe and clear lists."""
        cluster = _make_cluster(started=False)
        # Simulate leftover state from a failed start (result_queue closed
        # but cmd_queues/processes already cleaned by the except block).
        cluster._closed = False
        cluster.close()
        self.assertTrue(cluster._closed)
        self.assertEqual(cluster.cmd_queues, [])
        self.assertEqual(cluster.processes, [])

    def test_close_queue_swallows_already_closed_error(self):
        """_close_queue should not propagate ValueError from double-close."""
        q = RaisingFakeQueue()
        _close_queue(q)
        _close_queue(q)  # second call should not raise

    def test_closed_cluster_cannot_restart(self):
        """After close(), start() should raise RuntimeError."""
        cluster = _make_cluster(started=True)
        cluster.close()
        with self.assertRaises(RuntimeError, msg="TPCluster has been closed"):
            cluster.start()

    def test_find_free_port_returns_bound_socket(self):
        """find_free_port returns a (port, socket) where the socket is still bound."""
        port, sock = find_free_port()
        try:
            self.assertIsInstance(port, int)
            self.assertGreater(port, 0)
            self.assertEqual(sock.getsockname()[1], port)
            # Verify the port is in use — a second bind should fail.
            with self.assertRaises(OSError):
                s2 = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                try:
                    s2.bind(("127.0.0.1", port))
                finally:
                    s2.close()
        finally:
            sock.close()

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


if __name__ == "__main__":
    unittest.main()
