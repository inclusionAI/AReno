from __future__ import annotations

import unittest
import importlib.util
import sys
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


TPCluster = _load_protocol_module().TPCluster


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


if __name__ == "__main__":
    unittest.main()
