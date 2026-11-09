from __future__ import annotations

import threading
import unittest

from areno.experimental.async_policy.queueing import BoundedGroupQueue, GroupBatch, QueueClosed


def _group(index: int, policy_version: int = 0) -> GroupBatch:
    return GroupBatch(
        epoch=0,
        prompt_index=index,
        policy_version=policy_version,
        sequences=[index],
        rewards=[float(index)],
    )


class BoundedGroupQueueTest(unittest.TestCase):
    """Queue tests cover backpressure, batching, and close semantics."""

    def test_put_blocks_at_capacity_until_get(self):
        """Producers must block once max_groups is reached (backpressure)."""
        queue = BoundedGroupQueue(max_groups=1)
        queue.put(_group(0))
        done = threading.Event()

        def _produce():
            queue.put(_group(1))
            done.set()

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()
        self.assertFalse(done.wait(timeout=0.2), "producer should block while the queue is full")

        [item] = queue.get_many(1, timeout=1.0)
        self.assertEqual(item.prompt_index, 0)
        self.assertTrue(done.wait(timeout=2.0), "producer should resume once space frees up")
        producer.join(timeout=2.0)
        self.assertEqual(queue.depth, 1)

    def test_get_many_waits_for_full_batch(self):
        """Consumers wait for `count` groups instead of taking partial batches."""
        queue = BoundedGroupQueue(max_groups=4)
        queue.put(_group(0))
        received: list[list] = []

        def _consume():
            received.append(queue.get_many(2, timeout=5.0))

        consumer = threading.Thread(target=_consume, daemon=True)
        consumer.start()
        consumer.join(timeout=0.2)
        self.assertTrue(consumer.is_alive(), "consumer should wait for the second group")
        queue.put(_group(1))
        consumer.join(timeout=2.0)
        self.assertFalse(consumer.is_alive())
        self.assertEqual([item.prompt_index for item in received[0]], [0, 1])

    def test_get_many_returns_partial_batch_on_timeout(self):
        """A timeout with buffered groups trains on what is available."""
        queue = BoundedGroupQueue(max_groups=4)
        queue.put(_group(0))
        items = queue.get_many(2, timeout=0.1)
        self.assertEqual([item.prompt_index for item in items], [0])

    def test_close_drains_then_reports_empty(self):
        """Closed queues hand out the remaining groups, then return empty."""
        queue = BoundedGroupQueue(max_groups=4)
        queue.put(_group(0))
        queue.put(_group(1))
        queue.close()
        first = queue.get_many(2, timeout=0.1)
        self.assertEqual(len(first), 2)
        self.assertEqual(queue.get_many(2, timeout=0.1), [])
        self.assertTrue(queue.closed)

    def test_close_rejects_blocked_producer(self):
        """A blocked producer wakes with QueueClosed when the queue closes."""
        queue = BoundedGroupQueue(max_groups=1)
        queue.put(_group(0))
        errors: list[BaseException] = []
        started = threading.Event()

        def _produce():
            started.set()
            try:
                queue.put(_group(1))
            except QueueClosed as exc:
                errors.append(exc)

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()
        self.assertTrue(started.wait(timeout=1.0))
        queue.close()
        producer.join(timeout=2.0)
        self.assertFalse(producer.is_alive())
        self.assertEqual(len(errors), 1)
        with self.assertRaises(QueueClosed):
            queue.put(_group(2))

    def test_put_honors_stop_event(self):
        """Producers blocked on a full queue exit when their stop event sets."""
        queue = BoundedGroupQueue(max_groups=1)
        queue.put(_group(0))
        stop = threading.Event()
        errors: list[BaseException] = []

        def _produce():
            try:
                queue.put(_group(1), stop=stop)
            except QueueClosed as exc:
                errors.append(exc)

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()
        stop.set()
        producer.join(timeout=2.0)
        self.assertFalse(producer.is_alive())
        self.assertEqual(len(errors), 1)

    def test_invalid_capacity_rejected(self):
        with self.assertRaises(ValueError):
            BoundedGroupQueue(max_groups=0)


if __name__ == "__main__":
    unittest.main()
