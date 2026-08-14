"""CPU tests for the experimental async policy trainer primitives."""

from __future__ import annotations

import threading
import unittest

from areno.api.models import TrainSequence
from areno.experimental.async_policy import (
    AsyncTrainBatch,
    AsyncTrainBatchQueue,
    QueueClosed,
    is_stale,
    staleness_delta,
)


def _batch(*, version: int = 0, idx: int = 0) -> AsyncTrainBatch:
    return AsyncTrainBatch(
        train_sequences=[TrainSequence()],
        rollout_policy_version=version,
        epoch=0,
        prompt_batch_id=idx,
        prompt_count=1,
        sequence_count=1,
        token_count=0,
        reward_mean=None,
        rollout_time_s=0.0,
        reward_time_s=0.0,
        materialize_time_s=0.0,
    )


class StalenessTest(unittest.TestCase):
    """Staleness helpers compare rollout and training policy versions."""

    def test_delta_is_train_minus_rollout(self):
        self.assertEqual(staleness_delta(5, 3), 2)

    def test_below_bound_is_not_stale(self):
        self.assertFalse(is_stale(2, 1, max_staleness=2))

    def test_at_bound_is_not_stale(self):
        self.assertFalse(is_stale(3, 1, max_staleness=2))

    def test_above_bound_is_stale(self):
        self.assertTrue(is_stale(4, 1, max_staleness=2))

    def test_negative_max_staleness_rejected(self):
        with self.assertRaises(ValueError):
            is_stale(3, 2, max_staleness=-1)


class AsyncTrainBatchTest(unittest.TestCase):
    """AsyncTrainBatch carries the metadata the train loop needs."""

    def test_fields_populate_and_created_at_defaults(self):
        batch = _batch(version=2, idx=7)
        self.assertEqual(batch.rollout_policy_version, 2)
        self.assertEqual(batch.prompt_batch_id, 7)
        self.assertEqual(len(batch.train_sequences), 1)
        self.assertIsInstance(batch.created_at_s, float)
        self.assertGreater(batch.created_at_s, 0.0)


class AsyncTrainBatchQueueTest(unittest.TestCase):
    """The bounded queue enforces backpressure, timeouts, and error flow."""

    def test_fifo_order(self):
        q = AsyncTrainBatchQueue(maxsize=2)
        q.put(_batch(idx=0))
        q.put(_batch(idx=1))
        self.assertEqual(q.get().prompt_batch_id, 0)
        self.assertEqual(q.get().prompt_batch_id, 1)

    def test_rejects_nonpositive_maxsize(self):
        with self.assertRaises(ValueError):
            AsyncTrainBatchQueue(maxsize=0)

    def test_qsize_tracks_pending_batches(self):
        q = AsyncTrainBatchQueue(maxsize=2)
        self.assertEqual(q.qsize(), 0)
        q.put(_batch())
        self.assertEqual(q.qsize(), 1)

    def test_put_timeout_when_full(self):
        q = AsyncTrainBatchQueue(maxsize=1, timeout_s=0.05)
        q.put(_batch())
        with self.assertRaises(TimeoutError):
            q.put(_batch())

    def test_get_timeout_when_empty(self):
        q = AsyncTrainBatchQueue(maxsize=1, timeout_s=0.05)
        with self.assertRaises(TimeoutError):
            q.get()

    def test_put_blocks_until_slot_frees(self):
        q = AsyncTrainBatchQueue(maxsize=1)
        q.put(_batch(idx=0))
        producer = threading.Thread(target=lambda: q.put(_batch(idx=1)))
        producer.start()
        producer.join(timeout=0.2)
        self.assertTrue(producer.is_alive(), "producer should block on a full queue")
        q.get()
        producer.join(timeout=1.0)
        self.assertFalse(producer.is_alive(), "producer should finish once a slot frees")

    def test_get_blocks_until_item_arrives(self):
        q = AsyncTrainBatchQueue(maxsize=1)
        result = []

        def consume():
            result.append(q.get())

        consumer = threading.Thread(target=consume)
        consumer.start()
        consumer.join(timeout=0.2)
        self.assertTrue(consumer.is_alive(), "consumer should block on an empty queue")
        q.put(_batch(idx=3))
        consumer.join(timeout=1.0)
        self.assertFalse(consumer.is_alive(), "consumer should finish once an item arrives")
        self.assertEqual(result[0].prompt_batch_id, 3)

    def test_close_unblocks_blocked_put(self):
        q = AsyncTrainBatchQueue(maxsize=1)
        q.put(_batch())
        result = []

        def produce():
            try:
                q.put(_batch(idx=1))
            except Exception as e:
                result.append(e)

        producer = threading.Thread(target=produce)
        producer.start()
        producer.join(timeout=0.2)
        self.assertTrue(producer.is_alive(), "producer should block on a full queue")
        q.close()
        producer.join(timeout=1.0)
        self.assertFalse(producer.is_alive(), "producer should finish after close")
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], QueueClosed)

    def test_put_after_close_raises(self):
        q = AsyncTrainBatchQueue(maxsize=1)
        q.close()
        with self.assertRaises(QueueClosed):
            q.put(_batch())

    def test_get_after_close_raises(self):
        q = AsyncTrainBatchQueue(maxsize=1)
        q.close()
        with self.assertRaises(QueueClosed):
            q.get()

    def test_report_error_surfaces_on_put(self):
        q = AsyncTrainBatchQueue(maxsize=1)
        err = RuntimeError("rollout crashed")
        q.report_error(err)
        with self.assertRaises(RuntimeError) as ctx:
            q.put(_batch())
        self.assertIs(ctx.exception, err)

    def test_report_error_surfaces_on_get(self):
        q = AsyncTrainBatchQueue(maxsize=1)
        err = RuntimeError("rollout crashed")
        q.report_error(err)
        with self.assertRaises(RuntimeError) as ctx:
            q.get()
        self.assertIs(ctx.exception, err)

    def test_report_error_unblocks_blocked_get(self):
        q = AsyncTrainBatchQueue(maxsize=1)
        result = []

        def consume():
            try:
                q.get()
            except Exception as e:
                result.append(e)

        consumer = threading.Thread(target=consume)
        consumer.start()
        consumer.join(timeout=0.2)
        self.assertTrue(consumer.is_alive())
        err = RuntimeError("boom")
        q.report_error(err)
        consumer.join(timeout=1.0)
        self.assertFalse(consumer.is_alive())
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], err)


if __name__ == "__main__":
    unittest.main()
