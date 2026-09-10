"""Experimental asynchronous policy trainer primitives.

This namespace hosts the standalone building blocks for the async policy
trainer — batch metadata, a bounded queue, and staleness control — without
registering any experimental algorithms. Algorithm registration lands together
with the async trainer itself.
"""

from __future__ import annotations

from areno.experimental.async_policy.batch import AsyncTrainBatch
from areno.experimental.async_policy.queue import AsyncTrainBatchQueue, QueueClosed
from areno.experimental.async_policy.staleness import is_stale, staleness_delta

__all__ = [
    "AsyncTrainBatch",
    "AsyncTrainBatchQueue",
    "QueueClosed",
    "is_stale",
    "staleness_delta",
]
