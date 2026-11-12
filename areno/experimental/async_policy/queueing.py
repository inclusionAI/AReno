"""Bounded group queue for the experimental async policy pipeline.

The queue's atomic unit is a :class:`GroupBatch`: all `n_samples` rows of one
prompt group. Rows must travel together because their advantages were
normalized within the group and staleness is tracked per rollout policy
version.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


class QueueClosed(RuntimeError):
    """Raised by producers/consumers after the queue has been closed."""


@dataclass(slots=True)
class GroupBatch:
    """One prompt group's materialized training rows plus provenance."""

    epoch: int
    prompt_index: int
    policy_version: int
    sequences: list[Any]
    rewards: list[float]


class BoundedGroupQueue:
    """Thread-safe bounded FIFO with blocking backpressure.

    Producers block while the queue holds ``max_groups`` groups, which is the
    backpressure mechanism that keeps the rollout side from running ahead of
    training by more than the configured bound. Consumers pull up to ``count``
    groups per call, waiting for a full batch until ``timeout`` expires or the
    queue closes. ``close`` wakes every waiter: blocked producers raise
    :class:`QueueClosed`, consumers drain the remaining groups and then see an
    empty result.
    """

    def __init__(self, max_groups: int):
        if max_groups < 1:
            raise ValueError("max_groups must be >= 1")
        self._max_groups = max_groups
        self._items: deque[GroupBatch] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self.enqueued = 0
        self.dequeued = 0

    @property
    def closed(self) -> bool:
        """Return True once the queue is closed and fully drained."""

        with self._cv:
            return self._closed and not self._items

    @property
    def depth(self) -> int:
        """Return the number of groups currently buffered."""

        with self._cv:
            return len(self._items)

    def put(self, item: GroupBatch, *, stop: threading.Event | None = None) -> None:
        """Enqueue one group, blocking while the queue is at capacity.

        Raises :class:`QueueClosed` when the queue closes or when ``stop`` is
        set while the producer is waiting for space.
        """

        with self._cv:
            while len(self._items) >= self._max_groups and not self._closed:
                if stop is not None and stop.is_set():
                    raise QueueClosed("producer stop requested")
                self._cv.wait(timeout=0.1)
            if self._closed:
                raise QueueClosed("queue is closed")
            self._items.append(item)
            self.enqueued += 1
            self._cv.notify_all()

    def get_many(self, count: int, *, timeout: float | None = None) -> list[GroupBatch]:
        """Dequeue up to ``count`` groups.

        Blocks until a full batch is available, the queue closes (draining
        whatever is left), or ``timeout`` seconds elapse. Returns an empty list
        only when nothing can be dequeued before the timeout or after the
        queue is closed and drained.
        """

        if count < 1:
            raise ValueError("count must be >= 1")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while True:
                if len(self._items) >= count or (self._closed and self._items):
                    break
                if self._closed:
                    return []
                if deadline is None:
                    self._cv.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(timeout=remaining)
            take = min(count, len(self._items))
            out = [self._items.popleft() for _ in range(take)]
            self.dequeued += len(out)
            self._cv.notify_all()
            return out

    def close(self) -> None:
        """Close the queue and wake all blocked producers/consumers."""

        with self._cv:
            self._closed = True
            self._cv.notify_all()
