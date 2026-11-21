"""Bounded queue decoupling rollout from training."""

from __future__ import annotations

import threading
import time
from collections import deque

from areno.experimental.async_policy.batch import AsyncTrainBatch


class QueueClosed(RuntimeError):
    """Raised when a producer or consumer touches a closed queue."""


class AsyncTrainBatchQueue:
    """Bounded FIFO queue between the rollout worker and the train loop.

    Producers block on :meth:`put` while the queue is full (backpressure);
    consumers block on :meth:`get` while it is empty. :meth:`close` unblocks all
    waiters and makes later operations raise :class:`QueueClosed`. A rollout
    worker failure is surfaced through :meth:`report_error`, which likewise
    unblocks waiters and makes the error visible to the next operation.
    """

    def __init__(self, maxsize: int, *, timeout_s: float | None = None) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._maxsize = maxsize
        self._timeout_s = timeout_s
        self._items: deque[AsyncTrainBatch] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self._error: Exception | None = None

    def put(self, batch: AsyncTrainBatch, *, timeout_s: float | None = None) -> None:
        """Append a batch, blocking while the queue is full."""

        with self._cond:
            self._wait_while(lambda: len(self._items) >= self._maxsize, timeout_s, "put")
            self._raise_if_unusable()
            self._items.append(batch)
            self._cond.notify_all()

    def get(self, *, timeout_s: float | None = None) -> AsyncTrainBatch:
        """Remove and return the oldest batch, blocking while the queue is empty."""

        with self._cond:
            self._wait_while(lambda: not self._items, timeout_s, "get")
            self._raise_if_unusable()
            batch = self._items.popleft()
            self._cond.notify_all()
            return batch

    def report_error(self, error: Exception) -> None:
        """Record a worker failure, unblocking all waiters.

        The first reported error is retained; subsequent operations raise it.
        """

        with self._cond:
            if self._error is None:
                self._error = error
            self._cond.notify_all()

    def close(self) -> None:
        """Close the queue, unblocking all waiters."""

        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def qsize(self) -> int:
        """Return the number of batches currently queued."""

        return len(self._items)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def error(self) -> Exception | None:
        return self._error

    def _wait_while(self, blocked, timeout_s: float | None, op: str) -> None:
        timeout = self._timeout_s if timeout_s is None else timeout_s
        deadline = None if timeout is None else time.monotonic() + timeout
        while blocked() and not self._closed and self._error is None:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"{op} timed out")
            self._cond.wait(remaining)

    def _raise_if_unusable(self) -> None:
        if self._error is not None:
            raise self._error
        if self._closed:
            raise QueueClosed("queue is closed")
