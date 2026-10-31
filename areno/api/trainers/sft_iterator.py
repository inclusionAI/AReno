"""Resumable iteration over the raw rows of one SFT epoch."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np


class SFTBatchIterator(Iterator[list[Any]]):
    """Track the actual row order and cursor, including filtered rows."""

    def __init__(
        self,
        dataset: Any,
        convert: Callable[[Any], Any],
        batch_size: int,
        rng: np.random.RandomState,
        *,
        shuffle: bool,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.convert = convert
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError("SFT batch_size must be positive")
        size = len(dataset)
        if state is None:
            self.order = rng.permutation(size).tolist() if shuffle else list(range(size))
            self.next_index = 0
            self.accepted = 0
            self.skipped = 0
        else:
            self.order = state["order"]
            self.next_index = state["next_index"]
            self.accepted = state["accepted"]
            self.skipped = state["skipped"]
            if (
                not isinstance(self.order, list)
                or sorted(self.order) != list(range(size))
                or not isinstance(self.next_index, int)
                or not 0 <= self.next_index <= size
                or not isinstance(self.accepted, int)
                or not isinstance(self.skipped, int)
                or self.accepted < 0
                or self.skipped < 0
                or self.accepted + self.skipped != self.next_index
            ):
                raise ValueError("invalid SFT dataset iterator state")

    def __next__(self) -> list[Any]:
        batch = []
        while self.next_index < len(self.order) and len(batch) < self.batch_size:
            row = self.dataset[self.order[self.next_index]]
            sequence = self.convert(row)
            self.next_index += 1
            if sequence is None:
                self.skipped += 1
            else:
                self.accepted += 1
                batch.append(sequence)
        if batch:
            return batch
        if self.accepted == 0:
            raise ValueError(
                "SFT dataset produced no valid training rows after filtering: "
                f"scanned {len(self.dataset)} row(s), skipped {self.skipped} as empty, over-budget, or all-prompt "
                "examples. Check dataset quality, --max-prompt-tokens, and --max-new-tokens."
            )
        raise StopIteration

    def state_dict(self) -> dict[str, Any]:
        """Return the row permutation and the next raw row position."""

        return {
            "order": list(self.order),
            "next_index": self.next_index,
            "accepted": self.accepted,
            "skipped": self.skipped,
        }
