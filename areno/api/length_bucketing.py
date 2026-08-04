"""Length-bucketed batch sampler for reducing padding waste.

When a batch mixes very short and very long sequences, right-padding to the
batch max length wastes compute on positions that loss masks will ignore
anyway.  This module groups similar-length items into the same batch so the
per-batch max length stays low.

The core function :func:`bucketed_batch_indices` is pure Python, depends on
no external libraries, and works with any ``(indices, lengths)`` pair so
both the RL path (``PromptItem.input_tokens``) and the SFT path
(``TrainSequence.tokens``) can reuse it.
"""

from __future__ import annotations

import random


def bucketed_batch_indices(
    indices: list[int],
    lengths: list[int],
    *,
    batch_size: int,
    num_buckets: int | None = None,
    seed: int,
    drop_last: bool = False,
) -> list[list[int]]:
    """Return a list of batches, each a list of dataset indices.

    Algorithm:
      1. Sort *indices* by *lengths* [index] ascending.
      2. Divide the sorted list into *num_buckets* contiguous slices
         (default ``min(len(indices) // batch_size, 128)``).
      3. Shuffle within each bucket using ``random.Random(seed)``.
      4. Shuffle the bucket order with the same RNG instance.
      5. Flatten and chunk into batches of *batch_size*.
      6. If the last batch is partial: drop it when *drop_last* is ``True``,
         otherwise keep it.

    Guarantees:
      - Every index appears exactly once across all returned batches
        (unless ``drop_last=True`` removes a partial final batch).
      - The same *seed* always reproduces the same batch order.
      - Similar-length indices tend to land in the same batch, reducing
        the per-batch max length and thus padding.

    Args:
        indices: Dataset row indices to sample from.
        lengths: Parallel list of sequence lengths (``len(lengths) == len(indices)``).
        batch_size: Number of items per batch.
        num_buckets: Number of length-sorted buckets.  ``None`` uses
            ``min(len(indices) // batch_size, 128)``.
        seed: RNG seed for deterministic shuffle.
        drop_last: If ``True``, discard a partial final batch.

    Returns:
        A list of batches, each a list of dataset indices.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if len(indices) != len(lengths):
        raise ValueError(f"indices and lengths must have equal length, got {len(indices)} vs {len(lengths)}")
    if not indices:
        return []

    # --- Step 1: sort by length ascending ---
    sorted_pairs = sorted(zip(lengths, indices), key=lambda pair: pair[0])
    sorted_indices = [idx for _, idx in sorted_pairs]

    # --- Step 2: divide into contiguous buckets ---
    if num_buckets is None:
        num_buckets = min(len(sorted_indices) // batch_size, 128)
    num_buckets = max(num_buckets, 1)  # at least one bucket
    num_buckets = min(num_buckets, len(sorted_indices))  # no more buckets than items

    bucket_size = len(sorted_indices) // num_buckets
    remainder = len(sorted_indices) % num_buckets

    buckets: list[list[int]] = []
    start = 0
    for i in range(num_buckets):
        # Distribute the remainder across the first *remainder* buckets
        extra = 1 if i < remainder else 0
        end = start + bucket_size + extra
        buckets.append(sorted_indices[start:end])
        start = end

    # --- Steps 3 & 4: shuffle within buckets, then shuffle bucket order ---
    rng = random.Random(seed)
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)

    # --- Step 5: flatten and chunk into batches ---
    flat: list[int] = []
    for bucket in buckets:
        flat.extend(bucket)

    batches: list[list[int]] = []
    for i in range(0, len(flat), batch_size):
        batch = flat[i : i + batch_size]
        if len(batch) < batch_size and drop_last:
            continue
        batches.append(batch)

    return batches
