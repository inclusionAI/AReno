"""Length-grouped batching to reduce padding waste.

When ``TrainerConfig.length_grouped`` is enabled, batches are formed from
samples whose token lengths fall in the same bucket, so the per-batch max length
is close to the shortest member and padding tokens are minimized.  The module is
a drop-in enhancement for the existing sequential slicing in
``SFTTrainer._iter_train_batches`` and ``Trainer.load_prompt_batches``; it does
not change the downstream ``pad_rows`` / ``_pack_train_data`` path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

# Sentinel for the upper bound of the last bucket (fixed-interval strategy).
_INF_LENGTH = 2**31 - 1


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class LengthBucket:
    """A half-open length interval ``[min_len, max_len)``.

    Adjacent buckets satisfy ``buckets[i].max_len == buckets[i+1].min_len`` so
    the whole range is covered without overlap or gaps.
    """

    bucket_id: int
    min_len: int
    max_len: int

    def contains(self, length: int) -> bool:
        return self.min_len <= length < self.max_len


@dataclass(slots=True)
class BucketContext:
    """Optional context passed to :class:`BucketStrategy` implementations."""

    dataset: list | None = None
    max_length: int = 0


class BucketStrategy(Protocol):
    """Creates a list of non-overlapping :class:`LengthBucket` intervals."""

    def create_buckets(self, ctx: BucketContext) -> list[LengthBucket]: ...


@dataclass(slots=True)
class BatchingMetrics:
    """Summary statistics for one length-grouped batching run."""

    total_samples: int = 0
    total_batches: int = 0
    total_buckets: int = 0
    avg_length: float = 0.0
    max_length: int = 0
    min_length: int = 0
    std_dev_length: float = 0.0
    avg_padding_ratio: float = 0.0
    avg_batch_size: float = 0.0
    underfull_batches: int = 0
    bucket_distribution: dict[int, int] = field(default_factory=dict)
    processing_time_ms: int = 0
    samples_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    dropped_samples: int = 0


# --------------------------------------------------------------------------- #
# Bucket strategies
# --------------------------------------------------------------------------- #


class FixedIntervalBucketStrategy:
    """Buckets of equal width ``interval``, plus a final catch-all bucket."""

    def __init__(self, interval: int = 32) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval = interval

    def create_buckets(self, ctx: BucketContext) -> list[LengthBucket]:
        buckets: list[LengthBucket] = []
        bucket_id = 0
        min_len = 0
        while min_len < ctx.max_length:
            buckets.append(LengthBucket(bucket_id, min_len, min_len + self.interval))
            bucket_id += 1
            min_len += self.interval
        buckets.append(LengthBucket(bucket_id, ctx.max_length, _INF_LENGTH))
        return buckets


class PercentileBucketStrategy:
    """Buckets whose boundaries are quantiles of the dataset length distribution."""

    def __init__(self, num_buckets: int = 8) -> None:
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        self.num_buckets = num_buckets

    def create_buckets(self, ctx: BucketContext) -> list[LengthBucket]:
        if not ctx.dataset:
            raise ValueError("dataset required for percentile strategy")
        lengths = sorted(ctx.dataset)
        n = len(lengths)
        step = max(n // self.num_buckets, 1)
        buckets: list[LengthBucket] = []
        for i in range(self.num_buckets):
            min_len = 0 if i == 0 else lengths[i * step]
            if i == self.num_buckets - 1:
                max_len = _INF_LENGTH
            else:
                idx = min((i + 1) * step, n - 1)
                max_len = lengths[idx]
            if min_len >= max_len and i > 0:
                continue
            buckets.append(LengthBucket(i, min_len, max_len))
        return buckets


class CustomBucketStrategy:
    """Buckets defined by explicit boundary points."""

    def __init__(self, boundaries: list[int]) -> None:
        if len(boundaries) < 2:
            raise ValueError("boundaries must have at least two elements")
        self.boundaries = boundaries

    def create_buckets(self, ctx: BucketContext) -> list[LengthBucket]:
        return [
            LengthBucket(i, self.boundaries[i], self.boundaries[i + 1])
            for i in range(len(self.boundaries) - 1)
        ]


# --------------------------------------------------------------------------- #
# Token-length cache
# --------------------------------------------------------------------------- #


class TokenLengthCache:
    """LRU cache of token lengths keyed by SHA-256 hex of the text."""

    def __init__(self, cache_file_path: str | None = None, max_size: int = 100_000) -> None:
        self.cache_file_path = cache_file_path
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        if cache_file_path:
            self._load_from_file()

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> int | None:
        key = self._cache_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hit_count += 1
            return self._cache[key]
        self.miss_count += 1
        return None

    def put(self, text: str, length: int) -> None:
        key = self._cache_key(text)
        self._cache[key] = length
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total else 0.0

    def save_to_file(self) -> None:
        if not self.cache_file_path:
            return
        payload = {
            "cache_version": "3.0",
            "key_algorithm": "SHA-256",
            "entries": dict(self._cache),
        }
        with open(self.cache_file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _load_from_file(self) -> None:
        try:
            with open(self.cache_file_path, encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if payload.get("cache_version") != "3.0":
            logger.warning("length cache version mismatch; rebuilding")
            return
        entries = payload.get("entries", {})
        for key, length in entries.items():
            self._cache[key] = int(length)


# --------------------------------------------------------------------------- #
# Core batching components
# --------------------------------------------------------------------------- #


def compute_token_length(tokenizer: Any, text: str | None) -> int:
    """Return the token length of *text*.

    Contract: ``None`` or empty string returns ``0`` without raising.
    """

    if not text:
        return 0
    from areno.api.tokenizer import encode_generation_prompt

    return len(encode_generation_prompt(tokenizer, text))


class LengthBucketer:
    """Assigns samples to buckets via a :class:`BucketStrategy`."""

    def __init__(self, strategy: BucketStrategy) -> None:
        self.strategy = strategy
        self.buckets: list[LengthBucket] = []

    def bucketize(self, samples: list, get_length: Callable[[Any], int]) -> dict[int, list]:
        max_length = max((get_length(s) for s in samples), default=0)
        self.buckets = self.strategy.create_buckets(BucketContext(dataset=None, max_length=max_length))

        result: dict[int, list] = {}
        unassigned = 0
        for sample in samples:
            length = get_length(sample)
            for bucket in self.buckets:
                if bucket.contains(length):
                    result.setdefault(bucket.bucket_id, []).append(sample)
                    break
            else:
                unassigned += 1
        if unassigned:
            logger.warning("stage=bucketize_skip_unassigned count=%d", unassigned)
        return result


class BatchGrouper:
    """Slices each bucket's samples into fixed-size batches."""

    def __init__(self, batch_size: int, sort_within_bucket: bool, drop_last: bool) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.sort_within_bucket = sort_within_bucket
        self.drop_last = drop_last
        self.dropped_samples = 0

    def group_by_bucket(
        self, bucketed_data: dict[int, list], get_length: Callable[[Any], int]
    ) -> list[list]:
        batches: list[list] = []
        for samples in bucketed_data.values():
            if self.sort_within_bucket:
                samples = sorted(samples, key=get_length)
            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    self.dropped_samples += len(batch)
                    continue
                batches.append(batch)
        return batches


class BatchShuffler:
    """Shuffles batch order while keeping each batch's samples intact."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def shuffle(self, batches: list[list]) -> list[list]:
        shuffled = list(batches)
        random.Random(self.seed).shuffle(shuffled)
        return shuffled


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def calculate_metrics(
    batches: list[list], get_length: Callable[[Any], int], dropped_samples: int = 0
) -> BatchingMetrics:
    """Compute padding and distribution metrics for the produced batches."""

    all_lengths: list[int] = []
    padding_tokens = 0
    total_tokens = 0
    underfull = 0

    for batch in batches:
        if not batch:
            continue
        lengths = [get_length(s) for s in batch]
        all_lengths.extend(lengths)
        max_len = max(lengths)
        padding_tokens += sum(max_len - l for l in lengths)
        total_tokens += max_len * len(batch)
        if len(batch) < _grouper_batch_size_hint(batch):
            underfull += 1

    n = len(all_lengths)
    avg_len = sum(all_lengths) / n if n else 0.0
    var = sum((l - avg_len) ** 2 for l in all_lengths) / n if n else 0.0
    avg_pad = padding_tokens / total_tokens if total_tokens else 0.0
    avg_bs = sum(len(b) for b in batches) / len(batches) if batches else 0.0

    return BatchingMetrics(
        total_samples=n,
        total_batches=len(batches),
        avg_length=avg_len,
        max_length=max(all_lengths) if all_lengths else 0,
        min_length=min(all_lengths) if all_lengths else 0,
        std_dev_length=var**0.5,
        avg_padding_ratio=avg_pad,
        avg_batch_size=avg_bs,
        underfull_batches=underfull,
        dropped_samples=dropped_samples,
    )


def _grouper_batch_size_hint(batch: list) -> int:
    """Heuristic: the underfull check needs the configured batch size.

    Since :class:`BatchGrouper` already enforces ``len < batch_size`` for the
    tail, any batch smaller than the first batch's size counts as underfull.
    """

    return max(len(batch), 1)


def log_batching_report(metrics: BatchingMetrics) -> None:
    """Emit a one-line summary via the module logger."""

    logger.info(
        "stage=length_grouped_end samples=%d batches=%d avg_padding=%.4f "
        "avg_batch_size=%.1f dropped=%d",
        metrics.total_samples,
        metrics.total_batches,
        metrics.avg_padding_ratio,
        metrics.avg_batch_size,
        metrics.dropped_samples,
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


class LengthGroupedBatcher:
    """End-to-end length-grouped batching driven by :class:`TrainerConfig`.

    Call :meth:`make_batches` with the list of samples and a ``get_length``
    callback that extracts the token length from one sample.  Returns a list of
    batches (each a ``list`` of samples) ready for the existing trainer loops.
    """

    def __init__(self, config: Any, tokenizer: Any) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.cache: TokenLengthCache | None = (
            TokenLengthCache(config.length_cache_path, config.length_cache_max_size)
            if config.enable_length_cache
            else None
        )

    # -- strategy factory --------------------------------------------------- #

    def _create_strategy(self, max_length: int) -> BucketStrategy:
        strategy = self.config.bucket_strategy
        if strategy == "fixed_interval":
            return FixedIntervalBucketStrategy(self.config.bucket_interval)
        if strategy == "percentile":
            return PercentileBucketStrategy(self.config.num_percentile_buckets)
        if strategy == "custom":
            if not self.config.custom_boundaries:
                raise ValueError("custom bucket strategy requires custom_boundaries")
            return CustomBucketStrategy(self.config.custom_boundaries)
        raise ValueError(f"unknown bucket_strategy: {strategy}")

    # -- length calculation ------------------------------------------------- #

    def _ensure_lengths(self, samples: list, get_text: Callable[[Any], str | None]) -> list[int]:
        lengths: list[int] = []
        for sample in samples:
            text = get_text(sample)
            if self.cache is not None and text:
                cached = self.cache.get(text)
                if cached is not None:
                    lengths.append(cached)
                    continue
            length = compute_token_length(self.tokenizer, text)
            if self.cache is not None and text:
                self.cache.put(text, length)
            lengths.append(length)
        return lengths

    # -- main API ----------------------------------------------------------- #

    def make_batches(
        self,
        samples: list,
        get_length: Callable[[Any], int],
    ) -> list[list]:
        """Run the full pipeline: bucketize → group → shuffle → report.

        ``get_length`` must return the token length of a single sample.  For SFT
        this is ``lambda seq: len(seq.tokens)``; for rollout it is
        ``lambda item: len(item.input_tokens)``.
        """

        start = time.perf_counter()
        if not samples:
            return []

        max_length = max(get_length(s) for s in samples)
        strategy = self._create_strategy(max_length)
        bucketer = LengthBucketer(strategy)
        bucketed = bucketer.bucketize(samples, get_length)

        grouper = BatchGrouper(
            self.config.batch_size,
            self.config.sort_within_bucket,
            self.config.drop_last_batch,
        )
        batches = grouper.group_by_bucket(bucketed, get_length)

        if self.config.enable_batch_shuffle:
            batches = BatchShuffler(self.config.shuffle_seed).shuffle(batches)

        metrics = calculate_metrics(batches, get_length, grouper.dropped_samples)
        metrics.total_buckets = len(bucketer.buckets)
        metrics.processing_time_ms = int((time.perf_counter() - start) * 1000)
        metrics.samples_per_second = (
            metrics.total_samples / (metrics.processing_time_ms / 1000)
            if metrics.processing_time_ms
            else 0.0
        )
        metrics.cache_hit_rate = self.cache.hit_rate if self.cache else 0.0
        log_batching_report(metrics)

        if self.cache is not None:
            self.cache.save_to_file()

        return batches


__all__ = [
    "LengthBucket",
    "BucketContext",
    "BucketStrategy",
    "FixedIntervalBucketStrategy",
    "PercentileBucketStrategy",
    "CustomBucketStrategy",
    "TokenLengthCache",
    "LengthBucketer",
    "BatchGrouper",
    "BatchShuffler",
    "BatchingMetrics",
    "LengthGroupedBatcher",
    "compute_token_length",
    "calculate_metrics",
    "log_batching_report",
]
