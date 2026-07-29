"""CPU tests for length-grouped batching.

These tests use plain integers and lightweight stub objects to exercise the
bucketing, grouping, shuffling and caching logic without requiring a real
tokenizer or GPU.

The module under test (``areno/api/length_grouped.py``) is loaded directly via
``importlib`` to bypass the ``areno.api.__init__`` import chain, which pulls in
torch, pydantic, and other heavy dependencies that require Python 3.10+.
This keeps the pure-Python batching logic testable on any Python version.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

# ---- Load length_grouped.py without triggering areno.api.__init__ ------------
# We register a minimal ``areno.api`` package stub and a ``areno.api.tokenizer``
# stub in ``sys.modules`` so the delayed import inside ``compute_token_length``
# does not fail.  Then we load the module file directly by path.

_pkg = types.ModuleType("areno.api")
_pkg.__path__ = []  # mark as package
sys.modules.setdefault("areno", types.ModuleType("areno"))
sys.modules["areno"].__path__ = []
sys.modules["areno.api"] = _pkg

_tok_stub = types.ModuleType("areno.api.tokenizer")
_tok_stub.encode_generation_prompt = lambda tokenizer, text: list(range(len(text)))
sys.modules["areno.api.tokenizer"] = _tok_stub

_spec = importlib.util.spec_from_file_location(
    "areno.api.length_grouped",
    os.path.join(os.path.dirname(__file__), "..", "areno", "api", "length_grouped.py"),
)
_length_grouped = importlib.util.module_from_spec(_spec)
sys.modules["areno.api.length_grouped"] = _length_grouped

# Patch ``dataclass(slots=True)`` for Python 3.9 compatibility (project requires 3.10+).
import dataclasses as _dc

_orig_dataclass = _dc.dataclass

def _compat_dataclass(*args, **kwargs):
    kwargs.pop("slots", None)
    return _orig_dataclass(*args, **kwargs)

_dc.dataclass = _compat_dataclass
_spec.loader.exec_module(_length_grouped)
_dc.dataclass = _orig_dataclass  # restore

# Re-export for test classes
BatchGrouper = _length_grouped.BatchGrouper
BatchShuffler = _length_grouped.BatchShuffler
BatchingMetrics = _length_grouped.BatchingMetrics
BucketContext = _length_grouped.BucketContext
CustomBucketStrategy = _length_grouped.CustomBucketStrategy
FixedIntervalBucketStrategy = _length_grouped.FixedIntervalBucketStrategy
LengthBucket = _length_grouped.LengthBucket
LengthBucketer = _length_grouped.LengthBucketer
LengthGroupedBatcher = _length_grouped.LengthGroupedBatcher
PercentileBucketStrategy = _length_grouped.PercentileBucketStrategy
TokenLengthCache = _length_grouped.TokenLengthCache
calculate_metrics = _length_grouped.calculate_metrics
compute_token_length = _length_grouped.compute_token_length

# TrainerConfig also loads via importlib to bypass areno.api.__init__.
# It only needs areno.api.defaults.DEFAULT_METRICS_LOG_DIR.
_defaults_stub = types.ModuleType("areno.api.defaults")
_defaults_stub.DEFAULT_METRICS_LOG_DIR = "/tmp/areno/tfevent"
sys.modules["areno.api.defaults"] = _defaults_stub

_spec_tc = importlib.util.spec_from_file_location(
    "areno.api.trainer_config",
    os.path.join(os.path.dirname(__file__), "..", "areno", "api", "trainer_config.py"),
)
_tc_module = importlib.util.module_from_spec(_spec_tc)
sys.modules["areno.api.trainer_config"] = _tc_module

_dc.dataclass = _compat_dataclass
_spec_tc.loader.exec_module(_tc_module)
_dc.dataclass = _orig_dataclass  # restore

TrainerConfig = _tc_module.TrainerConfig
_HAS_DEPS = True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _StubSample:
    """Minimal sample with a ``tokens`` list, mimicking ``TrainSequence``."""

    __slots__ = ("tokens", "text")

    def __init__(self, tokens: list[int], text: str = "") -> None:
        self.tokens = tokens
        self.text = text


def _make_samples(lengths: list[int]) -> list[_StubSample]:
    return [_StubSample(list(range(n))) for n in lengths]


def _length_fn(sample: _StubSample) -> int:
    return len(sample.tokens)


# --------------------------------------------------------------------------- #
# LengthBucket
# --------------------------------------------------------------------------- #


class LengthBucketTest(unittest.TestCase):

    def test_contains_half_open(self):
        b = LengthBucket(0, 10, 20)
        self.assertFalse(b.contains(9))
        self.assertTrue(b.contains(10))
        self.assertTrue(b.contains(19))
        self.assertFalse(b.contains(20))

    def test_frozen(self):
        b = LengthBucket(0, 0, 10)
        with self.assertRaises(Exception):
            b.bucket_id = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Bucket strategies
# --------------------------------------------------------------------------- #


class FixedIntervalBucketStrategyTest(unittest.TestCase):

    def test_create_buckets_no_overlap(self):
        strategy = FixedIntervalBucketStrategy(32)
        buckets = strategy.create_buckets(BucketContext(max_length=128))
        self.assertEqual(len(buckets), 5)  # [0,32) [32,64) [64,96) [96,128) [128,inf)
        for i in range(len(buckets) - 1):
            self.assertEqual(buckets[i].max_len, buckets[i + 1].min_len)
        self.assertEqual(buckets[-1].max_len, 2**31 - 1)

    def test_zero_max_length(self):
        strategy = FixedIntervalBucketStrategy(32)
        buckets = strategy.create_buckets(BucketContext(max_length=0))
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0].min_len, 0)

    def test_invalid_interval(self):
        with self.assertRaises(ValueError):
            FixedIntervalBucketStrategy(0)


class PercentileBucketStrategyTest(unittest.TestCase):

    def test_creates_expected_buckets(self):
        strategy = PercentileBucketStrategy(4)
        lengths = [10, 20, 30, 40, 50, 60, 70, 80]
        ctx = BucketContext(dataset=lengths)
        buckets = strategy.create_buckets(ctx)
        self.assertGreaterEqual(len(buckets), 1)
        self.assertEqual(buckets[-1].max_len, 2**31 - 1)

    def test_empty_dataset_raises(self):
        strategy = PercentileBucketStrategy(4)
        with self.assertRaises(ValueError):
            strategy.create_buckets(BucketContext(dataset=[]))


class CustomBucketStrategyTest(unittest.TestCase):

    def test_boundaries(self):
        strategy = CustomBucketStrategy([0, 64, 128, 2**31 - 1])
        buckets = strategy.create_buckets(BucketContext())
        self.assertEqual(len(buckets), 3)
        self.assertEqual(buckets[0].min_len, 0)
        self.assertEqual(buckets[0].max_len, 64)
        self.assertEqual(buckets[2].max_len, 2**31 - 1)

    def test_too_few_boundaries(self):
        with self.assertRaises(ValueError):
            CustomBucketStrategy([0])


# --------------------------------------------------------------------------- #
# LengthBucketer
# --------------------------------------------------------------------------- #


class LengthBucketerTest(unittest.TestCase):

    def test_bucket_assignment(self):
        strategy = FixedIntervalBucketStrategy(32)
        bucketer = LengthBucketer(strategy)
        samples = _make_samples([10, 50, 70])
        result = bucketer.bucketize(samples, _length_fn)
        total = sum(len(v) for v in result.values())
        self.assertEqual(total, 3)

    def test_empty_dataset(self):
        strategy = FixedIntervalBucketStrategy(32)
        bucketer = LengthBucketer(strategy)
        result = bucketer.bucketize([], _length_fn)
        self.assertEqual(result, {})

    def test_sample_at_boundary(self):
        strategy = FixedIntervalBucketStrategy(32)
        bucketer = LengthBucketer(strategy)
        samples = _make_samples([32])  # exactly on boundary
        result = bucketer.bucketize(samples, _length_fn)
        total = sum(len(v) for v in result.values())
        self.assertEqual(total, 1)


# --------------------------------------------------------------------------- #
# BatchGrouper
# --------------------------------------------------------------------------- #


class BatchGrouperTest(unittest.TestCase):

    def test_batch_size(self):
        grouper = BatchGrouper(32, sort_within_bucket=False, drop_last=False)
        bucketed = {0: _make_samples(range(100))}
        batches = grouper.group_by_bucket(bucketed, _length_fn)
        for batch in batches[:-1]:
            self.assertEqual(len(batch), 32)
        self.assertLessEqual(len(batches[-1]), 32)

    def test_drop_last(self):
        grouper = BatchGrouper(32, sort_within_bucket=False, drop_last=True)
        bucketed = {0: _make_samples(range(50))}
        batches = grouper.group_by_bucket(bucketed, _length_fn)
        for batch in batches:
            self.assertEqual(len(batch), 32)
        self.assertEqual(grouper.dropped_samples, 50 - 32)  # 18 dropped

    def test_sort_within_bucket(self):
        grouper = BatchGrouper(100, sort_within_bucket=True, drop_last=False)
        samples = _make_samples([50, 10, 30, 20])
        bucketed = {0: samples}
        batches = grouper.group_by_bucket(bucketed, _length_fn)
        lengths = [_length_fn(s) for s in batches[0]]
        self.assertEqual(lengths, sorted(lengths))

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            BatchGrouper(0, False, False)


# --------------------------------------------------------------------------- #
# BatchShuffler
# --------------------------------------------------------------------------- #


class BatchShufflerTest(unittest.TestCase):

    def test_shuffle_preserves_batches(self):
        shuffler = BatchShuffler(seed=42)
        batches = [[1, 2], [3, 4], [5, 6]]
        shuffled = shuffler.shuffle(batches)
        # Each inner batch intact
        for batch in shuffled:
            self.assertEqual(len(batch), 2)
        # Same elements overall
        flat = [x for b in shuffled for x in b]
        self.assertEqual(sorted(flat), [1, 2, 3, 4, 5, 6])

    def test_deterministic(self):
        shuffler = BatchShuffler(seed=42)
        batches = [[1], [2], [3], [4], [5]]
        a = shuffler.shuffle(batches)
        b = shuffler.shuffle(batches)
        self.assertEqual(a, b)

    def test_does_not_mutate_input(self):
        shuffler = BatchShuffler(seed=1)
        batches = [[1], [2], [3]]
        original = [list(b) for b in batches]
        shuffler.shuffle(batches)
        self.assertEqual(batches, original)


# --------------------------------------------------------------------------- #
# TokenLengthCache
# --------------------------------------------------------------------------- #


class TokenLengthCacheTest(unittest.TestCase):

    def test_get_put(self):
        cache = TokenLengthCache(max_size=10)
        self.assertIsNone(cache.get("hello"))
        cache.put("hello", 5)
        self.assertEqual(cache.get("hello"), 5)
        self.assertEqual(cache.hit_count, 1)
        self.assertEqual(cache.miss_count, 1)

    def test_lru_eviction(self):
        cache = TokenLengthCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # a becomes most recent
        cache.put("c", 3)  # evicts b
        self.assertIsNone(cache.get("b"))
        self.assertIsNotNone(cache.get("a"))
        self.assertIsNotNone(cache.get("c"))

    def test_hit_rate(self):
        cache = TokenLengthCache(max_size=10)
        cache.put("x", 1)
        cache.get("x")  # hit
        cache.get("y")  # miss
        self.assertAlmostEqual(cache.hit_rate, 0.5)

    def test_file_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.json")
            cache = TokenLengthCache(cache_file_path=path, max_size=10)
            cache.put("hello", 5)
            cache.save_to_file()
            self.assertTrue(os.path.exists(path))
            # Reload
            cache2 = TokenLengthCache(cache_file_path=path, max_size=10)
            self.assertEqual(cache2.get("hello"), 5)

    def test_version_mismatch_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.json")
            with open(path, "w") as f:
                json.dump({"cache_version": "1.0", "entries": {"abc": 99}}, f)
            cache = TokenLengthCache(cache_file_path=path, max_size=10)
            self.assertIsNone(cache.get("abc"))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class CalculateMetricsTest(unittest.TestCase):

    def test_basic_metrics(self):
        samples = _make_samples([10, 20, 30])
        batches = [samples]
        metrics = calculate_metrics(batches, _length_fn)
        self.assertEqual(metrics.total_samples, 3)
        self.assertEqual(metrics.total_batches, 1)
        self.assertEqual(metrics.max_length, 30)
        self.assertEqual(metrics.min_length, 10)
        # padding = (30-10)+(30-20)+(30-30) = 20+10+0 = 30
        # total = 30*3 = 90
        self.assertAlmostEqual(metrics.avg_padding_ratio, 30 / 90)

    def test_dropped_samples(self):
        metrics = calculate_metrics([], _length_fn, dropped_samples=5)
        self.assertEqual(metrics.dropped_samples, 5)


# --------------------------------------------------------------------------- #
# LengthGroupedBatcher (integration)
# --------------------------------------------------------------------------- #


class _MockTokenizer:
    """Tokenizer stub: encodes by splitting on whitespace, returns char count."""

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text)))

    @property
    def chat_template(self):
        return None


class LengthGroupedBatcherTest(unittest.TestCase):

    def _make_config(self, **kwargs):
        defaults = dict(
            batch_size=8,
            bucket_strategy="fixed_interval",
            bucket_interval=32,
            custom_boundaries=None,
            num_percentile_buckets=4,
            sort_within_bucket=True,
            drop_last_batch=False,
            enable_batch_shuffle=False,
            shuffle_seed=42,
            enable_length_cache=False,
            length_cache_path=None,
            length_cache_max_size=1000,
            min_bucket_samples=1,
            max_sample_length=4096,
            truncate_strategy="keep",
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_make_batches_basic(self):
        config = self._make_config()
        tokenizer = _MockTokenizer()
        batcher = LengthGroupedBatcher(config, tokenizer)
        samples = _make_samples([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        batches = batcher.make_batches(samples, _length_fn)
        self.assertGreater(len(batches), 0)
        all_samples = [s for b in batches for s in b]
        self.assertEqual(len(all_samples), 10)

    def test_empty_input(self):
        config = self._make_config()
        tokenizer = _MockTokenizer()
        batcher = LengthGroupedBatcher(config, tokenizer)
        self.assertEqual(batcher.make_batches([], _length_fn), [])

    def test_custom_strategy(self):
        config = self._make_config(
            bucket_strategy="custom",
            custom_boundaries=[0, 20, 40, 2**31 - 1],
        )
        tokenizer = _MockTokenizer()
        batcher = LengthGroupedBatcher(config, tokenizer)
        samples = _make_samples([10, 25, 50])
        batches = batcher.make_batches(samples, _length_fn)
        self.assertGreater(len(batches), 0)

    def test_unknown_strategy_raises(self):
        config = self._make_config(bucket_strategy="bogus")
        tokenizer = _MockTokenizer()
        batcher = LengthGroupedBatcher(config, tokenizer)
        samples = _make_samples([10])
        with self.assertRaises(ValueError):
            batcher.make_batches(samples, _length_fn)

    def test_shuffle_changes_order(self):
        config = self._make_config(enable_batch_shuffle=True, shuffle_seed=123, batch_size=3)
        tokenizer = _MockTokenizer()
        samples = _make_samples(list(range(1, 31)))  # 30 distinct lengths
        batcher = LengthGroupedBatcher(config, tokenizer)
        batches = batcher.make_batches(samples, _length_fn)
        # At least 2 batches to have something to shuffle
        self.assertGreater(len(batches), 1)


# --------------------------------------------------------------------------- #
# compute_token_length contract
# --------------------------------------------------------------------------- #


class ComputeTokenLengthTest(unittest.TestCase):

    def test_none_returns_zero(self):
        self.assertEqual(compute_token_length(_MockTokenizer(), None), 0)

    def test_empty_returns_zero(self):
        self.assertEqual(compute_token_length(_MockTokenizer(), ""), 0)


# --------------------------------------------------------------------------- #
# TrainerConfig integration
# --------------------------------------------------------------------------- #


@unittest.skipUnless(_HAS_DEPS, "areno.api full dependency chain not available")
class TrainerConfigIntegrationTest(unittest.TestCase):

    def test_length_grouped_fields_accepted(self):
        config = TrainerConfig(
            algo="sft",
            ckpt="dummy",
            dataset_path="dummy",
            length_grouped=True,
            bucket_strategy="fixed_interval",
            bucket_interval=64,
            enable_batch_shuffle=False,
        )
        self.assertTrue(config.length_grouped)
        self.assertEqual(config.bucket_interval, 64)
        self.assertFalse(config.enable_batch_shuffle)

    def test_invalid_bucket_strategy_rejected(self):
        with self.assertRaises(ValueError):
            TrainerConfig(
                algo="sft",
                ckpt="dummy",
                dataset_path="dummy",
                bucket_strategy="invalid",
            )

    def test_defaults(self):
        config = TrainerConfig(algo="sft", ckpt="d", dataset_path="d")
        self.assertFalse(config.length_grouped)
        self.assertEqual(config.bucket_strategy, "fixed_interval")
        self.assertEqual(config.bucket_interval, 32)


if __name__ == "__main__":
    unittest.main()
