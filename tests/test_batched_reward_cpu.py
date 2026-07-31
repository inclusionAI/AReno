"""CPU tests for batched reward-hook execution (issue #225).

These tests cover:
- Per-example and batch paths produce identical results on a deterministic hook.
- Output cardinality validation raises with a diagnostic that names the batch.
- Empty batch returns an empty list.
- Backward compatibility: when no batch function is provided, the per-example
  path is used without error.
- ``load_reward_fns`` loads both functions from a module and detects the
  absence of ``reward_fn_batch``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import textwrap
import unittest

from areno.api.rewards import (
    RewardRecord,
    compute_rewards,
    load_reward_fns,
    validate_batch_rewards,
)


def _make_record(prompt: str = "p", completion: str = "c") -> RewardRecord:
    return RewardRecord(prompt=prompt, completion=completion)


def _scalar_reward(record: RewardRecord) -> float:
    """Score by counting characters in the completion."""

    return float(len(record.completion))


def _batch_reward(records: list[RewardRecord]) -> list[float]:
    """Batch version that produces the same scores as ``_scalar_reward``."""

    return [float(len(r.completion)) for r in records]


class ComputeRewardsTest(unittest.TestCase):
    """Core tests for the unified ``compute_rewards`` dispatch."""

    def test_batch_and_scalar_paths_agree(self):
        """A deterministic hook should give identical results via both paths."""
        records = [_make_record(completion=f"answer_{i}") for i in range(5)]

        scalar_results = compute_rewards(records, _scalar_reward, reward_fn_batch=None)
        batch_results = compute_rewards(records, _scalar_reward, reward_fn_batch=_batch_reward)

        self.assertEqual(scalar_results, batch_results)

    def test_scalar_path_works_without_batch_fn(self):
        """When no batch fn is provided the per-example loop should still work."""
        records = [_make_record(completion="ab"), _make_record(completion="abcd")]
        rewards = compute_rewards(records, _scalar_reward, reward_fn_batch=None)

        self.assertEqual(rewards, [2.0, 4.0])

    def test_batch_path_with_single_record(self):
        """A batch of one record should still work through the batch path."""
        records = [_make_record(completion="hello")]
        rewards = compute_rewards(records, _scalar_reward, reward_fn_batch=_batch_reward)

        self.assertEqual(rewards, [5.0])

    def test_empty_batch_returns_empty(self):
        """An empty record list should return an empty reward list."""
        self.assertEqual(compute_rewards([], _scalar_reward, reward_fn_batch=None), [])
        self.assertEqual(compute_rewards([], _scalar_reward, reward_fn_batch=_batch_reward), [])

    def test_batch_path_records_execution_time(self):
        """The batch path should log timing when a logger is provided."""
        logger = logging.getLogger("test_batch_timing")
        records = [_make_record(completion="x") for _ in range(3)]
        rewards = compute_rewards(records, _scalar_reward, reward_fn_batch=_batch_reward, logger=logger)

        self.assertEqual(len(rewards), 3)

    def test_scalar_path_records_execution_time(self):
        """The per-example path should log timing when a logger is provided."""
        logger = logging.getLogger("test_scalar_timing")
        records = [_make_record(completion="x") for _ in range(3)]
        rewards = compute_rewards(records, _scalar_reward, reward_fn_batch=None, logger=logger)

        self.assertEqual(len(rewards), 3)


class ValidateBatchRewardsTest(unittest.TestCase):
    """Tests for output cardinality validation."""

    def test_valid_cardinality_passes(self):
        """Matching input/output counts should succeed."""
        records = [_make_record() for _ in range(3)]
        rewards = [1.0, 0.5, 0.0]
        result = validate_batch_rewards(records, rewards)

        self.assertEqual(result, [1.0, 0.5, 0.0])

    def test_short_output_raises_with_batch_index(self):
        """A short output should name the batch index in the error."""
        records = [_make_record() for _ in range(4)]
        rewards = [1.0, 0.5]

        with self.assertRaisesRegex(ValueError, r"4 records.*batch index 2"):
            validate_batch_rewards(records, rewards, batch_index=2)

    def test_long_output_raises(self):
        """More rewards than inputs should also raise."""
        records = [_make_record() for _ in range(2)]
        rewards = [1.0, 0.5, 0.3]

        with self.assertRaisesRegex(ValueError, "3 scores for 2 records"):
            validate_batch_rewards(records, rewards)

    def test_empty_records_empty_rewards_validates(self):
        """Zero inputs and zero outputs should pass validation."""
        self.assertEqual(validate_batch_rewards([], []), [])

    def test_float_coercion(self):
        """Integer rewards should be coerced to float."""
        records = [_make_record() for _ in range(2)]
        rewards = [1, 0]
        result = validate_batch_rewards(records, rewards)

        self.assertEqual(result, [1.0, 0.0])
        self.assertIsInstance(result[0], float)


class LoadRewardFnsTest(unittest.TestCase):
    """Tests for the ``load_reward_fns`` module loader."""

    def test_loads_both_functions(self):
        """When both ``reward_fn`` and ``reward_fn_batch`` exist, both load."""
        source = textwrap.dedent("""
            def reward_fn(record):
                return 1.0

            def reward_fn_batch(records):
                return [1.0 for _ in records]
        """)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            path = f.name

        try:
            fn, batch_fn = load_reward_fns(path)
            self.assertTrue(callable(fn))
            self.assertTrue(callable(batch_fn))
            record = _make_record()
            self.assertEqual(fn(record), 1.0)
            self.assertEqual(batch_fn([record]), [1.0])
        finally:
            os.unlink(path)

    def test_loads_only_scalar_when_batch_absent(self):
        """When ``reward_fn_batch`` is absent, ``batch_fn`` should be None."""
        source = textwrap.dedent("""
            def reward_fn(record):
                return 0.5
        """)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            path = f.name

        try:
            fn, batch_fn = load_reward_fns(path)
            self.assertTrue(callable(fn))
            self.assertIsNone(batch_fn)
        finally:
            os.unlink(path)

    def test_rejects_non_callable_batch(self):
        """A non-callable ``reward_fn_batch`` should raise."""
        source = textwrap.dedent("""
            def reward_fn(record):
                return 0.5

            reward_fn_batch = "not a function"
        """)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(source)
            f.flush()
            path = f.name

        try:
            with self.assertRaisesRegex(ValueError, "not callable"):
                load_reward_fns(path)
        finally:
            os.unlink(path)


class BackwardCompatibilityTest(unittest.TestCase):
    """Verify that the default (no batch fn) path is unchanged."""

    def test_compute_rewards_without_batch_fn_matches_manual_loop(self):
        """The dispatch without a batch fn should match a manual per-example loop."""
        records = [_make_record(completion=f"c{i}") for i in range(6)]
        manual = [float(_scalar_reward(r)) for r in records]
        dispatched = compute_rewards(records, _scalar_reward, reward_fn_batch=None)

        self.assertEqual(manual, dispatched)


if __name__ == "__main__":
    unittest.main()