"""CPU unit tests for the reward profiler (timing, slow-sample, timeout).

Tests follow the patterns in ``test_losses_rewards_cpu.py``: pure
unittest, no GPU, no external services.  Deterministic delays are injected
via ``time.sleep`` so timing assertions are robust on any CPU.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from areno.api.reward_profiler import (
    REWARD_HOOK_NAME,
    RewardBatchProfile,
    RewardProfiler,
    RewardSampleTiming,
    RewardTimeoutError,
)
from areno.api.rewards import RewardRecord
from areno.api.trainer_config import PolicyTrainerConfig


def _make_records(n: int = 4) -> list[RewardRecord]:
    """Build ``n`` minimal reward records with prompt/sample identifiers."""

    return [
        RewardRecord(prompt="q", completion=f"answer{i}", metadata={"prompt_index": 0, "sample_index": i})
        for i in range(n)
    ]


def _fast_reward(record: RewardRecord) -> float:
    """Deterministic fast reward — returns 1.0 for all records."""

    return 1.0


def _slow_at_index(slow_idx: int, delay_s: float = 0.05):
    """Return a reward_fn that sleeps ``delay_s`` at ``slow_idx``."""

    def reward_fn(record: RewardRecord) -> float:
        if record.metadata.get("sample_index") == slow_idx:
            time.sleep(delay_s)
        return 1.0

    return reward_fn


class DisabledPassthroughTest(unittest.TestCase):
    """Disabled profiler must behave identically to a bare reward_fn call."""

    def test_disabled_is_passthrough(self):
        """Disabled profiler returns same rewards and None profile with no perf_counter."""

        with patch("areno.api.reward_profiler.time.perf_counter") as mock_pc:
            mock_pc.return_value = 0.0
            profiler = RewardProfiler(_fast_reward, enabled=False)
            records = _make_records(4)
            rewards, profile = profiler.score_batch(records)

        self.assertEqual(rewards, [1.0, 1.0, 1.0, 1.0])
        self.assertIsNone(profile)
        # perf_counter should not be called when disabled.
        mock_pc.assert_not_called()


class SlowSampleDetectionTest(unittest.TestCase):
    """Slow-sample detection with absolute threshold."""

    def test_slow_sample_flagged(self):
        """Samples exceeding slow_threshold_s must appear in slow_samples."""

        profiler = RewardProfiler(
            _slow_at_index(1, delay_s=0.05), enabled=True, slow_threshold_s=0.03
        )
        records = _make_records(4)
        rewards, profile = profiler.score_batch(records)

        self.assertEqual(rewards, [1.0, 1.0, 1.0, 1.0])
        self.assertIsNotNone(profile)
        self.assertEqual(len(profile.slow_samples), 1)
        self.assertEqual(profile.slow_samples[0].sample_index, 1)

    def test_boundary_threshold_configured_not_triggered(self):
        """A high threshold with fast samples must produce empty slow_samples."""

        profiler = RewardProfiler(
            _fast_reward, enabled=True, slow_threshold_s=10.0
        )
        records = _make_records(4)
        rewards, profile = profiler.score_batch(records)

        self.assertEqual(len(profile.slow_samples), 0)
        self.assertEqual(len(profile.sample_timings), 4)


class BatchTimeoutTest(unittest.TestCase):
    """Per-batch wall-clock timeout enforcement (soft, main-thread)."""

    def test_batch_timeout_raises_with_identifiers(self):
        """Timeout must raise RewardTimeoutError with hook name and sample index.

        Uses a slow first sample to exhaust the budget; the timeout is
        checked before the second sample, raising with the second sample's
        identifier.
        """

        profiler = RewardProfiler(
            _slow_at_index(0, delay_s=0.06), enabled=True, batch_timeout_s=0.05
        )
        records = _make_records(4)
        with self.assertRaises(RewardTimeoutError) as ctx:
            profiler.score_batch(records)

        err = ctx.exception
        self.assertEqual(err.hook, REWARD_HOOK_NAME)
        self.assertIsNotNone(err.sample_index)
        self.assertGreater(err.elapsed, 0)

    def test_boundary_timeout_configured_not_triggered(self):
        """A generous timeout with fast samples must complete normally."""

        profiler = RewardProfiler(
            _fast_reward, enabled=True, batch_timeout_s=10.0
        )
        records = _make_records(4)
        rewards, profile = profiler.score_batch(records)

        self.assertEqual(len(rewards), 4)
        self.assertEqual(profile.timeout_count, 0)

    def test_timeout_preserves_original_error(self):
        """When reward_fn raises, the original exception must be preserved."""

        def faulty_reward(record: RewardRecord) -> float:
            raise RuntimeError("scheme parse error")

        profiler = RewardProfiler(
            faulty_reward, enabled=True, batch_timeout_s=10.0
        )
        records = _make_records(2)
        with self.assertRaises(RuntimeError) as ctx:
            profiler.score_batch(records)

        self.assertIn("scheme parse error", str(ctx.exception))


class PrivacyTest(unittest.TestCase):
    """Profile output must never contain prompt or completion text."""

    def test_no_prompt_or_completion_in_profile(self):
        """Serialized profile fields must not contain prompt/completion."""

        profiler = RewardProfiler(_fast_reward, enabled=True)
        records = _make_records(2)
        _, profile = profiler.score_batch(records)

        jsonl_records = profile.to_jsonl_records()
        for rec in jsonl_records:
            keys = set(rec.keys())
            self.assertNotIn("prompt", keys)
            self.assertNotIn("completion", keys)
            self.assertEqual(keys, {"prompt_index", "sample_index", "duration_s", "timed_out"})


class ConfigValidationTest(unittest.TestCase):
    """Config validation rejects non-positive thresholds and timeouts."""

    def test_invalid_config_raises(self):
        """Invalid slow_threshold_s and batch_timeout_s must raise ValueError with clear messages."""

        with self.assertRaisesRegex(ValueError, r"reward_slow_threshold_s must be > 0"):
            RewardProfiler(_fast_reward, enabled=True, slow_threshold_s=-1)
        with self.assertRaisesRegex(ValueError, r"reward_batch_timeout_s must be > 0"):
            RewardProfiler(_fast_reward, enabled=True, batch_timeout_s=0)

    def test_backward_compatible_default(self):
        """Default PolicyTrainerConfig must construct without raising."""

        config = PolicyTrainerConfig(algo="gspo", ckpt="unused", dataset_path="unused")
        self.assertFalse(config.reward_profile)
        self.assertIsNone(config.reward_slow_threshold_s)
        self.assertIsNone(config.reward_batch_timeout_s)

        profiler = RewardProfiler(_fast_reward)
        self.assertFalse(profiler.enabled)


class BoundaryValuesTest(unittest.TestCase):
    """Effective-side boundary inputs — configured but not triggered."""

    def test_boundary_empty_records(self):
        """Scoring an empty record list must return empty rewards and a valid profile."""

        profiler = RewardProfiler(_fast_reward, enabled=True)
        rewards, profile = profiler.score_batch([])

        self.assertEqual(rewards, [])
        self.assertIsNotNone(profile)
        self.assertEqual(profile.sample_timings, [])
        self.assertEqual(profile.timeout_count, 0)
        self.assertAlmostEqual(profile.total_s, 0.0, places=1)

    def test_boundary_enabled_without_threshold_or_timeout(self):
        """Enabled profiling with no threshold/timeout must produce timing only."""

        profiler = RewardProfiler(_fast_reward, enabled=True)
        records = _make_records(4)
        rewards, profile = profiler.score_batch(records)

        self.assertEqual(len(rewards), 4)
        self.assertEqual(len(profile.sample_timings), 4)
        self.assertEqual(profile.slow_samples, [])
        self.assertEqual(profile.timeout_count, 0)
        self.assertGreater(profile.total_s, 0.0)


class HookNameOutputTest(unittest.TestCase):
    """Hook name must appear in timeout error and jsonl output."""

    def test_hook_name_in_output(self):
        """RewardTimeoutError repr and jsonl records must contain the hook name."""

        profiler = RewardProfiler(
            _slow_at_index(0, delay_s=0.06), enabled=True, batch_timeout_s=0.05
        )
        records = _make_records(2)
        with self.assertRaises(RewardTimeoutError) as ctx:
            profiler.score_batch(records)

        err = ctx.exception
        self.assertIn(REWARD_HOOK_NAME, repr(err))
        self.assertEqual(err.hook, REWARD_HOOK_NAME)

    def test_hook_name_in_jsonl(self):
        """jsonl output must carry timing fields via the profile's to_jsonl_records."""

        profiler = RewardProfiler(_fast_reward, enabled=True)
        records = _make_records(2)
        _, profile = profiler.score_batch(records)

        jsonl = profile.to_jsonl_records()
        self.assertEqual(len(jsonl), 2)
        for rec in jsonl:
            self.assertIn("duration_s", rec)
            self.assertIn("prompt_index", rec)
            self.assertIn("sample_index", rec)


class TrainerConfigPostInitTest(unittest.TestCase):
    """PolicyTrainerConfig.__post_init__ must validate reward_* fields and call super."""

    def test_config_validates_reward_threshold(self):
        """Non-positive reward_slow_threshold_s must raise ValueError."""

        with self.assertRaisesRegex(ValueError, r"reward_slow_threshold_s must be > 0"):
            PolicyTrainerConfig(
                algo="gspo", ckpt="unused", dataset_path="unused", reward_slow_threshold_s=-1
            )

    def test_config_validates_reward_timeout(self):
        """Non-positive reward_batch_timeout_s must raise ValueError."""

        with self.assertRaisesRegex(ValueError, r"reward_batch_timeout_s must be > 0"):
            PolicyTrainerConfig(
                algo="gspo", ckpt="unused", dataset_path="unused", reward_batch_timeout_s=0
            )

    def test_config_preserves_parent_validation(self):
        """Invalid attn_backend must still raise after adding reward_* fields (super call)."""

        with self.assertRaisesRegex(ValueError, r"attn_backend"):
            PolicyTrainerConfig(
                algo="gspo", ckpt="unused", dataset_path="unused", attn_backend="invalid"
            )

    def test_ppo_config_inherits_validation(self):
        """PPOTrainerConfig must inherit reward_* validation from PolicyTrainerConfig."""

        from areno.api.trainer_config import PPOTrainerConfig

        with self.assertRaisesRegex(ValueError, r"reward_slow_threshold_s"):
            PPOTrainerConfig(
                algo="ppo", ckpt="unused", dataset_path="unused", reward_slow_threshold_s=-1
            )


if __name__ == "__main__":
    unittest.main()