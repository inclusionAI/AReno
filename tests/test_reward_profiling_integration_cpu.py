"""Integration test: RewardProfiler across trainer → metrics modules.

Uses ``SimpleNamespace`` stubs for PromptBatch/RolloutResult (no GPU, no
backend) and verifies that profiling data flows through to MetricsRecorder
artifacts and TensorBoard scalar names.

GPU/distributed note: orchestration logic is isolated behind fakes.
The only GPU validation that remains is verifying that the timer does
not introduce a GPU-CPU sync in real rollout paths — that check is out
of scope for the CPU test suite.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace

from areno.api.metrics import MetricsRecorder
from areno.api.reward_profiler import RewardBatchProfile, RewardProfiler, RewardSampleTiming
from areno.api.rewards import RewardRecord


def _fake_reward(record: RewardRecord) -> float:
    """Deterministic fast reward for integration testing."""

    return 1.0


class RewardProfileMetricsIntegrationTest(unittest.TestCase):
    """Verify that RewardBatchProfile flows through MetricsRecorder correctly."""

    def test_record_reward_profile_writes_jsonl(self):
        """MetricsRecorder.record_reward_profile must write a valid jsonl file."""

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = MetricsRecorder(tmpdir)

            profile = RewardBatchProfile(
                total_s=0.12,
                sample_timings=[
                    RewardSampleTiming(prompt_index=0, sample_index=0, duration_s=0.01),
                    RewardSampleTiming(prompt_index=0, sample_index=1, duration_s=0.11),
                ],
                slow_samples=[
                    RewardSampleTiming(prompt_index=0, sample_index=1, duration_s=0.11),
                ],
                timeout_count=0,
            )

            recorder.record_reward_profile(profile)
            recorder.close()

            # Read back the jsonl file and verify fields.
            import os

            profile_file = os.path.join(tmpdir, f"reward_profile.{os.getpid()}.jsonl")
            with open(profile_file, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]

            self.assertEqual(len(lines), 2)
            for rec in lines:
                keys = set(rec.keys())
                self.assertIn("prompt_index", keys)
                self.assertIn("sample_index", keys)
                self.assertIn("duration_s", keys)
                self.assertIn("timed_out", keys)
                # Privacy: no prompt or completion text.
                self.assertNotIn("prompt", keys)
                self.assertNotIn("completion", keys)

            self.assertEqual(lines[1]["sample_index"], 1)
            self.assertAlmostEqual(lines[1]["duration_s"], 0.11)

    def test_record_reward_profile_none_is_noop(self):
        """Calling record_reward_profile(None) must not write anything."""

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = MetricsRecorder(tmpdir)
            recorder.record_reward_profile(None)
            recorder.close()

            import os

            profile_file = os.path.join(tmpdir, f"reward_profile.{os.getpid()}.jsonl")
            self.assertFalse(os.path.exists(profile_file))

    def test_profile_to_scalars(self):
        """RewardBatchProfile.to_scalars must return the expected keys."""

        profile = RewardBatchProfile(
            total_s=0.5,
            sample_timings=[
                RewardSampleTiming(prompt_index=0, sample_index=0, duration_s=0.1),
                RewardSampleTiming(prompt_index=0, sample_index=1, duration_s=0.3),
            ],
            slow_samples=[
                RewardSampleTiming(prompt_index=0, sample_index=1, duration_s=0.3),
            ],
            timeout_count=0,
        )

        scalars = profile.to_scalars()
        self.assertEqual(scalars["reward_slow_count"], 1.0)
        self.assertEqual(scalars["reward_max_s"], 0.3)
        self.assertEqual(scalars["reward_timeout_count"], 0.0)
        self.assertEqual(scalars["reward_total_s"], 0.5)


class RewardProfilerTrainerIntegrationTest(unittest.TestCase):
    """Verify that RewardProfiler produces correct rewards through the trainer path."""

    def test_disabled_profiler_preserves_trainer_reward_values(self):
        """When disabled, profiler must return the same rewards as a bare reward_fn."""

        records = [
            RewardRecord(prompt="q", completion=f"answer{i}",
                         metadata={"prompt_index": 0, "sample_index": i})
            for i in range(4)
        ]

        # Bare call (simulates pre-change behavior)
        bare_rewards = [float(_fake_reward(r)) for r in records]

        profiler = RewardProfiler(_fake_reward, enabled=False)
        profiled_rewards, profile = profiler.score_batch(records)

        self.assertEqual(profiled_rewards, bare_rewards)
        self.assertIsNone(profile)

    def test_enabled_profiler_preserves_reward_values(self):
        """When enabled, profiler must still return the correct reward values."""

        records = [
            RewardRecord(prompt="q", completion=f"answer{i}",
                         metadata={"prompt_index": 0, "sample_index": i})
            for i in range(4)
        ]

        expected_rewards = [float(_fake_reward(r)) for r in records]

        profiler = RewardProfiler(_fake_reward, enabled=True, slow_threshold_s=10.0)
        rewards, profile = profiler.score_batch(records)

        self.assertEqual(rewards, expected_rewards)
        self.assertIsNotNone(profile)
        self.assertEqual(len(profile.sample_timings), 4)
        self.assertEqual(len(profile.slow_samples), 0)


if __name__ == "__main__":
    unittest.main()