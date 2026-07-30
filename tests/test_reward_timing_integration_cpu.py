"""Integration test for reward timing in the trainer loop (issue #242).

This test uses fakes to exercise the orchestration logic that wires
``TimedRewardFn`` into ``PolicyOnlyTrainer``, verifying that timing reports
are produced and dashboard state is recorded -- all on CPU without a GPU.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from areno.api.reward_timing import RewardTimingConfig, TimedRewardFn
from areno.api.rewards import RewardRecord


class TrainerTimingIntegrationTest(unittest.TestCase):
    """Verify PolicyOnlyTrainer wires TimedRewardFn and reports correctly."""

    def test_build_timed_reward_fn_disabled_by_default(self):
        """The default config should produce a transparent passthrough."""

        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        config = SimpleNamespace(reward_timing_enabled=False, reward_slow_threshold_s=None, reward_timeout_s=None)
        trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
        trainer.config = config

        def reward_fn(record):
            return 0.5

        timed = trainer._build_timed_reward_fn(reward_fn)
        self.assertIsInstance(timed, TimedRewardFn)
        self.assertFalse(timed.config.enabled)
        # The public reward_fn should be replaced.
        self.assertIs(trainer.reward_fn, timed)

    def test_build_timed_reward_fn_enabled(self):
        """When enabled, the wrapper should carry the config settings."""

        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        config = SimpleNamespace(
            reward_timing_enabled=True,
            reward_slow_threshold_s=0.1,
            reward_timeout_s=5.0,
        )
        trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
        trainer.config = config

        timed = trainer._build_timed_reward_fn(lambda r: 1.0)
        self.assertTrue(timed.config.enabled)
        self.assertEqual(timed.config.slow_threshold_s, 0.1)
        self.assertEqual(timed.config.timeout_s, 5.0)

    def test_finalize_reward_timing_records_dashboard_state(self):
        """_finalize_reward_timing should record dashboard state when enabled."""

        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        config = SimpleNamespace(reward_timing_enabled=True, reward_slow_threshold_s=None, reward_timeout_s=None)
        trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
        trainer.config = config
        trainer.areno = MagicMock()

        # Build a real timed fn and prime it with one sample.
        timed = trainer._build_timed_reward_fn(lambda r: 1.0)
        timed(RewardRecord(prompt="x", completion="y", metadata={"prompt_index": 0, "sample_index": 0}))
        trainer._finalize_reward_timing(7)

        # Dashboard state should have been recorded with timing data.
        trainer.areno.record_dashboard_state.assert_called_once()
        call_kwargs = trainer.areno.record_dashboard_state.call_args
        self.assertEqual(call_kwargs.kwargs["stage"], "reward_timing")
        self.assertEqual(call_kwargs.kwargs["step"], 7)
        timing_dict = call_kwargs.kwargs["extra"]["reward_timing"]
        self.assertEqual(timing_dict["num_samples"], 1)
        self.assertEqual(timing_dict["hook_name"], "reward_fn")

    def test_finalize_reward_timing_noop_when_disabled(self):
        """When disabled, _finalize_reward_timing should not record dashboard state."""

        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        config = SimpleNamespace(reward_timing_enabled=False, reward_slow_threshold_s=None, reward_timeout_s=None)
        trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
        trainer.config = config
        trainer.areno = MagicMock()

        timed = trainer._build_timed_reward_fn(lambda r: 1.0)
        timed(RewardRecord(prompt="x", completion="y"))
        trainer._finalize_reward_timing(0)

        trainer.areno.record_dashboard_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
