from __future__ import annotations

import unittest

from areno.api.algorithms import describe_loss_fn, get_algorithm
from areno.api.data import PromptBatch, PromptItem
from areno.api.models import RolloutResult, RolloutSequence
from areno.experimental.async_policy.config import AsyncPolicyTrainerConfig
from areno.experimental.async_policy.losses import async_grpo_loss_fn
from areno.experimental.async_policy.trainer import AsyncPolicyTrainer


class FakeTokenizer:
    eos_token_id = 0

    def batch_decode(self, token_lists):
        return [" ".join(str(token) for token in tokens) for tokens in token_lists]


class _FakeRolloutSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class FakeInstance:
    """Stands in for `areno.Trainer` with canned rollout/train behavior.

    ``group_versions`` fixes the policy version stamped on each prompt group,
    which keeps staleness assertions deterministic regardless of how the
    producer and consumer threads interleave.
    """

    def __init__(self, *, group_versions, n_samples=2, train_version_step=1, rollout_error_at=None):
        self.group_versions = list(group_versions)
        self.n_samples = n_samples
        self.train_version_step = train_version_step
        self.rollout_error_at = rollout_error_at
        self.rollout_calls = 0
        self.train_calls: list[list] = []
        self.overlap_enabled = False
        self.closed = False
        self._pending_batches: list[tuple[list[PromptItem], list[int]]] = []
        self._version = 0

    def init(self):
        pass

    def close(self):
        self.closed = True

    def get_tokenizer(self):
        return FakeTokenizer()

    def get_processor(self):
        return FakeTokenizer()

    def enable_async_overlap(self):
        self.overlap_enabled = True

    def load_prompt_batches(self, dataset, *, batch_size, max_prompt_tokens):
        del dataset, max_prompt_tokens
        for start in range(0, len(self.group_versions), batch_size):
            versions = self.group_versions[start : start + batch_size]
            items = [
                PromptItem(
                    prompt=f"prompt-{start + offset}",
                    solutions=["42"],
                    input_tokens=[1, 2],
                    record={"prompt": f"prompt-{start + offset}", "solutions": ["42"]},
                )
                for offset in range(len(versions))
            ]
            self._pending_batches.append((items, versions))
            yield PromptBatch(items=items, scanned=len(items), skipped_long=0, total_skipped_long=0)

    def rollout_session(self, **kwargs):
        del kwargs
        return _FakeRolloutSession()

    async def rollout_token_batch_async(self, prompt_tokens, n_samples, sampling_params, prompt_features=None):
        del prompt_tokens, sampling_params, prompt_features
        self.rollout_calls += 1
        if self.rollout_error_at is not None and self.rollout_calls == self.rollout_error_at:
            raise RuntimeError("fake rollout failure")
        _, versions = self._pending_batches.pop(0)
        return [
            RolloutResult(
                sequences=[
                    RolloutSequence(resp_tokens=[3, 4], resp_logprobs=[-0.1, -0.2]) for _ in range(n_samples)
                ],
                # Decoy adapter version: staleness must stamp from
                # `policy_version`, so a regression trips the drop tests.
                adapter_version=10_000 + version,
                policy_version=version,
            )
            for version in versions
        ]

    def train(self, batch_data, loss_fn, mini_bs=8, gradient_accumulation_steps=None):
        del loss_fn, mini_bs, gradient_accumulation_steps
        self.train_calls.append(list(batch_data))
        self._version += self.train_version_step
        return {"adapter_version": self._version, "loss": 0.0}


def make_config(**overrides) -> AsyncPolicyTrainerConfig:
    base = dict(
        algo="async-grpo",
        ckpt="fake-model",
        dataset_path="fake-dataset",
        backend="cuda",
        epochs=1,
        batch_size=2,
        mini_bs=2,
        n_samples=2,
        world_size=2,
        tp_size=1,
        rollout_devices=[0],
        train_devices=[1, 2],
        queue_max_groups=8,
        queue_get_timeout_s=30.0,
    )
    base.update(overrides)
    return AsyncPolicyTrainerConfig(**base)


def _constant_reward(record):
    del record
    return 1.0


class AsyncPolicyTrainerTest(unittest.TestCase):
    """Pipeline tests cover overlap wiring, staleness control, and shutdown."""

    def _run_fit(self, instance: FakeInstance, config: AsyncPolicyTrainerConfig) -> AsyncPolicyTrainer:
        trainer = AsyncPolicyTrainer(
            config,
            instance=instance,
            dataset=object(),
            reward_fn=_constant_reward,
            loss_fn=async_grpo_loss_fn,
        )
        trainer.fit()
        return trainer

    def test_fit_trains_all_groups_and_enables_overlap(self):
        """Every produced group trains exactly once and versions advance."""
        instance = FakeInstance(group_versions=[0, 0, 1, 1])
        trainer = self._run_fit(instance, make_config())

        self.assertTrue(instance.overlap_enabled)
        self.assertTrue(instance.closed)
        self.assertEqual(instance.rollout_calls, 2)
        self.assertEqual(len(instance.train_calls), 2)
        self.assertEqual(sum(len(call) for call in instance.train_calls), 8)
        self.assertEqual(trainer._current_policy_version, 2)
        self.assertEqual(trainer._stale_dropped, 0)

    def test_stale_groups_are_dropped(self):
        """Groups lagging more than max_staleness policy versions are dropped."""
        instance = FakeInstance(group_versions=[0, 0, 1, 1])
        trainer = self._run_fit(instance, make_config(batch_size=1, max_staleness=0))

        # g0 (v0, current 0) trains; g1 (v0, current 1) drops;
        # g2 (v1, current 1) trains; g3 (v1, current 2) drops.
        self.assertEqual(len(instance.train_calls), 2)
        self.assertEqual(sum(len(call) for call in instance.train_calls), 4)
        self.assertEqual(trainer._stale_dropped, 2)
        self.assertEqual(trainer._current_policy_version, 2)

    def test_stale_policy_error_aborts(self):
        """stale_policy='error' turns staleness into a hard failure."""
        instance = FakeInstance(group_versions=[0, 0, 1, 1])
        config = make_config(batch_size=1, max_staleness=0, stale_policy="error")
        with self.assertRaisesRegex(RuntimeError, "exceeding max_staleness"):
            self._run_fit(instance, config)

    def test_producer_error_propagates(self):
        """A rollout failure surfaces to the caller of fit()."""
        instance = FakeInstance(group_versions=[0, 0], rollout_error_at=1)
        with self.assertRaisesRegex(RuntimeError, "fake rollout failure"):
            self._run_fit(instance, make_config())
        self.assertTrue(instance.closed)

    def test_max_steps_stops_training(self):
        """max_steps bounds optimizer updates and shuts the producer down."""
        instance = FakeInstance(group_versions=[0, 0, 1, 1])
        trainer = self._run_fit(instance, make_config(batch_size=1, max_steps=1, queue_max_groups=1))

        self.assertEqual(len(instance.train_calls), 1)
        self.assertTrue(instance.closed)
        self.assertEqual(trainer._current_policy_version, 1)


class AsyncAlgorithmRegistrationTest(unittest.TestCase):
    """Registry tests cover experimental algorithm discovery and loss binding."""

    def test_async_algorithms_are_registered_as_experimental(self):
        for name in ("async-grpo", "async-gspo"):
            spec = get_algorithm(name)
            self.assertTrue(spec.experimental, name)
            self.assertTrue(spec.requires_rollout, name)
            self.assertIs(spec.resolve_trainer_cls(), AsyncPolicyTrainer, name)

    def test_async_grpo_loss_binding_round_trips(self):
        spec = get_algorithm("async-grpo")
        loss = spec.make_loss_fn(make_config(grpo_clip_eps=0.3))
        described = describe_loss_fn(loss)
        self.assertEqual(described.name, "async-grpo")
        self.assertEqual(described.kwargs, {"clip_eps": 0.3})

    def test_async_gspo_loss_binding_round_trips(self):
        spec = get_algorithm("async-gspo")
        loss = spec.make_loss_fn(make_config(gspo_clip_eps=1.0e-3))
        described = describe_loss_fn(loss)
        self.assertEqual(described.name, "async-gspo")
        self.assertEqual(described.kwargs, {"clip_eps": 1.0e-3})

    def test_config_requires_separate_rollout_engine(self):
        with self.assertRaisesRegex(ValueError, "rollout_devices"):
            make_config(rollout_devices=None)


if __name__ == "__main__":
    unittest.main()
