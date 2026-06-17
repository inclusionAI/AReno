from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

import areno.api.trainer as trainer_mod
from areno import Trainer
from areno.api.context import Context
from areno.api.models import SamplingParams
from areno.api.trainer import Trainer as ApiTrainer
from tests.helpers import PatchedContext


def _encode_from_record_prompt(_tokenizer, prompt: str) -> list[int]:
    """Deterministic tokenizer stub for prompt batch tests."""

    tokens_by_prompt = {
        "short": [1, 2],
        "long": [1, 2, 3, 4, 5],
        "next": [3],
        "a": [10],
        "b": [11],
        "c": [12],
    }
    return tokens_by_prompt[prompt]


class TrainerPromptBatchTest(unittest.TestCase):
    """Prompt batching tests avoid backend initialization and tokenizer loading."""

    def test_top_level_trainer_export_matches_api_trainer(self):
        self.assertIs(Trainer, ApiTrainer)

    def test_load_prompt_batches_skips_long_prompts_and_keeps_records(self):
        """Overlong prompts should be skipped without dropping record metadata."""
        trainer = Trainer(world_size=1, model_path="unused")
        trainer._tokenizer = object()
        dataset = [
            {"prompt": "long", "solutions": ["skip"], "answer": "x"},
            {"prompt": "short", "solutions": ["ok"], "answer": "2"},
            {"prompt": "next", "answer": "3"},
        ]

        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            batches = list(trainer.load_prompt_batches(dataset, batch_size=2, max_prompt_tokens=3))

        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(batch.prompts, ["short", "next"])
        self.assertEqual(batch.scanned, 3)
        self.assertEqual(batch.skipped_long, 1)
        self.assertEqual(batch.total_skipped_long, 1)
        self.assertEqual(batch.items[0].input_tokens, [1, 2])
        self.assertEqual(batch.items[0].solutions, ["ok"])
        self.assertIsNone(batch.items[1].solutions)
        self.assertEqual(batch.items[0].record, {"prompt": "short", "solutions": ["ok"], "answer": "2"})

    def test_load_prompt_batches_yields_partial_final_batch(self):
        """The final accepted rows should be yielded even if the batch is short."""
        trainer = Trainer(world_size=1, model_path="unused")
        trainer._tokenizer = object()
        dataset = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]

        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            batches = list(trainer.load_prompt_batches(dataset, batch_size=2, max_prompt_tokens=4))

        self.assertEqual([batch.prompts for batch in batches], [["a", "b"], ["c"]])
        self.assertEqual([batch.scanned for batch in batches], [2, 1])
        self.assertEqual([batch.skipped_long for batch in batches], [0, 0])

    def test_load_prompt_batches_stops_when_only_long_prompts_remain(self):
        """A tail containing only skipped rows should not emit an empty batch."""
        trainer = Trainer(world_size=1, model_path="unused")
        trainer._tokenizer = object()
        dataset = [{"prompt": "long"}]

        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_record_prompt):
            batches = list(trainer.load_prompt_batches(dataset, batch_size=1, max_prompt_tokens=3))

        self.assertEqual(batches, [])

    def test_load_prompt_batches_requires_prompt_field(self):
        """Online RL datasets should expose canonical prompt rows."""
        trainer = Trainer(world_size=1, model_path="unused")
        trainer._tokenizer = object()

        with self.assertRaisesRegex(ValueError, "`prompt`"):
            list(trainer.load_prompt_batches([{"question": "raw"}], batch_size=1, max_prompt_tokens=3))

    def test_rollout_token_batch_passes_pre_tokenized_prompts_to_backend(self):
        """RL trainers should reuse PromptItem.input_tokens instead of re-encoding."""

        class BackendStub:
            def __init__(self):
                self.prompt_tokens = None

            def begin_rollout_session(self, _ctx):
                return None

            def end_rollout_session(self, _ctx):
                return None

            async def begin_rollout_session_async(self, _ctx):
                self.begin_rollout_session(_ctx)

            async def sync_rollout_session_async(self, _ctx):
                self.synced = True

            async def end_rollout_session_async(self, _ctx):
                self.end_rollout_session(_ctx)

            def rollout_batch(self, _ctx, prompt_tokens, n_samples, _sampling_params):
                self.prompt_tokens = prompt_tokens
                self.n_samples = n_samples
                return []

        backend = BackendStub()
        trainer = Trainer(world_size=1, model_path="unused")
        trainer._backend = backend
        trainer._ctx = Context(1, "unused", object())

        async def run_rollout():
            async with trainer.rollout_session(sampling_params=SamplingParams(), proxy=False):
                return trainer.rollout_token_batch([SimpleNamespace(ids=[1, 2]), [3]], 4, SamplingParams())

        result = asyncio.run(run_rollout())

        self.assertEqual(result, [])
        self.assertEqual(backend.prompt_tokens, [[1, 2], [3]])
        self.assertEqual(backend.n_samples, 4)

    def test_rollout_session_sync_requires_active_session(self):
        """Agentic pre-rollout sync should be forwarded only inside rollout sessions."""

        class BackendStub:
            def __init__(self):
                self.synced = False

            async def begin_rollout_session_async(self, _ctx):
                return None

            async def sync_rollout_session_async(self, _ctx):
                self.synced = True

            async def end_rollout_session_async(self, _ctx):
                return None

        backend = BackendStub()
        trainer = Trainer(world_size=1, model_path="unused")
        trainer._backend = backend
        trainer._ctx = Context(1, "unused", object())

        with self.assertRaisesRegex(RuntimeError, "sync_rollout_session_async"):
            asyncio.run(trainer.sync_rollout_session_async())

        async def run_sync():
            async with trainer.rollout_session(sampling_params=SamplingParams(), proxy=False):
                await trainer.sync_rollout_session_async()

        asyncio.run(run_sync())
        self.assertTrue(backend.synced)

    def test_consecutive_rollouts_share_one_context_step_until_train(self):
        """The trainer, not the backend, owns step increments across rollout/train."""

        class BackendStub:
            def begin_rollout_session(self, _ctx):
                return None

            def end_rollout_session(self, _ctx):
                return None

            async def begin_rollout_session_async(self, _ctx):
                self.begin_rollout_session(_ctx)

            async def end_rollout_session_async(self, _ctx):
                self.end_rollout_session(_ctx)

            def rollout_batch(self, _ctx, _prompt_tokens, _n_samples, _sampling_params):
                return []

            def train(self, _ctx, _batch_data, _loss_fn, _mini_bs, _gradient_accumulation_steps):
                return {"loss": 0.0}

        trainer = Trainer(world_size=1, model_path="unused")
        trainer._backend = BackendStub()
        trainer._ctx = Context(1, "unused", object())

        async def run_rollout(prompt_tokens):
            async with trainer.rollout_session(sampling_params=SamplingParams(), proxy=False):
                trainer.rollout_token_batch(prompt_tokens, 1, SamplingParams())

        asyncio.run(run_rollout([[1]]))
        asyncio.run(run_rollout([[2]]))

        self.assertEqual(trainer._ctx.global_step, 0)
        trainer.train([], lambda _pack, _logprobs: None, mini_bs=1)
        asyncio.run(run_rollout([[3]]))
        self.assertEqual(trainer._ctx.global_step, 1)

    def test_rollout_token_batch_requires_explicit_session(self):
        """Rollout callers must own the rollout session lifecycle explicitly."""

        class BackendStub:
            def rollout_batch(self, _ctx, _prompt_tokens, _n_samples, _sampling_params):
                return []

        trainer = Trainer(world_size=1, model_path="unused")
        trainer._backend = BackendStub()
        trainer._ctx = Context(1, "unused", object())

        with self.assertRaisesRegex(RuntimeError, "rollout_session"):
            trainer.rollout_token_batch([[1]], 1, SamplingParams())

    def test_train_without_rollout_opens_context_step(self):
        """Train-only algorithms should still record their first update as step 0."""

        class BackendStub:
            def train(self, _ctx, _batch_data, _loss_fn, _mini_bs, _gradient_accumulation_steps):
                return {"loss": 0.0}

        trainer = Trainer(world_size=1, model_path="unused")
        trainer._backend = BackendStub()
        trainer._ctx = Context(1, "unused", object())

        trainer.train([], lambda _pack, _logprobs: None, mini_bs=1)

        self.assertEqual(trainer._ctx.global_step, 0)


if __name__ == "__main__":
    unittest.main()
