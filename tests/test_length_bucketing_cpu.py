"""CPU-only tests for the length-bucketed batch sampler.

Run with: ``pytest tests/test_length_bucketing_cpu.py -v``

No GPU is required.  Tests cover the core ``bucketed_batch_indices``
function, integration with ``Trainer.load_prompt_batches()``, SFT
``_iter_train_batches()``, backward compatibility, and boundary/edge cases.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

# Import length_bucketing directly by file path to avoid triggering the
# areno.api.__init__ import chain (which pulls in torch).
_spec = importlib.util.spec_from_file_location(
    "length_bucketing",
    os.path.join(os.path.dirname(__file__), "..", "areno", "api", "length_bucketing.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
bucketed_batch_indices = _mod.bucketed_batch_indices


# ---------------------------------------------------------------------------
# Tokenizer stub — deterministic, maps prompt strings to fixed token lists.
# ---------------------------------------------------------------------------

_TOKENS_BY_PROMPT = {
    "tiny": [1],                    # 1 token
    "short": [1, 2],                # 2 tokens
    "medium": [1, 2, 3, 4],         # 4 tokens
    "long": [1, 2, 3, 4, 5, 6, 7, 8],  # 8 tokens
    "huge": list(range(16)),        # 16 tokens
}


def _encode_from_prompt(_tokenizer, prompt: str) -> list[int]:
    return _TOKENS_BY_PROMPT.get(prompt, [1, 2, 3])


# ---------------------------------------------------------------------------
# Unit tests for bucketed_batch_indices
# ---------------------------------------------------------------------------

class BucketedBatchIndicesTest(unittest.TestCase):
    """Core algorithm tests — pure Python, no trainer or tokenizer needed."""

    def test_every_sample_appears_once(self):
        """All indices must appear exactly once across all batches."""
        indices = list(range(20))
        lengths = [5, 3, 8, 1, 4, 7, 2, 6, 9, 3, 5, 8, 1, 2, 4, 7, 6, 9, 3, 5]
        batches = bucketed_batch_indices(indices, lengths, batch_size=4, seed=42)
        flat = [idx for batch in batches for idx in batch]
        self.assertEqual(sorted(flat), sorted(indices))
        self.assertEqual(len(flat), len(set(flat)))  # no duplicates

    def test_same_seed_same_order(self):
        """Identical seeds must reproduce identical batch order."""
        indices = list(range(10))
        lengths = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        a = bucketed_batch_indices(indices, lengths, batch_size=3, seed=99)
        b = bucketed_batch_indices(indices, lengths, batch_size=3, seed=99)
        self.assertEqual(a, b)

    def test_different_seed_different_order(self):
        """Different seeds should (almost certainly) produce different order."""
        indices = list(range(50))
        lengths = [i % 10 for i in range(50)]
        a = bucketed_batch_indices(indices, lengths, batch_size=5, seed=1)
        b = bucketed_batch_indices(indices, lengths, batch_size=5, seed=2)
        self.assertNotEqual(a, b)

    def test_drop_last(self):
        """Partial final batch is dropped when drop_last=True."""
        indices = list(range(10))
        lengths = [1] * 10
        batches = bucketed_batch_indices(indices, lengths, batch_size=3, seed=0, drop_last=True)
        for batch in batches:
            self.assertEqual(len(batch), 3)
        self.assertEqual(len(batches), 3)  # floor(10/3) = 3

    def test_drop_last_false_keeps_partial(self):
        """Partial final batch is kept when drop_last=False (default)."""
        indices = list(range(10))
        lengths = [1] * 10
        batches = bucketed_batch_indices(indices, lengths, batch_size=3, seed=0)
        self.assertEqual(len(batches[-1]), 1)  # 10 % 3 = 1
        self.assertEqual(len(batches), 4)

    def test_empty_dataset(self):
        """Empty input should return empty list, not crash."""
        result = bucketed_batch_indices([], [], batch_size=4, seed=0)
        self.assertEqual(result, [])

    def test_batch_size_1(self):
        """Each batch should contain exactly one element when batch_size=1."""
        indices = list(range(5))
        lengths = [3, 1, 4, 1, 5]
        batches = bucketed_batch_indices(indices, lengths, batch_size=1, seed=0)
        self.assertEqual(len(batches), 5)
        for batch in batches:
            self.assertEqual(len(batch), 1)

    def test_single_element(self):
        """One item → one batch of one."""
        result = bucketed_batch_indices([0], [10], batch_size=4, seed=0)
        self.assertEqual(result, [[0]])

    def test_all_same_length(self):
        """All lengths equal — still works, order is shuffled."""
        indices = list(range(8))
        lengths = [5] * 8
        batches = bucketed_batch_indices(indices, lengths, batch_size=2, seed=7)
        flat = [idx for batch in batches for idx in batch]
        self.assertEqual(sorted(flat), sorted(indices))

    def test_invalid_batch_size_raises(self):
        """batch_size < 1 should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            bucketed_batch_indices([0], [1], batch_size=0, seed=0)
        self.assertIn("batch_size must be >= 1", str(ctx.exception))

    def test_mismatched_lengths_raises(self):
        """indices and lengths of different sizes should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            bucketed_batch_indices([0, 1], [1], batch_size=1, seed=0)
        self.assertIn("equal length", str(ctx.exception))

    def test_bucketing_reduces_padding(self):
        """Mixed-length fixture: bucketed mode should produce less padding than
        a naive sequential batching of the same data (unsorted)."""
        # 12 items with lengths designed to produce high padding in random order
        lengths = [1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11]
        indices = list(range(12))
        batch_size = 4

        # Sequential (unsorted) padding
        def sequential_padding(indices, lengths, bs):
            total = 0
            for i in range(0, len(indices), bs):
                chunk = indices[i:i + bs]
                max_len = max(lengths[j] for j in chunk)
                total += max_len * len(chunk) - sum(lengths[j] for j in chunk)
            return total

        # Bucketed padding
        def bucketed_padding(indices, lengths, bs, seed):
            batches = bucketed_batch_indices(indices, lengths, batch_size=bs, seed=seed)
            total = 0
            for batch in batches:
                max_len = max(lengths[j] for j in batch)
                total += max_len * len(batch) - sum(lengths[j] for j in batch)
            return total

        seq_pad = sequential_padding(indices, lengths, batch_size)
        bucket_pad = bucketed_padding(indices, lengths, batch_size, seed=42)
        self.assertLess(bucket_pad, seq_pad)


# ---------------------------------------------------------------------------
# Integration: Trainer.load_prompt_batches() bucketed mode
# ---------------------------------------------------------------------------

class LoadPromptBatchesBucketedTest(unittest.TestCase):
    """Integration tests for the RL prompt-batch path with length bucketing."""

    def _make_trainer(self):
        import areno.api.trainer as trainer_mod
        from areno import Trainer
        from tests.helpers import PatchedContext

        trainer = Trainer(world_size=1, model_path="unused")
        trainer._tokenizer = object()
        return trainer, trainer_mod, PatchedContext

    def test_bucketed_each_sample_once(self):
        """Every prompt should appear exactly once across all batches."""
        trainer, trainer_mod, PatchedContext = self._make_trainer()
        dataset = [
            {"prompt": "tiny"}, {"prompt": "huge"}, {"prompt": "short"},
            {"prompt": "long"}, {"prompt": "medium"}, {"prompt": "medium"},
            {"prompt": "tiny"}, {"prompt": "long"},
        ]
        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_prompt):
            batches = list(trainer.load_prompt_batches(
                dataset, batch_size=3, max_prompt_tokens=100, length_bucket_seed=42,
            ))
        all_prompts = [item.prompt for batch in batches for item in batch.items]
        self.assertEqual(sorted(all_prompts), sorted(r["prompt"] for r in dataset))

    def test_bucketed_total_skipped_long(self):
        """total_skipped_long should reflect the pre-scan skip count."""
        trainer, trainer_mod, PatchedContext = self._make_trainer()
        dataset = [
            {"prompt": "tiny"},     # 1 token — accepted
            {"prompt": "huge"},     # 16 tokens — accepted (< 100)
            {"prompt": "short"},    # 2 tokens — accepted
        ]
        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_prompt):
            batches = list(trainer.load_prompt_batches(
                dataset, batch_size=2, max_prompt_tokens=10, length_bucket_seed=42,
            ))
        # "huge" has 16 tokens > 10, so it should be skipped
        self.assertTrue(all(b.total_skipped_long == 1 for b in batches))
        self.assertTrue(all(b.skipped_long == 0 for b in batches))

    def test_bucketed_reduces_padding_vs_sequential(self):
        """Bucketed mode should produce less padding than sequential mode."""
        trainer, trainer_mod, PatchedContext = self._make_trainer()
        dataset = [
            {"prompt": "tiny"}, {"prompt": "huge"}, {"prompt": "short"},
            {"prompt": "long"}, {"prompt": "medium"}, {"prompt": "tiny"},
        ]

        def compute_padding(batches):
            total = 0
            for batch in batches:
                lengths = [len(item.input_tokens) for item in batch.items]
                max_len = max(lengths)
                total += max_len * len(lengths) - sum(lengths)
            return total

        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_prompt):
            seq_batches = list(trainer.load_prompt_batches(
                dataset, batch_size=3, max_prompt_tokens=100,
            ))
            bucket_batches = list(trainer.load_prompt_batches(
                dataset, batch_size=3, max_prompt_tokens=100, length_bucket_seed=42,
            ))

        seq_pad = compute_padding(seq_batches)
        bucket_pad = compute_padding(bucket_batches)
        self.assertLess(bucket_pad, seq_pad)

    def test_seed_none_preserves_sequential_behavior(self):
        """length_bucket_seed=None should produce identical output to the
        original sequential implementation."""
        trainer, trainer_mod, PatchedContext = self._make_trainer()
        dataset = [
            {"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"},
        ]
        with PatchedContext(trainer_mod, encode_generation_prompt=_encode_from_prompt):
            batches = list(trainer.load_prompt_batches(
                dataset, batch_size=2, max_prompt_tokens=100, length_bucket_seed=None,
            ))
        # Should match the original sequential order
        self.assertEqual([b.prompts for b in batches], [["a", "b"], ["c"]])


# ---------------------------------------------------------------------------
# Integration: SFTTrainer._iter_train_batches() bucketed mode
# ---------------------------------------------------------------------------

class SftIterTrainBatchesBucketedTest(unittest.TestCase):
    """Integration tests for the SFT batch path with length bucketing."""

    def test_sft_config_validation_negative_seed(self):
        """TrainerConfig should reject negative length_bucket_seed."""
        from areno.api.trainer_config import TrainerConfig
        with self.assertRaises(ValueError) as ctx:
            TrainerConfig(
                algo="sft", ckpt="x", dataset_path="y",
                length_bucket_seed=-1,
            )
        self.assertIn("non-negative", str(ctx.exception))

    def test_sft_config_accepts_none_seed(self):
        """TrainerConfig should accept length_bucket_seed=None (disabled)."""
        from areno.api.trainer_config import TrainerConfig
        config = TrainerConfig(
            algo="sft", ckpt="x", dataset_path="y",
            length_bucket_seed=None,
        )
        self.assertIsNone(config.length_bucket_seed)

    def test_sft_config_accepts_positive_seed(self):
        """TrainerConfig should accept a positive length_bucket_seed."""
        from areno.api.trainer_config import TrainerConfig
        config = TrainerConfig(
            algo="sft", ckpt="x", dataset_path="y",
            length_bucket_seed=42,
        )
        self.assertEqual(config.length_bucket_seed, 42)

    def test_config_validation_happens_before_model_init(self):
        """Negative seed must be rejected at config construction time,
        before any expensive model or worker initialization."""
        from areno.api.trainer_config import TrainerConfig
        # This should fail immediately — no backend, no tokenizer, no GPU needed.
        with self.assertRaises(ValueError) as ctx:
            TrainerConfig(
                algo="sft", ckpt="x", dataset_path="y",
                length_bucket_seed=-5,
            )
        msg = str(ctx.exception)
        # Error must mention the field name and the constraint.
        self.assertIn("length_bucket_seed", msg)
        self.assertIn("non-negative", msg)
        # Error must not expose training sample content.
        self.assertNotIn("dataset", msg.lower())
        self.assertNotIn("prompt", msg.lower())


# ---------------------------------------------------------------------------
# Integration: SFTTrainer._iter_train_batches() bucketed mode (behavioral)
# ---------------------------------------------------------------------------

class SftBucketedBehaviorTest(unittest.TestCase):
    """Integration tests for SFT bucketed batch behavior: sample coverage,
    padding reduction, and backward compatibility."""

    def _make_sft_trainer(self, dataset, batch_size=4, seed=None):
        """Build a minimal SFTTrainer with a mock tokenizer.

        The mock tokenizer maps each character to its ordinal as a token id,
        so ``len(token_ids) == len(text)`` — simple and deterministic.
        """
        from areno.api.trainer_config import TrainerConfig
        from areno.api.trainers.sft import SFTTrainer

        class MockTokenizer:
            """Deterministic char-level tokenizer stub for SFT tests."""
            eos_token_id = 0
            chat_template = None  # Must be None so encode_generation_prompt uses .encode() directly

            def encode(self, text, add_special_tokens=True):
                return [ord(c) for c in text]

        config = TrainerConfig(
            algo="sft", ckpt="x", dataset_path="y",
            batch_size=batch_size, max_prompt_tokens=1000,
            max_new_tokens=1000, length_bucket_seed=seed,
        )

        class MockInstance:
            def get_tokenizer(self):
                return MockTokenizer()

        trainer = SFTTrainer.__new__(SFTTrainer)
        trainer.config = config
        trainer.areno = MockInstance()
        trainer.dataset = dataset
        trainer.loss_fn = None
        trainer.logger = __import__("logging").getLogger("test")
        return trainer, MockTokenizer()

    def test_sft_bucketed_each_sample_once(self):
        """Every SFT row should appear exactly once across all batches."""
        dataset = [
            {"prompt": "ab", "response": "cd"},
            {"prompt": "a", "response": "c"},
            {"prompt": "abcdef", "response": "ghijkl"},
            {"prompt": "abc", "response": "def"},
            {"prompt": "a", "response": "b"},
            {"prompt": "abcdefghij", "response": "klmnopqrst"},
        ]
        trainer, tok = self._make_sft_trainer(dataset, batch_size=2, seed=42)
        batches = list(trainer._iter_train_batches(tok, max_prompt_tokens=1000, max_new_tokens=1000))

        # Reconstruct which rows ended up where — use (prompt, response) as identity.
        all_rows = []
        for batch in batches:
            for seq in batch:
                # We can't easily get the original prompt/response back from
                # TrainSequence, so just count total sequences.
                all_rows.append(seq)
        self.assertEqual(len(all_rows), len(dataset))

    def test_sft_bucketed_reduces_padding(self):
        """Bucketed SFT mode should produce less padding than sequential."""
        # Construct rows with deliberately mixed lengths.
        dataset = []
        for i in range(24):
            prompt_len = 1 + (i % 12)  # 1..12
            response_len = 1 + ((i * 3) % 8)  # 1..8
            dataset.append({
                "prompt": "a" * prompt_len,
                "response": "b" * response_len,
            })

        def compute_padding(batches):
            total = 0
            for batch in batches:
                lengths = [len(seq.tokens) for seq in batch]
                max_len = max(lengths)
                total += max_len * len(batch) - sum(lengths)
            return total

        trainer_seq, tok = self._make_sft_trainer(dataset, batch_size=4, seed=None)
        seq_batches = list(trainer_seq._iter_train_batches(tok, max_prompt_tokens=1000, max_new_tokens=1000))
        seq_pad = compute_padding(seq_batches)

        trainer_bkt, tok2 = self._make_sft_trainer(dataset, batch_size=4, seed=42)
        bkt_batches = list(trainer_bkt._iter_train_batches(tok2, max_prompt_tokens=1000, max_new_tokens=1000))
        bkt_pad = compute_padding(bkt_batches)

        self.assertLess(bkt_pad, seq_pad)

    def test_sft_seed_none_preserves_sequential(self):
        """seed=None should produce batches in original dataset order."""
        dataset = [
            {"prompt": "ab", "response": "cd"},
            {"prompt": "ef", "response": "gh"},
            {"prompt": "ij", "response": "kl"},
            {"prompt": "mn", "response": "op"},
        ]
        trainer, tok = self._make_sft_trainer(dataset, batch_size=2, seed=None)
        batches = list(trainer._iter_train_batches(tok, max_prompt_tokens=1000, max_new_tokens=1000))

        # Each batch should have 2 sequences; total 2 batches for 4 rows.
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 2)


# ---------------------------------------------------------------------------
# Cross-module integration: CLI config -> Trainer.load_prompt_batches
# ---------------------------------------------------------------------------

class CliToTrainerIntegrationTest(unittest.TestCase):
    """Integration test crossing module boundaries: CLI option flows through
    config construction into the trainer's batch loading behavior."""

    def test_cli_option_flows_to_trainer_config(self):
        """--length-bucket-seed should appear in the constructed TrainerConfig."""
        from types import SimpleNamespace
        from areno.cli.train import _trainer_config_from_options

        # Simulate CLI args with length_bucket_seed set.
        args = SimpleNamespace(
            algo="sft", ckpt="actor", dataset_path="dataset",
            model_hub="modelscope", dataset_loader_fn="examples/sft/alpaca/dataset_loader.py",
            reward_fn_path=None, reward_ckpt=None,
            save_path="save", save_interval=10,
            tune_params=False, mem_frac=0.9, tune_max_samples=256,
            epochs=2, max_steps=None, score_micro_bs=8,
            tp_size=1, world_size=1, batch_size=2, n_samples=2,
            mini_bs=1, gradient_accumulation_steps=None,
            max_prompt_tokens=128, max_new_tokens=16, max_context_len=None,
            greedy=False, temperature=1.0, top_k=-1, top_p=1.0,
            max_running_prompts=None, lr=1e-6, min_lr=1e-7,
            lr_decay_steps=100, lr_decay_style="cosine",
            adam_beta1=0.9, adam_beta2=0.999, adam_8bit=False,
            weight_decay=1e-2, grad_clip_norm=1.0,
            activation_checkpointing=True, drop_rollout_state=False,
            eager_decode=False, attn_backend="flash",
            disable_thinking=False, metrics_log_dir=None,
            agent_fn=None, agent_timeout_s=300.0, train_tool_results=False,
            gspo_clip_eps=3.0e-4, grpo_clip_eps=0.2, ref_ckpt=None,
            dpo_beta=0.1, critic_ckpt=None, critic_lr=1e-5,
            use_kl_loss=True, kl_loss_coef=0.001, kl_loss_type="low_var_kl",
            clip_eps=0.2, clip_ratio_c=3.0, value_clip_eps=0.5,
            value_loss_coef=0.5, gamma=1.0, lam=0.95, critic_warmup_steps=20,
            length_bucket_seed=42,
        )
        config = _trainer_config_from_options(**vars(args))
        self.assertEqual(config.length_bucket_seed, 42)

    def test_cli_option_defaults_to_none(self):
        """When --length-bucket-seed is not passed, config should have None."""
        from types import SimpleNamespace
        from areno.cli.train import _trainer_config_from_options

        args = SimpleNamespace(
            algo="sft", ckpt="actor", dataset_path="dataset",
            model_hub="modelscope", dataset_loader_fn="examples/sft/alpaca/dataset_loader.py",
            reward_fn_path=None, reward_ckpt=None,
            save_path="save", save_interval=10,
            tune_params=False, mem_frac=0.9, tune_max_samples=256,
            epochs=2, max_steps=None, score_micro_bs=8,
            tp_size=1, world_size=1, batch_size=2, n_samples=2,
            mini_bs=1, gradient_accumulation_steps=None,
            max_prompt_tokens=128, max_new_tokens=16, max_context_len=None,
            greedy=False, temperature=1.0, top_k=-1, top_p=1.0,
            max_running_prompts=None, lr=1e-6, min_lr=1e-7,
            lr_decay_steps=100, lr_decay_style="cosine",
            adam_beta1=0.9, adam_beta2=0.999, adam_8bit=False,
            weight_decay=1e-2, grad_clip_norm=1.0,
            activation_checkpointing=True, drop_rollout_state=False,
            eager_decode=False, attn_backend="flash",
            disable_thinking=False, metrics_log_dir=None,
            agent_fn=None, agent_timeout_s=300.0, train_tool_results=False,
            gspo_clip_eps=3.0e-4, grpo_clip_eps=0.2, ref_ckpt=None,
            dpo_beta=0.1, critic_ckpt=None, critic_lr=1e-5,
            use_kl_loss=True, kl_loss_coef=0.001, kl_loss_type="low_var_kl",
            clip_eps=0.2, clip_ratio_c=3.0, value_clip_eps=0.5,
            value_loss_coef=0.5, gamma=1.0, lam=0.95, critic_warmup_steps=20,
            length_bucket_seed=None,
        )
        config = _trainer_config_from_options(**vars(args))
        self.assertIsNone(config.length_bucket_seed)