from __future__ import annotations

import unittest
from types import SimpleNamespace

from areno.api import data_utils
from areno.api.trainers import dpo as dpo_mod
from areno.api.trainers import sft as sft_mod


class FakeTextTokenizer:
    """Tokenizer double for offline SFT/DPO row conversion helpers."""

    eos_token_id = 99
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) % 50 + 1 for ch in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        ids = []
        for message in messages:
            ids.extend(self.encode(f"{message.get('role')}:{message.get('content')}"))
        return ids


class FakeSFTBackend:
    """Backend double that records whether SFT attempted to train."""

    def __init__(self):
        self.closed = False
        self.train_calls = 0

    def init(self):
        return None

    def close(self):
        self.closed = True

    def get_tokenizer(self):
        return FakeTextTokenizer()

    def train(self, _batch, _loss_fn, *, mini_bs, gradient_accumulation_steps):
        del mini_bs, gradient_accumulation_steps
        self.train_calls += 1
        return {}


def _sft_config(**overrides):
    """Return the minimal config shape SFTTrainer reads in CPU tests."""

    defaults = {
        "batch_size": 2,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "max_new_tokens": 2,
        "max_prompt_tokens": 2,
        "mini_bs": 1,
        "save_interval": 1,
        "save_path": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TrainerDatasetUtilityTest(unittest.TestCase):
    """Dataset helper tests convert rows to TrainSequence without backend init."""

    def test_sft_prompt_response_row_masks_only_response_suffix(self):
        """SFT loader rows should train on response tokens plus EOS."""
        tokenizer = FakeTextTokenizer()

        seq = sft_mod._record_to_train_sequence(
            {"prompt": "q", "response": "a"}, tokenizer, max_prompt_tokens=16, max_new_tokens=16
        )

        self.assertIsNotNone(seq)
        self.assertEqual(seq.eos_token_id, 99)
        self.assertEqual(seq.tokens[-1], 99)
        self.assertEqual(seq.prompt_mask.count(True), 1)
        self.assertTrue(any(not item for item in seq.prompt_mask[1:]))

    def test_sft_rejects_raw_rows_that_loader_did_not_normalize(self):
        """SFT requires the dataset loader to emit prompt/response rows."""
        tokenizer = FakeTextTokenizer()

        with self.assertRaisesRegex(ValueError, "dataset loader must return rows"):
            sft_mod._record_to_train_sequence({"text": "abc"}, tokenizer, max_prompt_tokens=16, max_new_tokens=16)

    def test_sft_drops_rows_without_target_tokens(self):
        """Rows that cannot produce a next-token target should be filtered out."""
        tokenizer = FakeTextTokenizer()

        seq = sft_mod._record_to_train_sequence(
            {"prompt": "a", "response": ""}, tokenizer, max_prompt_tokens=16, max_new_tokens=16
        )

        self.assertIsNone(seq)

    def test_sft_drops_rows_with_none_prompt_or_response(self):
        """None values should not be converted to the literal token string."""
        tokenizer = FakeTextTokenizer()

        none_prompt = sft_mod._record_to_train_sequence(
            {"prompt": None, "response": "answer"}, tokenizer, max_prompt_tokens=16, max_new_tokens=16
        )
        none_response = sft_mod._record_to_train_sequence(
            {"prompt": "question", "response": None}, tokenizer, max_prompt_tokens=16, max_new_tokens=16
        )

        self.assertIsNone(none_prompt)
        self.assertIsNone(none_response)

    def test_sft_enforces_prompt_and_response_budgets_independently(self):
        """SFT should reject over-budget prompts and responses separately."""
        tokenizer = FakeTextTokenizer()

        too_long_prompt = sft_mod._record_to_train_sequence(
            {"prompt": "abc", "response": "d"},
            tokenizer,
            max_prompt_tokens=2,
            max_new_tokens=10,
        )
        too_long_response = sft_mod._record_to_train_sequence(
            {"prompt": "q", "response": "abc"},
            tokenizer,
            max_prompt_tokens=10,
            max_new_tokens=3,
        )
        exact_budget = sft_mod._record_to_train_sequence(
            {"prompt": "ab", "response": "cd"},
            tokenizer,
            max_prompt_tokens=2,
            max_new_tokens=3,
        )

        self.assertIsNone(too_long_prompt)
        self.assertIsNone(too_long_response)
        self.assertIsNotNone(exact_budget)

    def test_sft_fit_raises_when_all_rows_are_filtered(self):
        """SFT should fail loudly instead of finishing with zero train steps."""
        backend = FakeSFTBackend()
        trainer = sft_mod.SFTTrainer(
            _sft_config(),
            instance=backend,
            dataset=[{"prompt": "q", "response": ""}, {"prompt": "q", "response": ""}],
            reward_fn=None,
            loss_fn=lambda _pack, _logprobs: None,
        )

        with self.assertRaisesRegex(ValueError, "no valid training rows"):
            trainer.fit()

        self.assertEqual(backend.train_calls, 0)
        self.assertTrue(backend.closed)

    def test_dpo_requires_explicit_prompt_chosen_rejected_schema(self):
        """DPO rows should not guess preference or prompt field aliases."""
        tokenizer = FakeTextTokenizer()

        with self.assertRaisesRegex(ValueError, "`chosen` and `rejected`"):
            dpo_mod._record_to_train_pair({"winner": "yes", "loser": "no"}, tokenizer, max_seq_len=32)
        with self.assertRaisesRegex(ValueError, "`prompt`"):
            dpo_mod._record_to_train_pair({"chosen": "yes", "rejected": "no"}, tokenizer, max_seq_len=32)

    def test_dpo_record_to_train_pair_keeps_chosen_rejected_adjacent(self):
        """DPO rows are emitted as chosen then rejected with shared prompt masks."""
        tokenizer = FakeTextTokenizer()

        pair = dpo_mod._record_to_train_pair(
            {"prompt": "q", "chosen": "good", "rejected": "bad"}, tokenizer, max_seq_len=32
        )

        self.assertEqual(len(pair), 2)
        self.assertEqual(pair[0].eos_token_id, 99)
        self.assertEqual(pair[1].eos_token_id, 99)
        self.assertEqual(pair[0].prompt_mask[0], True)
        self.assertEqual(pair[1].prompt_mask[0], True)
        self.assertTrue(any(not item for item in pair[0].prompt_mask[1:]))

    def test_common_prefix_len_stops_at_first_difference(self):
        """Chat-style DPO uses common-prefix length as prompt context."""
        self.assertEqual(dpo_mod._common_prefix_len([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(dpo_mod._common_prefix_len([1, 2], [1, 2, 3]), 2)

    def test_dpo_make_sequence_filters_too_long_rows(self):
        """DPO sequence builder should drop over-budget preference examples."""
        seq = dpo_mod._make_sequence([1, 2, 3], [True, False, False], eos_token_id=99, max_seq_len=2)

        self.assertIsNone(seq)


class SharedDataUtilityTest(unittest.TestCase):
    """Shared tokenizer helpers should preserve SFT and DPO row semantics."""

    def test_prompt_response_helper_appends_eos_and_masks_prompt(self):
        """Prompt/response tokenization should train only response tokens."""
        tokenizer = FakeTextTokenizer()

        tokens, mask = data_utils.prompt_response_to_tokens_and_mask("q", "a", tokenizer, tokenizer.eos_token_id)

        self.assertEqual(tokens[-1], tokenizer.eos_token_id)
        self.assertEqual(mask, [True, False, False])

    def test_first_value_returns_only_explicit_keys(self):
        """Shared first_value remains a literal-key utility, not schema guessing."""
        messages = [{"role": "assistant", "content": "ok"}]

        self.assertEqual(data_utils.first_value({"chosen": messages}, ("chosen",)), messages)
        self.assertIsNone(data_utils.first_value({}, ("chosen",)))


if __name__ == "__main__":
    unittest.main()
