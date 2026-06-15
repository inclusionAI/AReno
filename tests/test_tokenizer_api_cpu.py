from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from areno.api.config import ArenoConfig, coerce_backend_config, resolve_backend_type
from areno.api.models import BackendType, SamplingParams, TrainSequence
from areno.api.rewards import load_reward_fn
from areno.api.tokenizer import _looks_chat_formatted, encode_generation_prompt, eos_token_ids


class FakeTokenizer:
    """Small tokenizer double that exposes only the API helpers need."""

    eos_token_id = 1
    chat_template = "template"

    def __init__(self):
        self.encoded = []
        self.templated = []

    def encode(self, prompt):
        self.encoded.append(prompt)
        return [ord(prompt[0])]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.templated.append((messages, tokenize, add_generation_prompt))
        return [7, 8, 9]


class TokenizerApiTest(unittest.TestCase):
    """API helper tests that do not instantiate HuggingFace tokenizers."""

    def test_eos_token_ids_merges_tokenizer_and_nested_config(self):
        """Rollout stop ids should include tokenizer, top-level, and text_config EOS."""
        tokenizer = FakeTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(
                json.dumps({"eos_token_id": [2, 1], "text_config": {"eos_token_id": [3, 2]}}),
                encoding="utf-8",
            )

            ids = eos_token_ids(tmp, tokenizer)

        self.assertEqual(ids, (1, 2, 3))

    def test_encode_generation_prompt_applies_template_once(self):
        """Already formatted chat prompts must not be wrapped a second time."""
        tokenizer = FakeTokenizer()

        templated = encode_generation_prompt(tokenizer, "plain prompt")
        already_formatted = encode_generation_prompt(tokenizer, "<start_of_turn>user\nhello")

        self.assertEqual(templated, [7, 8, 9])
        self.assertEqual(already_formatted, [ord("<")])
        self.assertEqual(len(tokenizer.templated), 1)
        self.assertEqual(tokenizer.encoded, ["<start_of_turn>user\nhello"])
        self.assertTrue(_looks_chat_formatted("<|im_start|>user"))

    def test_sampling_params_defaults_are_backend_agnostic(self):
        """Public sampling defaults are part of the backend API contract."""
        params = SamplingParams()

        self.assertEqual(params.top_k, -1)
        self.assertEqual(params.max_new_tokens, 16)
        self.assertFalse(params.ignore_eos)

    def test_train_sequence_defaults_are_independent_lists(self):
        """Pydantic default factories must not share mutable lists across rows."""
        first = TrainSequence()
        second = TrainSequence()

        first.tokens.append(1)

        self.assertEqual(first.tokens, [1])
        self.assertEqual(second.tokens, [])

    def test_backend_config_coercion_rejects_wrong_type(self):
        """Typed backend configs should fail early when callers pass the wrong type."""
        cfg = ArenoConfig(tp_size=2)

        self.assertIs(resolve_backend_type(None, None), BackendType.Areno)
        self.assertIs(coerce_backend_config(BackendType.Areno, cfg), cfg)
        with self.assertRaisesRegex(TypeError, "requires its typed backend config"):
            coerce_backend_config(BackendType.Areno, object())

    def test_load_reward_fn_imports_callable_from_file(self):
        """Reward functions are loaded from plain Python files at runtime."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "reward_file.py")
            path.write_text("def reward_fn(record):\n    return len(record.completion)\n", encoding="utf-8")

            fn = load_reward_fn(str(path))

        self.assertEqual(fn(SimpleNamespace(completion="abc")), 3)

    def test_load_reward_fn_requires_callable(self):
        """Misconfigured reward files should fail before training starts."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "bad_reward.py")
            path.write_text("reward_fn = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "callable reward_fn"):
                load_reward_fn(str(path))


if __name__ == "__main__":
    unittest.main()
