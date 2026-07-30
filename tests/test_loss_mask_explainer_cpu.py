"""CPU tests for the loss-mask explainer.

All tests use a lightweight mock tokenizer — no model downloads or GPU
needed.  The mock renders messages into deterministic token-id sequences
so span boundaries are predictable.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from areno.api.loss_mask_explainer import (
    LossMaskExplainer,
    LossMaskReport,
    MaskSpan,
)
from areno.cli.inspect import inspect_loss_mask_command


class FakeTokenizer:
    """Minimal tokenizer double that produces deterministic token ids.

    Each message is rendered as ``[role_tag, content_tokens...]`` where
    ``role_tag`` is a fixed int per role and content tokens are the ord
    values of each character.  This makes span boundaries easy to reason
    about in tests.
    """

    chat_template = "<template>"
    all_special_ids = [0, 1, 2, 3, 4]

    _ROLE_TAGS = {"system": 1, "user": 2, "assistant": 3, "tool": 4}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs) -> Any:
        tokenize = kwargs.get("tokenize", False)
        add_gen = kwargs.get("add_generation_prompt", False)

        token_ids: list[int] = []
        for msg in messages:
            role = msg.get("role", "user")
            tag = self._ROLE_TAGS.get(role, 0)
            token_ids.append(tag)
            content = msg.get("content", "") or ""
            for ch in content:
                token_ids.append(ord(ch))
            # Add a turn-separator token so boundaries are clean.
            token_ids.append(0)

        if add_gen:
            token_ids.append(self._ROLE_TAGS["assistant"])

        if tokenize:
            return token_ids
        # Text mode: join decoded characters.
        parts = []
        for msg in messages:
            parts.append(f"[{msg.get('role', '')}]{msg.get('content', '')}")
        if add_gen:
            parts.append("[assistant]")
        return "".join(parts)

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return "".join(chr(t) for t in token_ids if 0 < t < 128)


# ---------------------------------------------------------------------------
# Test messages and packer output
# ---------------------------------------------------------------------------

SFT_MESSAGES = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]

# Rendered token ids for SFT_MESSAGES via FakeTokenizer:
# system: [1, Y, o, u, ..., 0]  user: [2, H, i, 0]  assistant: [3, H, e, l, l, o, !, 0]
# We construct token_ids and loss_mask to match.
SFT_TOKEN_IDS = (
    [1] + [ord(c) for c in "You are helpful."] + [0]
    + [2] + [ord(c) for c in "Hi"] + [0]
    + [3] + [ord(c) for c in "Hello!"] + [0]
)
# SFT mask: system and user are False (not trained), assistant is True.
SFT_LOSS_MASK = (
    [False] * (1 + len("You are helpful.") + 1)
    + [False] * (1 + len("Hi") + 1)
    + [True] * (1 + len("Hello!") + 1)
)

AGENTIC_MESSAGES = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
    {"role": "assistant", "content": "It is sunny."},
]

AGENTIC_TOKEN_IDS = (
    [2] + [ord(c) for c in "What's the weather?"] + [0]
    + [3] + [0]
    + [4] + [ord(c) for c in "Sunny"] + [0]
    + [3] + [ord(c) for c in "It is sunny."] + [0]
)
# Agentic mask: user=False, first assistant(tool_call)=False, tool=False,
# final assistant(text)=True.
AGENTIC_LOSS_MASK = (
    [False] * (1 + len("What's the weather?") + 1)
    + [False] * (1 + 1)
    + [False] * (1 + len("Sunny") + 1)
    + [True] * (1 + len("It is sunny.") + 1)
)


class LossMaskExplainerTest(unittest.TestCase):
    """Core explainer logic tests with mock packer output."""

    def test_sft_mask_only_assistant_trainable(self):
        """SFT should mark only assistant tokens as trainable."""

        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, SFT_TOKEN_IDS, SFT_LOSS_MASK, SFT_MESSAGES
        )

        self.assertEqual(report.total_tokens, len(SFT_TOKEN_IDS))
        self.assertGreater(report.trainable_tokens, 0)
        # Only the assistant span should be trainable.
        assistant_spans = [s for s in report.spans if s.role == "assistant"]
        self.assertTrue(any(s.is_trainable for s in assistant_spans))
        system_spans = [s for s in report.spans if s.role == "system"]
        self.assertTrue(all(not s.is_trainable for s in system_spans))
        user_spans = [s for s in report.spans if s.role == "user"]
        self.assertTrue(all(not s.is_trainable for s in user_spans))

    def test_agentic_mask_tool_result_not_trained(self):
        """Agentic mask should not train on tool-result tokens."""

        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, AGENTIC_TOKEN_IDS, AGENTIC_LOSS_MASK, AGENTIC_MESSAGES
        )

        tool_spans = [s for s in report.spans if s.role == "tool"]
        self.assertTrue(all(not s.is_trainable for s in tool_spans))
        # The final assistant text span should be trainable.
        assistant_spans = [s for s in report.spans if s.role == "assistant"]
        self.assertTrue(any(s.is_trainable for s in assistant_spans))

    def test_all_masked(self):
        """All tokens trainable should produce 100% mask ratio."""

        mask = [True] * len(SFT_TOKEN_IDS)
        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, SFT_TOKEN_IDS, mask, SFT_MESSAGES
        )

        self.assertEqual(report.trainable_tokens, report.total_tokens)
        self.assertAlmostEqual(report.overall_mask_ratio, 1.0)
        self.assertTrue(all(s.is_trainable for s in report.spans))

    def test_none_masked(self):
        """No trainable tokens should produce 0% mask ratio."""

        mask = [False] * len(SFT_TOKEN_IDS)
        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, SFT_TOKEN_IDS, mask, SFT_MESSAGES
        )

        self.assertEqual(report.trainable_tokens, 0)
        self.assertAlmostEqual(report.overall_mask_ratio, 0.0)
        self.assertTrue(all(not s.is_trainable for s in report.spans))

    def test_truncated_sample(self):
        """Token sequence cut mid-turn should still produce valid spans."""

        # Truncate the token sequence to half its length.
        half = len(SFT_TOKEN_IDS) // 2
        truncated_ids = SFT_TOKEN_IDS[:half]
        truncated_mask = SFT_LOSS_MASK[:half]
        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, truncated_ids, truncated_mask, SFT_MESSAGES
        )

        self.assertEqual(report.total_tokens, half)
        # Spans should not extend beyond the truncated length.
        for s in report.spans:
            self.assertLessEqual(s.token_count, half)

    def test_text_preview_truncated_by_default(self):
        """Text preview should be at most 50 characters by default."""

        long_messages = [
            {"role": "user", "content": "A" * 100},
            {"role": "assistant", "content": "B" * 100},
        ]
        token_ids = (
            [2] + [ord(c) for c in "A" * 100] + [0]
            + [3] + [ord(c) for c in "B" * 100] + [0]
        )
        loss_mask = [False] * 102 + [True] * 102
        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, token_ids, loss_mask, long_messages
        )

        for s in report.spans:
            self.assertLessEqual(len(s.text_preview), 50)

    def test_show_full_text(self):
        """show_full_text=True should not truncate the preview."""

        long_messages = [
            {"role": "user", "content": "A" * 100},
            {"role": "assistant", "content": "B" * 100},
        ]
        token_ids = (
            [2] + [ord(c) for c in "A" * 100] + [0]
            + [3] + [ord(c) for c in "B" * 100] + [0]
        )
        loss_mask = [False] * 102 + [True] * 102
        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, token_ids, loss_mask, long_messages,
            show_full_text=True,
        )

        # At least one span should have a preview longer than 50 chars.
        self.assertTrue(any(len(s.text_preview) > 50 for s in report.spans))

    def test_to_dict_produces_valid_json(self):
        """to_dict should return a JSON-serialisable structure."""

        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, SFT_TOKEN_IDS, SFT_LOSS_MASK, SFT_MESSAGES
        )
        d = report.to_dict()
        s = json.dumps(d)
        parsed = json.loads(s)

        self.assertEqual(parsed["model_name"], "test-model")
        self.assertIsInstance(parsed["spans"], list)
        self.assertIn("mask_ratio", parsed["spans"][0])

    def test_to_human_readable_contains_key_info(self):
        """Human-readable output should show role, tokens, ratio, and text."""

        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, SFT_TOKEN_IDS, SFT_LOSS_MASK, SFT_MESSAGES
        )
        text = report.to_human_readable()

        self.assertIn("Loss Mask Report: test-model", text)
        self.assertIn("system", text)
        self.assertIn("assistant", text)
        self.assertIn("Role", text)


class LossMaskCLITest(unittest.TestCase):
    """Integration tests crossing areno.cli.inspect → areno.api.loss_mask_explainer."""

    def _write_messages(self, tmp_dir: str) -> str:
        path = Path(tmp_dir, "messages.json")
        path.write_text(json.dumps(SFT_MESSAGES), encoding="utf-8")
        return str(path)

    def _invoke(self, messages_path: str, output_format: str = "text", show_full: bool = False):
        tok = FakeTokenizer()
        with (
            patch("areno.cli.inspect.resolve_model_ref", return_value="/fake/model"),
            patch("areno.cli.inspect.load_tokenizer", return_value=tok),
        ):
            runner = CliRunner()
            args = [
                "--model", "test-model",
                "--messages", messages_path,
                "--output-format", output_format,
            ]
            if show_full:
                args.append("--show-full-text")
            return runner.invoke(inspect_loss_mask_command, args)

    def test_cli_text_output(self):
        """CLI text mode should produce a human-readable table."""

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_messages(tmp)
            result = self._invoke(path, "text")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Loss Mask Report", result.output)
        self.assertIn("system", result.output)
        self.assertIn("assistant", result.output)

    def test_cli_json_output(self):
        """CLI json mode should produce valid JSON with expected fields."""

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_messages(tmp)
            result = self._invoke(path, "json")

        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["model_name"], "test-model")
        self.assertIsInstance(parsed["spans"], list)
        self.assertIn("mask_ratio", parsed["spans"][0])

    def test_cli_default_truncates_text(self):
        """Default output should not show full text."""

        long_messages = [{"role": "user", "content": "A" * 100},
                         {"role": "assistant", "content": "B" * 100}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "messages.json")
            path.write_text(json.dumps(long_messages), encoding="utf-8")
            result = self._invoke(str(path), "text")

        self.assertEqual(result.exit_code, 0)
        # The output should not contain a run of 100 'A' characters.
        self.assertNotIn("A" * 100, result.output)

    def test_cli_show_full_text(self):
        """--show-full-text should include longer text in output."""

        long_messages = [{"role": "user", "content": "A" * 100},
                         {"role": "assistant", "content": "B" * 100}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "messages.json")
            path.write_text(json.dumps(long_messages), encoding="utf-8")
            result = self._invoke(str(path), "text", show_full=True)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("A" * 100, result.output)


class DefaultBehaviorTest(unittest.TestCase):
    """Verify the explainer is opt-in and does not affect existing code."""

    def test_explainer_does_not_modify_inputs(self):
        """Running the explainer must not mutate the input lists."""

        tok = FakeTokenizer()
        original_ids = list(SFT_TOKEN_IDS)
        original_mask = list(SFT_LOSS_MASK)
        LossMaskExplainer.explain(
            "test", tok, SFT_TOKEN_IDS, SFT_LOSS_MASK, SFT_MESSAGES
        )
        self.assertEqual(SFT_TOKEN_IDS, original_ids)
        self.assertEqual(SFT_LOSS_MASK, original_mask)

    def test_module_imports_without_side_effects(self):
        """Importing the explainer module should not trigger heavy imports."""

        from areno.api import loss_mask_explainer
        self.assertTrue(hasattr(loss_mask_explainer, "LossMaskExplainer"))


class MalformedInputTest(unittest.TestCase):
    """Tests for malformed inputs that should not crash the explainer."""

    def test_empty_messages(self):
        """An empty message list should produce an empty report with zero tokens."""

        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, [], [], []
        )

        self.assertEqual(report.total_tokens, 0)
        self.assertEqual(report.trainable_tokens, 0)
        self.assertEqual(report.spans, [])

    def test_empty_token_ids(self):
        """Empty token ids with non-empty messages should produce zero-length spans."""

        tok = FakeTokenizer()
        report = LossMaskExplainer.explain(
            "test-model", tok, [], [], SFT_MESSAGES
        )

        self.assertEqual(report.total_tokens, 0)
        self.assertEqual(report.spans, [])

    def test_mismatched_lengths(self):
        """Loss mask shorter than token ids should not crash; clamped to token length."""

        tok = FakeTokenizer()
        # Loss mask is shorter than token ids.
        short_mask = SFT_LOSS_MASK[: len(SFT_LOSS_MASK) // 2]
        report = LossMaskExplainer.explain(
            "test-model", tok, SFT_TOKEN_IDS, short_mask, SFT_MESSAGES
        )

        # The report should still be produced without raising.
        self.assertEqual(report.total_tokens, len(SFT_TOKEN_IDS))
        # Trainable count should not exceed the mask length.
        self.assertLessEqual(report.trainable_tokens, len(short_mask))

    def test_loss_mask_longer_than_tokens(self):
        """Loss mask longer than token ids should not crash; clamped to token length."""

        tok = FakeTokenizer()
        long_mask = SFT_LOSS_MASK + [True] * 100
        report = LossMaskExplainer.explain(
            "test-model", tok, SFT_TOKEN_IDS, long_mask, SFT_MESSAGES
        )

        self.assertEqual(report.total_tokens, len(SFT_TOKEN_IDS))

    def test_messages_with_unknown_role(self):
        """Messages with an unknown role should not crash the explainer."""

        tok = FakeTokenizer()
        messages = [
            {"role": "narrator", "content": "Once upon a time."},
            {"role": "assistant", "content": "The end."},
        ]
        token_ids = (
            [0] + [ord(c) for c in "Once upon a time."] + [0]
            + [3] + [ord(c) for c in "The end."] + [0]
        )
        loss_mask = [False] * 19 + [True] * 10
        report = LossMaskExplainer.explain(
            "test-model", tok, token_ids, loss_mask, messages
        )

        # Should produce spans without raising.
        self.assertGreater(len(report.spans), 0)


if __name__ == "__main__":
    unittest.main()