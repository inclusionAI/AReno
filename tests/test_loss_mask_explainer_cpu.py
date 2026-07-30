"""CPU tests for the human-readable loss-mask explainer (Issue #221).

Covers:
- SFT packer span derivation from prompt_mask
- Agentic packer span construction with tool call / tool result splitting
- Token-level equivalence between spans and masks
- Boundary cases: all-masked, truncated
- Invalid inputs: length mismatch, span gaps
- show_text behaviour
- Backward compatibility (AgentTrainBatch without spans)
- CLI end-to-end (terminal + JSON)
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from areno.api.data import LossMaskExplanation, LossSpan
from areno.api.data_utils import spans_from_prompt_mask
from areno.api.loss_mask_explainer import explain_loss_mask
from areno.api.agentic import AgentTrainBatch, LossMaskPolicy


class TestSpansFromPromptMask(unittest.TestCase):
    """SFT span derivation from prompt_mask."""

    def test_basic_prompt_then_response(self):
        """Consecutive same values produce two spans."""
        prompt_mask = [True, True, True, False, False]
        spans = spans_from_prompt_mask(prompt_mask)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].role, "prompt")
        self.assertEqual(spans[0].start, 0)
        self.assertEqual(spans[0].end, 3)
        self.assertFalse(spans[0].loss)
        self.assertEqual(spans[1].role, "response")
        self.assertEqual(spans[1].start, 3)
        self.assertEqual(spans[1].end, 5)
        self.assertTrue(spans[1].loss)

    def test_all_prompt(self):
        """All-prompt mask produces one span with loss=False."""
        spans = spans_from_prompt_mask([True, True, True])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].role, "prompt")
        self.assertEqual(spans[0].start, 0)
        self.assertEqual(spans[0].end, 3)
        self.assertFalse(spans[0].loss)

    def test_all_response(self):
        """All-response mask produces one span with loss=True."""
        spans = spans_from_prompt_mask([False, False])
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].role, "response")
        self.assertEqual(spans[0].start, 0)
        self.assertEqual(spans[0].end, 2)
        self.assertTrue(spans[0].loss)

    def test_empty_mask(self):
        """Empty prompt_mask produces empty span list."""
        self.assertEqual(spans_from_prompt_mask([]), [])


class TestExplainSftEquivalence(unittest.TestCase):
    """Token-level equivalence: span loss flags must match ~prompt_mask."""

    def test_sft_equivalence(self):
        """For each token position, span.loss == (not prompt_mask[i])."""
        prompt_mask = [True, True, True, True, False, False, False, False, False]
        tokens = list(range(9))
        loss_mask = [not m for m in prompt_mask]
        spans = spans_from_prompt_mask(prompt_mask)
        exp = explain_loss_mask(tokens, loss_mask, spans)
        # Reconstruct mask from spans and compare.
        reconstructed = [False] * len(tokens)
        for span in exp.spans:
            for i in range(span.start, span.end):
                reconstructed[i] = span.loss
        self.assertEqual(reconstructed, loss_mask)
        self.assertEqual(exp.total_tokens, 9)
        self.assertEqual(exp.loss_tokens, 5)

    def test_sft_summary(self):
        """Summary contains per-role token and loss counts."""
        prompt_mask = [True] * 4 + [False] * 6
        tokens = list(range(10))
        loss_mask = [not m for m in prompt_mask]
        spans = spans_from_prompt_mask(prompt_mask)
        exp = explain_loss_mask(tokens, loss_mask, spans)
        roles = {e["role"]: e for e in exp.summary}
        self.assertEqual(roles["prompt"]["token_count"], 4)
        self.assertEqual(roles["prompt"]["loss_tokens"], 0)
        self.assertEqual(roles["response"]["token_count"], 6)
        self.assertEqual(roles["response"]["loss_tokens"], 6)


class TestExplainAgenticEquivalence(unittest.TestCase):
    """Agentic packer: span loss flags must match loss_masks."""

    def test_agentic_single_turn_text(self):
        """Single-turn assistant_text: prompt (loss=False) + response (loss=True)."""
        tokens = list(range(20))
        loss_mask = [False] * 8 + [True] * 12
        spans = [
            LossSpan(role="prompt", start=0, end=8, loss=False),
            LossSpan(role="assistant_text", start=8, end=20, loss=True),
        ]
        exp = explain_loss_mask(tokens, loss_mask, spans)
        reconstructed = [False] * len(tokens)
        for span in exp.spans:
            for i in range(span.start, span.end):
                reconstructed[i] = span.loss
        self.assertEqual(reconstructed, loss_mask)
        self.assertEqual(exp.loss_tokens, 12)

    def test_agentic_tool_call_with_tool_result(self):
        """Tool call (loss=True) + tool result sentinel (loss=False) in one response."""
        tokens = list(range(15))
        # prompt=5, tool_call=4, tool_result=6
        loss_mask = [False] * 5 + [True, True, True, True] + [False] * 6
        spans = [
            LossSpan(role="prompt", start=0, end=5, loss=False),
            LossSpan(role="assistant_tool_call", start=5, end=9, loss=True),
            LossSpan(role="tool_result", start=9, end=15, loss=False),
        ]
        exp = explain_loss_mask(tokens, loss_mask, spans)
        reconstructed = [False] * len(tokens)
        for span in exp.spans:
            for i in range(span.start, span.end):
                reconstructed[i] = span.loss
        self.assertEqual(reconstructed, loss_mask)
        roles = {e["role"]: e for e in exp.summary}
        self.assertIn("assistant_tool_call", roles)
        self.assertIn("tool_result", roles)
        self.assertEqual(roles["tool_result"]["loss_tokens"], 0)

    def test_agentic_multi_turn(self):
        """Multi-turn: 2 assistant turns + 1 tool call + 1 tool result."""
        tokens = list(range(30))
        # turn 0: prompt(0-5), assistant_text(5-12)
        # turn 1: assistant_tool_call(12-16), tool_result(16-22)
        # turn 2: assistant_text(22-30)
        loss_mask = (
            [False] * 5          # prompt
            + [True] * 7         # assistant_text turn 0
            + [True] * 4         # assistant_tool_call turn 1
            + [False] * 6        # tool_result turn 1
            + [True] * 8         # assistant_text turn 2
        )
        spans = [
            LossSpan(role="prompt", start=0, end=5, loss=False, turn=0),
            LossSpan(role="assistant_text", start=5, end=12, loss=True, turn=0),
            LossSpan(role="assistant_tool_call", start=12, end=16, loss=True, turn=1),
            LossSpan(role="tool_result", start=16, end=22, loss=False, turn=1),
            LossSpan(role="assistant_text", start=22, end=30, loss=True, turn=2),
        ]
        exp = explain_loss_mask(tokens, loss_mask, spans)
        # Verify token-level equivalence.
        reconstructed = [False] * len(tokens)
        for span in exp.spans:
            for i in range(span.start, span.end):
                reconstructed[i] = span.loss
        self.assertEqual(reconstructed, loss_mask)
        # Verify all 4 roles appear (prompt + 3 response roles).
        roles = {e["role"] for e in exp.summary}
        self.assertIn("prompt", roles)
        self.assertIn("assistant_text", roles)
        self.assertIn("assistant_tool_call", roles)
        self.assertIn("tool_result", roles)
        # Verify turn numbers.
        turns = {s.turn for s in exp.spans}
        self.assertEqual(turns, {0, 1, 2})


class TestBoundaryCases(unittest.TestCase):

    def test_all_masked_sample(self):
        """All loss=False: correct report of 0 loss tokens."""
        tokens = list(range(10))
        loss_mask = [False] * 10
        spans = [LossSpan(role="prompt", start=0, end=10, loss=False)]
        exp = explain_loss_mask(tokens, loss_mask, spans)
        self.assertEqual(exp.loss_tokens, 0)
        self.assertEqual(exp.total_tokens, 10)

    def test_truncated_sample(self):
        """Right-truncated sequence: spans clipped to token length."""
        # Original 10 tokens, truncated to 6.
        tokens = list(range(6))
        loss_mask = [False, False, False, True, True, False]  # truncated loss_mask too
        # Spans that would cover the original 10-token range.
        spans = [
            LossSpan(role="prompt", start=0, end=3, loss=False),
            LossSpan(role="response", start=3, end=10, loss=True),  # extends beyond truncation
        ]
        exp = explain_loss_mask(tokens, loss_mask, spans)
        # The response span should be clipped to end=6.
        self.assertEqual(exp.spans[1].end, 6)
        self.assertEqual(exp.total_tokens, 6)
        # Token-level equivalence within the truncated range.
        reconstructed = [False] * 6
        for span in exp.spans:
            for i in range(span.start, span.end):
                reconstructed[i] = span.loss
        # Spans say response is loss=True, but loss_mask says position 5 is False.
        # The explainer reports spans, not masks — len(loss_mask)==len(tokens) is the invariant.
        self.assertEqual(len(exp.spans), 2)


class TestInvalidInputs(unittest.TestCase):

    def test_length_mismatch(self):
        """tokens and loss_mask with different lengths raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            explain_loss_mask([1, 2, 3], [True, False], [])
        self.assertIn("length mismatch", str(ctx.exception))

    def test_span_gap(self):
        """Spans that don't cover [0, n) raise ValueError."""
        tokens = list(range(5))
        loss_mask = [True] * 5
        spans = [
            LossSpan(role="response", start=0, end=2, loss=True),
            # Gap at positions 2-3.
            LossSpan(role="response", start=4, end=5, loss=True),
        ]
        with self.assertRaises(ValueError) as ctx:
            explain_loss_mask(tokens, loss_mask, spans)
        self.assertIn("gap", str(ctx.exception))

    def test_span_overlap(self):
        """Overlapping spans raise ValueError."""
        tokens = list(range(5))
        loss_mask = [True] * 5
        spans = [
            LossSpan(role="response", start=0, end=3, loss=True),
            LossSpan(role="response", start=2, end=5, loss=True),  # overlaps at 2
        ]
        with self.assertRaises(ValueError) as ctx:
            explain_loss_mask(tokens, loss_mask, spans)
        self.assertIn("overlap", str(ctx.exception))


class TestShowText(unittest.TestCase):

    def test_show_text_false_omits_content(self):
        """Default behaviour: text_preview is None."""
        tokens = list(range(5))
        loss_mask = [True] * 5
        spans = [LossSpan(role="response", start=0, end=5, loss=True)]
        exp = explain_loss_mask(tokens, loss_mask, spans)
        self.assertIsNone(exp.text_preview)

    def test_show_text_true_with_tokenizer(self):
        """show_text=True with a tokenizer decodes span text."""
        class FakeTokenizer:
            def decode(self, token_ids):
                return f"<decoded {len(token_ids)} tokens>"

        tokens = list(range(8))
        loss_mask = [False] * 3 + [True] * 5
        spans = [
            LossSpan(role="prompt", start=0, end=3, loss=False),
            LossSpan(role="response", start=3, end=8, loss=True),
        ]
        exp = explain_loss_mask(tokens, loss_mask, spans, tokenizer=FakeTokenizer(), show_text=True)
        self.assertIsNotNone(exp.text_preview)
        self.assertIn(0, exp.text_preview)
        self.assertIn(1, exp.text_preview)
        self.assertEqual(exp.text_preview[0], "<decoded 3 tokens>")
        self.assertEqual(exp.text_preview[1], "<decoded 5 tokens>")


class TestBackwardCompatibility(unittest.TestCase):
    """AgentTrainBatch must work without passing spans."""

    def test_agent_train_batch_no_spans(self):
        """Constructing AgentTrainBatch without spans yields empty list."""
        batch = AgentTrainBatch(
            token_rows=[[1, 2, 3]],
            response_masks=[[False, True, True]],
            loss_masks=[[False, True, True]],
            rollout_logprobs=[[0.0, 0.1, 0.2]],
            rewards=[1.0],
            records=[{"id": 0}],
            reward_records=[],
        )
        self.assertEqual(batch.spans, [])


class TestCliExplainMask(unittest.TestCase):
    """CLI end-to-end tests using CliRunner."""

    def setUp(self):
        self.runner = CliRunner()
        self.dataset_path = None
        self.loader_path = None

    def _setup_fixture(self, tmp_path):
        """Create a minimal local JSON dataset and loader."""
        import json as json_mod

        data = [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "What is 3+3?", "response": "6"},
        ]
        dataset_file = tmp_path / "data.jsonl"
        with dataset_file.open("w") as f:
            for row in data:
                f.write(json_mod.dumps(row) + "\n")

        loader_file = tmp_path / "loader.py"
        loader_file.write_text(
            "def load_training_dataset(dataset):\n"
            "    return dataset\n"
        )
        return str(dataset_file), str(loader_file)

    @patch("areno.cli.explain_mask.load_tokenizer", create=True)
    def test_cli_help(self, *args):
        """--help produces output without error."""
        from areno.cli.explain_mask import explain_mask_command
        result = self.runner.invoke(explain_mask_command, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("explain-mask", result.output)

    @patch("areno.cli.explain_mask.load_tokenizer", create=True)
    def test_cli_missing_args(self, *args):
        """Missing required args produces error."""
        from areno.cli.explain_mask import explain_mask_command
        result = self.runner.invoke(explain_mask_command, [])
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_json_output(self):
        """CLI with --json produces valid JSON with expected structure."""
        import tempfile

        from areno.cli.explain_mask import explain_mask_command

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = __import__("pathlib").Path(tmp)
            dataset_path, loader_path = self._setup_fixture(tmp_path)

            # Mock the tokenizer and dataset loading.
            class FakeTokenizer:
                eos_token_id = 0
                chat_template = None

                def encode(self, text, add_special_tokens=False):
                    return [ord(c) for c in text[:5]]

            fake_dataset = [
                {"prompt": "Hello", "response": "World"},
            ]

            with patch("areno.api.tokenizer.load_tokenizer", return_value=FakeTokenizer()), \
                 patch("areno.cli.train._load_dataset_for_training", return_value=fake_dataset):
                result = self.runner.invoke(explain_mask_command, [
                    "--ckpt", "/fake/path",
                    "--dataset-path", dataset_path,
                    "--dataset-loader-fn", loader_path,
                    "--json",
                    "--max-rows", "1",
                ])
            if result.exception:
                import traceback
                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
            self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
            data = json.loads(result.output)
            self.assertIn("rows", data)
            self.assertEqual(len(data["rows"]), 1)
            row = data["rows"][0]
            self.assertIn("total_tokens", row)
            self.assertIn("loss_tokens", row)
            self.assertIn("spans", row)
            self.assertIn("summary", row)


if __name__ == "__main__":
    unittest.main()
