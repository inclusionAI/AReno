"""CPU tests for effective trainable token counting (issue #227).

Tests cover:
- Packed layout: response mask → counts.
- Padded layout: lengths + prompt_mask → counts.
- Padded layout with loss_mask: prompt_mask + loss_mask → counts.
- Zero-token batch (all lengths = 0).
- Single token per sequence (length = 1, no actions).
- Empty packed response mask.
- Mixed prompt/response in same sequence.
- Multiple sequences with varying lengths.
- Deterministic output.
- to_dict() structure.
- Invalid inputs (missing required args).
"""

from __future__ import annotations

import unittest

from areno.engine.token_counts import compute_token_counts


class TestPackedLayout(unittest.TestCase):
    """Packed layout: pass packed_response_mask and num_sequences."""

    def test_basic_packed(self):
        # 3 response tokens out of 5 actions, 2 sequences.
        # total_tokens = 5 + 2 = 7, masked = 7 - 3 = 4.
        counts = compute_token_counts(
            lengths=[],
            packed_response_mask=[True, False, True, False, True],
            num_sequences=2,
        )
        self.assertEqual(counts.effective_loss_tokens, 3)
        self.assertEqual(counts.total_input_tokens, 7)
        self.assertEqual(counts.masked_tokens, 4)
        self.assertEqual(counts.num_sequences, 2)
        self.assertAlmostEqual(counts.mean_effective_length, 1.5)

    def test_all_effective_packed(self):
        counts = compute_token_counts(
            lengths=[],
            packed_response_mask=[True, True, True],
            num_sequences=1,
        )
        self.assertEqual(counts.effective_loss_tokens, 3)
        self.assertEqual(counts.masked_tokens, 1)  # 4 total - 3 effective
        self.assertEqual(counts.total_input_tokens, 4)

    def test_none_effective_packed(self):
        counts = compute_token_counts(
            lengths=[],
            packed_response_mask=[False, False],
            num_sequences=1,
        )
        self.assertEqual(counts.effective_loss_tokens, 0)
        self.assertEqual(counts.masked_tokens, 3)
        self.assertAlmostEqual(counts.mean_effective_length, 0.0)

    def test_empty_packed_mask(self):
        counts = compute_token_counts(
            lengths=[],
            packed_response_mask=[],
            num_sequences=0,
        )
        self.assertEqual(counts.effective_loss_tokens, 0)
        # num_sequences=0 defaults to 1, so total = 0 + 1 = 1.
        self.assertEqual(counts.total_input_tokens, 1)

    def test_num_sequences_defaults_to_1(self):
        counts = compute_token_counts(
            lengths=[],
            packed_response_mask=[True],
            num_sequences=None,
        )
        self.assertEqual(counts.num_sequences, 1)


class TestPaddedLayout(unittest.TestCase):
    """Padded layout: pass lengths, prompt_mask_rows."""

    def test_basic_padded(self):
        # 1 sequence, length 5: [P, P, P, R, R]
        # Actions at positions 1-4: pos1=P(masked), pos2=P(masked), pos3=R(eff), pos4=R(eff)
        counts = compute_token_counts(
            lengths=[5],
            prompt_mask_rows=[[True, True, True, False, False]],
        )
        self.assertEqual(counts.total_input_tokens, 5)
        self.assertEqual(counts.effective_loss_tokens, 2)  # pos 3,4
        self.assertEqual(counts.masked_tokens, 2)  # pos 1,2
        self.assertEqual(counts.num_sequences, 1)
        self.assertAlmostEqual(counts.mean_effective_length, 2.0)

    def test_multiple_sequences(self):
        # seq0: length 4 [P,P,R,R] → actions pos1=P(masked), pos2-3=R(eff)
        # seq1: length 3 [P,R,R] → actions pos1=R(eff), pos2=R(eff)
        counts = compute_token_counts(
            lengths=[4, 3],
            prompt_mask_rows=[
                [True, True, False, False],
                [True, False, False],
            ],
        )
        self.assertEqual(counts.total_input_tokens, 7)
        self.assertEqual(counts.effective_loss_tokens, 4)  # 2 + 2
        self.assertEqual(counts.masked_tokens, 1)  # 1 from seq0
        self.assertAlmostEqual(counts.mean_effective_length, 2.0)

    def test_with_loss_mask(self):
        # seq0: length 4, prompt_mask=[T,T,F,F], loss_mask=[F,F,T,F]
        # pos1: prompt=True → masked
        # pos2: prompt=False, loss_mask=True → effective
        # pos3: prompt=False, loss_mask=False → masked
        counts = compute_token_counts(
            lengths=[4],
            prompt_mask_rows=[[True, True, False, False]],
            loss_mask_rows=[[False, False, True, False]],
        )
        self.assertEqual(counts.effective_loss_tokens, 1)  # only pos2
        self.assertEqual(counts.masked_tokens, 2)  # pos1 + pos3

    def test_single_token_sequence(self):
        # length=1 means no actions (positions 1..0 is empty).
        counts = compute_token_counts(
            lengths=[1],
            prompt_mask_rows=[[True]],
        )
        self.assertEqual(counts.total_input_tokens, 1)
        self.assertEqual(counts.effective_loss_tokens, 0)
        self.assertEqual(counts.masked_tokens, 0)
        self.assertAlmostEqual(counts.mean_effective_length, 0.0)


class TestZeroTokenBatch(unittest.TestCase):
    """Zero-token batches should not crash."""

    def test_all_zero_lengths(self):
        counts = compute_token_counts(
            lengths=[0, 0, 0],
            prompt_mask_rows=[[], [], []],
        )
        self.assertEqual(counts.total_input_tokens, 0)
        self.assertEqual(counts.effective_loss_tokens, 0)
        self.assertEqual(counts.masked_tokens, 0)
        self.assertAlmostEqual(counts.mean_effective_length, 0.0)

    def test_empty_lengths(self):
        with self.assertRaises(ValueError):
            compute_token_counts(lengths=[], prompt_mask_rows=[])


class TestInvalidInputs(unittest.TestCase):
    """Invalid inputs should raise clear errors."""

    def test_missing_prompt_mask_for_padded(self):
        with self.assertRaises(ValueError):
            compute_token_counts(lengths=[5])

    def test_missing_both_layouts(self):
        with self.assertRaises(ValueError):
            compute_token_counts(lengths=[])


class TestDeterminism(unittest.TestCase):
    """Same inputs produce same outputs."""

    def test_deterministic_packed(self):
        kwargs = dict(
            lengths=[],
            packed_response_mask=[True, False, True],
            num_sequences=1,
        )
        c1 = compute_token_counts(**kwargs)
        c2 = compute_token_counts(**kwargs)
        self.assertEqual(c1, c2)

    def test_deterministic_padded(self):
        kwargs = dict(
            lengths=[4, 3],
            prompt_mask_rows=[[True, True, False, False], [True, False, False]],
        )
        c1 = compute_token_counts(**kwargs)
        c2 = compute_token_counts(**kwargs)
        self.assertEqual(c1, c2)


class TestToDict(unittest.TestCase):
    """to_dict should return float metrics."""

    def test_to_dict_fields(self):
        counts = compute_token_counts(
            lengths=[3],
            prompt_mask_rows=[[True, False, False]],
        )
        d = counts.to_dict()
        self.assertIn("total_input_tokens", d)
        self.assertIn("masked_tokens", d)
        self.assertIn("effective_loss_tokens", d)
        self.assertIn("mean_effective_length", d)
        self.assertIsInstance(d["total_input_tokens"], float)
        self.assertIsInstance(d["effective_loss_tokens"], float)


class TestMixedPromptResponse(unittest.TestCase):
    """Prompt and response can be interleaved (agentic trajectories)."""

    def test_interleaved(self):
        # [P, R, R, P, R] — prompt and response alternate
        # Actions at pos 1-4:
        # pos1: prompt=False → effective
        # pos2: prompt=False → effective
        # pos3: prompt=True → masked
        # pos4: prompt=False → effective
        counts = compute_token_counts(
            lengths=[5],
            prompt_mask_rows=[[True, False, False, True, False]],
        )
        self.assertEqual(counts.effective_loss_tokens, 3)
        self.assertEqual(counts.masked_tokens, 1)


if __name__ == "__main__":
    unittest.main()
