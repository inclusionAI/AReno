"""CPU tests for degenerate sample detection utilities.

These tests cover the detection helpers in ``areno.api.data`` without
requiring a GPU, a real tokenizer backend, or model checkpoints. A minimal
fake tokenizer is used where token-level checks are needed.
"""

from __future__ import annotations

import unittest

from areno.api.data import (
    DegenerateFilterConfig,
    DegeneratePolicy,
    DegenerateReason,
    SampleQualityReport,
    apply_degenerate_policy,
    check_preference_pair,
    check_prompt_text,
    check_response_text,
    check_tokenized_prompt,
    check_trainable_tokens,
    format_degenerate_reasons,
    record_degenerate_reason,
)


class _FakeTokenizer:
    """Minimal tokenizer stub exposing ``all_special_ids``."""

    def __init__(self, special_ids: list[int] | None = None):
        self.all_special_ids = special_ids or []


class CheckPromptTextTest(unittest.TestCase):
    """Text-level prompt checks detect empty and whitespace-only inputs."""

    def test_normal_prompt_passes(self):
        """A non-empty prompt with content should not be flagged."""
        report = check_prompt_text("Hello world")
        self.assertFalse(report.is_degenerate)

    def test_empty_string_detected(self):
        """An empty string prompt should be flagged as EMPTY."""
        report = check_prompt_text("")
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.EMPTY)
        self.assertEqual(report.stage, "pre_tokenization")

    def test_whitespace_only_detected(self):
        """A whitespace-only prompt should be flagged as WHITESPACE_ONLY."""
        report = check_prompt_text("   \n\t  ")
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.WHITESPACE_ONLY)
        self.assertEqual(report.stage, "pre_tokenization")


class CheckResponseTextTest(unittest.TestCase):
    """Text-level response checks mirror the prompt checks."""

    def test_normal_response_passes(self):
        """A non-empty response with content should not be flagged."""
        report = check_response_text("The answer is 42.")
        self.assertFalse(report.is_degenerate)

    def test_empty_response_detected(self):
        """An empty string response should be flagged as EMPTY."""
        report = check_response_text("")
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.EMPTY)

    def test_whitespace_response_detected(self):
        """A whitespace-only response should be flagged as WHITESPACE_ONLY."""
        report = check_response_text("\n\n  \t")
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.WHITESPACE_ONLY)


class CheckTokenizedPromptTest(unittest.TestCase):
    """Token-level prompt checks detect zero-length and special-token-only rows."""

    def test_normal_tokens_pass(self):
        """A mix of regular and special tokens should not be flagged."""
        tok = _FakeTokenizer(special_ids=[0, 1, 2])
        report = check_tokenized_prompt([0, 5, 10, 2], tok)
        self.assertFalse(report.is_degenerate)

    def test_empty_tokens_detected(self):
        """Zero-length token list should be flagged as EMPTY."""
        tok = _FakeTokenizer(special_ids=[0, 1])
        report = check_tokenized_prompt([], tok)
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.EMPTY)
        self.assertEqual(report.stage, "post_tokenization")

    def test_all_special_tokens_detected(self):
        """Tokens that are all special IDs should be flagged as SPECIAL_TOKENS_ONLY."""
        tok = _FakeTokenizer(special_ids=[0, 1, 2])
        report = check_tokenized_prompt([0, 1, 2], tok)
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.SPECIAL_TOKENS_ONLY)
        self.assertEqual(report.stage, "post_tokenization")

    def test_no_special_ids_in_tokenizer_passes(self):
        """If the tokenizer has no special ids, the check should pass."""
        tok = _FakeTokenizer(special_ids=[])
        report = check_tokenized_prompt([5, 10, 15], tok)
        self.assertFalse(report.is_degenerate)


class CheckTrainableTokensTest(unittest.TestCase):
    """Token-level response checks detect all-prompt (no trainable) masks."""

    def test_normal_mask_passes(self):
        """A mask with at least one trainable position after the prefix passes."""
        report = check_trainable_tokens([True, True, False, False])
        self.assertFalse(report.is_degenerate)

    def test_all_prompt_detected(self):
        """A mask where every position is prompt should be flagged as NO_TRAINABLE_TOKENS."""
        report = check_trainable_tokens([True, True, True])
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.NO_TRAINABLE_TOKENS)
        self.assertEqual(report.stage, "post_tokenization")

    def test_single_element_mask_detected(self):
        """A single-element mask has no trainable positions after the prefix."""
        report = check_trainable_tokens([True])
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.NO_TRAINABLE_TOKENS)


class CheckPreferencePairTest(unittest.TestCase):
    """DPO preference pair checks detect identical branches."""

    def test_different_branches_pass(self):
        """Different chosen and rejected values should not be flagged."""
        report = check_preference_pair("good answer", "bad answer")
        self.assertFalse(report.is_degenerate)

    def test_identical_string_branches_detected(self):
        """Identical string branches should be flagged."""
        report = check_preference_pair("same", "same")
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.IDENTICAL_PREFERENCE_BRANCHES)

    def test_identical_list_branches_detected(self):
        """Identical message-list branches should be flagged."""
        chosen = [{"role": "user", "content": "hi"}]
        rejected = [{"role": "user", "content": "hi"}]
        report = check_preference_pair(chosen, rejected)
        self.assertTrue(report.is_degenerate)
        self.assertEqual(report.reason, DegenerateReason.IDENTICAL_PREFERENCE_BRANCHES)


class ApplyDegeneratePolicyTest(unittest.TestCase):
    """Policy application respects enabled, SKIP, and ERROR modes."""

    def test_normal_sample_not_skipped(self):
        """A non-degenerate report should never trigger a skip."""
        config = DegenerateFilterConfig()
        self.assertFalse(apply_degenerate_policy(SampleQualityReport.ok(), config))

    def test_policy_skip_returns_true(self):
        """SKIP policy on a degenerate sample should return True (skip it)."""
        report = SampleQualityReport.degenerate(DegenerateReason.EMPTY, "pre_tokenization", "test")
        config = DegenerateFilterConfig(policy=DegeneratePolicy.SKIP)
        self.assertTrue(apply_degenerate_policy(report, config))

    def test_policy_error_raises(self):
        """ERROR policy on a degenerate sample should raise ValueError."""
        report = SampleQualityReport.degenerate(
            DegenerateReason.WHITESPACE_ONLY, "pre_tokenization", "test detail"
        )
        config = DegenerateFilterConfig(policy=DegeneratePolicy.ERROR)
        with self.assertRaisesRegex(ValueError, "degenerate sample detected.*test detail"):
            apply_degenerate_policy(report, config)

    def test_disabled_config_passes(self):
        """When enabled=False, even degenerate samples should not be skipped."""
        report = SampleQualityReport.degenerate(DegenerateReason.EMPTY, "pre_tokenization", "test")
        config = DegenerateFilterConfig(enabled=False)
        self.assertFalse(apply_degenerate_policy(report, config))

    def test_default_config_is_skip_enabled(self):
        """The default DegenerateFilterConfig should be enabled with SKIP policy."""
        config = DegenerateFilterConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.policy, DegeneratePolicy.SKIP)


class DegenerateReasonCountingTest(unittest.TestCase):
    """Reason counting and formatting helpers produce correct output."""

    def test_record_and_format_reasons(self):
        """record_degenerate_reason should increment and format_degenerate_reasons should render."""
        counts: dict[str, int] = {}
        report_empty = SampleQualityReport.degenerate(DegenerateReason.EMPTY, "pre", "")
        report_ws = SampleQualityReport.degenerate(DegenerateReason.WHITESPACE_ONLY, "pre", "")
        record_degenerate_reason(counts, report_empty)
        record_degenerate_reason(counts, report_empty)
        record_degenerate_reason(counts, report_ws)
        self.assertEqual(counts["empty"], 2)
        self.assertEqual(counts["whitespace_only"], 1)
        formatted = format_degenerate_reasons(counts)
        self.assertIn("empty=2", formatted)
        self.assertIn("whitespace_only=1", formatted)

    def test_format_empty_counts(self):
        """format_degenerate_reasons on an empty dict should return an empty string."""
        self.assertEqual(format_degenerate_reasons({}), "")


if __name__ == "__main__":
    unittest.main()
