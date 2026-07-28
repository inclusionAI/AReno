"""CPU tests for near-duplicate training example detection (issue #218).

These tests exercise the pure-Python dedup module without any GPU or torch
dependency.  They cover:

* Text normalisation (exact, formatting-only, empty, unicode).
* Exact-mode duplicate detection (exact, formatting-only, no duplicates).
* Near-mode duplicate detection (paraphrases, minor edits, threshold boundary).
* Invalid inputs (bad mode, bad threshold, non-list records).
* Boundary values (empty list, single record, all duplicates).
* Deterministic output.
* Structured (to_dict) and human-readable (format_duplicate_report) output.
* Scope modes (prompt vs full).
* text_keys customisation.
"""

from __future__ import annotations

import unittest

from areno.dedup import (
    find_duplicates,
    format_duplicate_report,
    jaccard_similarity,
    normalize_text,
    shingle_signature,
)

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


class TestNormalizeText(unittest.TestCase):
    """normalize_text should produce canonical forms for duplicate detection."""

    def test_lowercase(self):
        self.assertEqual(normalize_text("Hello World"), "hello world")

    def test_strip_punctuation(self):
        self.assertEqual(normalize_text("Hello, World!"), "hello world")

    def test_collapse_whitespace(self):
        self.assertEqual(normalize_text("Hello   World\t\n"), "hello world")

    def test_empty_string(self):
        self.assertEqual(normalize_text(""), "")

    def test_none_like(self):
        # Empty string is the falsy case; normalize_text handles it.
        self.assertEqual(normalize_text(""), "")

    def test_unicode_nfc(self):
        # NFC normalisation: composed vs decomposed forms should match.
        composed = "caf\u00e9"  # é as single char
        decomposed = "cafe\u0301"  # e + combining accent
        self.assertEqual(normalize_text(composed), normalize_text(decomposed))

    def test_formatting_only_difference(self):
        a = "What is 2 + 2?"
        b = "what is 2 + 2"
        self.assertEqual(normalize_text(a), normalize_text(b))


# ---------------------------------------------------------------------------
# Shingling and Jaccard similarity
# ---------------------------------------------------------------------------


class TestShingling(unittest.TestCase):
    """shingle_signature and jaccard_similarity should work correctly."""

    def test_identical_texts_have_similarity_1(self):
        sig = shingle_signature("the quick brown fox")
        self.assertEqual(jaccard_similarity(sig, sig), 1.0)

    def test_disjoint_texts_have_similarity_0(self):
        sig_a = shingle_signature("aaaaaaa")
        sig_b = shingle_signature("zzzzzzz")
        self.assertEqual(jaccard_similarity(sig_a, sig_b), 0.0)

    def test_empty_text_returns_empty_set(self):
        self.assertEqual(shingle_signature(""), frozenset())

    def test_short_text_returns_single_shingle(self):
        sig = shingle_signature("hi", ngram_size=5)
        self.assertEqual(sig, frozenset({"hi"}))

    def test_max_features_caps_set_size(self):
        long_text = "word " * 1000
        sig = shingle_signature(long_text, max_features=10)
        self.assertLessEqual(len(sig), 10)

    def test_jaccard_both_empty(self):
        self.assertEqual(jaccard_similarity(frozenset(), frozenset()), 0.0)


# ---------------------------------------------------------------------------
# Exact-mode duplicate detection
# ---------------------------------------------------------------------------


class TestExactMode(unittest.TestCase):
    """find_duplicates in exact mode should catch formatting-only and exact dupes."""

    def setUp(self):
        self.records = [
            {"prompt": "What is 2 + 2?", "answer": "4"},
            {"prompt": "what is 2 + 2", "answer": "4"},  # formatting-only duplicate
            {"prompt": "Solve: 3 * 5", "answer": "15"},
            {"prompt": "What is 2 + 2?", "answer": "4"},  # exact duplicate of record 0
            {"prompt": "Explain photosynthesis", "answer": "..."},
        ]

    def test_finds_exact_duplicates(self):
        report = find_duplicates(self.records, mode="exact")
        self.assertEqual(len(report.groups), 1)
        self.assertEqual(report.groups[0].record_indices, [0, 1, 3])
        self.assertEqual(report.groups[0].similarity, 1.0)
        self.assertEqual(report.groups[0].match_type, "exact")

    def test_duplicate_count(self):
        report = find_duplicates(self.records, mode="exact")
        self.assertEqual(report.duplicate_records, 2)  # 3 in group, minus 1 representative
        self.assertEqual(report.unique_records, 3)  # 5 total - 2 duplicates

    def test_total_records(self):
        report = find_duplicates(self.records, mode="exact")
        self.assertEqual(report.total_records, 5)

    def test_no_duplicates(self):
        records = [
            {"prompt": "unique question one"},
            {"prompt": "unique question two"},
            {"prompt": "unique question three"},
        ]
        report = find_duplicates(records, mode="exact")
        self.assertEqual(len(report.groups), 0)
        self.assertEqual(report.duplicate_records, 0)
        self.assertEqual(report.unique_records, 3)

    def test_empty_list(self):
        report = find_duplicates([], mode="exact")
        self.assertEqual(report.groups, [])
        self.assertEqual(report.total_records, 0)
        self.assertEqual(report.duplicate_fraction, 0.0)

    def test_single_record(self):
        report = find_duplicates([{"prompt": "hello"}], mode="exact")
        self.assertEqual(report.groups, [])
        self.assertEqual(report.total_records, 1)

    def test_all_duplicates(self):
        records = [{"prompt": "same"} for _ in range(5)]
        report = find_duplicates(records, mode="exact")
        self.assertEqual(len(report.groups), 1)
        self.assertEqual(len(report.groups[0].record_indices), 5)
        self.assertEqual(report.duplicate_records, 4)


# ---------------------------------------------------------------------------
# Near-mode duplicate detection
# ---------------------------------------------------------------------------


class TestNearMode(unittest.TestCase):
    """find_duplicates in near mode should catch paraphrases and minor edits."""

    def setUp(self):
        self.records = [
            {"prompt": "What is the capital of France?"},
            {"prompt": "What is the capital of France"},  # minor edit (missing ?)
            {"prompt": "Name the capital city of France"},  # paraphrase
            {"prompt": "How does photosynthesis work in plants?"},
            {"prompt": "Explain the process of photosynthesis in plants"},
        ]

    def test_finds_near_duplicates(self):
        report = find_duplicates(self.records, mode="near", threshold=0.3)
        self.assertGreater(len(report.groups), 0)

    def test_high_threshold_fewer_groups(self):
        low_thresh = find_duplicates(self.records, mode="near", threshold=0.1)
        high_thresh = find_duplicates(self.records, mode="near", threshold=0.9)
        self.assertGreaterEqual(len(low_thresh.groups), len(high_thresh.groups))

    def test_threshold_1_only_exact(self):
        report = find_duplicates(self.records, mode="near", threshold=1.0)
        # Only truly identical (after normalisation) should match.
        for g in report.groups:
            self.assertEqual(g.similarity, 1.0)

    def test_similarity_in_range(self):
        report = find_duplicates(self.records, mode="near", threshold=0.1)
        for g in report.groups:
            self.assertGreaterEqual(g.similarity, 0.0)
            self.assertLessEqual(g.similarity, 1.0)

    def test_match_type_is_near(self):
        report = find_duplicates(self.records, mode="near", threshold=0.3)
        for g in report.groups:
            self.assertEqual(g.match_type, "near")


# ---------------------------------------------------------------------------
# Precision: exact, formatting-only, near-duplicate groups
# ---------------------------------------------------------------------------


class TestPrecisionScenarios(unittest.TestCase):
    """Evaluate precision on synthetic exact, formatting-only, and near-duplicate groups."""

    def test_exact_group(self):
        records = [
            {"prompt": "Solve x + 1 = 2"},
            {"prompt": "Solve x + 1 = 2"},  # exact copy
        ]
        report = find_duplicates(records, mode="exact")
        self.assertEqual(len(report.groups), 1)
        self.assertEqual(report.groups[0].record_indices, [0, 1])

    def test_formatting_only_group(self):
        records = [
            {"prompt": "Solve:  x + 1 = 2!"},
            {"prompt": "solve x+1=2"},  # formatting-only variation
        ]
        report = find_duplicates(records, mode="exact")
        self.assertEqual(len(report.groups), 1)

    def test_near_duplicate_group(self):
        records = [
            {"prompt": "What is the capital of France?"},
            {"prompt": "What is the capital of France"},  # missing punctuation
        ]
        report = find_duplicates(records, mode="near", threshold=0.5)
        self.assertEqual(len(report.groups), 1)

    def test_non_duplicate_not_grouped(self):
        records = [
            {"prompt": "What is the capital of France?"},
            {"prompt": "How do I bake a chocolate cake?"},
        ]
        report = find_duplicates(records, mode="near", threshold=0.5)
        self.assertEqual(len(report.groups), 0)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


class TestInvalidInputs(unittest.TestCase):
    """find_duplicates should raise on invalid parameters."""

    def test_non_list_records(self):
        with self.assertRaises(TypeError):
            find_duplicates("not a list")  # type: ignore[arg-type]

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            find_duplicates([], mode="invalid")  # type: ignore[arg-type]

    def test_bad_scope(self):
        with self.assertRaises(ValueError):
            find_duplicates([], scope="invalid")  # type: ignore[arg-type]

    def test_threshold_zero(self):
        with self.assertRaises(ValueError):
            find_duplicates([], mode="near", threshold=0.0)

    def test_threshold_above_one(self):
        with self.assertRaises(ValueError):
            find_duplicates([], mode="near", threshold=1.5)

    def test_threshold_negative(self):
        with self.assertRaises(ValueError):
            find_duplicates([], mode="near", threshold=-0.1)

    def test_bad_ngram_size(self):
        with self.assertRaises(ValueError):
            find_duplicates([], mode="near", ngram_size=0)

    def test_bad_max_features(self):
        with self.assertRaises(ValueError):
            find_duplicates([], mode="near", max_features=0)


# ---------------------------------------------------------------------------
# Scope: prompt vs full
# ---------------------------------------------------------------------------


class TestScopeModes(unittest.TestCase):
    """prompt scope compares only the primary text; full compares all fields."""

    def test_prompt_scope_ignores_different_answers(self):
        records = [
            {"prompt": "same question", "answer": "answer A"},
            {"prompt": "same question", "answer": "answer B"},
        ]
        report = find_duplicates(records, mode="exact", scope="prompt")
        self.assertEqual(len(report.groups), 1)

    def test_full_scope_detects_different_answers(self):
        records = [
            {"prompt": "same question", "answer": "answer A"},
            {"prompt": "same question", "answer": "answer B"},
        ]
        report = find_duplicates(records, mode="exact", scope="full")
        self.assertEqual(len(report.groups), 0)


# ---------------------------------------------------------------------------
# text_keys customisation
# ---------------------------------------------------------------------------


class TestTextKeys(unittest.TestCase):
    """Custom text_keys should control which fields are extracted."""

    def test_custom_text_keys(self):
        records = [
            {"my_field": "duplicate text", "other": "A"},
            {"my_field": "duplicate text", "other": "B"},
        ]
        report = find_duplicates(records, mode="exact", text_keys=["my_field"])
        self.assertEqual(len(report.groups), 1)

    def test_custom_text_keys_not_found_falls_back(self):
        records = [
            {"my_field": "unique A", "prompt": "same"},
            {"my_field": "unique B", "prompt": "same"},
        ]
        # text_keys not found -> falls back to any string value.
        # Fallback may pick different fields for different records,
        # so duplicates are not guaranteed.  Just verify it doesn't crash.
        report = find_duplicates(records, mode="exact", text_keys=["nonexistent"])
        self.assertEqual(report.total_records, 2)


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------


class TestDeterministicOutput(unittest.TestCase):
    """The same inputs should produce the same outputs every time."""

    def setUp(self):
        self.records = [
            {"prompt": "hello world"},
            {"prompt": "hello world"},
            {"prompt": "unique question"},
        ]

    def test_exact_mode_deterministic(self):
        r1 = find_duplicates(self.records, mode="exact")
        r2 = find_duplicates(self.records, mode="exact")
        self.assertEqual(r1.to_dict(), r2.to_dict())

    def test_near_mode_deterministic(self):
        r1 = find_duplicates(self.records, mode="near", threshold=0.5)
        r2 = find_duplicates(self.records, mode="near", threshold=0.5)
        self.assertEqual(r1.to_dict(), r2.to_dict())


# ---------------------------------------------------------------------------
# Structured output (to_dict)
# ---------------------------------------------------------------------------


class TestToDict(unittest.TestCase):
    """DuplicateReport.to_dict should produce a complete JSON structure."""

    def test_to_dict_has_required_fields(self):
        report = find_duplicates([{"prompt": "a"}, {"prompt": "a"}], mode="exact")
        d = report.to_dict()
        self.assertIn("groups", d)
        self.assertIn("total_records", d)
        self.assertIn("duplicate_records", d)
        self.assertIn("unique_records", d)
        self.assertIn("match_type", d)
        self.assertIn("threshold", d)
        self.assertIn("ngram_size", d)

    def test_to_dict_group_fields(self):
        report = find_duplicates([{"prompt": "a"}, {"prompt": "a"}], mode="exact")
        d = report.to_dict()
        self.assertTrue(len(d["groups"]) > 0)
        g = d["groups"][0]
        self.assertIn("group_id", g)
        self.assertIn("record_indices", g)
        self.assertIn("similarity", g)
        self.assertIn("match_type", g)


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------


class TestFormatReport(unittest.TestCase):
    """format_duplicate_report should produce readable output."""

    def test_contains_summary_stats(self):
        report = find_duplicates([{"prompt": "a"}, {"prompt": "a"}, {"prompt": "b"}], mode="exact")
        text = format_duplicate_report(report)
        self.assertIn("Total records: 3", text)
        self.assertIn("Duplicate records: 1", text)
        self.assertIn("Unique records: 2", text)

    def test_contains_group_info(self):
        report = find_duplicates([{"prompt": "a"}, {"prompt": "a"}], mode="exact")
        text = format_duplicate_report(report)
        self.assertIn("Group", text)
        self.assertIn("similarity", text)

    def test_no_groups_output(self):
        report = find_duplicates([{"prompt": "a"}, {"prompt": "b"}], mode="exact")
        text = format_duplicate_report(report)
        self.assertIn("Duplicate groups: 0", text)


# ---------------------------------------------------------------------------
# Backward compatibility: default behavior
# ---------------------------------------------------------------------------


class TestBackwardCompatible(unittest.TestCase):
    """Default behavior should not crash and should preserve existing data."""

    def test_does_not_mutate_records(self):
        records = [{"prompt": "hello"}, {"prompt": "hello"}]
        original = [dict(r) for r in records]
        find_duplicates(records, mode="exact")
        self.assertEqual(records, original)

    def test_default_mode_is_exact(self):
        report = find_duplicates([{"prompt": "a"}, {"prompt": "a"}])
        self.assertEqual(report.match_type, "exact")

    def test_duplicate_fraction(self):
        report = find_duplicates([{"prompt": "a"}, {"prompt": "a"}, {"prompt": "b"}], mode="exact")
        self.assertAlmostEqual(report.duplicate_fraction, 1 / 3)


if __name__ == "__main__":
    unittest.main()
