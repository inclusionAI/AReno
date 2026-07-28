"""CPU tests for configurable dataset field mapping and sample filtering.

These tests cover the core logic in ``areno.api.data_utils`` for:
- declarative field renaming (``apply_field_mapping``)
- constant-field injection (``apply_constant_fields``)
- sample filtering by field presence and text length (``check_sample_filter``)
- the combined ``transform_dataset`` pipeline
- JSON option parsing (``parse_json_option``)
- backward compatibility when no options are provided

All tests run on CPU without GPU, torch, or network access.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from areno.api.data_utils import (
    FilterSummary,
    apply_constant_fields,
    apply_field_mapping,
    check_sample_filter,
    parse_json_option,
    transform_dataset,
)


class FieldMappingTest(unittest.TestCase):
    """Tests for ``apply_field_mapping`` — declarative field renaming."""

    def test_renames_source_to_target(self):
        """A source field present should move to the target name."""
        record = {"question": "What is 1+1?", "answer": "2"}
        result = apply_field_mapping(dict(record), {"question": "prompt", "answer": "response"})
        self.assertIn("prompt", result)
        self.assertIn("response", result)
        self.assertNotIn("question", result)
        self.assertNotIn("answer", result)
        self.assertEqual(result["prompt"], "What is 1+1?")
        self.assertEqual(result["response"], "2")

    def test_does_not_overwrite_existing_target(self):
        """If target already exists, source is left in place."""
        record = {"question": "What?", "prompt": "Existing prompt"}
        result = apply_field_mapping(dict(record), {"question": "prompt"})
        self.assertEqual(result["prompt"], "Existing prompt")
        self.assertIn("question", result)

    def test_identity_mapping_is_noop(self):
        """Mapping a field to itself should not move or duplicate."""
        record = {"prompt": "hello"}
        result = apply_field_mapping(dict(record), {"prompt": "prompt"})
        self.assertEqual(result, record)

    def test_missing_source_is_silently_skipped(self):
        """If source is absent, nothing happens."""
        record = {"prompt": "hello"}
        result = apply_field_mapping(dict(record), {"question": "prompt"})
        self.assertEqual(result, {"prompt": "hello"})

    def test_preserves_unmapped_fields(self):
        """Fields not mentioned in mapping should survive untouched."""
        record = {"question": "Hi", "answer": "Hello", "metadata": {"id": 1}}
        result = apply_field_mapping(dict(record), {"question": "prompt"})
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"], {"id": 1})


class ConstantFieldsTest(unittest.TestCase):
    """Tests for ``apply_constant_fields`` — inject fixed key/values."""

    def test_injects_new_fields(self):
        record = {"prompt": "hello"}
        result = apply_constant_fields(dict(record), {"task_type": "math", "split": "train"})
        self.assertEqual(result["task_type"], "math")
        self.assertEqual(result["split"], "train")

    def test_does_not_overwrite_existing(self):
        record = {"prompt": "hello", "task_type": "logic"}
        result = apply_constant_fields(dict(record), {"task_type": "math"})
        self.assertEqual(result["task_type"], "logic")

    def test_empty_constants_is_noop(self):
        record = {"prompt": "hello"}
        result = apply_constant_fields(dict(record), {})
        self.assertEqual(result, record)


class SampleFilterTest(unittest.TestCase):
    """Tests for ``check_sample_filter`` — predicate-based record acceptance."""

    def test_keeps_record_with_all_required_fields(self):
        record = {"prompt": "hello", "response": "world"}
        keep, reason = check_sample_filter(record, {"require_fields": ["prompt", "response"]})
        self.assertTrue(keep)
        self.assertIsNone(reason)

    def test_drops_record_missing_required_field(self):
        record = {"prompt": "hello"}
        keep, reason = check_sample_filter(record, {"require_fields": ["prompt", "response"]})
        self.assertFalse(keep)
        self.assertIn("missing field: response", reason)

    def test_drops_record_with_empty_required_field(self):
        record = {"prompt": "hello", "response": ""}
        keep, reason = check_sample_filter(record, {"require_fields": ["response"]})
        self.assertFalse(keep)
        self.assertIn("empty field: response", reason)

    def test_min_prompt_chars_keeps_long_enough(self):
        record = {"prompt": "a" * 10}
        keep, reason = check_sample_filter(record, {"min_prompt_chars": 5})
        self.assertTrue(keep)

    def test_min_prompt_chars_drops_too_short(self):
        record = {"prompt": "ab"}
        keep, reason = check_sample_filter(record, {"min_prompt_chars": 5})
        self.assertFalse(keep)
        self.assertIn("prompt too short", reason)

    def test_max_prompt_chars_drops_too_long(self):
        record = {"prompt": "a" * 100}
        keep, reason = check_sample_filter(record, {"max_prompt_chars": 50})
        self.assertFalse(keep)
        self.assertIn("prompt too long", reason)

    def test_min_response_chars_drops_too_short(self):
        record = {"prompt": "hello", "response": "x"}
        keep, reason = check_sample_filter(record, {"min_response_chars": 5})
        self.assertFalse(keep)
        self.assertIn("response too short", reason)

    def test_empty_filter_keeps_everything(self):
        record = {"anything": 1}
        keep, reason = check_sample_filter(record, {})
        self.assertTrue(keep)
        self.assertIsNone(reason)

    def test_boundary_exact_min_length(self):
        """A prompt exactly at the minimum length should be kept."""
        record = {"prompt": "a" * 5}
        keep, reason = check_sample_filter(record, {"min_prompt_chars": 5})
        self.assertTrue(keep)

    def test_boundary_exact_max_length(self):
        """A prompt exactly at the maximum length should be kept."""
        record = {"prompt": "a" * 50}
        keep, reason = check_sample_filter(record, {"max_prompt_chars": 50})
        self.assertTrue(keep)


class TransformDatasetTest(unittest.TestCase):
    """Integration tests for the full ``transform_dataset`` pipeline."""

    def _fixture_a(self) -> list[dict]:
        """Dataset using ``question``/``answer`` field names."""
        return [
            {"question": "What is 1+1?", "answer": "2"},
            {"question": "Hi", "answer": "Hello"},
            {"question": "ab", "answer": "cd"},
        ]

    def _fixture_b(self) -> list[dict]:
        """Dataset using ``query``/``response`` field names."""
        return [
            {"query": "Explain gravity", "response": "It attracts mass"},
            {"query": "x", "response": "y"},
        ]

    def test_two_datasets_converge_to_same_contract(self):
        """Both fixtures should produce ``prompt``/``response`` after mapping."""
        mapping_a = {"question": "prompt", "answer": "response"}
        mapping_b = {"query": "prompt"}

        kept_a, summary_a = transform_dataset(self._fixture_a(), field_mapping=mapping_a)
        kept_b, summary_b = transform_dataset(self._fixture_b(), field_mapping=mapping_b)

        for record in kept_a + kept_b:
            self.assertIn("prompt", record)
            self.assertIn("response", record)

    def test_deterministic_filtering(self):
        """Same input + same config should always produce same output."""
        mapping = {"question": "prompt", "answer": "response"}
        filt = {"min_prompt_chars": 3}

        kept1, summary1 = transform_dataset(self._fixture_a(), field_mapping=mapping, sample_filter=filt)
        kept2, summary2 = transform_dataset(self._fixture_a(), field_mapping=mapping, sample_filter=filt)

        self.assertEqual(kept1, kept2)
        self.assertEqual(summary1.total_kept, summary2.total_kept)
        self.assertEqual(summary1.total_dropped, summary2.total_dropped)

    def test_summary_reconciles_with_input(self):
        """kept + dropped must equal total_in."""
        mapping = {"question": "prompt", "answer": "response"}
        filt = {"min_prompt_chars": 3}

        kept, summary = transform_dataset(self._fixture_a(), field_mapping=mapping, sample_filter=filt)

        self.assertEqual(summary.total_kept + summary.total_dropped, summary.total_in)
        self.assertEqual(len(kept), summary.total_kept)

    def test_drop_reasons_are_human_readable(self):
        """Drop reasons should describe why a record was filtered."""
        mapping = {"question": "prompt", "answer": "response"}
        filt = {"min_prompt_chars": 3}

        _, summary = transform_dataset(self._fixture_a(), field_mapping=mapping, sample_filter=filt)

        self.assertGreater(len(summary.drop_reasons), 0)
        for reason in summary.drop_reasons:
            self.assertIsInstance(reason, str)
            self.assertTrue(reason)

    def test_constant_fields_appear_in_all_kept_records(self):
        """Constant fields should be present in every kept record."""
        mapping = {"question": "prompt", "answer": "response"}
        constants = {"task_type": "math"}

        kept, _ = transform_dataset(self._fixture_a(), field_mapping=mapping, constant_fields=constants)

        for record in kept:
            self.assertEqual(record["task_type"], "math")

    def test_does_not_mutate_input(self):
        """The original dataset records should remain unchanged."""
        original = self._fixture_a()
        original_copy = [dict(r) for r in original]
        mapping = {"question": "prompt", "answer": "response"}

        transform_dataset(original, field_mapping=mapping)

        self.assertEqual(original, original_copy)

    def test_backward_compatible_when_no_options(self):
        """Without any options, the dataset should be returned unchanged."""
        data = self._fixture_a()
        kept, summary = transform_dataset(data)

        self.assertEqual(len(kept), len(data))
        self.assertEqual(summary.total_kept, len(data))
        self.assertEqual(summary.total_dropped, 0)
        # The content should match (shallow copy, so different identity).
        for orig, kept_record in zip(data, kept, strict=True):
            self.assertEqual(orig, kept_record)

    def test_empty_dataset(self):
        """An empty dataset should produce an empty result and zero summary."""
        kept, summary = transform_dataset([])
        self.assertEqual(kept, [])
        self.assertEqual(summary.total_in, 0)
        self.assertEqual(summary.total_kept, 0)

    def test_all_filtered_out(self):
        """When everything is dropped, kept is empty and summary reports it."""
        data = [{"prompt": "a"}]
        kept, summary = transform_dataset(data, sample_filter={"min_prompt_chars": 100})
        self.assertEqual(kept, [])
        self.assertEqual(summary.total_dropped, 1)
        self.assertEqual(summary.total_kept, 0)


class JsonOptionParsingTest(unittest.TestCase):
    """Tests for ``parse_json_option`` — CLI JSON string parsing."""

    def test_none_returns_none(self):
        self.assertIsNone(parse_json_option(None, "--test"))

    def test_valid_json_object(self):
        result = parse_json_option('{"key": "value"}', "--test")
        self.assertEqual(result, {"key": "value"})

    def test_invalid_json_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            parse_json_option("{invalid", "--test")

    def test_non_object_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            parse_json_option("[1, 2, 3]", "--test")

    def test_plain_string_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            parse_json_option('"hello"', "--test")


class FilterSummaryTest(unittest.TestCase):
    """Tests for ``FilterSummary`` log output formatting."""

    def test_log_lines_contain_counts(self):
        summary = FilterSummary(
            total_in=10, total_kept=7, total_dropped=3,
            drop_reasons={"too short": 2, "missing field": 1},
        )
        lines = summary.as_log_lines()
        self.assertTrue(any("scanned=10" in line for line in lines))
        self.assertTrue(any("kept=7" in line for line in lines))
        self.assertTrue(any("dropped=3" in line for line in lines))
        self.assertTrue(any("too short" in line for line in lines))

    def test_zero_drops_produces_single_line(self):
        summary = FilterSummary(total_in=5, total_kept=5, total_dropped=0)
        lines = summary.as_log_lines()
        self.assertEqual(len(lines), 1)


class SftFixtureConversionTest(unittest.TestCase):
    """Integration-style test: convert two differently shaped JSONL fixtures
    into one SFT contract without modifying source files, per acceptance criteria.
    """

    def test_two_jsonl_fixtures_to_sft_contract(self):
        """Write two JSONL files with different schemas, then map both to
        ``prompt``/``response`` and verify deterministic filtering."""

        fixture_a = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "ab", "answer": "cd"},  # will be filtered (too short)
        ]
        fixture_b = [
            {"query": "Explain photosynthesis", "response": "Plants convert light to energy"},
            {"query": "x", "response": "y"},  # will be filtered (too short)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "data_a.jsonl"
            path_b = Path(tmpdir) / "data_b.jsonl"
            path_a.write_text("\n".join(json.dumps(r) for r in fixture_a), encoding="utf-8")
            path_b.write_text("\n".join(json.dumps(r) for r in fixture_b), encoding="utf-8")

            # Load both fixtures.
            records_a = [json.loads(line) for line in path_a.read_text(encoding="utf-8").splitlines() if line.strip()]
            records_b = [json.loads(line) for line in path_b.read_text(encoding="utf-8").splitlines() if line.strip()]

            # Convert fixture A: question → prompt, answer → response.
            kept_a, summary_a = transform_dataset(
                records_a,
                field_mapping={"question": "prompt", "answer": "response"},
                sample_filter={"require_fields": ["prompt", "response"], "min_prompt_chars": 5},
            )

            # Convert fixture B: query → prompt (response already correct).
            kept_b, summary_b = transform_dataset(
                records_b,
                field_mapping={"query": "prompt"},
                sample_filter={"require_fields": ["prompt", "response"], "min_prompt_chars": 5},
            )

            # Both should produce prompt/response records.
            all_kept = kept_a + kept_b
            for record in all_kept:
                self.assertIn("prompt", record)
                self.assertIn("response", record)
                self.assertGreaterEqual(len(record["prompt"]), 5)

            # Summary counts reconcile.
            self.assertEqual(summary_a.total_kept + summary_a.total_dropped, summary_a.total_in)
            self.assertEqual(summary_b.total_kept + summary_b.total_dropped, summary_b.total_in)

            # Source files are not modified.
            self.assertEqual(
                [json.loads(line) for line in path_a.read_text(encoding="utf-8").splitlines() if line.strip()],
                fixture_a,
            )
            self.assertEqual(
                [json.loads(line) for line in path_b.read_text(encoding="utf-8").splitlines() if line.strip()],
                fixture_b,
            )


if __name__ == "__main__":
    unittest.main()