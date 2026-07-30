"""Integration tests for dedup CLI command (issue #218).

These tests create temporary JSONL/JSON fixture files and invoke the
``areno dedup`` CLI command through Click's test runner, verifying
end-to-end behaviour including file loading, detection, and output
formatting.

No GPU or torch dependency is required.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from click.testing import CliRunner

from areno.cli.dedup import dedup_command
from areno.cli.main import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_JSONL_WITH_DUPES = (
    '{"prompt": "What is 2 + 2?", "answer": "4"}\n'
    '{"prompt": "what is 2 + 2", "answer": "4"}\n'
    '{"prompt": "Solve: 3 * 5", "answer": "15"}\n'
    '{"prompt": "What is 2 + 2?", "answer": "4"}\n'
    '{"prompt": "Explain photosynthesis", "answer": "..."}\n'
)

_JSONL_NO_DUPES = (
    '{"prompt": "unique question one"}\n{"prompt": "unique question two"}\n{"prompt": "unique question three"}\n'
)

_JSONL_NEAR_DUPES = (
    '{"prompt": "What is the capital of France?"}\n'
    '{"prompt": "What is the capital of France"}\n'
    '{"prompt": "Name the capital city of France"}\n'
    '{"prompt": "How do I bake a cake?"}\n'
)

_JSON_EMPTY_ARRAY = "[]"

_JSON_ARRAY_WITH_DUPES = json.dumps(
    [
        {"prompt": "hello world"},
        {"prompt": "hello world"},
        {"prompt": "unique"},
    ]
)


def _write_temp_file(content: str, suffix: str = ".jsonl") -> str:
    """Write content to a temp file and return the path."""

    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, content.encode("utf-8"))
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# CLI: exact mode
# ---------------------------------------------------------------------------


class TestCLIExactMode(unittest.TestCase):
    """areno dedup --mode exact should find exact and formatting-only duplicates."""

    def setUp(self):
        self.runner = CliRunner()
        self.path = _write_temp_file(_JSONL_WITH_DUPES)

    def tearDown(self):
        os.unlink(self.path)

    def test_finds_duplicates_human_readable(self):
        result = self.runner.invoke(dedup_command, ["--data-path", self.path])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Duplicate detection", result.output)
        self.assertIn("Total records: 5", result.output)
        self.assertIn("Duplicate records: 2", result.output)

    def test_finds_duplicates_json(self):
        result = self.runner.invoke(dedup_command, ["--data-path", self.path, "--json"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data["total_records"], 5)
        self.assertEqual(data["match_type"], "exact")
        self.assertTrue(len(data["groups"]) >= 1)
        self.assertIn("record_indices", data["groups"][0])

    def test_top_level_cli_dispatches_to_dedup(self):
        result = self.runner.invoke(main, ["dedup", "--data-path", self.path, "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(json.loads(result.output)["duplicate_records"], 2)

    def test_no_duplicates(self):
        path = _write_temp_file(_JSONL_NO_DUPES)
        try:
            result = self.runner.invoke(dedup_command, ["--data-path", path])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Duplicate groups: 0", result.output)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI: near mode
# ---------------------------------------------------------------------------


class TestCLINearMode(unittest.TestCase):
    """areno dedup --mode near should find approximate matches."""

    def setUp(self):
        self.runner = CliRunner()
        self.path = _write_temp_file(_JSONL_NEAR_DUPES)

    def tearDown(self):
        os.unlink(self.path)

    def test_near_mode_finds_groups(self):
        result = self.runner.invoke(
            dedup_command,
            ["--data-path", self.path, "--mode", "near", "--threshold", "0.3"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("near", result.output)

    def test_near_mode_json(self):
        result = self.runner.invoke(
            dedup_command,
            [
                "--data-path",
                self.path,
                "--mode",
                "near",
                "--threshold",
                "0.3",
                "--max-features",
                "64",
                "--max-comparisons",
                "100",
                "--json",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(data["match_type"], "near")
        self.assertEqual(data["threshold"], 0.3)
        self.assertEqual(data["max_features"], 64)
        self.assertEqual(data["max_comparisons"], 100)
        self.assertGreater(data["candidate_comparisons"], 0)
        self.assertEqual(data["scope"], "prompt")


# ---------------------------------------------------------------------------
# CLI: scope modes
# ---------------------------------------------------------------------------


class TestCLIScopeModes(unittest.TestCase):
    """--scope prompt vs --scope full should change comparison behaviour."""

    def setUp(self):
        self.runner = CliRunner()
        self.path = _write_temp_file('{"prompt": "same", "answer": "A"}\n{"prompt": "same", "answer": "B"}\n')

    def tearDown(self):
        os.unlink(self.path)

    def test_prompt_scope_finds_duplicate(self):
        result = self.runner.invoke(dedup_command, ["--data-path", self.path, "--scope", "prompt"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Duplicate records: 1", result.output)

    def test_full_scope_no_duplicate(self):
        result = self.runner.invoke(dedup_command, ["--data-path", self.path, "--scope", "full"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Duplicate groups: 0", result.output)


# ---------------------------------------------------------------------------
# CLI: JSON array file
# ---------------------------------------------------------------------------


class TestCLIJsonArray(unittest.TestCase):
    """areno dedup should support .json array files."""

    def setUp(self):
        self.runner = CliRunner()

    def test_json_array_with_dupes(self):
        path = _write_temp_file(_JSON_ARRAY_WITH_DUPES, suffix=".json")
        try:
            result = self.runner.invoke(dedup_command, ["--data-path", path])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Duplicate records: 1", result.output)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI: error handling
# ---------------------------------------------------------------------------


class TestCLIErrorHandling(unittest.TestCase):
    """areno dedup should handle errors gracefully."""

    def setUp(self):
        self.runner = CliRunner()

    def test_file_not_found(self):
        result = self.runner.invoke(dedup_command, ["--data-path", "/nonexistent/file.jsonl"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("File not found", result.output)

    def test_empty_file(self):
        path = _write_temp_file("")
        try:
            result = self.runner.invoke(dedup_command, ["--data-path", path])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("No records", result.output)
        finally:
            os.unlink(path)

    def test_empty_json_array(self):
        path = _write_temp_file(_JSON_EMPTY_ARRAY, suffix=".json")
        try:
            result = self.runner.invoke(dedup_command, ["--data-path", path])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("No records", result.output)
        finally:
            os.unlink(path)

    def test_unsupported_file_type(self):
        path = _write_temp_file("data", suffix=".txt")
        try:
            result = self.runner.invoke(dedup_command, ["--data-path", path])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Unsupported file type", result.output)
        finally:
            os.unlink(path)

    def test_invalid_json_line(self):
        path = _write_temp_file('{"valid": "json"}\n{invalid json}\n')
        try:
            result = self.runner.invoke(dedup_command, ["--data-path", path])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Invalid JSON", result.output)
        finally:
            os.unlink(path)

    def test_invalid_threshold_is_a_cli_error(self):
        path = _write_temp_file(_JSONL_NO_DUPES)
        try:
            result = self.runner.invoke(
                dedup_command,
                ["--data-path", path, "--mode", "near", "--threshold", "0"],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--threshold", result.output)
            self.assertNotIsInstance(result.exception, ValueError)
        finally:
            os.unlink(path)

    def test_invalid_ngram_size_is_a_cli_error(self):
        path = _write_temp_file(_JSONL_NO_DUPES)
        try:
            result = self.runner.invoke(
                dedup_command,
                ["--data-path", path, "--mode", "near", "--ngram-size", "0"],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--ngram-size", result.output)
            self.assertNotIsInstance(result.exception, ValueError)
        finally:
            os.unlink(path)

    def test_invalid_max_comparisons_is_a_cli_error(self):
        path = _write_temp_file(_JSONL_NO_DUPES)
        try:
            result = self.runner.invoke(
                dedup_command,
                ["--data-path", path, "--mode", "near", "--max-comparisons", "0"],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--max-comparisons", result.output)
            self.assertNotIsInstance(result.exception, ValueError)
        finally:
            os.unlink(path)

    def test_missing_prompt_field_is_reported_without_sample_content(self):
        path = _write_temp_file('{"answer": "private answer"}\n')
        try:
            result = self.runner.invoke(dedup_command, ["--data-path", path])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("record at index 0", result.output)
            self.assertNotIn("private answer", result.output)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# CLI: custom text_keys
# ---------------------------------------------------------------------------


class TestCLITextKeys(unittest.TestCase):
    """--text-keys should control which fields are compared."""

    def setUp(self):
        self.runner = CliRunner()
        self.path = _write_temp_file(
            '{"custom_field": "duplicate", "prompt": "different A"}\n'
            '{"custom_field": "duplicate", "prompt": "different B"}\n'
        )

    def tearDown(self):
        os.unlink(self.path)

    def test_custom_text_keys_finds_duplicates(self):
        result = self.runner.invoke(
            dedup_command,
            ["--data-path", self.path, "--text-keys", "custom_field"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Duplicate records: 1", result.output)

    def test_custom_text_keys_are_trimmed(self):
        result = self.runner.invoke(
            dedup_command,
            ["--data-path", self.path, "--text-keys", " custom_field , prompt "],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Duplicate records: 1", result.output)

    def test_default_text_keys_no_duplicates(self):
        """With default text_keys (prompt), these records are unique."""
        result = self.runner.invoke(dedup_command, ["--data-path", self.path])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Duplicate groups: 0", result.output)


# ---------------------------------------------------------------------------
# CLI: source data not mutated
# ---------------------------------------------------------------------------


class TestCLINoMutation(unittest.TestCase):
    """areno dedup must not modify the source file."""

    def setUp(self):
        self.runner = CliRunner()

    def test_source_file_unchanged(self):
        path = _write_temp_file(_JSONL_WITH_DUPES)
        with open(path) as f:
            original_content = f.read()
        try:
            self.runner.invoke(dedup_command, ["--data-path", path])
            with open(path) as f:
                self.assertEqual(f.read(), original_content)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
