"""CPU tests for the streaming JSONL / loader-output quality scanner.

These cover the core scan logic in ``areno.api.data`` (success, malformed
input, boundary values, redaction, bounded issues) and the
``areno scan-dataset`` CLI command (file/stdin/json output/loader-fn paths).
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from areno.api.data import (
    SCAN_BLANK,
    SCAN_JSON_ERROR,
    SCAN_NON_OBJECT,
    SCAN_SCHEMA,
    format_scan_report,
    scan_jsonl_stream,
    scan_loader_output,
)
from areno.cli.diagnostics import scan_dataset_command

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(tmp: str, lines: list[str], name: str = "data.jsonl") -> str:
    path = Path(tmp, name)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Core scanner: scan_jsonl_stream
# ---------------------------------------------------------------------------


class ScanJsonlStreamTest(unittest.TestCase):
    def test_clean_file_reports_ok(self):
        lines = [
            json.dumps({"prompt": "hello", "response": "world"}),
            json.dumps({"prompt": "foo", "response": "bar"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path)

        self.assertTrue(report.ok)
        self.assertEqual(report.total_lines, 2)
        self.assertEqual(report.object_lines, 2)
        self.assertEqual(report.issues, [])

    def test_blank_lines_are_reported(self):
        lines = [
            json.dumps({"a": 1}),
            "",
            "   ",
            json.dumps({"a": 2}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path)

        self.assertFalse(report.ok)
        self.assertEqual(report.blank_lines, 2)
        self.assertEqual(report.total_lines, 4)
        self.assertEqual(report.object_lines, 2)
        blank_issues = [i for i in report.issues if i.category == SCAN_BLANK]
        self.assertEqual(len(blank_issues), 2)
        self.assertEqual(blank_issues[0].line_number, 2)
        self.assertEqual(blank_issues[1].line_number, 3)

    def test_json_parse_errors_are_reported(self):
        lines = [
            '{"a": 1}',
            '{"a": }',  # malformed JSON
            '{"b": 2}',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path)

        self.assertEqual(report.json_error_lines, 1)
        err = [i for i in report.issues if i.category == SCAN_JSON_ERROR]
        self.assertEqual(len(err), 1)
        self.assertEqual(err[0].line_number, 2)
        self.assertIn("invalid JSON", err[0].detail)

    def test_non_object_records_are_reported(self):
        lines = [
            '{"a": 1}',
            "[1, 2, 3]",  # array, not object
            "42",  # scalar
            '{"b": 2}',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path)

        self.assertEqual(report.non_object_lines, 2)
        non_obj = [i for i in report.issues if i.category == SCAN_NON_OBJECT]
        self.assertEqual(len(non_obj), 2)
        self.assertEqual(non_obj[0].line_number, 2)
        self.assertEqual(non_obj[0].detail, "expected JSON object, got list")
        self.assertEqual(non_obj[1].detail, "expected JSON object, got int")

    def test_required_keys_schema_check(self):
        lines = [
            '{"prompt": "hi", "response": "yo"}',
            '{"prompt": "missing response"}',
            '{"response": "missing prompt"}',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path, required_keys=("prompt", "response"))

        self.assertEqual(report.object_lines, 1)
        self.assertEqual(report.schema_error_lines, 2)
        schema = [i for i in report.issues if i.category == SCAN_SCHEMA]
        self.assertEqual(len(schema), 2)
        self.assertEqual(schema[0].line_number, 2)
        self.assertIn("missing required keys: response", schema[0].detail)
        self.assertIn("missing required keys: prompt", schema[1].detail)

    def test_scanning_continues_after_recoverable_errors(self):
        """A bad line should not stop the scan; subsequent good lines count."""

        lines = [
            '{"a": 1}',
            "BROKEN",
            '{"a": 2}',
            "",
            '{"a": 3}',
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path)

        self.assertEqual(report.total_lines, 5)
        self.assertEqual(report.object_lines, 3)
        self.assertEqual(report.json_error_lines, 1)
        self.assertEqual(report.blank_lines, 1)

    def test_preview_is_truncated(self):
        long_value = "x" * 500
        lines = [f'{{"key": "{long_value}"}}']  # valid but we'll make it invalid
        # Make it invalid by truncating the closing quote
        lines[0] = lines[0][:-2]  # remove closing "} -- now broken JSON
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path)

        err = [i for i in report.issues if i.category == SCAN_JSON_ERROR]
        self.assertEqual(len(err), 1)
        # Preview should be capped at _PREVIEW_MAX_CHARS + "..."
        self.assertLessEqual(len(err[0].raw_preview), 203)
        self.assertTrue(err[0].raw_preview.endswith("..."))

    def test_secret_values_are_redacted_in_preview(self):
        lines = ['{"api_key": "sk-secret-12345", broken']
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path)

        err = [i for i in report.issues if i.category == SCAN_JSON_ERROR]
        self.assertEqual(len(err), 1)
        self.assertIn("<redacted>", err[0].raw_preview)
        self.assertNotIn("sk-secret-12345", err[0].raw_preview)

    def test_max_issues_bounds_stored_issues(self):
        lines = [f"broken_line_{i}" for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path, max_issues=3)

        self.assertEqual(len(report.issues), 3)
        self.assertEqual(report.truncated_issues, 7)
        self.assertEqual(report.json_error_lines, 10)

    def test_max_issues_zero_stores_nothing(self):
        lines = ["broken", '{"a": 1}']
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            report = scan_jsonl_stream(path, max_issues=0)

        self.assertEqual(report.issues, [])
        self.assertEqual(report.truncated_issues, 1)
        self.assertEqual(report.json_error_lines, 1)
        self.assertEqual(report.object_lines, 1)

    def test_negative_max_issues_raises(self):
        with self.assertRaises(ValueError):
            scan_jsonl_stream(io.StringIO(""), max_issues=-1)

    def test_stream_input_without_path(self):
        stream = io.StringIO('{"a": 1}\n{"a": 2}\n')
        report = scan_jsonl_stream(stream)

        self.assertTrue(report.ok)
        self.assertEqual(report.object_lines, 2)
        self.assertEqual(report.source, "<stdin>")

    def test_source_is_recorded_from_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, ['{"a": 1}'])
            report = scan_jsonl_stream(path)

        self.assertEqual(report.source, path)


# ---------------------------------------------------------------------------
# Core scanner: scan_loader_output
# ---------------------------------------------------------------------------


class ScanLoaderOutputTest(unittest.TestCase):
    def test_clean_records_report_ok(self):
        records = [{"prompt": "a", "response": "b"}, {"prompt": "c", "response": "d"}]
        report = scan_loader_output(records, required_keys=("prompt", "response"))

        self.assertTrue(report.ok)
        self.assertEqual(report.object_lines, 2)

    def test_non_dict_records_reported(self):
        records = [{"a": 1}, [1, 2], "string", 42]
        report = scan_loader_output(records)

        self.assertEqual(report.object_lines, 1)
        self.assertEqual(report.non_object_lines, 3)
        non_obj = [i for i in report.issues if i.category == SCAN_NON_OBJECT]
        self.assertEqual(len(non_obj), 3)
        self.assertEqual(non_obj[0].line_number, 2)
        self.assertEqual(non_obj[0].detail, "expected dict, got list")

    def test_schema_missing_keys_reported(self):
        records = [{"prompt": "hi"}, {"prompt": "a", "response": "b"}]
        report = scan_loader_output(records, required_keys=("prompt", "response"))

        self.assertEqual(report.object_lines, 1)
        self.assertEqual(report.schema_error_lines, 1)

    def test_source_label(self):
        report = scan_loader_output([], source="my_loader")
        self.assertEqual(report.source, "my_loader")


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


class FormatScanReportTest(unittest.TestCase):
    def test_json_output_is_valid_json(self):
        stream = io.StringIO('{"a": 1}\nbroken\n')
        report = scan_jsonl_stream(stream)
        text = format_scan_report(report, json_output=True)

        parsed = json.loads(text)
        self.assertEqual(parsed["total_lines"], 2)
        self.assertEqual(parsed["object_lines"], 1)
        self.assertEqual(len(parsed["issues"]), 1)

    def test_human_output_contains_status(self):
        stream = io.StringIO('{"a": 1}\n')
        report = scan_jsonl_stream(stream)
        text = format_scan_report(report, json_output=False)

        self.assertIn("status: OK", text)
        self.assertIn("total_lines: 1", text)

    def test_human_output_shows_issue_lines(self):
        stream = io.StringIO("broken\n")
        report = scan_jsonl_stream(stream)
        text = format_scan_report(report, json_output=False)

        self.assertIn("status: ISSUES FOUND", text)
        self.assertIn("[json_error]", text)
        self.assertIn("line 1:", text)


# ---------------------------------------------------------------------------
# CLI: areno scan-dataset
# ---------------------------------------------------------------------------


class ScanDatasetCliTest(unittest.TestCase):
    def test_cli_clean_file_exits_zero(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, [json.dumps({"prompt": "hi", "response": "yo"})])
            result = runner.invoke(scan_dataset_command, [path])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("status: OK", result.output)

    def test_cli_bad_file_exits_one(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, ["broken_json"])
            result = runner.invoke(scan_dataset_command, [path])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("status: ISSUES FOUND", result.output)
        self.assertIn("[json_error]", result.output)

    def test_cli_required_keys_option(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, ['{"prompt": "hi"}'])
            result = runner.invoke(scan_dataset_command, [path, "--required-keys", "prompt,response"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("schema_errors: 1", result.output)

    def test_cli_json_output(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, ['{"a": 1}', "broken"])
            result = runner.invoke(scan_dataset_command, [path, "--json"])

        self.assertEqual(result.exit_code, 1)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["total_lines"], 2)
        self.assertEqual(parsed["json_error_lines"], 1)

    def test_cli_stdin_input(self):
        runner = CliRunner()
        result = runner.invoke(scan_dataset_command, input='{"a": 1}\n{"a": 2}\n')

        self.assertEqual(result.exit_code, 0)
        self.assertIn("objects: 2", result.output)

    def test_cli_max_issues_option(self):
        runner = CliRunner()
        lines = ["broken"] * 10
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, lines)
            result = runner.invoke(scan_dataset_command, [path, "--max-issues", "3"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("truncated 7", result.output)

    def test_cli_lists_in_top_level_help(self):
        from areno.cli.main import main

        result = CliRunner().invoke(main, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("scan-dataset", result.output)

    def test_cli_loader_fn_scans_output(self):
        """The --loader-fn path loads a .py loader and scans its records."""

        runner = CliRunner()
        loader_code = (
            "def load_training_dataset(path, *, default_loader, **_):\n"
            "    return [\n"
            "        {'prompt': 'a', 'response': 'b'},\n"
            "        {'prompt': 'c'},               # missing response\n"
            "        [1, 2],                          # non-dict\n"
            "    ]\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            loader_path = Path(tmp, "my_loader.py")
            loader_path.write_text(loader_code, encoding="utf-8")
            data_path = Path(tmp, "data.jsonl")
            data_path.write_text('{"x": 1}\n', encoding="utf-8")
            result = runner.invoke(
                scan_dataset_command,
                [str(data_path), "--loader-fn", str(loader_path), "--required-keys", "prompt,response"],
            )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("non_object: 1", result.output)
        self.assertIn("schema_errors: 1", result.output)
        self.assertIn("objects: 1", result.output)

    def test_cli_loader_fn_missing_file_raises_usage_error(self):
        runner = CliRunner()
        result = runner.invoke(scan_dataset_command, ["--loader-fn", "/nonexistent/loader.py"])

        self.assertEqual(result.exit_code, 2)
        self.assertIn("loader file not found", result.output)


# ---------------------------------------------------------------------------
# Bounded memory / large-file simulation
# ---------------------------------------------------------------------------


class BoundedMemoryTest(unittest.TestCase):
    def test_large_file_line_count_is_correct(self):
        """Simulate a large file (many lines) and verify counts.

        The scanner reads line by line, so memory does not scale with file
        size.  We verify the counts match the input exactly.
        """

        good = json.dumps({"prompt": "x", "response": "y"})
        lines = [good, "", "broken", good, "[1, 2, 3]"]
        # Repeat the pattern to simulate a larger file.
        all_lines = lines * 1000
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_jsonl(tmp, all_lines)
            report = scan_jsonl_stream(path, max_issues=5)

        self.assertEqual(report.total_lines, 5000)
        # Per pattern: 2 good objects, 1 blank, 1 json error, 1 non-object (array)
        self.assertEqual(report.object_lines, 2000)
        self.assertEqual(report.blank_lines, 1000)
        self.assertEqual(report.json_error_lines, 1000)
        self.assertEqual(report.non_object_lines, 1000)
        # Issues are capped at 5
        self.assertEqual(len(report.issues), 5)
        self.assertEqual(report.truncated_issues, 2995)


if __name__ == "__main__":
    unittest.main()
