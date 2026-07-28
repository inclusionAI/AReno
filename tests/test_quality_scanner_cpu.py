"""CPU tests for the streaming JSONL quality scanner."""

from __future__ import annotations

import io
import json

import pytest

# Import directly from the module file to avoid pulling in heavy engine deps
# (torch, triton, ...) via areno.api.__init__.
import sys
from pathlib import Path

_api_dir = Path(__file__).resolve().parent.parent / "areno" / "api"
sys.path.insert(0, str(_api_dir))

from quality_scanner import (  # noqa: E402
    ErrorType,
    ScanResult,
    render_json,
    render_table,
    scan_jsonl,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jsonl(lines: list[str]) -> io.StringIO:
    """Build a StringIO from a list of raw line strings."""

    return io.StringIO("\n".join(lines) + "\n" if lines else "")


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestValidJsonl:
    def test_all_valid_records(self):
        data = _make_jsonl([
            json.dumps({"prompt": "hello", "response": "world"}),
            json.dumps({"prompt": "foo", "response": "bar"}),
        ])
        result = scan_jsonl(data)
        assert result.total_lines == 2
        assert result.valid_records == 2
        assert result.total_errors == 0

    def test_single_line(self):
        data = _make_jsonl([json.dumps({"a": 1})])
        result = scan_jsonl(data)
        assert result.total_lines == 1
        assert result.valid_records == 1

    def test_empty_file(self):
        data = _make_jsonl([])
        result = scan_jsonl(data)
        assert result.total_lines == 0
        assert result.valid_records == 0
        assert result.total_errors == 0


# ---------------------------------------------------------------------------
# Error detection
# ---------------------------------------------------------------------------

class TestErrorDetection:
    def test_blank_lines(self):
        data = _make_jsonl(["", json.dumps({"a": 1}), "", ""])
        result = scan_jsonl(data)
        assert result.blank_lines == 3
        assert result.valid_records == 1
        assert len(result.errors) == 3
        assert all(e.error_type == ErrorType.BLANK_LINE for e in result.errors)

    def test_json_parse_error(self):
        data = _make_jsonl(["{bad json", json.dumps({"a": 1}), '{"a":}'])
        result = scan_jsonl(data)
        assert result.json_errors == 2
        assert result.valid_records == 1
        assert result.errors[0].error_type == ErrorType.JSON_PARSE
        assert result.errors[0].line_number == 1

    def test_non_object_record(self):
        data = _make_jsonl([
            json.dumps([1, 2, 3]),
            json.dumps("just a string"),
            json.dumps(42),
            json.dumps({"a": 1}),
        ])
        result = scan_jsonl(data)
        assert result.non_object_records == 3
        assert result.valid_records == 1
        assert result.errors[0].detail == "type is list"

    def test_mixed_errors(self):
        data = _make_jsonl([
            json.dumps({"prompt": "hi", "response": "yo"}),
            "",
            "{broken",
            json.dumps([1, 2]),
            json.dumps({"prompt": "", "response": "ok"}),
        ])
        result = scan_jsonl(data)
        assert result.total_lines == 5
        assert result.blank_lines == 1
        assert result.json_errors == 1
        assert result.non_object_records == 1


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------

class TestSchemaChecks:
    def test_missing_required_field(self):
        data = _make_jsonl([
            json.dumps({"prompt": "hi"}),  # missing response
            json.dumps({"prompt": "hi", "response": "yo"}),
        ])
        result = scan_jsonl(data, required_fields=["prompt", "response"])
        assert result.schema_issues == 1
        assert result.valid_records == 1
        assert result.errors[0].error_type == ErrorType.SCHEMA_MISSING_FIELD

    def test_empty_required_field(self):
        data = _make_jsonl([
            json.dumps({"prompt": "hi", "response": ""}),
            json.dumps({"prompt": "hi", "response": "  "}),
        ])
        result = scan_jsonl(data, required_fields=["prompt", "response"])
        assert result.schema_issues == 2
        assert result.errors[0].error_type == ErrorType.SCHEMA_EMPTY_FIELD

    def test_no_schema_check_when_none(self):
        data = _make_jsonl([json.dumps({"a": 1})])
        result = scan_jsonl(data, required_fields=None)
        assert result.schema_issues == 0
        assert result.valid_records == 1


# ---------------------------------------------------------------------------
# Bounded preview
# ---------------------------------------------------------------------------

class TestBoundedPreview:
    def test_max_errors_limit(self):
        lines = ["{bad"] * 200
        data = _make_jsonl(lines)
        result = scan_jsonl(data, max_errors=10)
        assert len(result.errors) == 10
        assert result.errors_truncated == 190
        assert result.json_errors == 200

    def test_default_max_errors(self):
        lines = [""] * 150
        data = _make_jsonl(lines)
        result = scan_jsonl(data)
        assert len(result.errors) == 100
        assert result.errors_truncated == 50


# ---------------------------------------------------------------------------
# File path input
# ---------------------------------------------------------------------------

class TestFileInput:
    def test_file_path(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        filepath.write_text(
            json.dumps({"a": 1}) + "\n" +
            "{bad\n" +
            json.dumps({"b": 2}) + "\n",
            encoding="utf-8",
        )
        result = scan_jsonl(str(filepath))
        assert result.total_lines == 3
        assert result.valid_records == 2
        assert result.json_errors == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        data1 = _make_jsonl(["{bad", json.dumps({"a": 1}), ""])
        data2 = _make_jsonl(["{bad", json.dumps({"a": 1}), ""])
        r1 = scan_jsonl(data1)
        r2 = scan_jsonl(data2)
        assert r1.to_dict() == r2.to_dict()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_table_output_contains_counts(self):
        data = _make_jsonl(["{bad", json.dumps({"a": 1}), ""])
        result = scan_jsonl(data)
        table = render_table(result)
        assert "Total lines:" in table
        assert "JSON errors:" in table
        assert "Blank lines:" in table

    def test_json_output_is_valid_json(self):
        data = _make_jsonl(["{bad", json.dumps({"a": 1})])
        result = scan_jsonl(data)
        output = render_json(result)
        parsed = json.loads(output)
        assert parsed["total_lines"] == 2
        assert parsed["json_errors"] == 1

    def test_table_shows_error_preview(self):
        data = _make_jsonl(["{bad", "", json.dumps([1])])
        result = scan_jsonl(data)
        table = render_table(result)
        assert "Error preview" in table
        assert "line" in table

    def test_table_truncation_notice(self):
        lines = ["{bad"] * 200
        data = _make_jsonl(lines)
        result = scan_jsonl(data, max_errors=10)
        table = render_table(result)
        assert "additional errors not shown" in table


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------

class TestBoundaryCases:
    def test_very_long_line(self):
        long_value = "x" * 10000
        data = _make_jsonl([json.dumps({"prompt": long_value, "response": "ok"})])
        result = scan_jsonl(data)
        assert result.valid_records == 1

    def test_unicode_content(self):
        data = _make_jsonl([json.dumps({"prompt": "你好世界", "response": "再见"})])
        result = scan_jsonl(data)
        assert result.valid_records == 1

    def test_nested_objects(self):
        data = _make_jsonl([json.dumps({"a": {"b": {"c": 1}}})])
        result = scan_jsonl(data)
        assert result.valid_records == 1

    def test_redacted_detail_length(self):
        data = _make_jsonl(["{'a': " + "x" * 200 + "}"])
        result = scan_jsonl(data)
        for err in result.errors:
            assert len(err.detail) <= 100  # redacted


# ---------------------------------------------------------------------------
# Large file (bounded memory)
# ---------------------------------------------------------------------------

class TestLargeFile:
    def test_thousand_lines(self, tmp_path):
        filepath = tmp_path / "large.jsonl"
        lines = []
        for i in range(1000):
            if i % 100 == 50:
                lines.append("{bad")
            elif i % 100 == 51:
                lines.append("")
            else:
                lines.append(json.dumps({"id": i, "text": f"line {i}"}))
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = scan_jsonl(str(filepath))
        assert result.total_lines == 1000
        assert result.valid_records == 980
        assert result.json_errors == 10
        assert result.blank_lines == 10
