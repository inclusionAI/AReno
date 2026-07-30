"""CPU-only unit tests for the post-loader data contract validator.

These tests cover SFT, DPO, online RL, and agentic contracts — exercising
valid fixtures, invalid fixtures, and boundary / failure paths.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from areno.api.data_contract import (
    ContractError,
    ContractReport,
    ContractSpec,
    FieldSpec,
    get_contract_spec,
    list_contract_modes,
    validate_contract,
)

FIXTURES = Path(__file__).parent / "fixtures" / "data_contract"


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL fixture file into a list of dicts."""

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


class ListContractModesTest(unittest.TestCase):
    """The contract mode registry should expose all four training modes."""

    def test_modes_contains_all_expected(self):
        modes = list_contract_modes()
        self.assertIn("sft", modes)
        self.assertIn("dpo", modes)
        self.assertIn("online_rl", modes)
        self.assertIn("agentic", modes)

    def test_get_contract_spec_returns_spec_for_each_mode(self):
        for mode in list_contract_modes():
            spec = get_contract_spec(mode)
            self.assertIsInstance(spec, ContractSpec)
            self.assertEqual(spec.mode, mode)

    def test_get_contract_spec_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "unknown contract mode"):
            get_contract_spec("bogus")

    def test_get_contract_spec_normalises_case(self):
        spec = get_contract_spec("SFT")
        self.assertEqual(spec.mode, "sft")


class SFTContractTest(unittest.TestCase):
    """SFT contracts must validate prompt+response or messages variants."""

    def test_valid_prompt_response(self):
        """Valid SFT records with prompt+response should produce no errors."""
        records = _load_jsonl(FIXTURES / "sft_valid.jsonl")
        report = validate_contract(records, mode="sft")
        self.assertTrue(report.ok)
        self.assertEqual(report.total_scanned, 3)
        self.assertEqual(len(report.errors), 0)

    def test_invalid_mix(self):
        """Invalid SFT records should aggregate multiple errors."""
        records = _load_jsonl(FIXTURES / "sft_invalid.jsonl")
        report = validate_contract(records, mode="sft")
        self.assertFalse(report.ok)
        self.assertGreater(len(report.errors), 0)

        # sample 0: missing 'response'
        self.assertTrue(
            any(e.sample_index == 0 and e.field_path == "response" and e.actual == "missing" for e in report.errors)
        )
        # sample 1: prompt is int
        self.assertTrue(
            any(e.sample_index == 1 and e.field_path == "prompt" and e.actual == "int" for e in report.errors)
        )
        # sample 2: response is empty string — not an error per current spec (str type passes)
        # sample 3 is valid
        self.assertFalse(
            any(e.sample_index == 3 for e in report.errors)
        )

    def test_messages_variant_accepted(self):
        """SFT records with messages list should be accepted as the messages variant."""
        records = [{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}]
        report = validate_contract(records, mode="sft")
        self.assertTrue(report.ok)

    def test_messages_missing_role_error(self):
        """A messages variant with a missing role should be flagged."""
        records = [{"messages": [{"content": "Hello"}]}]
        report = validate_contract(records, mode="sft")
        self.assertFalse(report.ok)
        self.assertTrue(
            any("messages[0].role" in e.field_path for e in report.errors)
        )


class DPOContractTest(unittest.TestCase):
    """DPO contracts must handle both string and chat-list chosen/rejected variants."""

    def test_valid_str_variant(self):
        """Valid DPO records with string chosen/rejected should pass."""
        records = _load_jsonl(FIXTURES / "dpo_valid_str.jsonl")
        report = validate_contract(records, mode="dpo")
        self.assertTrue(report.ok)

    def test_valid_list_variant(self):
        """Valid DPO records with chat-list chosen/rejected should pass."""
        records = _load_jsonl(FIXTURES / "dpo_valid_list.jsonl")
        report = validate_contract(records, mode="dpo")
        self.assertTrue(report.ok)

    def test_invalid_str_without_prompt(self):
        """DPO str variant requires prompt; missing prompt is an error."""
        records = _load_jsonl(FIXTURES / "dpo_invalid.jsonl")
        report = validate_contract(records, mode="dpo")
        self.assertFalse(report.ok)
        # sample 0: missing prompt (str variant)
        self.assertTrue(
            any(e.sample_index == 0 and e.field_path == "prompt" and e.actual == "missing" for e in report.errors)
        )
        # sample 1: missing chosen
        self.assertTrue(
            any(e.sample_index == 1 and e.field_path == "chosen" and e.actual == "missing" for e in report.errors)
        )
        # sample 2: chosen is int
        self.assertTrue(
            any(e.sample_index == 2 and e.field_path == "chosen" and e.actual == "int" for e in report.errors)
        )

    def test_list_without_optional_prompt_passes(self):
        """DPO list variant doesn't require prompt; it's optional."""
        records = [
            {
                "chosen": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
                "rejected": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Bye"}],
            }
        ]
        report = validate_contract(records, mode="dpo")
        self.assertTrue(report.ok)


class OnlineRLContractTest(unittest.TestCase):
    """Online RL contracts require prompt and optional solutions."""

    def test_valid_with_solutions(self):
        """Valid online RL records with optional solutions should pass."""
        records = _load_jsonl(FIXTURES / "online_rl_valid.jsonl")
        report = validate_contract(records, mode="online_rl")
        self.assertTrue(report.ok)
        self.assertEqual(report.total_scanned, 2)

    def test_invalid_null_prompt(self):
        """Null prompt should be an error."""
        records = _load_jsonl(FIXTURES / "online_rl_invalid.jsonl")
        report = validate_contract(records, mode="online_rl")
        self.assertFalse(report.ok)
        # sample 0: prompt is None
        self.assertTrue(
            any(e.sample_index == 0 and e.field_path == "prompt" and e.actual == "NoneType" for e in report.errors)
        )
        # sample 1: prompt is missing
        self.assertTrue(
            any(e.sample_index == 1 and e.field_path == "prompt" and e.actual == "missing" for e in report.errors)
        )
        # sample 2: prompt is int
        self.assertTrue(
            any(e.sample_index == 2 and e.field_path == "prompt" and e.actual == "int" for e in report.errors)
        )

    def test_solutions_with_wrong_element_type(self):
        """A solutions list with non-string elements should be an error."""
        records = [{"prompt": "Solve this.", "solutions": [1, 2]}]
        report = validate_contract(records, mode="online_rl")
        self.assertFalse(report.ok)
        self.assertTrue(
            any("solutions[0]" in e.field_path for e in report.errors)
        )


class AgenticContractTest(unittest.TestCase):
    """Agentic contracts accept prompt-only or messages-format records."""

    def test_valid_prompt_variant(self):
        """Agentic records with just a prompt field should pass."""
        records = _load_jsonl(FIXTURES / "agentic_valid.jsonl")
        report = validate_contract(records, mode="agentic")
        self.assertTrue(report.ok)

    def test_valid_messages_variant(self):
        """Agentic records with OpenAI-style messages should pass."""
        records = _load_jsonl(FIXTURES / "agentic_valid_messages.jsonl")
        report = validate_contract(records, mode="agentic")
        self.assertTrue(report.ok)

    def test_invalid_messages(self):
        """Agentic records with malformed messages should aggregate errors."""
        records = _load_jsonl(FIXTURES / "agentic_invalid.jsonl")
        report = validate_contract(records, mode="agentic")
        self.assertFalse(report.ok)
        # sample 0: invalid role 'narrator'
        self.assertTrue(
            any(
                e.sample_index == 0
                and "role" in e.field_path
                and "narrator" in e.actual
                for e in report.errors
            )
        )
        # sample 1: messages[0] missing content
        self.assertTrue(
            any(e.sample_index == 1 and "content" in e.field_path and e.actual == "missing" for e in report.errors)
        )
        # sample 2: tool_calls is not a list
        self.assertTrue(
            any(e.sample_index == 2 and "tool_calls" in e.field_path for e in report.errors)
        )
        # sample 3: valid prompt-only record
        self.assertFalse(any(e.sample_index == 3 for e in report.errors))

    def test_tool_calls_missing_function(self):
        """An assistant message with tool_calls missing 'function' should be flagged."""
        records = [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "1", "type": "function"}],
                    }
                ]
            }
        ]
        report = validate_contract(records, mode="agentic")
        self.assertFalse(report.ok)
        self.assertTrue(
            any("function" in e.field_path for e in report.errors)
        )


class BoundedErrorsTest(unittest.TestCase):
    """The validator should respect max_samples and max_errors limits."""

    def test_max_samples_truncates_scan(self):
        """Only the first max_samples records should be scanned."""
        records = [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(50)]
        report = validate_contract(records, mode="sft", max_samples=10)
        self.assertEqual(report.total_scanned, 10)
        self.assertTrue(report.ok)

    def test_max_errors_truncates_error_list(self):
        """Error collection should stop at max_errors with a warning."""
        records = [{"prompt": i} for i in range(50)]  # all invalid: prompt is int, missing response
        report = validate_contract(records, mode="sft", max_samples=50, max_errors=5)
        self.assertLessEqual(len(report.errors), 6)  # may slightly overshoot per-record
        self.assertFalse(report.ok)
        # a truncation warning should be present
        self.assertTrue(
            any(w.field_path == "<report>" and "truncated" in w.hint for w in report.warnings)
        )


class NonDictRecordTest(unittest.TestCase):
    """Non-dict records should produce a root-level error."""

    def test_string_record(self):
        """A non-dict record should produce a helpful root error."""
        report = validate_contract(["just a string"], mode="sft")
        self.assertFalse(report.ok)
        self.assertTrue(
            any(e.field_path == "<root>" and e.actual == "str" for e in report.errors)
        )

    def test_none_record(self):
        """A None record should produce a root error."""
        report = validate_contract([None], mode="sft")
        self.assertFalse(report.ok)
        self.assertTrue(
            any(e.field_path == "<root>" and e.actual == "NoneType" for e in report.errors)
        )


class EmptyDatasetTest(unittest.TestCase):
    """An empty dataset should produce an OK report with zero scanned."""

    def test_empty_list(self):
        report = validate_contract([], mode="sft")
        self.assertTrue(report.ok)
        self.assertEqual(report.total_scanned, 0)
        self.assertEqual(len(report.errors), 0)


class ReportSerializationTest(unittest.TestCase):
    """ContractReport.to_dict should produce a JSON-serializable dict."""

    def test_to_dict_structure(self):
        records = [{"prompt": "valid", "response": "ok"}]
        report = validate_contract(records, mode="sft")
        d = report.to_dict()
        self.assertEqual(d["mode"], "sft")
        self.assertEqual(d["total_scanned"], 1)
        self.assertTrue(d["ok"])
        self.assertEqual(d["errors"], [])
        self.assertEqual(d["warnings"], [])

    def test_to_dict_with_errors(self):
        records = [{"prompt": 42}]
        report = validate_contract(records, mode="sft")
        d = report.to_dict()
        self.assertFalse(d["ok"])
        self.assertEqual(len(d["errors"]), 2)  # wrong type + missing response
        err = d["errors"][0]
        self.assertIn("sample_index", err)
        self.assertIn("field_path", err)
        self.assertIn("expected", err)
        self.assertIn("actual", err)
        self.assertIn("hint", err)

    def test_to_dict_json_roundtrip(self):
        """The dict should be JSON serializable."""
        records = [{"prompt": 42}]
        report = validate_contract(records, mode="sft")
        json_str = json.dumps(report.to_dict())
        self.assertIsInstance(json_str, str)
        restored = json.loads(json_str)
        self.assertEqual(restored["mode"], "sft")


class ErrorPrivacyTest(unittest.TestCase):
    """Error messages should not contain raw sample values."""

    def test_no_raw_values_in_errors(self):
        sensitive_prompt = "SECRET-API-KEY-12345"
        records = [{"prompt": sensitive_prompt, "response": "ok"}]
        # This is valid so no errors, but let's make it invalid
        records = [{"prompt": sensitive_prompt}]  # missing response
        report = validate_contract(records, mode="sft")
        self.assertFalse(report.ok)
        for err in report.errors:
            self.assertNotIn(sensitive_prompt, err.hint)
            self.assertNotIn(sensitive_prompt, err.expected)
            self.assertNotIn(sensitive_prompt, err.actual)
            self.assertNotIn(sensitive_prompt, err.field_path)


class FieldSpecTest(unittest.TestCase):
    """FieldSpec is the building block for contract definitions."""

    def test_field_spec_basic(self):
        spec = FieldSpec("test", required=True, expected_type=str)
        self.assertTrue(spec.required)
        self.assertEqual(spec.name, "test")
        self.assertIsNone(spec.element_type)
        self.assertIsNone(spec.min_length)

    def test_field_spec_with_element_type(self):
        spec = FieldSpec("solutions", required=False, expected_type=list, element_type=str, nullable=True)
        self.assertFalse(spec.required)
        self.assertEqual(spec.element_type, str)
        self.assertTrue(spec.nullable)


if __name__ == "__main__":
    unittest.main()
