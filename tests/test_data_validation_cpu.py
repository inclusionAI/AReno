from __future__ import annotations

import json
import sys
import unittest
from types import ModuleType

# Load data_validation directly to avoid the areno.api import chain which
# requires torch.  data_validation itself only depends on stdlib.
_mod = ModuleType("areno.api.data_validation")
_mod.__file__ = __file__
sys.modules["areno.api.data_validation"] = _mod
with open("areno/api/data_validation.py") as _f:
    exec(_f.read(), _mod.__dict__)

ValidationError = _mod.ValidationError
ValidationResult = _mod.ValidationResult
validate_dataset = _mod.validate_dataset
format_validation_result = _mod.format_validation_result
format_validation_result_json = _mod.format_validation_result_json


class DataValidationSFTTest(unittest.TestCase):
    """SFT mode data contract validation."""

    def test_valid_dataset_passes(self):
        result = validate_dataset([{"prompt": "hi", "response": "hello"}], "sft")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.valid_rows, 1)
        self.assertEqual(result.total_rows, 1)

    def test_missing_response_detected(self):
        result = validate_dataset([{"prompt": "hi"}], "sft")
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].field_path, "response")
        self.assertEqual(result.errors[0].expected_type, "str")
        self.assertIsNotNone(result.errors[0].hint)

    def test_empty_response_detected(self):
        result = validate_dataset([{"prompt": "hi", "response": ""}], "sft")
        self.assertFalse(result.is_valid)

    def test_response_wrong_type_detected(self):
        result = validate_dataset([{"prompt": "hi", "response": 123}], "sft")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].actual_type, "int")

    def test_prompt_wrong_type_detected(self):
        result = validate_dataset([{"prompt": 123, "response": "ok"}], "sft")
        self.assertFalse(result.is_valid)

    def test_mixed_errors_aggregated(self):
        result = validate_dataset(
            [
                {"prompt": "hi"},  # missing response
                {"prompt": "q", "response": ""},  # empty response
                {"prompt": 1, "response": "a"},  # wrong prompt type
            ],
            "sft",
        )
        self.assertEqual(len(result.errors), 3)

    def test_valid_rows_count(self):
        result = validate_dataset(
            [
                {"prompt": "a", "response": "b"},
                {"prompt": "c"},  # invalid
                {"prompt": "d", "response": "e"},
            ],
            "sft",
        )
        self.assertEqual(result.valid_rows, 2)
        self.assertEqual(result.total_rows, 3)


class DataValidationDPOTest(unittest.TestCase):
    """DPO mode data contract validation."""

    def test_prompt_response_style_passes(self):
        result = validate_dataset(
            [{"prompt": "hi", "chosen": "good", "rejected": "bad"}], "dpo"
        )
        self.assertTrue(result.is_valid)

    def test_chat_message_list_style_passes(self):
        result = validate_dataset(
            [
                {
                    "chosen": [{"role": "user", "content": "hi"}],
                    "rejected": [{"role": "user", "content": "no"}],
                }
            ],
            "dpo",
        )
        self.assertTrue(result.is_valid)

    def test_mismatched_types_detected(self):
        result = validate_dataset(
            [{"chosen": "good", "rejected": ["bad"]}], "dpo"
        )
        self.assertFalse(result.is_valid)

    def test_missing_chosen_detected(self):
        result = validate_dataset([{"rejected": "bad"}], "dpo")
        self.assertFalse(result.is_valid)

    def test_none_chosen_detected(self):
        result = validate_dataset([{"chosen": None, "rejected": "bad"}], "dpo")
        self.assertFalse(result.is_valid)


class DataValidationOnlineRLTest(unittest.TestCase):
    """Online RL (GSPO/GRPO/PPO) data contract validation."""

    def test_gspo_valid_passes(self):
        result = validate_dataset([{"question": "1+1", "answer": "2"}], "gspo")
        self.assertTrue(result.is_valid)

    def test_grpo_valid_passes(self):
        result = validate_dataset([{"q": "test"}], "grpo")
        self.assertTrue(result.is_valid)

    def test_ppo_valid_passes(self):
        result = validate_dataset([{"q": "test"}], "ppo")
        self.assertTrue(result.is_valid)

    def test_non_dict_row_detected(self):
        result = validate_dataset(["not a dict"], "grpo")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].field_path, "(row)")


class DataValidationAgenticTest(unittest.TestCase):
    """Agentic mode auto-detection."""

    def test_agent_fn_triggers_agentic_mode(self):
        result = validate_dataset([{"task": "solve"}], "gspo", agent_fn="some/agent.py")
        self.assertEqual(result.mode, "agentic")


class DataValidationBoundaryTest(unittest.TestCase):
    """Boundary and error handling."""

    def test_max_errors_limit(self):
        result = validate_dataset([{"prompt": "hi"}] * 100, "sft", max_errors=5)
        self.assertEqual(len(result.errors), 5)
        self.assertEqual(result.total_rows, 5)

    def test_empty_dataset_passes(self):
        result = validate_dataset([], "sft")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.total_rows, 0)

    def test_unknown_algo_raises(self):
        with self.assertRaisesRegex(ValueError, "不支持"):
            validate_dataset([], "unknown")


class DataValidationOutputTest(unittest.TestCase):
    """Human-readable and JSON output formatting."""

    def setUp(self):
        self.result = validate_dataset([{"prompt": "hi"}], "sft")

    def test_human_readable_contains_mode(self):
        text = format_validation_result(self.result)
        self.assertIn("SFT", text)

    def test_human_readable_contains_row_index(self):
        text = format_validation_result(self.result)
        self.assertIn("[行 0]", text)

    def test_human_readable_contains_hint(self):
        text = format_validation_result(self.result)
        self.assertIn("提示", text)

    def test_json_output_contains_index(self):
        j = format_validation_result_json(self.result)
        self.assertIn('"index"', j)

    def test_json_output_contains_field_path(self):
        j = format_validation_result_json(self.result)
        self.assertIn('"field_path"', j)

    def test_json_output_is_parseable(self):
        j = format_validation_result_json(self.result)
        parsed = json.loads(j)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["mode"], "sft")
        self.assertEqual(len(parsed["errors"]), 1)


if __name__ == "__main__":
    unittest.main()