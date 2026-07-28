"""CPU tests for conversation role normalization and tool-message pairing.

Covers:
  - Role alias mapping (ShareGPT, OpenAI, Anthropic style)
  - Tool-call / tool-response pairing (single, parallel, nested)
  - Malformed inputs (missing response, orphan response, duplicate)
  - Role alternation rules (consecutive, misplaced system, misplaced tool)
  - Batch normalization with human-readable and structured output
  - Default / disabled behavior (backward compatibility)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


def _load_normalizer_module():
    """Load conversation_normalizer directly, bypassing torch-dependent __init__."""
    path = Path(__file__).resolve().parents[1] / "areno" / "engine" / "data" / "conversation_normalizer.py"
    spec = importlib.util.spec_from_file_location("conversation_normalizer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_api_data_module(normalizer_module):
    """Create a stub for areno.api.data re-export to avoid torch/slots on < 3.10."""
    import types

    stub = types.ModuleType("areno.api.data")
    stub.normalize_conversation = normalizer_module.normalize_conversation
    stub.normalize_dataset = normalizer_module.normalize_dataset
    stub.normalize_dataset_iter = normalizer_module.normalize_dataset_iter
    stub.normalize_role = normalizer_module.normalize_role
    stub.ConversationValidationError = normalizer_module.ConversationValidationError
    stub.BatchNormalizeReport = normalizer_module.BatchNormalizeReport
    stub.NormalizeResult = normalizer_module.NormalizeResult
    sys.modules["areno.api.data"] = stub
    return stub


_mod = _load_normalizer_module()
_api_mod = _load_api_data_module(_mod)

BatchNormalizeReport = _mod.BatchNormalizeReport
ConversationValidationError = _mod.ConversationValidationError
NormalizeResult = _mod.NormalizeResult
normalize_conversation = _mod.normalize_conversation
normalize_dataset = _mod.normalize_dataset
normalize_dataset_iter = _mod.normalize_dataset_iter
normalize_role = _mod.normalize_role

api_normalize_conversation = _api_mod.normalize_conversation
api_normalize_dataset = _api_mod.normalize_dataset


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _sharegpt_sample() -> list[dict]:
    """A typical ShareGPT-style conversation with role aliases."""
    return [
        {"role": "human", "content": "What's the weather in Beijing?"},
        {"role": "bot", "content": None, "tool_calls": [
            {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}}
        ]},
        {"role": "function", "content": "25C sunny", "tool_call_id": "call_1"},
        {"role": "bot", "content": "Beijing is 25C and sunny today."},
    ]


def _openai_sample() -> list[dict]:
    """A standard OpenAI-format conversation (already uses standard roles)."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Calculate 2+2."},
        {"role": "assistant", "content": "2+2 = 4."},
    ]


def _parallel_tool_sample() -> list[dict]:
    """Assistant issues two tool calls in parallel; both get responses."""
    return [
        {"role": "user", "content": "What's the weather in Beijing and Shanghai?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Beijing"}}},
            {"id": "c2", "function": {"name": "get_weather", "arguments": {"city": "Shanghai"}}},
        ]},
        {"role": "tool", "content": "25C", "tool_call_id": "c1"},
        {"role": "tool", "content": "28C", "tool_call_id": "c2"},
        {"role": "assistant", "content": "Beijing 25C, Shanghai 28C."},
    ]


def _nested_tool_sample() -> list[dict]:
    """Tool result triggers a new tool call (nested)."""
    return [
        {"role": "user", "content": "Search and then summarize."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "s1", "function": {"name": "search", "arguments": {"q": "AI"}}}
        ]},
        {"role": "tool", "content": "Found 3 results", "tool_call_id": "s1"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "s2", "function": {"name": "summarize", "arguments": {"ids": [1, 2, 3]}}}
        ]},
        {"role": "tool", "content": "Summary: AI is growing", "tool_call_id": "s2"},
        {"role": "assistant", "content": "Here is the summary: AI is growing."},
    ]


# ---------------------------------------------------------------------------
# Role mapping tests
# ---------------------------------------------------------------------------


class RoleMappingTest(unittest.TestCase):

    def test_sharegpt_aliases_mapped(self):
        result = normalize_conversation(_sharegpt_sample())
        self.assertTrue(result.ok)
        roles = [m["role"] for m in result.messages]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])

    def test_openai_standard_roles_unchanged(self):
        result = normalize_conversation(_openai_sample())
        self.assertTrue(result.ok)
        roles = [m["role"] for m in result.messages]
        self.assertEqual(roles, ["system", "user", "assistant"])

    def test_individual_alias_mapping(self):
        cases = {
            "human": "user",
            "person": "user",
            "bot": "assistant",
            "gpt": "assistant",
            "model": "assistant",
            "function": "tool",
            "tool_result": "tool",
            "system": "system",
            "instruction": "system",
        }
        for alias, expected in cases.items():
            self.assertEqual(normalize_role(alias), expected)

    def test_unknown_role_raises(self):
        msg = [{"role": "agent", "content": "hi"}]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg, sample_index=0)
        self.assertEqual(ctx.exception.error_type, "unknown_role")
        self.assertIn("agent", str(ctx.exception))

    def test_unknown_role_collected_not_raised(self):
        msg = [{"role": "agent", "content": "hi"}]
        result = normalize_conversation(msg, sample_index=5, raise_on_error=False)
        self.assertFalse(result.ok)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].sample_index, 5)
        self.assertEqual(result.errors[0].error_type, "unknown_role")

    def test_case_insensitive_role(self):
        msg = [
            {"role": "Human", "content": "hi"},
            {"role": "Bot", "content": "hello"},
        ]
        result = normalize_conversation(msg)
        self.assertTrue(result.ok)
        self.assertEqual(result.messages[0]["role"], "user")
        self.assertEqual(result.messages[1]["role"], "assistant")

    def test_role_normalization_preserves_content(self):
        result = normalize_conversation(_sharegpt_sample())
        self.assertTrue(result.ok)
        self.assertEqual(result.messages[0]["content"], "What's the weather in Beijing?")
        self.assertEqual(result.messages[3]["content"], "Beijing is 25C and sunny today.")


# ---------------------------------------------------------------------------
# Tool-call pairing tests
# ---------------------------------------------------------------------------


class ToolPairingTest(unittest.TestCase):

    def test_single_tool_call_paired(self):
        result = normalize_conversation(_sharegpt_sample())
        self.assertTrue(result.ok)

    def test_parallel_tool_calls_paired(self):
        result = normalize_conversation(_parallel_tool_sample())
        self.assertTrue(result.ok)

    def test_nested_tool_calls_paired(self):
        result = normalize_conversation(_nested_tool_sample())
        self.assertTrue(result.ok)

    def test_missing_tool_response_detected(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "x1", "function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "assistant", "content": "done"},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg, sample_index=2)
        self.assertEqual(ctx.exception.error_type, "missing_tool_response")
        self.assertEqual(ctx.exception.sample_index, 2)

    def test_missing_tool_response_at_end_detected(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "x1", "function": {"name": "f", "arguments": {}}}
            ]},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg)
        self.assertEqual(ctx.exception.error_type, "missing_tool_response")

    def test_orphan_tool_response_detected(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "result", "tool_call_id": "nonexistent"},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg)
        self.assertEqual(ctx.exception.error_type, "orphan_tool_response")

    def test_tool_response_missing_call_id(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "tool", "content": "result"},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg)
        self.assertEqual(ctx.exception.error_type, "missing_tool_call_id")

    def test_parallel_with_one_missing_response(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "function": {"name": "f1", "arguments": {}}},
                {"id": "b", "function": {"name": "f2", "arguments": {}}},
            ]},
            {"role": "tool", "content": "r1", "tool_call_id": "a"},
            {"role": "assistant", "content": "done"},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg)
        self.assertEqual(ctx.exception.error_type, "missing_tool_response")

    def test_user_interrupts_pending_tool_calls(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "user", "content": "never mind"},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg)
        self.assertEqual(ctx.exception.error_type, "interrupted_tool_call")

    def test_tool_response_id_order_does_not_matter(self):
        """Parallel tool responses may arrive in any order."""
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f1", "arguments": {}}},
                {"id": "c2", "function": {"name": "f2", "arguments": {}}},
            ]},
            {"role": "tool", "content": "r2", "tool_call_id": "c2"},
            {"role": "tool", "content": "r1", "tool_call_id": "c1"},
            {"role": "assistant", "content": "done"},
        ]
        result = normalize_conversation(msg)
        self.assertTrue(result.ok)

    def test_tool_calls_arguments_string_parsed(self):
        """arguments as JSON string should be parsed to dict."""
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": '{"key": "val"}'}}
            ]},
            {"role": "tool", "content": "ok", "tool_call_id": "c1"},
            {"role": "assistant", "content": "done"},
        ]
        result = normalize_conversation(msg)
        self.assertTrue(result.ok)
        args = result.messages[1]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args, {"key": "val"})


# ---------------------------------------------------------------------------
# Role alternation tests
# ---------------------------------------------------------------------------


class RoleSequenceTest(unittest.TestCase):

    def test_standard_alternation_passes(self):
        msg = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "bye"},
        ]
        result = normalize_conversation(msg)
        self.assertTrue(result.ok)

    def test_consecutive_user_detected(self):
        msg = [
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg)
        self.assertEqual(ctx.exception.error_type, "consecutive_user")

    def test_consecutive_assistant_without_tool_calls_detected(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "assistant", "content": "bye"},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg)
        self.assertEqual(ctx.exception.error_type, "consecutive_assistant")

    def test_system_in_middle_detected(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "extra instruction"},
            {"role": "assistant", "content": "hello"},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg)
        self.assertEqual(ctx.exception.error_type, "misplaced_system")

    def test_tool_without_preceding_assistant_detected(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "result", "tool_call_id": "x"},
        ]
        with self.assertRaises(ConversationValidationError) as ctx:
            normalize_conversation(msg)
        # The orphan tool response error is caught first in phase 2.
        self.assertIn(ctx.exception.error_type, ("orphan_tool_response", "invalid_tool_position"))

    def test_single_turn_passes(self):
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = normalize_conversation(msg)
        self.assertTrue(result.ok)

    def test_empty_conversation_passes(self):
        result = normalize_conversation([])
        self.assertTrue(result.ok)
        self.assertEqual(result.messages, [])


# ---------------------------------------------------------------------------
# Batch normalization tests
# ---------------------------------------------------------------------------


class BatchNormalizeTest(unittest.TestCase):

    def test_batch_all_pass(self):
        samples = [_openai_sample(), _sharegpt_sample(), _parallel_tool_sample()]
        report = normalize_dataset(samples)
        self.assertEqual(report.total, 3)
        self.assertEqual(report.passed, 3)
        self.assertEqual(report.failed, 0)
        self.assertEqual(len(report.normalized), 3)

    def test_batch_mixed_pass_and_fail(self):
        samples = [
            _openai_sample(),
            [{"role": "agent", "content": "bad"}],  # unknown role
            _sharegpt_sample(),
        ]
        report = normalize_dataset(samples)
        self.assertEqual(report.total, 3)
        self.assertEqual(report.passed, 2)
        self.assertEqual(report.failed, 1)
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(report.errors[0]["sample"], 1)
        self.assertEqual(report.errors[0]["type"], "unknown_role")

    def test_batch_error_includes_sample_and_turn(self):
        samples = [
            _openai_sample(),
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "function": {"name": "f", "arguments": {}}}
                ]},
                # Missing tool response for c1.
                {"role": "assistant", "content": "done"},
            ],
        ]
        report = normalize_dataset(samples)
        self.assertEqual(report.failed, 1)
        err = report.errors[0]
        self.assertEqual(err["sample"], 1)
        self.assertEqual(err["turn"], 2)
        self.assertEqual(err["type"], "missing_tool_response")

    def test_human_readable_output(self):
        samples = [_openai_sample(), [{"role": "bad", "content": "x"}]]
        report = normalize_dataset(samples)
        text = report.to_human_string()
        self.assertIn("Total: 2", text)
        self.assertIn("Passed: 1", text)
        self.assertIn("Failed: 1", text)
        self.assertIn("unknown_role", text)
        self.assertIn("sample #1", text)

    def test_structured_output(self):
        samples = [_openai_sample(), [{"role": "bad", "content": "x"}]]
        report = normalize_dataset(samples)
        d = report.to_dict()
        self.assertEqual(d["total"], 2)
        self.assertEqual(d["passed"], 1)
        self.assertEqual(d["failed"], 1)
        self.assertEqual(len(d["errors"]), 1)
        self.assertIn("type", d["errors"][0])
        self.assertIn("detail", d["errors"][0])
        self.assertIn("sample", d["errors"][0])

    def test_structured_json_output(self):
        report = normalize_dataset([_openai_sample()])
        j = json.loads(report.to_json())
        self.assertEqual(j["total"], 1)
        self.assertEqual(j["passed"], 1)

    def test_skip_invalid_false_raises(self):
        samples = [_openai_sample(), [{"role": "bad", "content": "x"}]]
        with self.assertRaises(ConversationValidationError):
            normalize_dataset(samples, skip_invalid=False)

    def test_iter_mode(self):
        samples = [_openai_sample(), _sharegpt_sample()]
        results = list(normalize_dataset_iter(iter(samples)))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 0)  # sample index
        self.assertIsNotNone(results[0][1])  # normalized messages
        self.assertEqual(results[0][2], [])  # no errors
        self.assertEqual(results[1][0], 1)
        self.assertIsNotNone(results[1][1])

    def test_iter_mode_with_errors(self):
        samples = [_openai_sample(), [{"role": "bad", "content": "x"}]]
        results = list(normalize_dataset_iter(iter(samples), raise_on_error=False))
        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0][1])  # first ok
        self.assertIsNone(results[1][1])     # second failed
        self.assertTrue(len(results[1][2]) > 0)


# ---------------------------------------------------------------------------
# Backward compatibility / default behavior
# ---------------------------------------------------------------------------


class DefaultBehaviorTest(unittest.TestCase):

    def test_api_reexport_works(self):
        """The re-export from areno.api.data should work identically."""
        result = api_normalize_conversation(_sharegpt_sample())
        self.assertTrue(result.ok)
        self.assertEqual(result.messages[0]["role"], "user")

    def test_api_dataset_reexport_works(self):
        report = api_normalize_dataset([_openai_sample()])
        self.assertEqual(report.passed, 1)

    def test_content_none_normalized_to_empty_string(self):
        """OpenAI assistant tool-call messages with content=null get ''."""
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "tool", "content": "ok", "tool_call_id": "c1"},
            {"role": "assistant", "content": "done"},
        ]
        result = normalize_conversation(msg)
        self.assertTrue(result.ok)
        self.assertEqual(result.messages[1]["content"], "")

    def test_flat_tool_call_shape_normalized(self):
        """Tool calls in {name, arguments} flat shape get wrapped in function sub-dict."""
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "name": "search", "arguments": {"q": "test"}}
            ]},
            {"role": "tool", "content": "results", "tool_call_id": "c1"},
            {"role": "assistant", "content": "found it"},
        ]
        result = normalize_conversation(msg)
        self.assertTrue(result.ok)
        tc = result.messages[1]["tool_calls"][0]
        self.assertIn("function", tc)
        self.assertEqual(tc["function"]["name"], "search")
        self.assertEqual(tc["function"]["arguments"], {"q": "test"})

    def test_non_dict_message_handled(self):
        msg = ["not a dict"]
        result = normalize_conversation(msg, sample_index=0, raise_on_error=False)
        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0].error_type, "invalid_message")

    def test_non_list_input_handled(self):
        result = normalize_conversation("not a list", sample_index=0, raise_on_error=False)  # type: ignore[arg-type]
        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0].error_type, "invalid_input")


# ---------------------------------------------------------------------------
# Integration-style test: normalize then verify agentic contract
# ---------------------------------------------------------------------------


class AgenticContractTest(unittest.TestCase):
    """Verify normalized output passes the existing agentic message contract."""

    def test_normalized_output_roles_are_standard(self):
        """Every role in normalized output must be in STANDARD_ROLES."""
        from conversation_normalizer_under_test import STANDARD_ROLES

        samples = [_sharegpt_sample(), _openai_sample(), _parallel_tool_sample(), _nested_tool_sample()]
        report = normalize_dataset(samples)
        self.assertEqual(report.failed, 0)
        for messages in report.normalized:
            for msg in messages:
                self.assertIn(msg["role"], STANDARD_ROLES)

    def test_normalized_tool_calls_have_function_key(self):
        """Every tool call must have a 'function' sub-dict after normalization."""
        msg = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "name": "f", "arguments": {}}
            ]},
            {"role": "tool", "content": "ok", "tool_call_id": "c1"},
            {"role": "assistant", "content": "done"},
        ]
        result = normalize_conversation(msg)
        self.assertTrue(result.ok)
        for tc in result.messages[1]["tool_calls"]:
            self.assertIn("function", tc)
            self.assertIn("name", tc["function"])

    def test_normalized_output_compatible_with_openai_normalize(self):
        """Normalized output should be compatible with the openai_chat.normalize_messages contract.

        The openai_chat module requires torch via its import chain, so we
        inline the two key checks that normalize_messages performs:
        1. content=None is replaced with ''
        2. tool_calls are kept and their function.arguments parsed
        """
        result = normalize_conversation(_sharegpt_sample())
        self.assertTrue(result.ok)
        messages = result.messages

        # Verify content=None was replaced (normalize_messages does the same).
        for msg in messages:
            self.assertIsNotNone(msg.get("content"))

        # Verify tool_calls are present and properly shaped.
        assistant_with_calls = messages[1]
        self.assertIn("tool_calls", assistant_with_calls)
        for tc in assistant_with_calls["tool_calls"]:
            self.assertIn("function", tc)
            self.assertIsInstance(tc["function"]["arguments"], dict)

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "user")


# ---------------------------------------------------------------------------
# CLI output tests
# ---------------------------------------------------------------------------


class CliOutputTest(unittest.TestCase):
    """Test the human-readable and JSON structured output formats."""

    def test_human_readable_includes_summary_and_errors(self):
        samples = [_openai_sample(), [{"role": "bad", "content": "x"}]]
        report = normalize_dataset(samples)
        text = report.to_human_string()
        self.assertIn("Total:", text)
        self.assertIn("Passed:", text)
        self.assertIn("Failed:", text)
        self.assertIn("unknown_role", text)

    def test_json_output_has_required_fields(self):
        samples = [_openai_sample(), [{"role": "bad", "content": "x"}]]
        report = normalize_dataset(samples)
        d = report.to_dict()
        self.assertIn("total", d)
        self.assertIn("passed", d)
        self.assertIn("failed", d)
        self.assertIn("errors", d)
        self.assertEqual(len(d["errors"]), 1)
        self.assertIn("type", d["errors"][0])
        self.assertIn("detail", d["errors"][0])
        self.assertIn("sample", d["errors"][0])
        self.assertIn("turn", d["errors"][0])

    def test_json_output_is_valid_json(self):
        report = normalize_dataset([_openai_sample()])
        parsed = json.loads(report.to_json())
        self.assertEqual(parsed["total"], 1)
        self.assertEqual(parsed["passed"], 1)


if __name__ == "__main__":
    unittest.main()