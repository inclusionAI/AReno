"""CPU tests for the chat-template compatibility inspector.

All tests use lightweight mock tokenizers — no model downloads or GPU needed.
The mock patterns follow ``FakeTokenizer`` from ``test_tokenizer_api_cpu.py``.
"""

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from areno.api.chat_template_inspector import (
    CANONICAL_SCENARIOS,
    ChatTemplateInspector,
    DiagnosticResult,
    InspectionReport,
    check_duplicate_special_tokens,
    check_generation_boundary,
    check_role_support,
    check_template_exists,
    check_tool_schema,
)
from areno.cli.inspect import inspect_chat_template_command


# ---------------------------------------------------------------------------
# Mock tokenizers
# ---------------------------------------------------------------------------


class CompatibleTokenizer:
    """A mock tokenizer whose template correctly renders all message types."""

    chat_template = "<template>"
    all_special_ids = [0, 1, 2]

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs) -> str:
        parts = []
        add_gen = kwargs.get("add_generation_prompt", False)
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    func = call.get("function", {})
                    parts.append(f"[{role}][tool_call:{func.get('name', '')}({func.get('arguments', '')})]")
            if content:
                parts.append(f"[{role}]{content}")
        if add_gen:
            parts.append("[assistant]")
        return "".join(parts)

    def encode(self, text: str) -> list[int]:
        # Deterministic pseudo-encode: each char -> its ord, capped at 10.
        return [min(ord(c), 9) for c in text]


class NoTemplateTokenizer:
    """A tokenizer with no chat_template defined."""

    chat_template = None
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages, **kwargs):
        raise ValueError("No chat template configured")

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text[:10]]


class SystemRoleUnsupportedTokenizer:
    """A tokenizer whose template fails on the ``system`` role."""

    chat_template = "<template>"
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages, **kwargs) -> str:
        for msg in messages:
            if msg.get("role") == "system":
                raise ValueError("system role is not supported by this template")
        return "".join(
            f"[{m.get('role', '')}]{m.get('content', '')}" for m in messages
        )

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text[:10]]


class BoundaryBrokenTokenizer:
    """A tokenizer where add_generation_prompt output is not a prefix."""

    chat_template = "<template>"
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages, **kwargs) -> str:
        add_gen = kwargs.get("add_generation_prompt", False)
        text = "".join(
            f"[{m.get('role', '')}]{m.get('content', '')}" for m in messages
        )
        if add_gen:
            # Deliberately prepend a non-prefix marker.
            return "[BROKEN]" + text
        return text

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text[:10]]


class ToolDroppingTokenizer:
    """A tokenizer that silently ignores tool_calls and tool messages."""

    chat_template = "<template>"
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages, **kwargs) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "tool":
                continue  # silently drop tool messages
            if msg.get("tool_calls"):
                continue  # silently drop tool_calls
            if content:
                parts.append(f"[{role}]{content}")
        return "".join(parts)

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text[:10]]


class DuplicateSpecialTokenizer:
    """A tokenizer that produces consecutive duplicate special token IDs."""

    chat_template = "<template>"
    all_special_ids = [99]

    def apply_chat_template(self, messages, **kwargs) -> str:
        parts = []
        add_gen = kwargs.get("add_generation_prompt", False)
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    func = call.get("function", {})
                    parts.append(f"[{role}][tool_call:{func.get('name', '')}({func.get('arguments', '')})]")
            if content:
                parts.append(f"[{role}]{content}")
        if add_gen:
            parts.append("[assistant]")
        return "".join(parts)

    def encode(self, text: str) -> list[int]:
        # Produce a token sequence with a consecutive duplicate of special id 99.
        return [1, 2, 99, 99, 3, 4]


# ---------------------------------------------------------------------------
# Tests: individual checks
# ---------------------------------------------------------------------------


class CheckTemplateExistsTest(unittest.TestCase):
    def test_pass_when_template_exists(self):
        tok = CompatibleTokenizer()
        result = check_template_exists(tok)
        self.assertEqual(result.status, "pass")
        self.assertTrue(result.detail["has_chat_template"])

    def test_fail_when_template_missing(self):
        tok = NoTemplateTokenizer()
        result = check_template_exists(tok)
        self.assertEqual(result.status, "fail")
        self.assertIn("no chat_template", result.message.lower())
        self.assertFalse(result.detail["has_chat_template"])


class CheckRoleSupportTest(unittest.TestCase):
    def test_pass_with_compatible_template(self):
        tok = CompatibleTokenizer()
        scenario = CANONICAL_SCENARIOS[0]  # single_turn_basic
        result = check_role_support(tok, scenario)
        self.assertEqual(result.status, "pass")

    def test_fail_when_system_role_unsupported(self):
        tok = SystemRoleUnsupportedTokenizer()
        scenario = CANONICAL_SCENARIOS[0]  # single_turn_basic (has system)
        result = check_role_support(tok, scenario)
        self.assertEqual(result.status, "fail")
        self.assertEqual(result.detail.get("suspected_role"), "system")

    def test_pass_for_scenario_without_system(self):
        tok = SystemRoleUnsupportedTokenizer()
        scenario = CANONICAL_SCENARIOS[3]  # no_system_role
        result = check_role_support(tok, scenario)
        self.assertEqual(result.status, "pass")


class CheckGenerationBoundaryTest(unittest.TestCase):
    def test_pass_when_prefix_consistent(self):
        tok = CompatibleTokenizer()
        scenario = CANONICAL_SCENARIOS[0]  # single_turn_basic
        result = check_generation_boundary(tok, scenario)
        self.assertEqual(result.status, "pass")

    def test_fail_when_prefix_broken(self):
        tok = BoundaryBrokenTokenizer()
        scenario = CANONICAL_SCENARIOS[0]
        result = check_generation_boundary(tok, scenario)
        self.assertEqual(result.status, "fail")
        self.assertIn("NOT a prefix", result.message)

    def test_skip_when_last_not_assistant(self):
        tok = CompatibleTokenizer()
        # Construct a scenario where last message is user.
        scenario = {
            "name": "truncated",
            "messages": [{"role": "user", "content": "Hi"}],
            "expected_roles": {"user"},
        }
        result = check_generation_boundary(tok, scenario)
        self.assertEqual(result.status, "pass")
        self.assertIn("Skipped", result.message)


class CheckToolSchemaTest(unittest.TestCase):
    def test_pass_with_tool_rendered(self):
        tok = CompatibleTokenizer()
        scenario = CANONICAL_SCENARIOS[2]  # tool_call_request
        result = check_tool_schema(tok, scenario)
        self.assertEqual(result.status, "pass")

    def test_fail_when_tool_dropped(self):
        tok = ToolDroppingTokenizer()
        scenario = CANONICAL_SCENARIOS[2]  # tool_call_request
        result = check_tool_schema(tok, scenario)
        self.assertEqual(result.status, "fail")
        self.assertIn("function_name:get_weather", result.detail["missing_parts"][0])

    def test_skip_for_non_tool_scenario(self):
        tok = CompatibleTokenizer()
        scenario = CANONICAL_SCENARIOS[0]  # single_turn_basic
        result = check_tool_schema(tok, scenario)
        self.assertEqual(result.status, "pass")
        self.assertIn("Skipped", result.message)


class CheckDuplicateSpecialTokensTest(unittest.TestCase):
    def test_pass_no_duplicates(self):
        tok = CompatibleTokenizer()
        scenario = CANONICAL_SCENARIOS[0]
        result = check_duplicate_special_tokens(tok, scenario)
        self.assertEqual(result.status, "pass")

    def test_warn_with_duplicate_special(self):
        tok = DuplicateSpecialTokenizer()
        scenario = CANONICAL_SCENARIOS[0]
        result = check_duplicate_special_tokens(tok, scenario)
        self.assertEqual(result.status, "warning")
        self.assertEqual(result.detail["duplicate_count"], 1)
        self.assertEqual(result.detail["first_duplicate"]["token_id"], 99)


# ---------------------------------------------------------------------------
# Tests: full inspection
# ---------------------------------------------------------------------------


class ChatTemplateInspectorTest(unittest.TestCase):
    def test_pass_with_compatible_template(self):
        tok = CompatibleTokenizer()
        report = ChatTemplateInspector.inspect("test-model", tok)
        self.assertEqual(report.overall_status, "pass")
        self.assertIn("passed", report.summary)
        # Should have 1 (template) + 5 scenarios * 4 checks = 21 results.
        self.assertEqual(len(report.results), 1 + len(CANONICAL_SCENARIOS) * 4)

    def test_fail_fast_with_missing_template(self):
        tok = NoTemplateTokenizer()
        report = ChatTemplateInspector.inspect("bad-model", tok)
        self.assertEqual(report.overall_status, "fail")
        # Should short-circuit: only 1 result (the template check).
        self.assertEqual(len(report.results), 1)
        self.assertEqual(report.results[0].check_name, "missing_template")

    def test_warning_when_duplicate_special_tokens(self):
        tok = DuplicateSpecialTokenizer()
        report = ChatTemplateInspector.inspect("dup-model", tok)
        self.assertEqual(report.overall_status, "warning")

    def test_fail_when_tool_schema_broken(self):
        tok = ToolDroppingTokenizer()
        report = ChatTemplateInspector.inspect("tool-broken", tok)
        self.assertEqual(report.overall_status, "fail")
        tool_results = [r for r in report.results if r.check_name == "tool_schema" and r.status == "fail"]
        self.assertGreaterEqual(len(tool_results), 1)

    def test_to_dict_produces_valid_json(self):
        tok = CompatibleTokenizer()
        report = ChatTemplateInspector.inspect("json-model", tok)
        d = report.to_dict()
        # Must be JSON serialisable.
        s = json.dumps(d)
        parsed = json.loads(s)
        self.assertEqual(parsed["model_name"], "json-model")
        self.assertEqual(parsed["overall_status"], "pass")
        self.assertIsInstance(parsed["results"], list)

    def test_to_human_readable_contains_key_info(self):
        tok = CompatibleTokenizer()
        report = ChatTemplateInspector.inspect("hr-model", tok)
        text = report.to_human_readable()
        self.assertIn("Chat Template Inspection: hr-model", text)
        self.assertIn("Overall: PASS", text)
        self.assertIn("OK", text)

    def test_to_human_readable_shows_failures(self):
        tok = NoTemplateTokenizer()
        report = ChatTemplateInspector.inspect("fail-model", tok)
        text = report.to_human_readable()
        self.assertIn("Overall: FAIL", text)
        self.assertIn("FAIL", text)
        self.assertIn("no chat_template", text.lower())


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class InspectChatTemplateCLITest(unittest.TestCase):
    def _invoke(self, model: str, output_format: str = "text"):
        """Invoke the CLI command with mocked tokenizer loading."""

        tok = CompatibleTokenizer()
        with (
            patch("areno.cli.inspect.resolve_model_ref", return_value=f"/fake/{model}"),
            patch("areno.cli.inspect.load_tokenizer", return_value=tok),
        ):
            runner = CliRunner()
            return runner.invoke(inspect_chat_template_command, [
                "--model", model,
                "--output-format", output_format,
            ])

    def test_cli_text_output_pass(self):
        result = self._invoke("compatible-model", "text")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Chat Template Inspection", result.output)
        self.assertIn("Overall: PASS", result.output)

    def test_cli_json_output_pass(self):
        result = self._invoke("compatible-model", "json")
        self.assertEqual(result.exit_code, 0)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["overall_status"], "pass")
        self.assertEqual(parsed["model_name"], "compatible-model")
        self.assertIsInstance(parsed["results"], list)

    def test_cli_exit_code_on_failure(self):
        tok = NoTemplateTokenizer()
        with (
            patch("areno.cli.inspect.resolve_model_ref", return_value="/fake/bad"),
            patch("areno.cli.inspect.load_tokenizer", return_value=tok),
        ):
            runner = CliRunner()
            result = runner.invoke(inspect_chat_template_command, [
                "--model", "bad-model",
                "--output-format", "text",
            ])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("FAIL", result.output)

    def test_cli_json_output_on_failure(self):
        tok = NoTemplateTokenizer()
        with (
            patch("areno.cli.inspect.resolve_model_ref", return_value="/fake/bad"),
            patch("areno.cli.inspect.load_tokenizer", return_value=tok),
        ):
            runner = CliRunner()
            result = runner.invoke(inspect_chat_template_command, [
                "--model", "bad-model",
                "--output-format", "json",
            ])
        self.assertEqual(result.exit_code, 1)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["overall_status"], "fail")
        self.assertEqual(len(parsed["results"]), 1)


# ---------------------------------------------------------------------------
# Tests: default behavior unchanged
# ---------------------------------------------------------------------------


class DefaultBehaviorTest(unittest.TestCase):
    def test_inspector_does_not_modify_tokenizer(self):
        """Running the inspector must not mutate the tokenizer object."""

        tok = CompatibleTokenizer()
        original_template = tok.chat_template
        original_special_ids = tok.all_special_ids
        ChatTemplateInspector.inspect("test", tok)
        self.assertEqual(tok.chat_template, original_template)
        self.assertEqual(tok.all_special_ids, original_special_ids)

    def test_inspector_not_invoked_when_not_called(self):
        """The inspector is opt-in; existing code paths are unaffected."""

        # Verify the module imports without side effects.
        from areno.api import chat_template_inspector
        self.assertTrue(hasattr(chat_template_inspector, "ChatTemplateInspector"))


if __name__ == "__main__":
    unittest.main()