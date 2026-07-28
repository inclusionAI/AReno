"""Chat-template compatibility inspector.

A lightweight pre-training diagnostic that loads only the tokenizer (not model
weights) and renders a set of canonical message scenarios through the model's
chat template.  It proactively detects five categories of compatibility issues
that would otherwise surface mid-training as mis-encoded data or crashes.

The inspector reuses existing AReno contracts (``load_tokenizer``,
``apply_chat_template_with_options``, ``normalize_messages``) and produces both
structured (dict/JSON) and human-readable (terminal table) output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from areno.api.tokenizer import apply_chat_template_with_options


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiagnosticResult:
    """Outcome of a single diagnostic check for one scenario.

    ``status`` is one of ``"pass"``, ``"fail"``, ``"warning"``.
    ``detail`` carries structured information for programmatic consumers.
    """

    check_name: str
    scenario_name: str
    status: str
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InspectionReport:
    """Aggregated report across all scenarios and checks.

    Use :meth:`to_dict` for structured (JSON) output and
    :meth:`to_human_readable` for terminal display.
    """

    model_name: str
    overall_status: str  # "pass" | "fail" | "warning"
    summary: str
    results: list[DiagnosticResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for programmatic consumption."""

        return {
            "model_name": self.model_name,
            "overall_status": self.overall_status,
            "summary": self.summary,
            "results": [
                {
                    "check_name": r.check_name,
                    "scenario_name": r.scenario_name,
                    "status": r.status,
                    "message": r.message,
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }

    def to_human_readable(self) -> str:
        """Return a terminal-friendly table for CLI display."""

        lines: list[str] = []
        lines.append(f"Chat Template Inspection: {self.model_name}")
        lines.append(f"Overall: {self.overall_status.upper()}")
        lines.append(self.summary)
        lines.append("-" * 60)
        for r in self.results:
            icon = {"pass": "OK  ", "fail": "FAIL", "warning": "WARN"}[r.status]
            lines.append(f"  [{icon}] {r.check_name} ({r.scenario_name})")
            if r.status != "pass" and r.message:
                # Indent and wrap at 72 cols for readability.
                lines.append(f"       {r.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canonical test scenarios
# ---------------------------------------------------------------------------

CANONICAL_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "single_turn_basic",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "expected_roles": {"system", "user", "assistant"},
    },
    {
        "name": "multi_turn",
        "messages": [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "And 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "expected_roles": {"user", "assistant"},
    },
    {
        "name": "tool_call_request",
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Sunny, 72F",
            },
            {"role": "assistant", "content": "It's sunny and 72F in NYC."},
        ],
        "expected_roles": {"user", "assistant", "tool"},
    },
    {
        "name": "no_system_role",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "expected_roles": {"user", "assistant"},
    },
    {
        "name": "empty_assistant",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
        ],
        "expected_roles": {"user", "assistant"},
    },
]


# ---------------------------------------------------------------------------
# Individual diagnostic checks
# ---------------------------------------------------------------------------


def check_template_exists(tokenizer: Any) -> DiagnosticResult:
    """Verify that the tokenizer has a ``chat_template`` attribute."""

    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return DiagnosticResult(
            check_name="missing_template",
            scenario_name="_global",
            status="fail",
            message="Tokenizer has no chat_template defined. "
            "Messages cannot be rendered without a template.",
            detail={"has_chat_template": False},
        )
    return DiagnosticResult(
        check_name="missing_template",
        scenario_name="_global",
        status="pass",
        message="",
        detail={"has_chat_template": True},
    )


def check_role_support(
    tokenizer: Any, scenario: dict[str, Any]
) -> DiagnosticResult:
    """Render the scenario and verify no role is dropped or causes an error."""

    name = scenario["name"]
    messages = scenario["messages"]
    expected_roles = scenario["expected_roles"]

    try:
        rendered = apply_chat_template_with_options(
            tokenizer, messages, tokenize=False
        )
    except Exception as exc:  # noqa: BLE001 — we need to catch template errors
        # Identify which role likely caused the failure by checking the
        # exception message for role names.
        exc_str = str(exc).lower()
        culprit = None
        for role in expected_roles:
            if role in exc_str:
                culprit = role
                break
        return DiagnosticResult(
            check_name="role_support",
            scenario_name=name,
            status="fail",
            message=(
                f"Template raised an error rendering scenario '{name}'. "
                f"{'Suspected unsupported role: ' + culprit + '.' if culprit else ''} "
                f"Error: {exc}"
            ),
            detail={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "suspected_role": culprit,
            },
        )

    if not isinstance(rendered, str) or len(rendered.strip()) == 0:
        return DiagnosticResult(
            check_name="role_support",
            scenario_name=name,
            status="fail",
            message=f"Template returned empty output for scenario '{name}'.",
            detail={"rendered_length": len(rendered) if rendered else 0},
        )

    # Verify that user content appears in the rendered text (a basic sanity
    # check — we do not require every role's content to be verbatim, but the
    # user message should always be present).
    for msg in messages:
        if msg["role"] == "user" and msg.get("content"):
            if msg["content"] not in rendered:
                return DiagnosticResult(
                    check_name="role_support",
                    scenario_name=name,
                    status="warning",
                    message=(
                        f"User content '{msg['content']}' is missing from "
                        f"the rendered text for scenario '{name}'. "
                        f"The template may be silently dropping user messages."
                    ),
                    detail={"missing_content": msg["content"]},
                )

    return DiagnosticResult(
        check_name="role_support",
        scenario_name=name,
        status="pass",
        message="",
    )


def check_generation_boundary(
    tokenizer: Any, scenario: dict[str, Any]
) -> DiagnosticResult:
    """Verify that ``add_generation_prompt`` output is a prefix of the full render."""

    name = scenario["name"]
    messages = scenario["messages"]

    # Only meaningful when the last message is a non-empty assistant turn
    # we can remove to create the "prompt" version.  Empty assistant turns
    # are a degenerate case where the boundary check is not applicable.
    if not messages or messages[-1]["role"] != "assistant":
        return DiagnosticResult(
            check_name="generation_boundary",
            scenario_name=name,
            status="pass",
            message="Skipped: last message is not an assistant turn.",
        )
    last_content = messages[-1].get("content", "")
    if not last_content:
        return DiagnosticResult(
            check_name="generation_boundary",
            scenario_name=name,
            status="pass",
            message="Skipped: last assistant turn has empty content.",
        )

    prompt_messages = messages[:-1]

    try:
        prompt_rendered = apply_chat_template_with_options(
            tokenizer, prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_rendered = apply_chat_template_with_options(
            tokenizer, messages, tokenize=False
        )
    except Exception as exc:  # noqa: BLE001
        return DiagnosticResult(
            check_name="generation_boundary",
            scenario_name=name,
            status="fail",
            message=(
                f"Template raised an error during generation-boundary check "
                f"for scenario '{name}'. Error: {exc}"
            ),
            detail={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )

    if not isinstance(prompt_rendered, str) or not isinstance(full_rendered, str):
        return DiagnosticResult(
            check_name="generation_boundary",
            scenario_name=name,
            status="warning",
            message=(
                f"Template returned non-string output for scenario '{name}'. "
                f"Cannot verify generation boundary."
            ),
        )

    if full_rendered.startswith(prompt_rendered):
        return DiagnosticResult(
            check_name="generation_boundary",
            scenario_name=name,
            status="pass",
            message="",
        )

    return DiagnosticResult(
        check_name="generation_boundary",
        scenario_name=name,
        status="fail",
        message=(
            f"add_generation_prompt output is NOT a prefix of the full "
            f"conversation render for scenario '{name}'. "
            f"Loss mask boundary will be incorrect."
        ),
        detail={
            "prompt_rendered_length": len(prompt_rendered),
            "full_rendered_length": len(full_rendered),
        },
    )


def check_tool_schema(
    tokenizer: Any, scenario: dict[str, Any]
) -> DiagnosticResult:
    """For tool-call scenarios, verify tool content appears in the render."""

    name = scenario["name"]
    messages = scenario["messages"]

    has_tool = any(
        msg.get("role") == "tool" or msg.get("tool_calls")
        for msg in messages
    )
    if not has_tool:
        return DiagnosticResult(
            check_name="tool_schema",
            scenario_name=name,
            status="pass",
            message="Skipped: scenario has no tool messages.",
        )

    try:
        rendered = apply_chat_template_with_options(
            tokenizer, messages, tokenize=False
        )
    except Exception as exc:  # noqa: BLE001
        return DiagnosticResult(
            check_name="tool_schema",
            scenario_name=name,
            status="fail",
            message=(
                f"Template raised an error rendering tool messages for "
                f"scenario '{name}'. Error: {exc}"
            ),
            detail={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )

    if not isinstance(rendered, str) or len(rendered.strip()) == 0:
        return DiagnosticResult(
            check_name="tool_schema",
            scenario_name=name,
            status="fail",
            message=f"Template returned empty output for tool scenario '{name}'.",
        )

    missing_parts: list[str] = []

    # Check that function names from tool_calls appear in the render.
    for msg in messages:
        calls = msg.get("tool_calls")
        if not calls:
            continue
        for call in calls:
            func = call.get("function", {})
            func_name = func.get("name", "")
            if func_name and func_name not in rendered:
                missing_parts.append(f"function_name:{func_name}")

    # Check that tool-role response content appears.
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("content"):
            if msg["content"] not in rendered:
                missing_parts.append(f"tool_response:{msg['content'][:30]}")

    if missing_parts:
        return DiagnosticResult(
            check_name="tool_schema",
            scenario_name=name,
            status="fail",
            message=(
                f"Tool schema rendering is incomplete for scenario '{name}'. "
                f"Missing: {', '.join(missing_parts)}"
            ),
            detail={"missing_parts": missing_parts},
        )

    return DiagnosticResult(
        check_name="tool_schema",
        scenario_name=name,
        status="pass",
        message="",
    )


def check_duplicate_special_tokens(
    tokenizer: Any, scenario: dict[str, Any]
) -> DiagnosticResult:
    """Tokenize the rendered text and check for consecutive duplicate special tokens."""

    name = scenario["name"]
    messages = scenario["messages"]

    try:
        rendered = apply_chat_template_with_options(
            tokenizer, messages, tokenize=False
        )
        token_ids = tokenizer.encode(rendered)
    except Exception as exc:  # noqa: BLE001
        return DiagnosticResult(
            check_name="duplicate_special_tokens",
            scenario_name=name,
            status="warning",
            message=(
                f"Could not tokenize rendered text for scenario '{name}'. "
                f"Error: {exc}"
            ),
            detail={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )

    # Normalise to a plain list of ints.
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    if hasattr(token_ids, "ids"):
        token_ids = token_ids.ids
    if not isinstance(token_ids, (list, tuple)):
        token_ids = list(token_ids)

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    if not special_ids:
        # If we cannot determine special tokens, skip the check.
        return DiagnosticResult(
            check_name="duplicate_special_tokens",
            scenario_name=name,
            status="pass",
            message="Skipped: tokenizer has no special token IDs.",
        )

    duplicates: list[dict[str, Any]] = []
    for i in range(1, len(token_ids)):
        if (
            token_ids[i] in special_ids
            and token_ids[i] == token_ids[i - 1]
        ):
            duplicates.append({
                "position": i,
                "token_id": token_ids[i],
            })

    if duplicates:
        # Decode the duplicated token for a friendlier message.
        try:
            token_str = tokenizer.decode([duplicates[0]["token_id"]])
        except Exception:  # noqa: BLE001
            token_str = "<unknown>"

        return DiagnosticResult(
            check_name="duplicate_special_tokens",
            scenario_name=name,
            status="warning",
            message=(
                f"Found {len(duplicates)} consecutive duplicate special "
                f"token(s) in scenario '{name}'. First: token_id="
                f"{duplicates[0]['token_id']} ({token_str!r}). "
                f"This may indicate a template bug."
            ),
            detail={
                "duplicate_count": len(duplicates),
                "first_duplicate": duplicates[0],
                "token_str": token_str,
            },
        )

    return DiagnosticResult(
        check_name="duplicate_special_tokens",
        scenario_name=name,
        status="pass",
        message="",
    )


# ---------------------------------------------------------------------------
# Inspector entry point
# ---------------------------------------------------------------------------

_ALL_CHECKS = [
    check_role_support,
    check_generation_boundary,
    check_tool_schema,
    check_duplicate_special_tokens,
]


class ChatTemplateInspector:
    """Orchestrates the five diagnostic checks across all canonical scenarios."""

    @staticmethod
    def inspect(model_name: str, tokenizer: Any) -> InspectionReport:
        """Run all checks and return an :class:`InspectionReport`.

        ``model_name`` is the user-provided model identifier (for reporting).
        ``tokenizer`` is a pre-loaded tokenizer instance.
        """

        results: list[DiagnosticResult] = []

        # Check 1: template existence (global, only once).
        template_result = check_template_exists(tokenizer)
        results.append(template_result)

        if template_result.status == "fail":
            # No point running further checks without a template.
            return InspectionReport(
                model_name=model_name,
                overall_status="fail",
                summary="1 check, 1 failed — no chat_template defined.",
                results=results,
            )

        # Checks 2-5: per scenario.
        for scenario in CANONICAL_SCENARIOS:
            for check_fn in _ALL_CHECKS:
                results.append(check_fn(tokenizer, scenario))

        # Aggregate overall status.
        fail_count = sum(1 for r in results if r.status == "fail")
        warn_count = sum(1 for r in results if r.status == "warning")
        pass_count = sum(1 for r in results if r.status == "pass")

        if fail_count:
            overall = "fail"
        elif warn_count:
            overall = "warning"
        else:
            overall = "pass"

        summary = (
            f"{len(results)} checks: {pass_count} passed, "
            f"{fail_count} failed, {warn_count} warnings."
        )

        return InspectionReport(
            model_name=model_name,
            overall_status=overall,
            summary=summary,
            results=results,
        )