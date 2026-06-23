"""Reward function for the agentic coding example."""

from __future__ import annotations

import json
from typing import Any


def reward_fn(record) -> float:
    """Reward successful test execution followed by a solved submission."""

    source = dict(record.source_record)
    required_commands = [str(command) for command in source.get("test_commands", [])]
    tool_calls = list(record.tool_calls)
    tool_results = [_decode_result(result.get("content")) for result in record.tool_results]
    submitted = _submitted_status(tool_calls)
    successful_commands = {
        str(result.get("command"))
        for result in tool_results
        if result.get("returncode") == 0 and isinstance(result.get("command"), str)
    }
    all_tests_passed = (
        all(command in successful_commands for command in required_commands) if required_commands else True
    )
    applied_patch = any(call.get("name") == "apply_patch" for call in tool_calls)
    if submitted == "solved" and all_tests_passed and applied_patch:
        return 1.0
    if all_tests_passed and required_commands:
        return 0.5
    if submitted == "solved":
        return -0.5
    return -1.0


def _decode_result(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _submitted_status(tool_calls: list[dict[str, Any]]) -> str | None:
    for call in reversed(tool_calls):
        if call.get("name") != "submit":
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if isinstance(args, dict):
            return str(args.get("status"))
    return None
