#!/usr/bin/env python3
"""Validate OpenAI-style assistant tool-call and tool-result ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args()
    value = json.loads(args.transcript.read_text(encoding="utf-8"))
    messages = value.get("messages", value) if isinstance(value, dict) else value
    errors = []
    pending: list[str] = []
    calls = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message {index} is not an object")
            continue
        role = message.get("role")
        tool_calls = message.get("tool_calls")
        if role == "assistant" and tool_calls is not None:
            if not isinstance(tool_calls, list) or not tool_calls:
                errors.append(f"assistant message {index} has empty tool_calls")
                continue
            for call in tool_calls:
                call_id = call.get("id") if isinstance(call, dict) else None
                function = call.get("function") if isinstance(call, dict) else None
                if not call_id or not isinstance(function, dict) or not function.get("name"):
                    errors.append(f"invalid tool call at message {index}")
                    continue
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        json.loads(arguments)
                    except json.JSONDecodeError:
                        errors.append(f"non-JSON arguments for call {call_id}")
                elif not isinstance(arguments, dict):
                    errors.append(f"non-JSON arguments for call {call_id}")
                pending.append(call_id)
                calls += 1
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                errors.append(f"unmatched tool result {call_id!r} at message {index}")
            else:
                pending.remove(call_id)
        elif pending:
            errors.append(f"message {index} appears before pending tool results {pending}")
    if pending:
        errors.append(f"missing tool results for {pending}")
    result = {"ok": not errors, "messages": len(messages), "tool_calls": calls, "errors": errors}
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
