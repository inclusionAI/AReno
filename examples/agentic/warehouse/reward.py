"""Graded distance reward for warehouse navigation trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    baseline_distance,
    build_state,
    execute_action,
    remaining_distance,
    state_metrics,
)


def reward_fn(record) -> float:
    """Replay tool calls and score completion plus navigation progress."""

    state = build_state(dict(record.source_record))
    baseline = baseline_distance(state)
    groups = _tool_call_groups(record)

    for calls in groups:
        if len(calls) != 1:
            state.invalid_actions += 1
            continue

        name, raw_arguments = _call_parts(calls[0])
        if name is None:
            state.invalid_actions += 1
            continue
        arguments, valid_json = _parse_arguments(raw_arguments)
        if not valid_json:
            state.invalid_actions += 1
            continue

        execute_action(state, name, arguments)

    metrics = state_metrics(state, baseline=baseline)

    if state.completed:
        score = 1.0
    else:
        remaining = remaining_distance(state)
        if baseline > 0:
            progress = max(0.0, 1.0 - remaining / baseline)
        else:
            progress = 1.0 if remaining == 0 else 0.0
        score = -0.5 + 0.3 * progress

    score -= 0.05 * state.invalid_actions
    return max(-1.0, min(1.0, score))


def _tool_call_groups(record) -> list[list[dict[str, Any]]]:
    """Preserve assistant response boundaries when extracting tool calls."""

    groups: list[list[dict[str, Any]]] = []
    for message in getattr(record, "messages", []) or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            groups.append([call for call in calls if isinstance(call, dict)])
    if groups:
        return groups

    fallback = [call for call in (getattr(record, "tool_calls", []) or []) if isinstance(call, dict)]
    return [[call] for call in fallback]


def _call_parts(call: dict[str, Any]) -> tuple[str | None, Any]:
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        return (
            name if isinstance(name, str) and name else None,
            function.get("arguments"),
        )

    name = call.get("name")
    return (
        name if isinstance(name, str) and name else None,
        call.get("arguments"),
    )


def _parse_arguments(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        return value, True
    if not isinstance(value, str):
        return None, False
    try:
        return json.loads(value), True
    except json.JSONDecodeError:
        return None, False