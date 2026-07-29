"""Reward function for the calendar scheduling agentic example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting tool calls and evaluating constraints.

    Evaluates two dimensions:
    1. Constraint satisfaction: timezone conversion, availability, conflicts, duration.
    2. Tool-call efficiency: minimal queries, single propose, single confirm.
    """
    source = record.source_record
    state = game.record_to_state(source)
    meeting_id = source.get("target_meeting_id", "")

    tool_calls = _extract_tool_calls(record)
    proposed = _extract_proposal(tool_calls, meeting_id)

    if proposed is None:
        # No valid proposal found — check if there was no solution.
        meeting = state.meeting_by_id(meeting_id)
        if meeting is not None:
            common = game.find_common_slots(meeting, state.participants)
            if not common:
                # No solution exists; reward neutral if agent didn't propose.
                return 0.0
        return -1.0

    utc_start, utc_end = proposed
    return game.compute_reward(state, meeting_id, utc_start, utc_end, tool_calls)


def _extract_tool_calls(record: Any) -> list[dict[str, Any]]:
    """Extract the list of tool calls from the agent record."""
    calls = getattr(record, "tool_calls", None)
    if calls is None:
        return []
    result = []
    for call in calls:
        if isinstance(call, dict):
            result.append(call)
    return result


def _extract_proposal(
    tool_calls: list[dict[str, Any]], meeting_id: str
) -> tuple[int, int] | None:
    """Extract the last proposed or confirmed UTC slot for the given meeting."""
    for call in reversed(tool_calls):
        name = call.get("name", "") if isinstance(call, dict) else ""
        if name not in ("propose_slot", "confirm_slot"):
            continue
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        if args.get("meeting_id") != meeting_id:
            continue
        try:
            utc_start = int(args.get("utc_start_hour", -1))
            utc_end = int(args.get("utc_end_hour", -1))
        except (TypeError, ValueError):
            continue
        if utc_start >= 0 and utc_end > utc_start:
            return utc_start, utc_end
    return None