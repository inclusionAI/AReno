"""Reward function for the multi-turn calendar scheduling agentic example.

Rewards the final confirmed slot, with a multi-turn tool-use bonus for
following the correct query → propose → confirm flow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Reward the final confirmed slot, with a multi-turn tool-use bonus.

    Evaluates two dimensions:
    1. Constraint satisfaction: the confirmed slot must pass all validation
       (timezone conversion, availability, conflicts, duration).
    2. Tool-call flow: the model should follow query → propose → confirm
       in the correct order.
    """
    source = record.source_record
    state = game.record_to_state(source)
    meeting_id = source.get("target_meeting_id", "")

    tool_calls = list(record.tool_calls)
    names = [call.get("name", "") for call in tool_calls if isinstance(call, dict)]

    # Extract the last confirmed or proposed slot.
    confirmed = _extract_confirmed_slot(tool_calls, meeting_id)
    if confirmed is None:
        # No confirmation found — check if there was no solution.
        meeting = state.meeting_by_id(meeting_id)
        if meeting is not None:
            common = game.find_common_slots(meeting, state.participants)
            if not common:
                # No solution exists; reward neutral if agent didn't confirm.
                return 0.0
        return -1.0

    utc_start, utc_end = confirmed
    # Check constraint satisfaction.
    error = game.validate_proposal(state, meeting_id, utc_start, utc_end)
    if error is not None:
        return -1.0

    # Check tool-call flow: query → propose → confirm.
    expected_flow = _has_correct_flow(names)
    if expected_flow:
        return 1.0
    # Correct slot but wrong flow → partial credit.
    return 0.5


def _extract_confirmed_slot(
    tool_calls: list[dict[str, Any]], meeting_id: str
) -> tuple[int, int] | None:
    """Extract the last confirmed (or proposed) UTC slot for the meeting.

    Prefer confirm_slot; fall back to propose_slot if no confirm was made.
    """
    # Look for confirm_slot first (in reverse).
    for call in reversed(tool_calls):
        name = call.get("name", "") if isinstance(call, dict) else ""
        if name != "confirm_slot":
            continue
        slot = _parse_slot_args(call, meeting_id)
        if slot is not None:
            return slot
    # Fall back to propose_slot.
    for call in reversed(tool_calls):
        name = call.get("name", "") if isinstance(call, dict) else ""
        if name != "propose_slot":
            continue
        slot = _parse_slot_args(call, meeting_id)
        if slot is not None:
            return slot
    return None


def _parse_slot_args(call: dict[str, Any], meeting_id: str) -> tuple[int, int] | None:
    """Parse UTC start/end from a tool call's arguments."""
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    if args.get("meeting_id") != meeting_id:
        return None
    try:
        utc_start = int(args.get("utc_start_hour", -1))
        utc_end = int(args.get("utc_end_hour", -1))
    except (TypeError, ValueError):
        return None
    if utc_start >= 0 and utc_end > utc_start:
        return utc_start, utc_end
    return None


def _has_correct_flow(names: list[str]) -> bool:
    """Check if the tool-call names follow the expected query → propose → confirm flow.

    The flow is: at least one query_availability, then propose_slot, then confirm_slot.
    The order must be: all queries before propose, propose before confirm.
    """
    # Find indices of each tool type.
    query_indices = [i for i, n in enumerate(names) if n == "query_availability"]
    propose_indices = [i for i, n in enumerate(names) if n == "propose_slot"]
    confirm_indices = [i for i, n in enumerate(names) if n == "confirm_slot"]

    # Must have at least one query, one propose, and one confirm.
    if not query_indices or not propose_indices or not confirm_indices:
        return False

    # All queries must come before the first propose.
    if max(query_indices) > min(propose_indices):
        return False
    # Propose must come before confirm.
    if min(propose_indices) > min(confirm_indices):
        return False

    return True