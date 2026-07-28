"""Reward function for the multi-turn maze tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one maze episode by replaying all act tool calls."""

    state = dataset_generator.record_to_state(record.source_record)
    actions = _extract_actions(record)
    total, _metrics = game.compute_trajectory_reward(state, actions)
    return total


def _extract_actions(record: Any) -> list[dict[str, str] | None]:
    """Pull the ordered list of act tool-call arguments from the record."""

    actions: list[dict[str, str] | None] = []
    for call in getattr(record, "tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        if call.get("name") != "act":
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                actions.append(None)
                continue
        if isinstance(args, dict):
            actions.append({"action": str(args.get("action", "")), "direction": str(args.get("direction", ""))})
        else:
            actions.append(None)
    return actions
