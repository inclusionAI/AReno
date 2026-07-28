"""Reward function for the maze tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one maze episode by replaying the act tool call actions."""

    source = record.source_record
    raw = source.get("state", source)
    state = dataset_generator.record_to_state(raw)
    actions = _extract_actions(record)
    total, _metrics = game.compute_trajectory_reward(state, actions)
    return total


def _extract_actions(record: Any) -> list[dict[str, str] | None]:
    """Pull the ordered action list from the act tool call."""

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
                return []
        if not isinstance(args, dict):
            continue
        raw_actions = args.get("actions", [])
        if not isinstance(raw_actions, list):
            continue
        actions: list[dict[str, str] | None] = []
        for item in raw_actions:
            if isinstance(item, dict):
                actions.append({
                    "action": str(item.get("action", "")),
                    "direction": str(item.get("direction", "")),
                })
            else:
                actions.append(None)
        return actions
    return []
