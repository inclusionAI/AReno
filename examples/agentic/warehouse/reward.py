"""Reward function for the warehouse-picking agentic RL example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import baseline_distance, build_state, score_task  # noqa: E402


def reward_fn(record) -> float:
    """AReno entry point: extract stats from RewardRecord, delegate to score_task."""

    source = dict(record.source_record)
    tool_calls = list(record.tool_calls)
    names = [call.get("name") for call in tool_calls]

    stats = _extract_stats_from_messages(record.messages)
    state = build_state(source)
    baseline = baseline_distance(state)

    trajectory_data = {
        "completed": stats["completed"],
        "distance": stats["distance"],
        "picking_errors": stats["picking_errors"],
        "invalid_actions": stats["invalid_actions"],
        "baseline_distance": baseline,
        "tool_names": names,
        "cart": stats.get("cart", {}),
    }
    return score_task(source, trajectory_data)


def _extract_stats_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract environment statistics from tool result messages."""

    stats: dict[str, Any] = {
        "completed": False,
        "distance": 0,
        "picking_errors": 0,
        "invalid_actions": 0,
        "cart": {},
    }
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        try:
            content = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        data = content.get("data", {})
        if data.get("completed"):
            stats["completed"] = True
        if "distance" in data:
            stats["distance"] = max(stats["distance"], data["distance"])
        if content.get("success") and "cart" in data:
            stats["cart"] = dict(data["cart"])
        if not content.get("success"):
            msg_text = content.get("message", "")
            if "unreachable" in msg_text or "unknown shelf" in msg_text:
                stats["invalid_actions"] += 1
            elif "insufficient" in msg_text or "not on shelf" in msg_text or "invalid qty" in msg_text:
                stats["picking_errors"] += 1
    return stats