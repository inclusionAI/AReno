"""Reward function for the Sudoku multi-turn tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import score_episode  # noqa: E402


def reward_fn(record) -> float:
    """Replay the tool-call trajectory and score the episode."""

    source = dict(record.source_record)
    actions = []
    for call in record.tool_calls:
        name = call.get("name")
        if not name:
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        actions.append({"name": name, "arguments": args})
    result = score_episode(
        source["puzzle"],
        actions,
        max_actions=int(source.get("max_actions", 120)),
    )
    return result["reward"]