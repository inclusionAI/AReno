"""Reward function for the DuelGrid tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the choose_action tool call."""

    state = dataset_generator.record_to_state(record.source_record["state"])
    return game.score_actions(state, _tool_actions(record))


def _tool_actions(record: Any) -> list[dict[str, str]]:
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "choose_action":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return []
        if isinstance(arguments, dict):
            return game.parse_actions(arguments)
    return game.parse_actions(getattr(record, "completion", None))
