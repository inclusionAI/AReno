"""Reward function for the multi-turn shopping kit tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import score_bundle  # noqa: E402


def reward_fn(record) -> float:
    """Reward the final submitted bundle, with a small multi-turn tool-use bonus."""

    source = dict(record.source_record)
    tool_calls = list(record.tool_calls)
    names = [call.get("name") for call in tool_calls]
    submitted = _submitted_item_ids(tool_calls)
    score = score_bundle(source, submitted)
    if score > 0 and names[:4] == ["search_catalog", "inspect_items", "check_kit", "submit_bundle"]:
        return score
    if score > 0:
        return 0.5
    return score


def _submitted_item_ids(tool_calls: list[dict[str, Any]]) -> list[str] | None:
    for call in reversed(tool_calls):
        if call.get("name") != "submit_bundle":
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if isinstance(args, dict) and isinstance(args.get("item_ids"), list):
            return [str(item_id) for item_id in args["item_ids"]]
    return None
