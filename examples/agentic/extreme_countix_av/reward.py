"""Reward for audiovisual repetition counting."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import count_similarity  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score only relative repetition-count accuracy."""

    prediction = _tool_prediction(getattr(record, "tool_calls", None))
    if prediction is None:
        prediction = _text_prediction(getattr(record, "completion", ""))
    if prediction is None:
        return -1.0
    source = record.source_record
    return round(count_similarity(prediction.get("repetition_count"), source["repetition_count"]), 6)


def _tool_prediction(tool_calls: Any) -> dict[str, Any] | None:
    for call in tool_calls or []:
        if not isinstance(call, dict) or call.get("name") != "report_repetitions":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict) and arguments.get("repetition_count") is not None:
            return {**arguments, "from_tool": True}
    return None


def _text_prediction(text: Any) -> dict[str, Any] | None:
    value = str(text).strip()
    match = re.search(r"(.+?)(?:\s*[:,-]\s*|\s+)(\d+)\s*(?:repetitions?|reps?)?\s*$", value, re.I)
    if not match:
        return None
    return {"action_class": match.group(1).strip(), "repetition_count": int(match.group(2)), "from_tool": False}
