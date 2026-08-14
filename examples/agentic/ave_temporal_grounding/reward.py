"""Numerical timestamp reward for AVE audiovisual temporal grounding."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import timestamp_reward  # noqa: E402


def reward_fn(record: Any) -> float:
    """Reward temporal overlap and boundary accuracy for one event interval."""

    prediction = _tool_prediction(getattr(record, "tool_calls", None))
    if prediction is None:
        return -1.0
    source = record.source_record
    return timestamp_reward(
        prediction.get("start_seconds"),
        prediction.get("end_seconds"),
        source["start_seconds"],
        source["end_seconds"],
    )


def _tool_prediction(tool_calls: Any) -> dict[str, Any] | None:
    matching = [
        call for call in tool_calls or [] if isinstance(call, dict) and call.get("name") == "report_event_range"
    ]
    if len(matching) != 1:
        return None
    arguments = matching[0].get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return arguments if isinstance(arguments, dict) else None
