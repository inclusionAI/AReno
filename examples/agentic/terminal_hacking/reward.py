"""Dense asymmetric reward for single-turn terminal candidate filtering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import candidate_filter_session  # noqa: E402


def reward_fn(record) -> float:
    calls = list(record.tool_calls)
    if len(calls) != 1 or calls[0].get("name") != "submit_candidates":
        return -1.0
    arguments = calls[0].get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return -1.0
    if not isinstance(arguments, dict) or set(arguments) != {"candidates"}:
        return -1.0
    predicted = arguments["candidates"]
    if (
        not isinstance(predicted, list)
        or not predicted
        or any(not isinstance(word, str) for word in predicted)
        or len(predicted) != len(set(predicted))
    ):
        return -1.0

    session = candidate_filter_session(dict(record.source_record))
    active = session.public_state()["active_candidates"]
    if any(word not in active for word in predicted):
        return -1.0
    expected_set = set(session.consistent_candidates())
    predicted_set = set(predicted)
    false_positives = len(predicted_set - expected_set)
    false_negatives = len(expected_set - predicted_set)
    shrink_ratio = false_negatives / max(len(expected_set), 1)

    # Extra inconsistent candidates are much worse than a conservative subset:
    # one extra costs 1.5 reward, while omitting the entire true set costs 0.1.
    score = 1.0 - 1.5 * false_positives - 0.1 * shrink_ratio
    return max(-1.0, min(1.0, score))
