"""Reward function for the multi-tool agentic example.

Emits per-dimension scores (tool_selection, arguments, order, final_answer)
and separate failure-class metrics via the reward record metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import score_task  # noqa: E402


def reward_fn(record) -> float:
    """Score the tool-call trajectory for a multi-tool task.

    Uses per-dimension scoring from game.score_task and returns the overall
    reward. Per-dimension breakdown is available in the returned score dict.
    """

    source = dict(record.source_record)
    tool_calls = list(record.tool_calls)
    score = score_task(source, tool_calls)
    return float(score["overall"])