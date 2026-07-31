"""Reward function for the multi-tool agentic example.

Emits per-dimension scores (tool_selection, arguments, order, final_answer)
and separate failure-class metrics via structured log output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import score_task  # noqa: E402

logger = logging.getLogger("areno.multitool.reward")


def reward_fn(record) -> float:
    """Score the tool-call trajectory for a multi-tool task.

    Uses per-dimension scoring from game.score_task and returns the overall
    reward. Per-dimension breakdown (tool_selection, arguments, order,
    final_answer) and failure classes are emitted via structured log output.
    """

    source = dict(record.source_record)
    tool_calls = list(record.tool_calls)
    score = score_task(source, tool_calls)
    logger.info(
        "multitool_reward overall=%.4f tool_selection=%.4f arguments=%.4f order=%.4f final_answer=%.4f failures=%s",
        score["overall"],
        score["tool_selection"],
        score["arguments"],
        score["order"],
        score["final_answer"],
        ",".join(score["failures"]) if score["failures"] else "none",
    )
    return float(score["overall"])