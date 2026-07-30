"""Reward function for the odd-ball balance-scale tool-call example.

Scoring formula:  R_end = K - T·a - P_repeat - P_invalid

  K (answer reward):
    - Full answer correct (ball + direction) → base_reward
    - Identity only correct (ball, wrong direction) → base_reward / 2
    - Submitted but completely wrong → 0
    - No submit_answer call → -1 (base penalty)

  base_reward = ceil(log3(num_balls * 2))  — information-theoretic lower bound,
    auto-scales to any number of balls.

  T (weighing cost):  each valid weighing costs `alpha` (default 0.15).

  P_repeat:  penalty for repeated identical weighings (same left+right sets).

  P_invalid:  penalty for invalid weighings (unequal size, overlapping, out
    of range) — counted but not fatal; the agent should learn to avoid them.

  Early termination protection:  if the agent submits within the information-
  theoretic minimum number of weighings (ceil(log3(num_balls*2))), no extra
  penalty is applied beyond the per-weighing cost.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

# --- Configurable constants ---
ALPHA = 0.15          # per-weighing cost
REPEAT_PENALTY = 0.3   # penalty per repeated weighing
INVALID_PENALTY = 0.2  # penalty per invalid weighing attempt
NO_SUBMIT_PENALTY = -1.0


def reward_fn(record: Any) -> float:
    """Score one completion using information-gain-aware reward."""

    source = record.source_record
    ball_set = game.BallSet(
        num_balls=source["num_balls"],
        odd_ball_index=source["odd_ball_index"],
        direction=source["direction"],
        max_weighings=source["max_weighings"],
    )

    tool_calls = _iter_tool_calls(record)
    weigh_calls = [c for c in tool_calls if c.get("name") == "weigh"]
    answer = _extract_answer(record)

    # Initialise answer reward; set when answer is not None.
    k: float | None = None

    # --- Analyse weighings ---
    valid_weighings = 0
    repeated_weighings = 0
    invalid_weighings = 0
    seen_weighings: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    for call in weigh_calls:
        args = _parse_arguments(call)
        if args is None:
            invalid_weighings += 1
            continue
        left = args.get("left")
        right = args.get("right")
        if not isinstance(left, list) or not isinstance(right, list):
            invalid_weighings += 1
            continue
        # Check validity
        is_valid = _is_valid_weighing(left, right, ball_set.num_balls)
        if not is_valid:
            invalid_weighings += 1
            continue
        # Check for repetition (normalised: sorted tuple pair, order-independent)
        key = _weighing_key(left, right)
        if key in seen_weighings:
            repeated_weighings += 1
        else:
            seen_weighings.add(key)
            valid_weighings += 1

    total_weighings = valid_weighings + repeated_weighings + invalid_weighings
    base_reward = _info_theoretic_base(ball_set.num_balls)
    min_weighings = base_reward  # ceil(log3(num_balls*2))

    # --- Compute reward ---
    if answer is None:
        reward = NO_SUBMIT_PENALTY
        full_acc = 0.0
        identity_acc = 0.0
    else:
        result = game.check_answer(ball_set, answer["ball_index"], answer["direction"])
        full_acc = 1.0 if result["full_correct"] else 0.0
        identity_acc = 1.0 if result["ball_correct"] else 0.0

        if result["full_correct"]:
            k = float(base_reward)
        elif result["ball_correct"]:
            k = float(base_reward) / 2.0
        else:
            k = 0.0

        # Weighing cost: only valid + repeated count toward T
        # (invalid weighings are penalised separately)
        t_cost = (valid_weighings + repeated_weighings) * ALPHA
        repeat_cost = repeated_weighings * REPEAT_PENALTY
        invalid_cost = invalid_weighings * INVALID_PENALTY

        reward = k - t_cost - repeat_cost - invalid_cost

    # --- Populate metadata ---
    metadata = getattr(record, "metadata", None)
    if metadata is None:
        metadata = {}
    metadata["weighings_used"] = total_weighings
    metadata["valid_weighings"] = valid_weighings
    metadata["repeated_weighings"] = repeated_weighings
    metadata["invalid_weighings"] = invalid_weighings
    metadata["max_weighings"] = ball_set.max_weighings
    metadata["min_weighings"] = min_weighings
    metadata["base_reward"] = base_reward
    metadata["full_answer_accuracy"] = full_acc
    metadata["identity_only_accuracy"] = identity_acc
    metadata["reward_components"] = {
        "k": k,
        "t_cost": (valid_weighings + repeated_weighings) * ALPHA if answer is not None else 0.0,
        "repeat_cost": repeated_weighings * REPEAT_PENALTY,
        "invalid_cost": invalid_weighings * INVALID_PENALTY,
    }
    _set_metadata(record, metadata)

    return reward


def _info_theoretic_base(num_balls: int) -> int:
    """Information-theoretic lower bound on weighings needed.

    Returns ceil(log3(num_balls * 2)) — the minimum number of ternary
    outcomes (left_heavy / right_heavy / balanced) to distinguish
    num_balls * 2 possibilities (each ball can be heavier or lighter).
    """

    if num_balls <= 1:
        return 1
    import math
    return max(1, math.ceil(math.log(num_balls * 2, 3)))


def _is_valid_weighing(left: list, right: list, num_balls: int) -> bool:
    """Check if a weighing is valid (equal size, disjoint, in range)."""

    if not left or not right:
        return False
    if len(left) != len(right):
        return False
    left_set = set(left)
    right_set = set(right)
    if left_set & right_set:
        return False
    for ball in left_set | right_set:
        if not isinstance(ball, int) or isinstance(ball, bool):
            return False
        if not (0 <= ball < num_balls):
            return False
    return True


def _weighing_key(left: list[int], right: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Normalised key for a weighing, order-independent (left/right swap = same)."""

    left_sorted = tuple(sorted(left))
    right_sorted = tuple(sorted(right))
    return (left_sorted, right_sorted) if left_sorted <= right_sorted else (right_sorted, left_sorted)


def _parse_arguments(call: dict) -> dict | None:
    """Parse tool call arguments, whether stored as str or dict."""

    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if isinstance(arguments, dict):
        return arguments
    return None


def _extract_answer(record: Any) -> dict[str, Any] | None:
    """Extract the submit_answer arguments from tool calls."""

    for call in _iter_tool_calls(record):
        if call.get("name") != "submit_answer":
            continue
        args = _parse_arguments(call)
        if args is None:
            return None
        ball_index = args.get("ball_index")
        direction = args.get("direction")
        try:
            return {"ball_index": int(ball_index), "direction": str(direction)}
        except (TypeError, ValueError):
            return None
    return None


def _iter_tool_calls(record: Any) -> list[dict[str, Any]]:
    """Return the list of tool call dicts from a reward record."""

    tool_calls = getattr(record, "tool_calls", None)
    if tool_calls is None:
        return []
    return tool_calls if isinstance(tool_calls, list) else []


def _set_metadata(record: Any, metadata: dict[str, Any]) -> None:
    """Best-effort write of metadata back onto the record."""

    if hasattr(record, "metadata"):
        if isinstance(record.metadata, dict):
            record.metadata.update(metadata)
        else:
            record.metadata = metadata
