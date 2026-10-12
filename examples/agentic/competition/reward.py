"""Reward function for the competition agentic example.

Three-dimensional scoring: user_score * 0.5 + self_score * 0.2 + peer_score * 0.3
Plus sandwich structure bonus, self-eval calibration penalty, and compute gain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    check_sandwich_structure,
    simulate_user_score,
    load_profile,
)


def reward_fn(record) -> float:
    """Compute reward from the agent's tool-call trajectory."""

    source = dict(record.source_record)
    tool_calls = list(record.tool_calls)

    # Extract generated content
    content = _extract_content(tool_calls)
    if not content.strip():
        return -1.0

    # Extract self and peer scores
    self_score = _extract_score(tool_calls, "self_score")
    peer_score = _received_peer_score(record)

    # Simulate user score
    diary = source.get("diary", "")
    profile = source.get("user_profile") or load_profile()
    user_score = simulate_user_score(content, diary, profile)

    # Three-dimensional weighted score
    base_score = user_score * 0.5 + self_score * 0.2 + peer_score * 0.3

    # Sandwich structure bonus
    structure_score = check_sandwich_structure(content)
    structure_bonus = structure_score * 0.2

    # Self-eval calibration penalty
    self_penalty = abs(self_score - base_score) * 0.3

    # Peer-eval calibration penalty (only penalize lowballing)
    peer_penalty = max(0, base_score - peer_score) * 0.2

    compute_gain = _compute_gain(record)

    total = base_score + structure_bonus - self_penalty - peer_penalty + compute_gain
    return total


def _extract_content(tool_calls: list) -> str:
    """Extract the generated content from generate_content tool calls."""
    for call in tool_calls:
        if call.get("name") != "generate_content":
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return ""
        if isinstance(args, dict):
            return str(args.get("content", ""))
    return ""


def _extract_score(tool_calls: list, tool_name: str) -> float:
    """Extract a score (0-1) from self_score or peer_score tool calls."""
    for call in tool_calls:
        if call.get("name") != tool_name:
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return 0.5
        if isinstance(args, dict):
            score = args.get("score", 0.5)
            try:
                return max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                return 0.5
    return 0.5


def _received_peer_score(record) -> float:
    source = dict(record.source_record)
    result = source.get("_competition_result") or {}
    scores = result.get("peer_scores_received") or {}
    sample_index = _sample_index(record)
    if sample_index is not None:
        score = scores.get(str(sample_index), scores.get(sample_index))
        if score is not None:
            try:
                return max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                return 0.5
    return _extract_score(list(record.tool_calls), "peer_score")


def _compute_gain(record) -> float:
    source = dict(record.source_record)
    result = source.get("_competition_result") or {}
    gains = result.get("compute_gains") or {}
    sample_index = _sample_index(record)
    if sample_index is None:
        return 0.0
    gain = gains.get(str(sample_index), gains.get(sample_index, 0.0))
    try:
        return float(gain)
    except (TypeError, ValueError):
        return 0.0


def _sample_index(record) -> int | None:
    metadata = getattr(record, "metadata", {}) or {}
    if not isinstance(metadata, dict) or "sample_index" not in metadata:
        return None
    try:
        return int(metadata["sample_index"])
    except (TypeError, ValueError):
        return None
