"""Reward function for the Sudoku agentic example.

Grading is purely structural: a solve is detected from the visible board state
revealed through tool results — never from a stored solution. The function
walks the ``place_digit`` tool results and scores one episode.

Curriculum-aware reward (gated on ``SUDOKU_CURRICULUM`` env var, default "on"):
- Solved reward scales with difficulty so harder boards pay more, pulling the
  policy toward harder bands as its ability grows:
    easy +1.0, medium +1.2, hard +1.5, extreme +2.0
- Un-solved but with legal progress: a small shaped reward proportional to
  legal placements made (0.1 * (legal_placements / empty_cells)), capped per
  band. This keeps hard/extreme from being pure zero-signal dead zones without
  ever reading the solution — progress is measured solely from visible legal
  placements, so the no-leak invariant is preserved.
- No legal placement at all: -0.1 (pure noise penalty).

Flat (legacy) behavior with ``SUDOKU_CURRICULUM=off``: solved=1.0, legal
progress=0.0, noise=-0.1.

Per-difficulty ``solve_rate`` and ``invalid_action_rate`` are derivable from
the same tool results grouped by ``record.source_record["difficulty"]``; wire
those into AReno's metric fields in the trainer config.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sudoku  # noqa: E402

# Solved-reward weight per difficulty (curriculum). Higher bands pay more.
SOLVED_REWARD: dict[str, float] = {
    "easy": 1.0,
    "medium": 1.2,
    "hard": 1.5,
    "extreme": 2.0,
}
# Progress-shaping scale per difficulty (cap on the shaped component). Kept
# small so solving always dominates merely-progressing.
PROGRESS_CAP: dict[str, float] = {
    "easy": 0.05,
    "medium": 0.15,
    "hard": 0.25,
    "extreme": 0.35,
}
NOISE_PENALTY = -0.1


def _curriculum_enabled() -> bool:
    return os.environ.get("SUDOKU_CURRICULUM", "on").lower() not in ("off", "0", "false", "no")


def reward_fn(record: Any) -> float:
    """Score one episode by replaying its place_digit tool results."""

    source = record.source_record
    difficulty = str(source.get("difficulty", sudoku.DEFAULT_DIFFICULTY)).lower()
    empty_cells = int(source.get("empty_cells", 0)) or 1

    place_results = _place_results(record)
    if not place_results:
        return NOISE_PENALTY

    solved = any(bool(result.get("solved")) for result in place_results)
    legal_placements = sum(1 for result in place_results if result.get("placed"))

    if not _curriculum_enabled():
        if solved:
            return 1.0
        if legal_placements:
            return 0.0
        return NOISE_PENALTY

    # Curriculum path.
    if solved:
        return SOLVED_REWARD.get(difficulty, 1.0)
    if legal_placements:
        progress = 0.1 * (legal_placements / empty_cells)
        cap = PROGRESS_CAP.get(difficulty, 0.1)
        return min(cap, progress)
    return NOISE_PENALTY


def _place_results(record: Any) -> list[dict[str, Any]]:
    """Decode the JSON content of every ``place_digit`` tool result."""

    results: list[dict[str, Any]] = []
    for call, content in zip(record.tool_calls, record.tool_results, strict=False):
        name = call.get("name") if isinstance(call, dict) else None
        if name != "place_digit":
            continue
        results.append(_decode(content.get("content") if isinstance(content, dict) else content))
    return results


def _decode(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}