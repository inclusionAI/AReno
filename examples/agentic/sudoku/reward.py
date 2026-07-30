"""Reward function for the Sudoku agentic example.

Grading is purely structural: a solve is detected from the visible board state
revealed through tool results — never from a stored solution.

Reward design (curriculum path, gated on ``SUDOKU_CURRICULUM`` env var,
default "on"):

    solved                     -> SOLVED_REWARD[difficulty]      (e.g. tutorial 0.8)
    legal progress, unsolved   -> SOLVED_REWARD * 0.15 * sqrt(filled/empty), capped
    tried place, all illegal   -> ATTEMPT_PENALTY      (-0.05)
    only inspected, no place   -> INSPECT_ONLY_PENALTY (-0.08)
    did nothing at all         -> NOISE_PENALTY        (-0.1)

Three deliberate trade-offs:

1. **Anti-deadlock (effort tiers).** With a purely sparse reward, when every
   sample in a rollout group fails to make any legal placement, all rewards
   collapse to the same value, group advantages become 0, gradients vanish,
   and RL stalls. Grading *effort* below the legal-progress tier guarantees
   within-group spread as long as samples differ in how hard they tried, so
   advantages (and gradients) stay nonzero. It never rewards illegal placement
   above legal progress.

2. **Slow progress curve (sqrt + low cap).** Progress reward grows as the
   square root of fill ratio and is capped at 0.15 * SOLVED_REWARD, far below
   the solved reward. This deliberately makes "greedily stuffing legal digits
   to farm progress" unattractive: the most progress can ever pay is ~0.12
   (tutorial), while a solve pays 0.8. The policy's main incentive stays
   "actually finish", not "fill cells".

3. **Invalid-action penalty.** A greedy stuffer produces many rejected
   placements (digit conflicts). We subtract ``INVALID_PENALTY_RATIO *
   (invalid_actions / total_actions)`` from any non-solved reward, so
   "thoughtless placing" is taxed while careful, low-invalid play is not.

Flat (legacy) behavior with ``SUDOKU_CURRICULUM=off``: solved=1.0, legal
progress=0.0, noise=-0.1.

Per-difficulty ``solve_rate`` and ``invalid_action_rate`` are derivable from
the same ``place_digit`` tool results grouped by ``record.source_record["difficulty"]``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sudoku  # noqa: E402

# --- reward weights ---------------------------------------------------------

# Solved-reward weight per difficulty (curriculum). Higher bands pay more.
SOLVED_REWARD: dict[str, float] = {
    "tutorial": 0.8,
    "easy": 1.0,
    "medium": 1.2,
    "hard": 1.5,
    "extreme": 2.0,
}

# Progress shaping: a legal-but-unsolved episode earns a small fraction of the
# solved weight, growing with the square root of the fill ratio. The sqrt curve
# is deliberately *sub-linear* so early fills pay relatively more (keeps a weak
# policy moving) but late fills pay relatively less (no incentive to stuff cells
# just to raise the fill ratio). The cap keeps progress far below solved.
PROGRESS_FRACTION = 0.15
PROGRESS_CAP: dict[str, float] = {
    band: weight * PROGRESS_FRACTION for band, weight in SOLVED_REWARD.items()
}

# Anti-deadlock effort tiers (see module docstring, trade-off 1).
ATTEMPT_PENALTY = -0.05       # tried place_digit but every attempt was illegal
INSPECT_ONLY_PENALTY = -0.08  # only inspected, never attempted a placement
NOISE_PENALTY = -0.1          # produced no tool call worth grading

# Tax on thoughtless placing (see module docstring, trade-off 3). Subtracted
# from any non-solved reward, proportional to the episode's invalid-action rate.
INVALID_PENALTY_RATIO = 0.1


# --- episode feature extraction --------------------------------------------


class EpisodeFeatures:
    """Derived signals from one rollout episode, used to pick its reward tier.

    A plain class (not ``@dataclass``) on purpose: AReno loads this file via
    ``importlib`` dynamic module exec, which does not register the module in
    ``sys.modules`` — and ``@dataclass`` needs that registration to resolve
    type annotations, so it would raise ``AttributeError`` at import time.
    """

    __slots__ = (
        "solved", "legal_placements", "total_place_attempts", "invalid_actions",
        "tried_place", "inspect_count", "empty_cells", "difficulty",
    )

    def __init__(
        self,
        *,
        solved: bool,
        legal_placements: int,
        total_place_attempts: int,
        invalid_actions: int,
        tried_place: bool,
        inspect_count: int,
        empty_cells: int,
        difficulty: str,
    ) -> None:
        self.solved = solved
        self.legal_placements = legal_placements            # place_digit calls that succeeded
        self.total_place_attempts = total_place_attempts    # place_digit calls (legal + illegal)
        self.invalid_actions = invalid_actions              # place_digit calls rejected by the env
        self.tried_place = tried_place                      # any place_digit was attempted at all
        self.inspect_count = inspect_count                  # inspect_candidates calls made
        self.empty_cells = empty_cells                      # board empties at episode start
        self.difficulty = difficulty


def _resolved_empty_cells(source: Any) -> int:
    """Empty-cell count for fill-ratio shaping, robust to records that omit it.

    Prefer the loader-computed ``empty_cells``; otherwise recount from the
    stored ``puzzle``. Always >= 1 so ``fill_ratio`` cannot divide by zero or
    explode if some future record path forgets both fields.
    """

    cached = int(source.get("empty_cells", 0)) if isinstance(source, dict) else 0
    if cached > 0:
        return cached
    puzzle = source.get("puzzle") if isinstance(source, dict) else None
    if puzzle:
        try:
            return max(1, sum(1 for row in puzzle for v in row if not v))
        except TypeError:
            return 1
    return 1


def _extract_features(record: Any) -> EpisodeFeatures:
    """Read tool calls/results from the rollout record into structured features."""

    source = record.source_record
    place_results = _place_results(record)
    legal = sum(1 for r in place_results if r.get("placed"))
    invalid = sum(1 for r in place_results if r.get("invalid_action"))

    return EpisodeFeatures(
        solved=any(bool(r.get("solved")) for r in place_results),
        legal_placements=legal,
        total_place_attempts=len(place_results),
        invalid_actions=invalid,
        tried_place=bool(place_results),
        inspect_count=_inspect_count(record),
        empty_cells=_resolved_empty_cells(source),
        difficulty=str(source.get("difficulty", sudoku.DEFAULT_DIFFICULTY)).lower(),
    )


# --- reward tiers -----------------------------------------------------------


def _solved_reward(features: EpisodeFeatures) -> float:
    return SOLVED_REWARD.get(features.difficulty, 1.0)


def _progress_reward(features: EpisodeFeatures) -> float:
    """Legal-but-unsolved: sqrt(fill) * weight, capped; minus invalid tax.

    ``fill_ratio`` is how much of the board the policy legally filled. The cap
    and the invalid tax together make "stuff cells to farm progress" pay poorly
    while still giving a weak policy a learnable gradient for partial progress.
    """

    weight = SOLVED_REWARD.get(features.difficulty, 1.0)
    fill_ratio = features.legal_placements / features.empty_cells
    base = weight * PROGRESS_FRACTION * math.sqrt(fill_ratio)
    capped = min(PROGRESS_CAP.get(features.difficulty, weight * PROGRESS_FRACTION), base)
    return capped - _invalid_penalty(features)


def _invalid_penalty(features: EpisodeFeatures) -> float:
    """Fraction of place attempts that were illegal, scaled. 0 if none tried."""

    if features.total_place_attempts == 0:
        return 0.0
    return INVALID_PENALTY_RATIO * (features.invalid_actions / features.total_place_attempts)


def _effort_reward(features: EpisodeFeatures) -> float:
    """No legal placement: grade effort to avoid a zero-advantage deadlock.

    Tried-to-place (all illegal) ranks above only-inspected, which ranks above
    pure noise. This keeps within-group reward spread (and thus gradients) alive
    even when every sample fails to place legally.
    """

    if features.tried_place:
        base = ATTEMPT_PENALTY
    elif features.inspect_count:
        base = INSPECT_ONLY_PENALTY
    else:
        base = NOISE_PENALTY
    return base - _invalid_penalty(features)


# --- public entry point -----------------------------------------------------


def reward_fn(record: Any) -> float:
    """Score one episode. Tier order: solved > legal-progress > effort > noise."""

    features = _extract_features(record)

    if not _curriculum_enabled():
        # Flat legacy behavior.
        if features.solved:
            return 1.0
        if features.legal_placements:
            return 0.0
        if features.tried_place:
            return ATTEMPT_PENALTY
        return NOISE_PENALTY

    if features.solved:
        return _solved_reward(features)
    if features.legal_placements:
        return _progress_reward(features)
    return _effort_reward(features)


def _curriculum_enabled() -> bool:
    return os.environ.get("SUDOKU_CURRICULUM", "on").lower() not in ("off", "0", "false", "no")


# --- tool-result decoding helpers ------------------------------------------


def _place_results(record: Any) -> list[dict[str, Any]]:
    """Decode the JSON content of every ``place_digit`` tool result."""

    results: list[dict[str, Any]] = []
    for call, content in zip(record.tool_calls, record.tool_results, strict=False):
        name = call.get("name") if isinstance(call, dict) else None
        if name != "place_digit":
            continue
        results.append(_decode(content.get("content") if isinstance(content, dict) else content))
    return results


def _inspect_count(record: Any) -> int:
    """How many inspect_candidates calls the episode made (effort signal)."""

    count = 0
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name == "inspect_candidates":
            count += 1
    return count


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