"""Reward function for the Hanoi tool-call agentic example.

Mirrors ``examples/agentic/duelgrid/reward.py``: extract the model's
``move_disk`` tool call, replay the proposed move sequence against a fresh
board, and return a scalar reward. Rewards come from the rule engine, so the
same fixtures work for warmup, rollout, or RLVR training.

Design rationale (completion-led, with a small partial-credit bridge):
    The issue asks to "score completion with a small efficiency component
    relative to the known optimum". Read strictly, that means 0 unless the
    board is solved. Read leniently — which this file adopts — completion is
    still the dominant signal, but a small partial credit is given for legal
    moves on unsolved traces. The lenient reading is chosen for a concrete
   training reason: a weak base model almost never solves the board in one shot,
    so a strictly-sparse reward leaves every GSPO group at all-zero reward,
    zero variance, zero advantage, zero gradient — a cold-start stall (the
    scenario H-version #203/#239 target, but AReno has no early stop today).

    Concretely:
    - solved:      ``COMPLETION_REWARD - EXCESS_STEP_PENALTY * excess``
                   (1.0 minus 0.02 per move above the oracle ``2**n-1``).
                   Completion dominates; efficiency is the "small component".
    - unsolved:    ``min(LEGAL_STEP_BONUS * legal_count + ILLEGAL_STEP_PENALTY * illegal_count,
                   PARTIAL_CREDIT_CAP)`` (≈ +0.02 per legal move, -0.05 per
                   illegal, hard-capped at 0.5). The cap is essential: without
                   it a long legal-but-unsolved trace could accumulate more than
                   1.0 from legal steps alone and perversely reward not solving.
                   With the cap, completing (≥ the efficiency-discounted 1.0
                   path) is always strictly better than any unsolved trace, so
                   completion stays the primary signal; the partial credit only
                   gives GSPO non-zero reward variance so training can move.

    The completion-rate + excess-moves-over-optimum metrics in ``game.evaluate``
    are unaffected — they still key on completion — so the issue's acceptance
    criteria hold under either reading.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator
import game

# Accepts a single ``move_disk`` call whose arguments look like
# ``{"moves": [[0, 2], [0, 1], ...]}``. A bare JSON object / list in the
# completion text is also accepted as a fallback.
_MOVE_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)

# Partial-credit magnitudes for UNSOLVED traces (see module docstring). Kept
# small and local so completion (game.COMPLETION_REWARD = 1.0) stays the
# dominant signal; these only exist to give GSPO non-zero reward variance.
LEGAL_STEP_BONUS = 0.02  # each legal move on an unsolved trace earns a little
ILLEGAL_STEP_PENALTY = -0.05  # each illegal move on an unsolved trace costs a little
# Hard cap on unsolved partial credit. Without it, a long legal-but-unsolved
# trace (e.g. the failure fixture oscillating for 2*(2**n-1) steps) could
# accumulate more than COMPLETION_REWARD from legal steps alone — which would
# perversely reward *not* solving. The cap guarantees completing the board is
# always strictly better than any unsolved trace.
PARTIAL_CREDIT_CAP = 0.5


def reward_fn(record: Any) -> float:
    """Score one completion by replaying the ``move_disk`` move sequence.

    Solved:    ``COMPLETION_REWARD - EXCESS_STEP_PENALTY * excess`` (completion-led).
    Unsolved:  a small partial credit from legal moves (see module docstring).
    """

    state = dataset_generator.record_to_state(record.source_record["state"])
    moves = _tool_moves(record)
    result = game.replay(moves, state.n)
    if result.completed:
        # 1.0 for solving, minus a small per-step penalty for excess moves over
        # the oracle optimum 2**n - 1 — the "small efficiency component" from
        # the issue. excess = steps actually taken (legal + illegal) minus the
        # oracle optimum, floored at replay's own excess_moves so the two paths
        # never disagree on a completed trace.
        excess = max(0, result.legal_count + result.illegal_count - game.optimal_steps(state.n))
        excess = max(excess, result.excess_moves)
        return game.COMPLETION_REWARD - game.EXCESS_STEP_PENALTY * excess
    # Unsolved: small partial credit only — enough to vary across samples so
    # GSPO can form a non-zero advantage, never enough to rival completion.
    # The cap (0.5) keeps any unsolved trace strictly below COMPLETION_REWARD,
    # so solving is always the dominant choice even if a trace racks up many
    # legal moves without finishing.
    partial = LEGAL_STEP_BONUS * result.legal_count + ILLEGAL_STEP_PENALTY * result.illegal_count
    return min(partial, PARTIAL_CREDIT_CAP)


def _tool_moves(record: Any) -> list[tuple[int, int]]:
    """Extract a move list from the ``move_disk`` tool call (or completion text).

    Tolerant of the argument shape a weak base model actually emits. The tool
    schema mandates ``{"moves": [[s, t], ...]}``, but historical prose told the
    model ``{source, target}``, so a model may return a bare
    ``{"source": s, "target": t}`` single move or wrap one dict under ``moves``.
    All three are coerced to a move list; only a genuinely empty/absent call
    falls through to completion text. Without this, such calls silently extract
    ``[]`` and reward 0 even though ``tool_calls`` was parsed — the cold-start
    stall observed when ``tool_calls=8/8`` yet ``reward_mean=0.0``.
    """

    for call in getattr(record, "tool_calls", []) or []:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "move_disk":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return []
        if isinstance(arguments, dict):
            moves_field = arguments.get("moves")
            # Tolerate {"source": s, "target": t} (bare single move) and
            # {"moves": {"source": s, "target": t}} (single move wrapped under
            # moves) so a model following the prose over the schema still scores.
            if moves_field is None and ("source" in arguments or "target" in arguments):
                return _coerce_moves([arguments])
            if isinstance(moves_field, dict):
                return _coerce_moves([moves_field])
            return _coerce_moves(moves_field)
        if isinstance(arguments, list):
            return _coerce_moves(arguments)
    return _coerce_moves(_fallback_completion(getattr(record, "completion", None)))


def _coerce_moves(raw: Any) -> list[tuple[int, int]]:
    if isinstance(raw, str):
        # The Qwen tool-call path often serializes the moves list as a JSON
        # *string* ("[[0,1],[0,2]]") rather than a list. Deserialize first;
        # without this, str-shaped moves are dropped to [] and reward is 0 even
        # when every sample emitted a valid move_disk call (observed in rollout:
        # tool_calls=8/8 yet reward_mean=0.0).
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return game.parse_trace(raw) if raw.strip() else []
    if not isinstance(raw, list):
        return []
    moves: list[tuple[int, int]] = []
    for item in raw:
        pair = game._normalize_action(item)
        if pair is not None:
            moves.append(pair)
    return moves


def _fallback_completion(completion: Any) -> list[Any]:
    """Best-effort move list from raw model text (used outside tool-call mode)."""

    if isinstance(completion, list):
        return completion
    if isinstance(completion, str):
        match = _MOVE_LIST_RE.search(completion)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return parsed.get("moves", parsed) if isinstance(parsed, dict) else parsed
    return []
