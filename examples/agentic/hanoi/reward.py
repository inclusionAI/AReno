"""Reward function for the Hanoi tool-call agentic example.

Mirrors ``examples/agentic/duelgrid/reward.py``: extract the model's
``move_disk`` tool call, replay the proposed move sequence against a fresh
board, and return a scalar reward. Rewards come from the rule engine, so the
same fixtures work for warmup, rollout, or RLVR training.

Design rationale (completion-led, with hybrid PROGRESS + tiny legal-floor partial credit):
    The issue asks to "score completion with a small efficiency component
    relative to the known optimum". Read strictly, that means 0 unless the
    board is solved. Read leniently — which this file adopts — completion is
    still the dominant signal, but a small partial credit is given for unsolved
    traces to keep GSPO gradients alive during cold start.

    The partial credit is a hybrid: PROGRESS-based (disks correctly stacked on
    peg 2 from the bottom, 0.02/disk, cap 0.5) as the main signal, plus a
    tiny legal-move floor (0.005/legal move, cap 0.02) to ensure at least a
    small non-zero gradient even when no progress has been made. The floor is
    deliberately too small to be worth freezing on — the collapse shortcut
    [[0,2],[0,1]] scores only 0.01 from the floor, making progress (which
    pays 0.02 for the first progress disk) strictly more rewarding and
    completion (1.0, globally optimal) the dominant long-term objective.

    Earlier iterations tried a pure legal-step bonus (collapsed: model locked
    0.04 and froze) and pure progress (too sparse: 0.8B could not produce
    >4 legal moves consistently, reward stayed 0). The hybrid sits between
    both extremes and has been validated in a 100-step Kaggle run.

    Concretely:
    - solved:    ``COMPLETION_REWARD - EXCESS_STEP_PENALTY * excess``
                 (1.0 minus 0.02 per move above the oracle ``2**n-1``).
                 Completion dominates; efficiency is the "small component".
    - unsolved:  ``min(PROGRESS_BONUS * progress_count, PARTIAL_CREDIT_CAP)``
                 (≈ +0.02 per disk correctly stacked on peg 2 from the bottom,
                 hard-capped at 0.5). The cap keeps any unsolved trace strictly
                 below COMPLETION_REWARD, so completing is always the dominant
                 choice; progress credit only gives variance + a direction.

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

# Partial credit for UNSOLVED traces is a hybrid of PROGRESS (main) and a
# tiny legal-move floor (gradient survival), see module docstring.
PROGRESS_BONUS = 0.02  # per disk correctly stacked on peg 2 from the bottom
LEGAL_FLOOR_BONUS = 0.005  # per legal move — tiny floor to keep gradient alive
LEGAL_FLOOR_CAP = 0.005  # floor cap: 1 legal move caps it; too small to steal, just keeps gradient alive
PARTIAL_CREDIT_CAP = 0.5  # global cap: completion (>=1.0) always dominates


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
    # Unsolved: small PROGRESS-based partial credit (see module docstring).
    # Only disks correctly stacked on peg 2 from the bottom count, so a couple
    # of legal-but-stagnant moves score 0 and cannot be "stolen" for a stable
    # reward. This keeps the gradient pointing at completion while still giving
    # GSPO non-zero reward variance across samples that reach different depths.
    # Unsolved: hybrid partial credit — PROGRESS (main) plus a tiny legal-move
    # floor to keep gradients alive during cold start (see module docstring).
    progress = PROGRESS_BONUS * _progress_count(result)
    floor = min(LEGAL_FLOOR_BONUS * result.legal_count, LEGAL_FLOOR_CAP)
    return min(max(progress, floor), PARTIAL_CREDIT_CAP)


def _progress_count(result: Any) -> int:
    """Disks correctly stacked on peg 2 from the bottom (0 if none match).

    The target peg is ``game.TARGET_PEG`` and the correct bottom-up order is
    ``(n, n-1, ..., 1)``. Only a contiguous correct prefix counts, so a disk in
    the right slot above a wrong one does not score — pushing the gradient to
    *build* the target stack in order rather than park any disk on peg 2.
    """

    peg2 = result.final_state.pegs[game.TARGET_PEG]
    n = result.final_state.n
    count = 0
    for i, disk in enumerate(peg2):
        if disk == n - i:
            count += 1
        else:
            break
    return count


def _tool_moves(record: Any) -> list[tuple[int, int]]:
    """Extract a consolidated move list from all ``move_disk`` tool calls.

    In multi-turn mode each call carries a single ``{source, target}`` move;
    all calls across all turns are accumulated into one sequential move list
    that ``replay`` consumes.  For backward compatibility the old single-turn
    ``{"moves": [[s, t], ...]}`` form is still accepted.  Only a complete
    absence of tool calls triggers a fallback to completion-text parsing.
    """

    all_moves: list[tuple[int, int]] = []
    for call in getattr(record, "tool_calls", []) or []:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "move_disk":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue  # skip malformed, don't abort the whole trajectory
        if isinstance(arguments, dict):
            # Multi-turn: {"source": s, "target": t}
            src = arguments.get("source")
            tgt = arguments.get("target")
            if src is not None and tgt is not None:
                pair = game._normalize_action((src, tgt))
                if pair is not None:
                    all_moves.append(pair)
                    continue
            # Legacy single-turn: {"moves": [[s, t], ...]} or {"moves": {s, t}}
            moves_field = arguments.get("moves")
            if isinstance(moves_field, dict):
                all_moves.extend(_coerce_moves([moves_field]))
                continue
            if moves_field is not None:
                all_moves.extend(_coerce_moves(moves_field))
                continue
        if isinstance(arguments, list):
            all_moves.extend(_coerce_moves(arguments))
    if all_moves:
        return all_moves
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
