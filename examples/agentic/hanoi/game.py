"""Towers of Hanoi rules engine for an agentic RL demo.

This module is intentionally small and self-contained: it has no AReno
dependency and can be exercised on CPU only. The public surface mirrors the
 DuelGrid example so the same generator / reward / agent files can wrap it:

- ``HanoiEnv``               — three-peg state, ``move(source, target)``, legality,
                               reward, terminal, configurable illegal-action policy.
- ``optimal_steps`` /        — oracle ground truth ``2**n - 1`` plus the recursive
  ``optimal_solution``         shortest action sequence and a validator.
- ``replay`` /               — deterministic text-trace replay used to turn an
  ``serialize_trace``           "is the policy good?" question into a fixed-trace
                               assertion that needs no training run.

n is restricted to 3..6 per the issue. All state stays local.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

NUM_PEGS = 3
MIN_DISKS = 3
MAX_DISKS = 6
SOURCE_PEG = 0
TARGET_PEG = 2

# Reward shaping constants. Kept small so the completion signal dominates.
COMPLETION_REWARD = 1.0
EXCESS_STEP_PENALTY = 0.02
ILLEGAL_PENALTY = -0.1

IllegalActionPolicy = Literal["penalize", "terminate"]

_TRACE_MOVE_RE = re.compile(r"(\d+)\s*->\s*(\d+)")


@dataclass(frozen=True)
class HanoiState:
    """Immutable Towers of Hanoi state.

    ``pegs[i]`` is a tuple listing disks on peg ``i`` bottom-first, so the last
    element is the top disk. Larger numbers mean larger disks.
    """

    pegs: tuple[tuple[int, ...], ...]
    n: int
    moves: int = 0
    max_moves: int = 256
    done: bool = False

    def top(self, peg: int) -> int | None:
        stack = self.pegs[peg]
        return stack[-1] if stack else None


def make_state(n: int, *, max_moves: int | None = None) -> HanoiState:
    """Build the canonical start state for ``n`` disks (3..6)."""

    n = int(n)
    if not MIN_DISKS <= n <= MAX_DISKS:
        raise ValueError(f"n must be in [{MIN_DISKS},{MAX_DISKS}], got {n}")
    if max_moves is None:
        # A generous ceiling well above the optimum 2**n-1 but bounded so a
        # wandering policy still terminates.
        max_moves = max(64, (2**n) * 4)
    pegs = (tuple(range(n, 0, -1)), (), ())
    return HanoiState(pegs=pegs, n=n, max_moves=max_moves)


def is_terminal(state: HanoiState) -> bool:
    """A state is terminal when all disks sit on the target peg, or the
    move budget is exhausted."""

    if state.done:
        return True
    if state.moves >= state.max_moves:
        return True
    return _is_completed(state)


def _is_completed(state: HanoiState) -> bool:
    target = state.pegs[TARGET_PEG]
    return len(target) == state.n and tuple(target) == tuple(range(state.n, 0, -1))


def legal_moves(state: HanoiState) -> list[tuple[int, int]]:
    """Return all legal ``(source, target)`` moves for the current state."""

    moves: list[tuple[int, int]] = []
    for src in range(NUM_PEGS):
        top = state.top(src)
        if top is None:
            continue
        for dst in range(NUM_PEGS):
            if dst == src:
                continue
            other = state.top(dst)
            if other is None or top < other:
                moves.append((src, dst))
    return moves


def is_legal(state: HanoiState, source: int, target: int) -> bool:
    """Check a single move's legality without applying it."""

    if not (0 <= source < NUM_PEGS and 0 <= target < NUM_PEGS):
        return False
    if source == target:
        return False
    top = state.top(source)
    if top is None:
        return False
    other = state.top(target)
    return other is None or top < other


def _normalize_action(action: Any) -> tuple[int, int] | None:
    """Accept ``move(s, t)``/``{"source","target"}``/``"s->t"`` forms."""

    if action is None:
        return None
    if isinstance(action, (tuple, list)) and len(action) == 2:
        return _coerce_pair(action[0], action[1])
    if isinstance(action, dict):
        return _coerce_pair(action.get("source"), action.get("target"))
    if isinstance(action, str):
        match = _TRACE_MOVE_RE.search(action)
        if not match:
            return None
        return _coerce_pair(match.group(1), match.group(2))
    return None


def _coerce_pair(src: Any, dst: Any) -> tuple[int, int] | None:
    try:
        return int(src), int(dst)
    except (TypeError, ValueError):
        return None


def step(
    state: HanoiState,
    action: Any,
    *,
    illegal_policy: IllegalActionPolicy = "penalize",
) -> tuple[HanoiState, float, bool, dict[str, Any]]:
    """Apply one move.

    Returns ``(next_state, reward, done, info)``. Illegal moves yield a small
    negative reward and (by default) leave the state unchanged so the agent can
    keep learning; with ``illegal_policy="terminate"`` they end the episode.

    Why ``penalize`` is the default: a single illegal move should not waste the
    whole episode — leaving the board unchanged and dishing a small penalty
    lets the policy keep exploring (matches DuelGrid's convention). ``terminate``
    is kept as an opt-in for users who want a stricter environment. This
    default is also the "safe" one required by the issue: existing behaviour is
    unchanged unless the caller explicitly opts into the stricter policy.
    """

    parsed = _normalize_action(action)
    if parsed is None:
        return _illegal(state, illegal_policy, reason="malformed", action=action)

    source, target = parsed
    if not (0 <= source < NUM_PEGS and 0 <= target < NUM_PEGS):
        return _illegal(state, illegal_policy, reason="out_of_range", action=parsed)
    if source == target:
        return _illegal(state, illegal_policy, reason="no_op", action=parsed)
    top = state.top(source)
    if top is None:
        return _illegal(state, illegal_policy, reason="empty_source", action=parsed)
    other = state.top(target)
    if other is not None and top > other:
        return _illegal(state, illegal_policy, reason="larger_on_smaller", action=parsed)

    pegs = [list(stack) for stack in state.pegs]
    disk = pegs[source].pop()
    pegs[target].append(disk)
    next_state = replace(
        state,
        pegs=tuple(tuple(stack) for stack in pegs),
        moves=state.moves + 1,
    )
    done = is_terminal(next_state)
    reward = _shaped_reward(next_state, done)
    return (
        next_state,
        reward,
        done,
        {
            "illegal": False,
            "move": parsed,
            "moves": next_state.moves,
            "completed": _is_completed(next_state),
        },
    )


def _illegal(
    state: HanoiState, policy: IllegalActionPolicy, *, reason: str, action: Any
) -> tuple[HanoiState, float, bool, dict[str, Any]]:
    info = {
        "illegal": True,
        "reason": reason,
        "action": action,
        "legal_moves": legal_moves(state),
    }
    if policy == "terminate":
        return replace(state, done=True), ILLEGAL_PENALTY, True, info
    return state, ILLEGAL_PENALTY, is_terminal(state), info


def _shaped_reward(state: HanoiState, done: bool) -> float:
    if done and _is_completed(state):
        # Completion dominates; subtract a small efficiency term so shorter
        # solutions score higher, relative to the oracle optimum.
        excess = max(0, state.moves - optimal_steps(state.n))
        return COMPLETION_REWARD - EXCESS_STEP_PENALTY * excess
    return 0.0


# --- Oracle -----------------------------------------------------------------


def optimal_steps(n: int) -> int:
    """Closed-form minimum number of moves for ``n`` disks: ``2**n - 1``."""

    n = int(n)
    if not MIN_DISKS <= n <= MAX_DISKS:
        raise ValueError(f"n must be in [{MIN_DISKS},{MAX_DISKS}], got {n}")
    return 2**n - 1


def optimal_solution(n: int, *, source: int = SOURCE_PEG, target: int = TARGET_PEG) -> list[tuple[int, int]]:
    """Yield one shortest solution as a list of ``(source, target)`` moves.

    The auxiliary peg is whichever peg is neither source nor target. The
    recursion is the classic three-step decomposition.
    """

    n = int(n)
    if not MIN_DISKS <= n <= MAX_DISKS:
        raise ValueError(f"n must be in [{MIN_DISKS},{MAX_DISKS}], got {n}")
    if source == target:
        return []
    aux = NUM_PEGS - source - target
    moves: list[tuple[int, int]] = []

    def _solve(k: int, frm: int, via: int, to: int) -> None:
        if k == 0:
            return
        _solve(k - 1, frm, to, via)
        moves.append((frm, to))
        _solve(k - 1, via, frm, to)

    _solve(n, source, aux, target)
    return moves


def validate_solution(moves: Iterable[tuple[int, int]], n: int) -> bool:
    """Return True iff ``moves`` is legal start-to-completed for ``n`` disks."""

    state = make_state(n)
    for mv in moves:
        parsed = _normalize_action(mv)
        if parsed is None or not is_legal(state, *parsed):
            return False
        state, _r, done, _info = step(state, parsed)
        if done and not _is_completed(state):
            return False
    return _is_completed(state) and is_terminal(state)


# --- Trace replay -----------------------------------------------------------


def serialize_trace(moves: Iterable[Any]) -> str:
    """Render a move list as a compact ``s->t,s->t`` text trace."""

    parts: list[str] = []
    for mv in moves:
        parsed = _normalize_action(mv)
        if parsed is None:
            parts.append("?->?")
        else:
            parts.append(f"{parsed[0]}->{parsed[1]}")
    return ",".join(parts)


def parse_trace(trace: str) -> list[tuple[int, int]]:
    """Parse a text trace into a list of integer ``(source, target)`` pairs."""

    pairs: list[tuple[int, int]] = []
    for match in _TRACE_MOVE_RE.finditer(trace):
        pairs.append((int(match.group(1)), int(match.group(2))))
    return pairs


@dataclass
class ReplayResult:
    """Structured outcome of replaying a trace against a fresh state."""

    steps: int
    legal_count: int
    illegal_count: int
    completed: bool
    excess_moves: int
    final_state: HanoiState
    events: list[dict[str, Any]] = field(default_factory=list)

    def as_text(self) -> str:
        status = "completed" if self.completed else "incomplete"
        lines = [
            f"replay: {self.steps} steps ({self.legal_count} legal, {self.illegal_count} illegal) -> {status}",
            f"excess_moves_over_optimum={self.excess_moves}",
        ]
        lines.extend(f"  step {i + 1}: {ev}" for i, ev in enumerate(self.events))
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "legal_count": self.legal_count,
            "illegal_count": self.illegal_count,
            "completed": self.completed,
            "excess_moves": self.excess_moves,
            "final_pegs": [list(stack) for stack in self.final_state.pegs],
        }


def replay(
    trace: str | Iterable[Any],
    n: int,
    *,
    illegal_policy: IllegalActionPolicy = "penalize",
) -> ReplayResult:
    """Drive a fresh ``n``-disk state with a trace (text or move list).

    With the default ``penalize`` policy, illegal moves are counted but do not
    end the replay, so a single trace can demonstrate both the successful path
    and a boundary/failure input — exactly the minimal example the issue asks
    for.
    """

    if isinstance(trace, str):
        moves = parse_trace(trace)
    else:
        moves = [parsed for mv in trace if (parsed := _normalize_action(mv)) is not None]

    state = make_state(n)
    events: list[dict[str, Any]] = []
    legal = illegal = 0
    done = False
    for mv in moves:
        if done:
            break
        next_state, reward, step_done, info = step(state, mv, illegal_policy=illegal_policy)
        events.append(
            {
                "move": mv,
                "illegal": info.get("illegal", False),
                "reason": info.get("reason"),
                "reward": reward,
                "pegs": [list(stack) for stack in next_state.pegs],
            }
        )
        if info.get("illegal"):
            illegal += 1
        else:
            legal += 1
        state = next_state
        done = step_done

    completed = _is_completed(state)
    # For completed traces, excess = steps beyond the oracle optimum (used by
    # the reward / evaluate). For incomplete traces there is no "optimum
    # excess" concept, so we just report steps taken as a diagnostic; it is NOT
    # consumed by reward_fn (which returns 0 for incomplete traces).
    excess = max(0, state.moves - optimal_steps(n)) if completed else max(0, state.moves)
    return ReplayResult(
        steps=legal + illegal,
        legal_count=legal,
        illegal_count=illegal,
        completed=completed,
        excess_moves=excess,
        final_state=state,
        events=events,
    )


# --- Evaluation -------------------------------------------------------------


def evaluate(
    traces: Iterable[str | Iterable[Any]],
    n: int,
    *,
    illegal_policy: IllegalActionPolicy = "penalize",
) -> dict[str, Any]:
    """Aggregate completion rate and excess moves over a set of traces.

    Returns a structured result suitable for CLI / metric output. Excess moves
    are reported relative to the oracle optimum ``2**n - 1`` and only counted
    for completed traces.
    """

    results = [replay(t, n, illegal_policy=illegal_policy) for t in traces]
    completed = [r for r in results if r.completed]
    total = len(results) or 1
    return {
        "n": n,
        "sample_count": len(results),
        "completion_rate": len(completed) / total,
        "avg_excess_moves": (sum(r.excess_moves for r in completed) / len(completed)) if completed else 0.0,
        "oracle_steps": optimal_steps(n),
        "results": [r.as_dict() for r in results],
    }


def render_state(state: HanoiState) -> str:
    """Pretty-print pegs top-down for prompts and human-readable output."""

    rows = []
    for peg, stack in enumerate(state.pegs):
        rows.append(f"peg{peg}: [{', '.join(str(d) for d in stack)}]")
    return "\n".join(rows)


def format_prompt(state: HanoiState) -> str:
    """Build the user turn for the LLM-controlled agent."""

    legal = legal_moves(state)
    return (
        "You move disks in a Towers of Hanoi puzzle.\n\n"
        "Rules:\n"
        "- Call the move_disk tool ONCE per turn with a single {source, target}.\n"
        "- source and target are integers in {0,1,2}.\n"
        "- Only move the top disk of a peg; never place a larger disk on a smaller one.\n"
        "- Win when all disks are stacked on peg 2 (largest at the bottom).\n\n"
        f"Disks: {state.n}. Moves so far: {state.moves}/{state.max_moves}.\n\n"
        f"Current pegs:\n{render_state(state)}\n\n"
        f"Legal moves: {json.dumps([list(m) for m in legal], separators=(',', ':'))}"
    )


__all__ = [
    "COMPLETION_REWARD",
    "EXCESS_STEP_PENALTY",
    "ILLEGAL_PENALTY",
    "MAX_DISKS",
    "MIN_DISKS",
    "NUM_PEGS",
    "SOURCE_PEG",
    "TARGET_PEG",
    "HanoiState",
    "IllegalActionPolicy",
    "ReplayResult",
    "evaluate",
    "format_prompt",
    "is_legal",
    "is_terminal",
    "legal_moves",
    "make_state",
    "optimal_solution",
    "optimal_steps",
    "parse_trace",
    "render_state",
    "replay",
    "serialize_trace",
    "step",
    "validate_solution",
]
