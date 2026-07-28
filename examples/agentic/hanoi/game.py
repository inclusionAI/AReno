"""Deterministic Towers of Hanoi rules for the agentic example."""

from __future__ import annotations

from typing import Any, Iterable

PEGS = ("A", "B", "C")
MIN_DISKS = 3
MAX_DISKS = 6
DEFAULT_SEED = 2026
DEFAULT_COUNT = 128

# Reward weighting: completion is the dominant signal; efficiency is a small
# component relative to the known optimum. Tune these in call sites (and reward
# consumers) without changing the scoring contract.
COMPLETION_WEIGHT = 0.5
EFFICIENCY_WEIGHT = 0.5

MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "move",
        "description": "Move the top disk of the source peg onto the target peg.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": list(PEGS),
                    "description": "The peg to take the top disk from (A, B, or C).",
                },
                "target": {
                    "type": "string",
                    "enum": list(PEGS),
                    "description": "The peg to place the disk onto (A, B, or C).",
                },
            },
            "required": ["source", "target"],
            "additionalProperties": False,
        },
    },
}


def initial_state(n: int) -> dict[str, list[int]]:
    """Return the canonical start state: all disks on peg A, largest on the bottom."""

    n = _validate_disk_count(n)
    return {"A": list(range(n, 0, -1)), "B": [], "C": []}


def validate_disks(n: int) -> int:
    """Validate and return a disk count in the supported range."""

    return _validate_disk_count(n)


def _validate_disk_count(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("disk count must be an integer")
    if n < MIN_DISKS or n > MAX_DISKS:
        raise ValueError(f"disk count must be in [{MIN_DISKS}, {MAX_DISKS}]")
    return n


def normalize_state(state: dict[str, Iterable[int]], *, n: int) -> dict[str, list[int]]:
    """Return a validated state with integer stacks and canonical peg keys."""

    n = _validate_disk_count(n)
    if not isinstance(state, dict) or set(state.keys()) != set(PEGS):
        raise ValueError(f"state must expose exactly pegs {PEGS}")
    normalized: dict[str, list[int]] = {}
    for peg in PEGS:
        stack = [int(disk) for disk in state[peg]]
        if any(disk < 1 or disk > n for disk in stack):
            raise ValueError(f"peg {peg} contains a disk outside [1, {n}]")
        if len(stack) != len(set(stack)):
            raise ValueError(f"peg {peg} contains duplicate disks")
        normalized[peg] = stack
    seen = sorted(disk for peg in PEGS for disk in normalized[peg])
    if seen != list(range(1, n + 1)):
        raise ValueError(f"state does not contain each disk 1..{n} exactly once")
    return normalized


def top_disk(state: dict[str, list[int]], peg: str) -> int | None:
    """Return the top disk of ``peg`` (last list element) or None if empty."""

    stack = state.get(peg)
    if not stack:
        return None
    return stack[-1]


def is_legal_move(state: dict[str, list[int]], source: Any, target: Any) -> bool:
    """Reject empty-source moves, larger-on-smaller moves, and invalid pegs."""

    try:
        state = normalize_state(state, n=_infer_n(state))
    except ValueError:
        return False
    if source not in PEGS or target not in PEGS:
        return False
    if source == target:
        return False
    source_top = top_disk(state, source)
    if source_top is None:
        return False
    target_top = top_disk(state, target)
    if target_top is not None and source_top > target_top:
        return False
    return True


def illegal_reason(state: dict[str, list[int]], source: Any, target: Any) -> str:
    """Return a human-readable reason for an illegal move."""

    if source not in PEGS or target not in PEGS:
        return f"peg names must be one of {PEGS}"
    if source == target:
        return "source and target are the same peg"
    if top_disk(state, source) is None:
        return f"peg {source} is empty"
    source_top = top_disk(state, source)
    target_top = top_disk(state, target)
    return f"cannot place disk {source_top} on smaller disk {target_top}"


def legal_moves(state: dict[str, list[int]]) -> list[tuple[str, str]]:
    """Return every legal (source, target) peg pair for ``state``."""

    result: list[tuple[str, str]] = []
    for source in PEGS:
        for target in PEGS:
            if is_legal_move(state, source, target):
                result.append((source, target))
    return result


def apply_move(state: dict[str, list[int]], source: str, target: str) -> dict[str, list[int]]:
    """Apply a legal move and return a new state. Raise on illegal moves."""

    n = _infer_n(state)
    state = normalize_state(state, n=n)
    if source not in PEGS or target not in PEGS:
        raise ValueError(f"peg names must be one of {PEGS}")
    if not is_legal_move(state, source, target):
        raise ValueError(illegal_reason(state, source, target))
    next_state = {peg: list(stack) for peg, stack in state.items()}
    next_state[target].append(next_state[source].pop())
    return next_state


def is_terminal(state: dict[str, list[int]], n: int) -> bool:
    """Return True when every disk is stacked on peg C in the solved order."""

    normalize_state(state, n=n)
    return list(state["C"]) == list(range(n, 0, -1))


def optimal_steps(n: int) -> int:
    """Return the closed-form minimum number of moves: 2**n - 1."""

    return 2 ** _validate_disk_count(n) - 1


def optimal_moves(n: int, *, source: str = "A", target: str = "C", auxiliary: str = "B") -> list[tuple[str, str]]:
    """Return the recursive optimal move sequence solving the puzzle."""

    _validate_disk_count(n)
    return _optimal_moves(n, source, target, auxiliary)


def _optimal_moves(n: int, source: str, target: str, auxiliary: str) -> list[tuple[str, str]]:
    if n == 0:
        return []
    moves: list[tuple[str, str]] = []
    moves.extend(_optimal_moves(n - 1, source, auxiliary, target))
    moves.append((source, target))
    moves.extend(_optimal_moves(n - 1, auxiliary, target, source))
    return moves


def default_max_moves(n: int) -> int:
    """Return a generous per-task move cap, twice the optimum plus a little slack."""

    return optimal_steps(n) * 2 + 2


def state_to_text(state: dict[str, list[int]]) -> str:
    """Render pegs bottom-up so the model can read the top disk at a glance."""

    lines = []
    for peg in PEGS:
        stack = state.get(peg, [])
        rendered = " ".join(str(disk) for disk in stack) if stack else "(empty)"
        lines.append(f"Peg {peg} (bottom->top): {rendered}")
    return "\n".join(lines)


def make_prompt(record: dict[str, Any]) -> str:
    """Build the per-task task prompt without leaking any solution."""

    n = int(record["n"])
    max_moves = int(record.get("max_moves", default_max_moves(n)))
    return (
        f"Solve the Towers of Hanoi puzzle with {n} disks. All disks start on peg A "
        "(largest on the bottom, smallest on top) and must end on peg C in the same order.\n\n"
        "Rules:\n"
        "- Move exactly one disk per turn by calling the move(source, target) tool.\n"
        "- source/target are peg names A, B, or C.\n"
        "- You may only move the top disk of a peg. Never place a larger disk on a smaller one.\n"
        "- An empty-source move or a larger-on-smaller move is illegal and ends the episode.\n"
        f"- You have at most {max_moves} moves.\n\n"
        f"Start state:\n{state_to_text(initial_state(n))}\n\nMove:"
    )


def score_episode(n: int, moves: list[tuple[Any, Any]]) -> dict[str, Any]:
    """Replay ``moves`` and score completion plus efficiency over the optimum.

    Illegal moves terminate the replay and score 0.0 exactly (per the design:
    an illegal action ends the rollout). Completion earns the full completion
    weight; the efficiency component decays as moves exceed the known optimum.
    """

    n = _validate_disk_count(n)
    optimum = optimal_steps(n)
    state = initial_state(n)
    for index, move in enumerate(moves):
        source, target = _move_pair(move)
        if not is_legal_move(state, source, target):
            return {
                "completed": False,
                "illegal": True,
                "num_moves": index,
                "excess": None,
                "efficiency": 0.0,
                "reward": 0.0,
            }
        state = apply_move(state, source, target)
        if is_terminal(state, n):
            num_moves = index + 1
            excess = max(0, num_moves - optimum)
            efficiency = max(0.0, 1.0 - excess / optimum)
            reward = COMPLETION_WEIGHT + EFFICIENCY_WEIGHT * efficiency
            return {
                "completed": True,
                "illegal": False,
                "num_moves": num_moves,
                "excess": excess,
                "efficiency": efficiency,
                "reward": reward,
            }
    return {
        "completed": False,
        "illegal": False,
        "num_moves": len(moves),
        "excess": None,
        "efficiency": 0.0,
        "reward": 0.0,
    }


def _move_pair(move: Any) -> tuple[Any, Any]:
    if isinstance(move, (tuple, list)) and len(move) == 2:
        return move[0], move[1]
    return None, None


def _infer_n(state: dict[str, Iterable[int]]) -> int:
    disks = [int(disk) for peg in PEGS for disk in state.get(peg, [])]
    if not disks:
        raise ValueError("state has no disks; pass n explicitly via normalize_state")
    return max(disks)
