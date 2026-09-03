"""Bash game (巴什博弈 / 取石子游戏): single-turn optimal-move oracle.

This is the classic take-away strategy game. There is one pile of ``n``
stones; two players alternate and on each turn a player must remove between
``1`` and ``m`` stones; the player who removes the last stone wins.

The winning strategy is a pure modular rule (it is *not* an XOR / GF(2)
system): the current position is losing (a P-position) iff ``n % (m + 1) == 0``;
otherwise it is winning and the unique optimal move is to take exactly
``n % (m + 1)`` stones, leaving a multiple of ``m + 1`` to the opponent.

Single-turn contract: ``state (n, m) -> one move (take k | resign)``.

The answer is fully determined by ``n`` and ``m``; there is no environment
step-back, no multi-turn trace, and no hidden search.
"""

from __future__ import annotations

import copy
import random
from typing import Any

VALID_MAX_TAKES = (2, 3, 4, 5, 6)   # difficulty axis 1: the max take m
DEFAULT_N_MAX = 40                  # default cap on pile size
DEFAULT_M = 3


class BashGame:
    """Pure functions for the Bash game (no hidden I/O, no RNG here)."""

    def __init__(self, n: int, m: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        self.n = int(n)
        self.m = int(m)

    @property
    def modulus(self) -> int:
        """The period of the game, m + 1."""
        return self.m + 1

    def is_winning(self) -> bool:
        """True iff the player to move has a winning strategy."""
        return self.n % self.modulus != 0

    def optimal_move(self) -> dict[str, Any]:
        """Return the unique optimal move.

        * If ``n % (m + 1) == 0`` the position is losing: ``{"resign": True}``.
        * Otherwise ``{"take": k}`` with ``k = n % (m + 1)`` (1 <= k <= m).
        """
        k = self.n % self.modulus
        if k == 0:
            return {"resign": True}
        return {"take": k}

    def after_move(self, take: int) -> "BashGame":
        """Apply a legal move and return the successor position."""
        if take < 1 or take > self.m or take > self.n:
            raise ValueError(f"illegal take {take} for n={self.n} m={self.m}")
        return BashGame(self.n - take, self.m)


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy a raw generator row into internal form."""
    record = copy.deepcopy(raw)
    n = int(record["n"])
    m = int(record["m"])
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if m < 1 or m not in VALID_MAX_TAKES:
        raise ValueError(f"m must be one of {VALID_MAX_TAKES}, got {m}")
    record["n"] = n
    record["m"] = m
    game = BashGame(n, m)
    move = game.optimal_move()
    record["winning"] = game.is_winning()
    record["oracle_move"] = move          # hidden: never shown to the model
    record["solution"] = move             # aliased for consistency with loader
    return record


def move_to_answer(move: dict[str, Any]) -> str:
    """Human-readable canonical answer string for an oracle move."""
    if move.get("resign"):
        return "resign"
    return f"take {move['take']}"


def format_prompt(record: dict[str, Any]) -> str:
    """Single-turn prompt exposing only the visible state ``(n, m)``."""
    n = record["n"]
    m = record["m"]
    return (
        "You are playing the Bash game (取石子游戏).\n\n"
        "Rules:\n"
        f"- There is one pile of {n} stones.\n"
        f"- On a turn you must take between 1 and {m} stones.\n"
        "- The player who takes the very last stone wins.\n"
        "- It is your turn. If you have a winning move, take exactly the number\n"
        "  of stones that leaves your opponent a losing position.\n"
        "- If your position is losing no matter what you take, resign instead.\n"
        "- Answer with exactly one submit_move tool call and nothing else.\n\n"
        f"Pile: {n} stones, max take {m}."
    )


def tool_schema() -> dict[str, Any]:
    """One closed tool call: either ``take`` k (1..m) or ``resign`` true."""
    return {
        "type": "function",
        "function": {
            "name": "submit_move",
            "description": "Submit your single move: take k stones (winning) or resign (losing position).",
            "parameters": {
                "type": "object",
                "properties": {
                    "take": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of stones to take this turn (omit if resigning).",
                    },
                    "resign": {
                        "type": "boolean",
                        "description": "Set true exactly when there is no winning move.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def parse_move(record_or_response: Any) -> dict[str, Any] | None:
    """Extract a single normalized move dict from a model response.

    Returns ``None`` for malformed / missing / multi-call inputs so the reward
    can fail safely instead of raising.
    """
    tool_calls = getattr(record_or_response, "tool_calls", None)
    if tool_calls is None and hasattr(record_or_response, "source_record"):
        # record-shaped input
        tool_calls = getattr(record_or_response, "tool_calls", None)
    if not tool_calls:
        return None
    # tolerate a response object with nested function name/arguments
    calls = []
    for call in tool_calls:
        fn = getattr(call, "function", None)
        if fn is not None:
            name = getattr(fn, "name", None)
            args = getattr(fn, "arguments", None)
        elif isinstance(call, dict):
            if "function" in call and isinstance(call["function"], dict):
                fn = call["function"]
                name = fn.get("name")
                args = fn.get("arguments")
            else:
                # flat dict produced by eval.py / FakeRecord
                name = call.get("name")
                args = call.get("arguments")
        else:
            continue
        calls.append((name, args))
    if len(calls) != 1:
        return None
    name, args = calls[0]
    if name != "submit_move":
        return None
    if isinstance(args, str):
        import json

        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(args, dict):
        return None
    # only the closed two keys; no extra text injection honoured
    take = args.get("take")
    resign = args.get("resign")
    if resign is True and take in (None, 0):
        return {"resign": True}
    if resign is not True and isinstance(take, int):
        return {"take": take}
    return None


def legal_moves(n: int, m: int) -> list[int]:
    return list(range(1, min(m, n) + 1))


def random_record(rng: random.Random, n_max: int = DEFAULT_N_MAX, m: int | None = None,
                  force_winning: bool | None = None) -> dict[str, Any]:
    """Sample a random position (used by tests / quick inspection)."""
    mm = m if m is not None else rng.choice(VALID_MAX_TAKES)
    n = rng.randint(1, n_max)
    if force_winning is True:
        while n % (mm + 1) == 0:
            n = rng.randint(1, n_max)
    elif force_winning is False:
        while n % (mm + 1) != 0:
            n = rng.randint(1, n_max)
    return normalize_record({"n": n, "m": mm})
