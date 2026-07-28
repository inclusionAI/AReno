"""Pure-Python Sudoku environment for the agentic RL example.

This module is intentionally free of GPU, network, and sandbox dependencies.
It provides:

- a generator for *uniquely solvable* Sudoku puzzles (backtracking solve,
  randomized digging, uniqueness check);
- three agent tools: ``inspect_candidates``, ``place_digit``, ``undo``;
- per-call validation of rows, columns, boxes, the action budget, and the
  terminal state.

The solution is held internally for generation-time uniqueness checks only.
It is **never** returned to the agent. Termination is decided purely from the
visible board: for a uniquely solvable Sudoku, "every cell filled with no
row/column/box conflict" is equivalent to "matches the unique solution", so
the environment can grade a solve without ever exposing the answer.

Design notes / TODO (confirm with the team before training):
- An *illegal* ``place_digit`` (digit conflicts with row/column/box, or the
  cell is a given/already filled) is **rejected**: the board is unchanged, the
  call still consumes one action from the budget, and the result is flagged
  ``invalid_action=True``. This keeps the board clean and lets the policy learn
  to avoid illegal moves. An alternative (accept + penalize) is described in
  the README; switch ``_reject_illegal`` if that is preferred.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

EMPTY = 0
DIGITS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
_ALL_DIGITS_MASK: int = 0
for _d in DIGITS:
    _ALL_DIGITS_MASK |= 1 << _d
ALL_COORDS: tuple[tuple[int, int], ...] = tuple((r, c) for r in range(9) for c in range(9))

# Number of givens retained per difficulty band. 17 is the theoretical minimum
# for a unique-solution Sudoku; bands are chosen so generation stays fast.
DIFFICULTY_CLUES: dict[str, int] = {
    "easy": 40,
    "medium": 32,
    "hard": 26,
    "extreme": 20,
}
DEFAULT_DIFFICULTY = "medium"
# Default action budget per episode. Plenty for a 41-empty-cell medium board
# while still bounded so truncation is observable.
DEFAULT_ACTION_BUDGET = 81


class SudokuError(ValueError):
    """Raised for malformed coordinates/digits or illegal undo at history start."""


def _box_index(row: int, col: int) -> int:
    return (row // 3) * 3 + col // 3


@dataclass(slots=True)
class SudokuEnv:
    """One Sudoku rollout episode.

    ``puzzle`` is the agent-visible board (0 = empty). ``solution`` and
    ``givens`` are internal and never serialized into tool results.
    """

    puzzle: list[list[int]]
    solution: list[list[int]] | None
    givens: set[tuple[int, int]]
    history: list[tuple[int, int, int]] = field(default_factory=list)
    action_budget: int = DEFAULT_ACTION_BUDGET
    actions_used: int = 0
    invalid_actions: int = 0
    difficulty: str = DEFAULT_DIFFICULTY
    seed: int = 0

    # ----- construction -------------------------------------------------

    @classmethod
    def generate(
        cls,
        *,
        difficulty: str = DEFAULT_DIFFICULTY,
        seed: int = 0,
        action_budget: int | None = None,
    ) -> "SudokuEnv":
        """Generate a new uniquely-solvable puzzle at the given difficulty."""

        difficulty = _coerce_difficulty(difficulty)
        rng = random.Random(seed)
        solution = _random_full_solution(rng)
        puzzle = _dig_holes(solution, rng, clues=DIFFICULTY_CLUES[difficulty])
        givens = {(r, c) for r, c in ALL_COORDS if puzzle[r][c] != EMPTY}
        budget = DEFAULT_ACTION_BUDGET if action_budget is None else int(action_budget)
        if budget <= 0:
            raise ValueError("action_budget must be positive")
        return cls(
            puzzle=[row[:] for row in puzzle],
            solution=solution,
            givens=givens,
            action_budget=budget,
            difficulty=difficulty,
            seed=seed,
        )

    @classmethod
    def from_puzzle(
        cls,
        puzzle: Sequence[Sequence[int]],
        *,
        difficulty: str = DEFAULT_DIFFICULTY,
        seed: int = 0,
        action_budget: int | None = None,
    ) -> "SudokuEnv":
        """Rebuild an episode from a stored puzzle (no solution needed).

        Used at rollout time so every sample starts from the exact dataset
        board. ``solution`` is left ``None``: ``is_solved`` does not consult it.
        """

        difficulty = _coerce_difficulty(difficulty)
        board = _normalize_puzzle(puzzle)
        givens = {(r, c) for r, c in ALL_COORDS if board[r][c] != EMPTY}
        budget = DEFAULT_ACTION_BUDGET if action_budget is None else int(action_budget)
        if budget <= 0:
            raise ValueError("action_budget must be positive")
        return cls(
            puzzle=board,
            solution=None,
            givens=givens,
            action_budget=budget,
            difficulty=difficulty,
            seed=seed,
        )

    # ----- agent tools --------------------------------------------------

    def inspect_candidates(self, row: int, col: int) -> dict[str, Any]:
        """Return the constraint-based candidate digits for an empty cell.

        Candidates are computed from the visible board only (row/column/box
        conflicts). The solution is never consulted, so this call cannot leak
        the answer.
        """

        self._check_coord(row, col)
        if self.puzzle[row][col] != EMPTY:
            raise SudokuError(f"cell ({row},{col}) is not empty")
        if self.is_terminal():
            raise SudokuError("episode is terminal; no further actions allowed")
        candidates = sorted(self._candidates(row, col))
        return {
            "action": "inspect_candidates",
            "row": row,
            "col": col,
            "candidates": candidates,
            "actions_remaining": self._actions_remaining(),
            "solved": self.is_solved(),
        }

    def place_digit(self, row: int, col: int, digit: int) -> dict[str, Any]:
        """Place a digit. Illegal placements are rejected but consume budget."""

        self._check_coord(row, col)
        if digit not in DIGITS:
            raise SudokuError(f"digit must be 1-9, got {digit!r}")
        if self.is_terminal():
            raise SudokuError("episode is terminal; no further actions allowed")

        self.actions_used += 1
        if self.puzzle[row][col] != EMPTY:
            return self._reject(row, col, digit, reason="cell_not_empty")
        if not self._is_placeable(row, col, digit):
            return self._reject(row, col, digit, reason="digit_conflicts")

        self.puzzle[row][col] = digit
        self.history.append((row, col, digit))
        solved = self.is_solved()
        return {
            "action": "place_digit",
            "row": row,
            "col": col,
            "digit": digit,
            "placed": True,
            "invalid_action": False,
            "solved": solved,
            "actions_remaining": self._actions_remaining(),
        }

    def undo(self) -> dict[str, Any]:
        """Undo the most recent successful placement."""

        if self.is_terminal():
            raise SudokuError("episode is terminal; no further actions allowed")
        if not self.history:
            raise SudokuError("undo at history start: no placement to revert")
        row, col, digit = self.history.pop()
        self.puzzle[row][col] = EMPTY
        self.actions_used += 1
        return {
            "action": "undo",
            "row": row,
            "col": col,
            "digit": digit,
            "undone": True,
            "actions_remaining": self._actions_remaining(),
            "solved": False,
        }

    # ----- validation / state ------------------------------------------

    def validate(self) -> dict[str, Any]:
        """Full validation of rows/columns/boxes/budget/terminal state."""

        conflicts = _find_conflicts(self.puzzle)
        return {
            "rows_ok": not conflicts["rows"],
            "cols_ok": not conflicts["cols"],
            "boxes_ok": not conflicts["boxes"],
            "conflicts": conflicts,
            "filled_cells": sum(1 for r, c in ALL_COORDS if self.puzzle[r][c] != EMPTY),
            "actions_used": self.actions_used,
            "invalid_actions": self.invalid_actions,
            "actions_remaining": self._actions_remaining(),
            "is_terminal": self.is_terminal(),
            "is_solved": self.is_solved(),
        }

    def is_solved(self) -> bool:
        """A uniquely-solvable board is solved iff filled with no conflicts."""

        return _is_complete_and_valid(self.puzzle)

    def is_terminal(self) -> bool:
        return self.is_solved() or self.actions_used >= self.action_budget

    # ----- rendering ----------------------------------------------------

    def board_text(self) -> str:
        """Render the visible board with box separators. ``.`` = empty."""

        lines = []
        for r in range(9):
            cells = [str(self.puzzle[r][c]) if self.puzzle[r][c] != EMPTY else "." for c in range(9)]
            row = " | ".join(" ".join(cells[i * 3 : i * 3 + 3]) for i in range(3))
            lines.append(row)
            if r in (2, 5):
                lines.append("-----+-----+-----")
        return "\n".join(lines)

    def public_state(self, include_candidates: bool = False) -> dict[str, Any]:
        """Return everything the agent is allowed to see (no solution)."""

        state: dict[str, Any] = {
            "difficulty": self.difficulty,
            "board": [row[:] for row in self.puzzle],
            "board_text": self.board_text(),
            "givens": sorted((r + 1, c + 1) for r, c in self.givens),
            "actions_used": self.actions_used,
            "invalid_actions": self.invalid_actions,
            "actions_remaining": self._actions_remaining(),
            "is_solved": self.is_solved(),
            "is_terminal": self.is_terminal(),
        }
        if include_candidates:
            state["candidates_by_cell"] = {
                f"{r+1},{c+1}": sorted(self._candidates(r, c))
                for r, c in ALL_COORDS
                if self.puzzle[r][c] == EMPTY
            }
        return state

    # ----- internals ----------------------------------------------------

    def _candidates(self, row: int, col: int) -> set[int]:
        used = set(self.puzzle[row]) | {self.puzzle[r][col] for r in range(9)}
        br, bc = (row // 3) * 3, (col // 3) * 3
        used.update(self.puzzle[br + i][bc + j] for i in range(3) for j in range(3))
        used.discard(EMPTY)
        return set(DIGITS) - used

    def _is_placeable(self, row: int, col: int, digit: int) -> bool:
        if digit in self.puzzle[row]:
            return False
        if digit in (self.puzzle[r][col] for r in range(9)):
            return False
        br, bc = (row // 3) * 3, (col // 3) * 3
        if digit in (self.puzzle[br + i][bc + j] for i in range(3) for j in range(3)):
            return False
        return True

    def _reject(self, row: int, col: int, digit: int, *, reason: str) -> dict[str, Any]:
        self.invalid_actions += 1
        return {
            "action": "place_digit",
            "row": row,
            "col": col,
            "digit": digit,
            "placed": False,
            "invalid_action": True,
            "reason": reason,
            "solved": False,
            "actions_remaining": self._actions_remaining(),
        }

    def _actions_remaining(self) -> int:
        return max(0, self.action_budget - self.actions_used)

    def _check_coord(self, row: int, col: int) -> None:
        if not (isinstance(row, int) and isinstance(col, int)):
            raise SudokuError(f"coordinates must be ints, got {row!r},{col!r}")
        if not (0 <= row < 9 and 0 <= col < 9):
            raise SudokuError(f"coordinates out of range 0..8, got ({row},{col})")


# ----- pure solver / generator helpers --------------------------------------


def _coerce_difficulty(difficulty: str | None) -> str:
    if difficulty is None:
        return DEFAULT_DIFFICULTY
    key = str(difficulty).lower()
    if key not in DIFFICULTY_CLUES:
        raise ValueError(f"unknown difficulty {difficulty!r}; choose from {sorted(DIFFICULTY_CLUES)}")
    return key


def _normalize_puzzle(puzzle: Sequence[Sequence[int]]) -> list[list[int]]:
    """Validate and copy a 9x9 board into a mutable int grid."""

    if len(puzzle) != 9 or any(len(row) != 9 for row in puzzle):
        raise ValueError("Sudoku puzzle must be 9x9")
    board: list[list[int]] = []
    for row in puzzle:
        cells: list[int] = []
        for v in row:
            iv = int(v)
            if iv != EMPTY and iv not in DIGITS:
                raise ValueError(f"cell value must be 0 or 1-9, got {v!r}")
            cells.append(iv)
        board.append(cells)
    return board


def _random_full_solution(rng: random.Random) -> list[list[int]]:
    """Fill a 9x9 grid with a random valid complete solution."""

    board = [[EMPTY] * 9 for _ in range(9)]
    row_used, col_used, box_used = _empty_masks()
    _fill_solution(board, rng, row_used, col_used, box_used)
    return board


def _empty_masks() -> tuple[list[int], list[int], list[int]]:
    return [0] * 9, [0] * 9, [0] * 9


def _apply_masks(masks: tuple[list[int], list[int], list[int]], board: list[list[int]]) -> None:
    row_used, col_used, box_used = masks
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v != EMPTY:
                bit = 1 << v
                row_used[r] |= bit
                col_used[c] |= bit
                box_used[_box_index(r, c)] |= bit


def _fill_solution(
    board: list[list[int]],
    rng: random.Random,
    row_used: list[int],
    col_used: list[int],
    box_used: list[int],
) -> bool:
    """Fill the board via MRV backtracking with random candidate order."""

    best = _mrv_cell(board, row_used, col_used, box_used)
    if best is None:
        return True  # no empty cell left -> solved
    r, c, cand_mask = best
    b = _box_index(r, c)
    digits = [d for d in DIGITS if cand_mask & (1 << d)]
    rng.shuffle(digits)
    for d in digits:
        bit = 1 << d
        board[r][c] = d
        row_used[r] |= bit
        col_used[c] |= bit
        box_used[b] |= bit
        if _fill_solution(board, rng, row_used, col_used, box_used):
            return True
        board[r][c] = EMPTY
        row_used[r] &= ~bit
        col_used[c] &= ~bit
        box_used[b] &= ~bit
    return False


def _mrv_cell(
    board: list[list[int]],
    row_used: list[int],
    col_used: list[int],
    box_used: list[int],
) -> tuple[int, int, int] | None:
    """Return (row, col, candidate_mask) for the empty cell with fewest candidates."""

    best: tuple[int, int, int] | None = None
    best_count = 10
    for r in range(9):
        for c in range(9):
            if board[r][c] != EMPTY:
                continue
            cand = _ALL_DIGITS_MASK & ~(row_used[r] | col_used[c] | box_used[_box_index(r, c)])
            count = _popcount(cand)
            if count == 0:
                return r, c, 0  # dead end: no legal digit
            if count < best_count:
                best_count = count
                best = (r, c, cand)
                if count == 1:
                    return best
    return best


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


def _dig_holes(
    solution: list[list[int]],
    rng: random.Random,
    *,
    clues: int,
) -> list[list[int]]:
    """Remove cells while preserving a unique solution until `clues` remain."""

    board = [row[:] for row in solution]
    cells = list(ALL_COORDS)
    rng.shuffle(cells)
    removed = 0
    target_remove = 81 - clues
    for r, c in cells:
        if removed >= target_remove:
            break
        if board[r][c] == EMPTY:
            continue
        saved = board[r][c]
        board[r][c] = EMPTY
        if _count_solutions(board, limit=2) == 1:
            removed += 1
        else:
            board[r][c] = saved  # not unique; restore
    return board


def _count_solutions(board: list[list[int]], *, limit: int = 2) -> int:
    """Count solutions up to `limit` (early-exit). Used for uniqueness checks.

    Uses row/col/box bitmasks and MRV ordering so that proving uniqueness
    (count == 1, which requires exhausting the search tree) stays fast even on
    boards with many empty cells.
    """

    work = [row[:] for row in board]
    row_used, col_used, box_used = _empty_masks()
    _apply_masks((row_used, col_used, box_used), work)
    counts = 0

    def backtrack() -> None:
        nonlocal counts
        if counts >= limit:
            return
        best = _mrv_cell(work, row_used, col_used, box_used)
        if best is None:
            counts += 1
            return
        r, c, cand_mask = best
        if cand_mask == 0:
            return  # dead end
        b = _box_index(r, c)
        d = 1
        while cand_mask:
            bit = cand_mask & -cand_mask  # lowest set bit
            cand_mask ^= bit
            d = bit.bit_length() - 1
            work[r][c] = d
            row_used[r] |= bit
            col_used[c] |= bit
            box_used[b] |= bit
            backtrack()
            work[r][c] = EMPTY
            row_used[r] &= ~bit
            col_used[c] &= ~bit
            box_used[b] &= ~bit
            if counts >= limit:
                return

    backtrack()
    return counts


def _is_placeable_static(board: list[list[int]], row: int, col: int, digit: int) -> bool:
    """Constraint check for a static board (kept for tests/simple callers)."""

    if digit in board[row]:
        return False
    if digit in (board[r][col] for r in range(9)):
        return False
    br, bc = (row // 3) * 3, (col // 3) * 3
    return digit not in (board[br + i][bc + j] for i in range(3) for j in range(3))


def _is_complete_and_valid(board: Sequence[Sequence[int]]) -> bool:
    """True iff every cell is filled and rows/cols/boxes have no conflict."""

    for row in board:
        if EMPTY in (c for c in row):
            return False
    if any(set(row) != set(DIGITS) for row in board):
        return False
    for c in range(9):
        col = {board[r][c] for r in range(9)}
        if col != set(DIGITS):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = {board[br + i][bc + j] for i in range(3) for j in range(3)}
            if box != set(DIGITS):
                return False
    return True


def _find_conflicts(board: Sequence[Sequence[int]]) -> dict[str, list[list[int]]]:
    """Return indices of rows/cols/boxes that contain duplicate non-empty digits."""

    def dup_index(groups: Iterable[Iterable[int]]) -> list[int]:
        dups = []
        for idx, group in enumerate(groups):
            seen = set()
            bad = False
            for v in group:
                if v == EMPTY:
                    continue
                if v in seen:
                    bad = True
                    break
                seen.add(v)
            if bad:
                dups.append(idx)
        return dups

    rows = [list(board[r]) for r in range(9)]
    cols = [[board[r][c] for r in range(9)] for c in range(9)]
    boxes = [
        [board[br + i][bc + j] for i in range(3) for j in range(3)]
        for br in range(0, 9, 3)
        for bc in range(0, 9, 3)
    ]
    return {"rows": dup_index(rows), "cols": dup_index(cols), "boxes": dup_index(boxes)}


def parse_coord(value: Any) -> tuple[int, int]:
    """Best-effort parse of (row, col) from tool-call arguments.

    Accepts 0-based or 1-based ints; values in 1..9 are treated as 1-based.
    Raises SudokuError on missing/malformed input so the agent loop can feed
    a structured error back to the policy instead of crashing the rollout.
    """

    if isinstance(value, (list, tuple)) and len(value) == 2:
        a, b = value[0], value[1]
    elif isinstance(value, str):
        parts = [p for p in value.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise SudokuError(f"cannot parse coordinate from {value!r}")
        a, b = parts[0], parts[1]
    else:
        raise SudokuError(f"cannot parse coordinate from {value!r}")
    if a is None or b is None:
        missing = "row" if a is None else "col"
        raise SudokuError(f"missing required argument '{missing}'")
    try:
        r, c = int(a), int(b)
    except (TypeError, ValueError):
        raise SudokuError(f"row/col must be integers, got ({a!r},{b!r})") from None
    if 1 <= r <= 9 and 1 <= c <= 9:
        return r - 1, c - 1
    if 0 <= r < 9 and 0 <= c < 9:
        return r, c
    raise SudokuError(f"coordinates out of range, got ({r},{c})")