"""Pure-Python uniquely-solvable Sudoku environment for agentic RL.

Provides a deterministic puzzle generator, board validation, three agent tools
(inspect_candidates, place_digit, undo), and an episode scorer.  The
environment validates rows, columns, boxes, action budget, and terminal state
after every tool call without revealing the solution.
"""

from __future__ import annotations

import random
from typing import Any

Board = list[list[int]]
EMPTY = 0
DIGITS = frozenset(range(1, 10))

# Difficulty -> number of empty cells.
DIFFICULTY_EMPTY: dict[str, int] = {
    "easy": 36,
    "medium": 46,
    "hard": 54,
}
DEFAULT_DIFFICULTY = "easy"
DEFAULT_MAX_ACTIONS = 120

# --- Tool schemas (OpenAI function-calling format) -------------------------

INSPECT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_candidates",
        "description": "Return the list of legal digits for an empty cell.",
        "parameters": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8,
                    "description": "Row index (0-8).",
                },
                "col": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8,
                    "description": "Column index (0-8).",
                },
            },
            "required": ["row", "col"],
            "additionalProperties": False,
        },
    },
}

PLACE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "place_digit",
        "description": "Place a digit in an empty cell. Validates row, column, and box constraints.",
        "parameters": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8,
                    "description": "Row index (0-8).",
                },
                "col": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8,
                    "description": "Column index (0-8).",
                },
                "digit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "The digit to place (1-9).",
                },
            },
            "required": ["row", "col", "digit"],
            "additionalProperties": False,
        },
    },
}

UNDO_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "undo",
        "description": "Undo the last placed digit and restore the previous board state.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

TOOLS = [INSPECT_TOOL, PLACE_TOOL, UNDO_TOOL]


# --- Board helpers ---------------------------------------------------------

def normalize_board(board: list[list[Any]]) -> Board:
    """Return a validated 9x9 Sudoku board with int cells (0 = empty)."""

    rows = [[int(cell) for cell in row] for row in board]
    if len(rows) != 9 or any(len(row) != 9 for row in rows):
        raise ValueError("Sudoku board must be 9x9")
    for row in rows:
        for cell in row:
            if cell != EMPTY and cell not in DIGITS:
                raise ValueError(f"cell value must be 0 or 1-9, got {cell}")
    return rows


def board_to_text(board: Board) -> str:
    """Render the board as a human-readable grid with box separators."""

    lines = []
    lines.append("    0  1  2 | 3  4  5 | 6  7  8")
    lines.append("  +---------+---------+---------+")
    for r in range(9):
        cells = []
        for c in range(9):
            val = board[r][c]
            cells.append(f" {val}" if val else " .")
        row_str = "".join(cells)
        # Insert box separators at columns 3 and 6.
        row_str = row_str[:8] + "|" + row_str[8:17] + "|" + row_str[17:]
        lines.append(f"{r} |{row_str}|")
        if r in (2, 5):
            lines.append("  +---------+---------+---------+")
    lines.append("  +---------+---------+---------+")
    return "\n".join(lines)


def is_complete(board: Board) -> bool:
    """Return True if every cell is filled."""

    return all(board[r][c] != EMPTY for r in range(9) for c in range(9))


def is_valid_board(board: Board) -> bool:
    """Return True if the full board satisfies all Sudoku constraints."""

    for i in range(9):
        row_vals = [board[i][c] for c in range(9) if board[i][c] != EMPTY]
        col_vals = [board[r][i] for r in range(9) if board[r][i] != EMPTY]
        if len(row_vals) != len(set(row_vals)):
            return False
        if len(col_vals) != len(set(col_vals)):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box_vals = [board[br + dr][bc + dc] for dr in range(3) for dc in range(3) if board[br + dr][bc + dc] != EMPTY]
            if len(box_vals) != len(set(box_vals)):
                return False
    return True


# --- Constraint validation (does NOT reveal the solution) ------------------

def validate_placement(board: Board, row: int, col: int, digit: int) -> dict[str, Any]:
    """Check whether placing *digit* at (row, col) is legal.

    Returns ``{"valid": True}`` or ``{"valid": False, "error": str}``.
    The solution is never referenced.
    """

    if not (0 <= row <= 8 and 0 <= col <= 8):
        return {"valid": False, "error": f"coordinates out of range: row={row}, col={col}"}
    if digit not in DIGITS:
        return {"valid": False, "error": f"digit must be 1-9, got {digit}"}
    if board[row][col] != EMPTY:
        return {"valid": False, "error": f"cell ({row}, {col}) is already filled"}
    # Row conflict.
    if digit in board[row]:
        return {"valid": False, "error": f"row {row} already contains {digit}"}
    # Column conflict.
    if any(board[r][col] == digit for r in range(9)):
        return {"valid": False, "error": f"column {col} already contains {digit}"}
    # Box conflict.
    br, bc = (row // 3) * 3, (col // 3) * 3
    for dr in range(3):
        for dc in range(3):
            if board[br + dr][bc + dc] == digit:
                return {"valid": False, "error": f"box ({br // 3}, {bc // 3}) already contains {digit}"}
    return {"valid": True}


def inspect_candidates(board: Board, row: int, col: int) -> dict[str, Any]:
    """Return the legal candidate digits for an empty cell."""

    if not (0 <= row <= 8 and 0 <= col <= 8):
        return {"valid": False, "error": f"coordinates out of range: row={row}, col={col}"}
    if board[row][col] != EMPTY:
        return {"valid": False, "error": f"cell ({row}, {col}) is already filled with {board[row][col]}"}
    candidates = [d for d in range(1, 10) if validate_placement(board, row, col, d)["valid"]]
    return {"valid": True, "candidates": candidates}


# --- Episode state manager -------------------------------------------------

class SudokuEpisode:
    """Mutable episode state with operation history for undo."""

    def __init__(self, puzzle: Board, *, max_actions: int = DEFAULT_MAX_ACTIONS) -> None:
        self.board = normalize_board(puzzle)
        self.original = [row[:] for row in self.board]
        self.history: list[tuple[int, int, int]] = []
        self.max_actions = max_actions
        self.actions_taken = 0

    def place(self, row: int, col: int, digit: int) -> dict[str, Any]:
        """Place a digit and record history for undo."""

        result = validate_placement(self.board, row, col, digit)
        if result["valid"]:
            self.board[row][col] = digit
            self.history.append((row, col, digit))
        self.actions_taken += 1
        return result

    def undo(self) -> dict[str, Any]:
        """Undo the last placed digit."""

        self.actions_taken += 1
        if not self.history:
            return {"valid": False, "error": "no moves to undo"}
        row, col, _digit = self.history.pop()
        self.board[row][col] = EMPTY
        return {"valid": True, "undid": {"row": row, "col": col}}

    def is_done(self) -> bool:
        """Return True when the board is full or the action budget is spent."""

        if self.actions_taken >= self.max_actions:
            return True
        return is_complete(self.board)

    def is_solved(self) -> bool:
        """Return True only when the board is correctly filled."""

        return is_complete(self.board) and is_valid_board(self.board)


# --- Puzzle generation -----------------------------------------------------

def _fill_board(rng: random.Random) -> Board:
    """Generate a complete valid Sudoku solution via backtracking with randomness."""

    board = [[0] * 9 for _ in range(9)]
    _fill_board_recursive(board, 0, 0, rng)
    return board


def _fill_board_recursive(board: Board, row: int, col: int, rng: random.Random) -> bool:
    """Fill the board cell by cell with randomized digit order."""

    if row == 9:
        return True
    next_row, next_col = (row, col + 1) if col < 8 else (row + 1, 0)
    if board[row][col] != EMPTY:
        return _fill_board_recursive(board, next_row, next_col, rng)
    digits = list(range(1, 10))
    rng.shuffle(digits)
    for digit in digits:
        if validate_placement(board, row, col, digit)["valid"]:
            board[row][col] = digit
            if _fill_board_recursive(board, next_row, next_col, rng):
                return True
            board[row][col] = EMPTY
    return False


def _count_solutions(board: Board, limit: int = 2) -> int:
    """Count solutions up to *limit* to verify uniqueness."""

    board = [row[:] for row in board]
    count = 0

    def solve() -> None:
        nonlocal count
        if count >= limit:
            return
        for r in range(9):
            for c in range(9):
                if board[r][c] == EMPTY:
                    for d in range(1, 10):
                        if validate_placement(board, r, c, d)["valid"]:
                            board[r][c] = d
                            solve()
                            board[r][c] = EMPTY
                            if count >= limit:
                                return
                    return
        count += 1

    solve()
    return count


def generate_puzzle(
    difficulty: str = DEFAULT_DIFFICULTY,
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    """Generate a uniquely solvable Sudoku puzzle.

    Returns ``{"puzzle": Board, "difficulty": str}``.  The solution is not
    included in the return value to avoid accidental leakage.
    """

    if difficulty not in DIFFICULTY_EMPTY:
        raise ValueError(f"difficulty must be one of {list(DIFFICULTY_EMPTY)}, got {difficulty!r}")
    target_empty = DIFFICULTY_EMPTY[difficulty]
    rng = random.Random(seed)

    solution = _fill_board(rng)
    puzzle = [row[:] for row in solution]

    # Collect all cell coordinates and shuffle for random removal.
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)

    removed = 0
    for r, c in cells:
        if removed >= target_empty:
            break
        backup = puzzle[r][c]
        puzzle[r][c] = EMPTY
        if _count_solutions(puzzle, limit=2) == 1:
            removed += 1
        else:
            puzzle[r][c] = backup

    return {"puzzle": puzzle, "difficulty": difficulty}


# --- Prompt building -------------------------------------------------------

def make_prompt(record: dict[str, Any]) -> str:
    """Build the user prompt for one Sudoku task without leaking the solution."""

    board = normalize_board(record["puzzle"])
    difficulty = str(record.get("difficulty", DEFAULT_DIFFICULTY))
    max_actions = int(record.get("max_actions", DEFAULT_MAX_ACTIONS))
    return (
        f"You are solving a {difficulty} Sudoku puzzle. Fill every empty cell (.) "
        f"with a digit 1-9 so that each row, column, and 3x3 box contains 1-9 with no repeats. "
        f"You have at most {max_actions} actions. Use inspect_candidates to check legal digits, "
        f"place_digit to fill a cell, and undo to revert the last placement.\n\n"
        f"{board_to_text(board)}"
    )


SYSTEM_PROMPT = (
    "You are a careful Sudoku solver. On every turn call exactly one tool: "
    "inspect_candidates, place_digit, or undo. Use inspect_candidates before placing "
    "when unsure. Never guess blindly if a cell has only one candidate — place it. "
    "If a placement leads to a dead end, use undo and try a different digit."
)


# --- Scoring ---------------------------------------------------------------

def score_episode(
    puzzle: Board,
    actions: list[dict[str, Any]],
    *,
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> dict[str, Any]:
    """Replay *actions* on a copy of *puzzle* and return metrics.

    Returns a dict with ``reward``, ``solved``, ``valid_actions``,
    ``invalid_actions``, ``undos``, ``total_actions``.
    """

    board = normalize_board(puzzle)
    original = [row[:] for row in board]
    history: list[tuple[int, int, int]] = []
    valid_actions = 0
    invalid_actions = 0
    undos = 0

    for action in actions[:max_actions]:
        name = action.get("name")
        args = action.get("arguments", {})
        if name == "place_digit":
            row = int(args.get("row", -1))
            col = int(args.get("col", -1))
            digit = int(args.get("digit", 0))
            result = validate_placement(board, row, col, digit)
            if result["valid"]:
                board[row][col] = digit
                history.append((row, col, digit))
                valid_actions += 1
            else:
                invalid_actions += 1
        elif name == "inspect_candidates":
            valid_actions += 1  # inspect is always a valid action
        elif name == "undo":
            if history:
                r, c, _ = history.pop()
                board[r][c] = EMPTY
                undos += 1
                valid_actions += 1
            else:
                invalid_actions += 1
        else:
            invalid_actions += 1

    solved = is_complete(board) and is_valid_board(board)
    filled = sum(
        1
        for r in range(9)
        for c in range(9)
        if board[r][c] != EMPTY and original[r][c] == EMPTY
    )
    total_empty = sum(1 for r in range(9) for c in range(9) if original[r][c] == EMPTY)
    fill_ratio = filled / max(total_empty, 1)

    if solved:
        efficiency = max(0.0, (max_actions - len(actions[:max_actions])) / max(max_actions - 1, 1))
        reward = 0.8 + 0.2 * efficiency
    elif fill_ratio > 0:
        reward = 0.3 * fill_ratio
    else:
        reward = -1.0

    if invalid_actions > 0:
        reward -= 0.1 * invalid_actions / max(len(actions), 1)

    return {
        "reward": round(reward, 4),
        "solved": solved,
        "valid_actions": valid_actions,
        "invalid_actions": invalid_actions,
        "undos": undos,
        "total_actions": len(actions),
        "fill_ratio": round(fill_ratio, 4),
    }