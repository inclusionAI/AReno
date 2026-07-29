"""Pure-Python 2048 board engine with seeded tile placement.

Provides the game logic for an agentic RL example: a 4x4 board, four
directional swipes, deterministic new-tile spawning via an external RNG,
merge scoring, terminal detection, and text rendering for prompts.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable

Board = list[list[int]]
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
EMPTY = 0
SIZE = 4
DEFAULT_MAX_MOVES = 50

SYSTEM_PROMPT = (
    "You are a 2048 game AI. The board is a 4x4 grid. Each turn you choose to "
    "swipe UP, DOWN, LEFT, or RIGHT. Identical tiles merge on collision. After "
    "each move a new 2 (90%) or 4 (10%) appears in a random empty cell.\n"
    "Keep your largest tile in a corner. "
    "You MUST call the move tool every turn with one direction."
)

MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "move",
        "description": "Swipe the 2048 board in one direction.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": list(DIRECTIONS),
                    "description": "The swipe direction: UP, DOWN, LEFT, or RIGHT.",
                }
            },
            "required": ["direction"],
            "additionalProperties": False,
        },
    },
}

_DIRECTION_RE = re.compile(r"\b(UP|DOWN|LEFT|RIGHT)\b", re.IGNORECASE)


# ------------------------------------------------------------------
# Board creation
# ------------------------------------------------------------------


def new_board(seed: int) -> Board:
    """Create a fresh 4x4 board with two seeded initial tiles."""

    rng = random.Random(seed)
    board = _empty_board()
    board = spawn_tile(board, rng)
    board = spawn_tile(board, rng)
    return board


def _empty_board() -> Board:
    return [[EMPTY] * SIZE for _ in range(SIZE)]


# ------------------------------------------------------------------
# Core move logic
# ------------------------------------------------------------------


def move(board: Board, direction: str, rng: random.Random) -> tuple[Board, int, bool, bool]:
    """Execute one swipe.

    Returns ``(new_board, merge_score, valid, terminal)`` where *valid*
    indicates whether the board changed and *terminal* whether no further
    moves are possible after spawning the new tile.
    """

    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    slid, score = _slide(board, direction)
    valid = slid != board
    if valid:
        slid = spawn_tile(slid, rng)
    terminal = is_terminal(slid)
    return slid, score, valid, terminal


def _slide(board: Board, direction: str) -> tuple[Board, int]:
    """Apply swipe + merge without spawning a new tile."""

    grid = [list(row) for row in board]
    score = 0

    if direction in ("LEFT", "RIGHT"):
        for row in grid:
            if direction == "RIGHT":
                row.reverse()
            gained, merged = _slide_row_left(row)
            score += gained
            if direction == "RIGHT":
                row.reverse()

    else:
        transposed = _transpose(grid)
        for col in transposed:
            if direction == "DOWN":
                col.reverse()
            gained, _ = _slide_row_left(col)
            score += gained
            if direction == "DOWN":
                col.reverse()
        grid = _transpose(transposed)

    return grid, score


def _slide_row_left(row: list[int]) -> tuple[int, list[int]]:
    """Compress and merge a single row to the left in-place.

    Returns ``(merge_score, row)``.
    """

    tiles = [v for v in row if v != EMPTY]
    merged: list[int] = []
    score = 0
    skip = False
    for i, v in enumerate(tiles):
        if skip:
            skip = False
            continue
        if i + 1 < len(tiles) and tiles[i + 1] == v:
            merged.append(v * 2)
            score += v * 2
            skip = True
        else:
            merged.append(v)
    row[:] = merged + [EMPTY] * (SIZE - len(merged))
    return score, row


def spawn_tile(board: Board, rng: random.Random) -> Board:
    """Place a 2 (90%) or 4 (10%) in a random empty cell."""

    empties = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == EMPTY]
    if not empties:
        return [list(row) for row in board]
    r, c = rng.choice(empties)
    result = [list(row) for row in board]
    result[r][c] = 4 if rng.random() < 0.1 else 2
    return result


# ------------------------------------------------------------------
# Terminal / legality
# ------------------------------------------------------------------


def is_terminal(board: Board) -> bool:
    """Return True when no empty cells and no adjacent merges remain."""

    board = normalize_board(board)
    if any(board[r][c] == EMPTY for r in range(SIZE) for c in range(SIZE)):
        return False
    for r in range(SIZE):
        for c in range(SIZE):
            v = board[r][c]
            if c + 1 < SIZE and board[r][c + 1] == v:
                return False
            if r + 1 < SIZE and board[r + 1][c] == v:
                return False
    return True


def legal_directions(board: Board) -> list[str]:
    """Return directions that would change the board."""

    return [d for d in DIRECTIONS if _slide(board, d)[0] != board]


def random_action(board: Board, rng: random.Random) -> str:
    """Pick a random legal direction; fall back to any direction if none legal."""

    legal = legal_directions(board)
    if legal:
        return rng.choice(legal)
    return rng.choice(DIRECTIONS)


# ------------------------------------------------------------------
# Board queries
# ------------------------------------------------------------------


def max_tile(board: Board) -> int:
    return max(max(row) for row in normalize_board(board))


def empty_count(board: Board) -> int:
    return sum(1 for row in normalize_board(board) for v in row if v == EMPTY)


def total_score(board: Board) -> int:
    """Return the sum of all tile values minus the initial tile contributions.

    Not used for gameplay — kept for potential diagnostics.
    """

    return sum(v for row in normalize_board(board) for v in row if v != EMPTY)


# ------------------------------------------------------------------
# Rendering / prompt
# ------------------------------------------------------------------


def board_to_text(board: Board) -> str:
    """Render the board as a 4-line text grid; empty cells are dots."""

    lines = []
    for row in normalize_board(board):
        cells = [str(v) if v != EMPTY else "." for v in row]
        lines.append("  ".join(f"{c:>4}" for c in cells))
    return "\n".join(lines)


def format_prompt(board: Board) -> str:
    """Build a user-facing prompt with the current board and instructions."""

    return f"Current board:\n{board_to_text(board)}"


# ------------------------------------------------------------------
# Action parsing
# ------------------------------------------------------------------


def parse_action(text: str) -> str | None:
    """Extract a direction keyword from model text output."""

    if not text:
        return None
    match = _DIRECTION_RE.search(text)
    return match.group(1).upper() if match else None


# ------------------------------------------------------------------
# Episode scoring
# ------------------------------------------------------------------


def score_episode(
    total_merge_score: int,
    max_tile_value: int,
    valid_moves: int,
    invalid_moves: int,
) -> dict:
    """Compute structured episode metrics and a scalar reward in [-1, 1].

    The reward combines normalised merge score (50%), max-tile progression
    (30%), and an invalid-move penalty (20% base - penalty).
    """

    total_moves = valid_moves + invalid_moves
    invalid_rate = invalid_moves / total_moves if total_moves > 0 else 0.0
    score_component = min(total_merge_score / 2000.0, 1.0)
    tile_component = min(math.log2(max(max_tile_value, 2)) / 13.0, 1.0)
    penalty = 0.3 * invalid_rate
    reward = 0.5 * score_component + 0.3 * tile_component + 0.2 - penalty
    return {
        "reward": reward,
        "total_score": total_merge_score,
        "max_tile": max_tile_value,
        "invalid_rate": invalid_rate,
        "valid_moves": valid_moves,
        "invalid_moves": invalid_moves,
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def generate_board(max_tile: int, pattern: str, seed: int) -> Board:
    """Generate a mid-game board with the given max tile and layout pattern.

    Args:
        max_tile: The largest tile value on the board (64, 128, 256, or 512).
        pattern: "corner" (large tiles clustered in a corner) or "scattered".
        seed: Deterministic seed for tile placement.
    """
    rng = random.Random(seed)
    board = _empty_board()
    tiles = _tile_set(max_tile, rng)
    if pattern == "corner":
        _place_corner(board, tiles, rng)
    else:
        _place_scattered(board, tiles, rng)
    return board


def _tile_set(max_tile: int, rng: random.Random) -> list[int]:
    """Return a sorted list of tile values for a mid-game board."""
    # Approximate distribution: one max tile, a few mid tiles, several small ones
    tiles = [max_tile]
    val = max_tile // 2
    while val >= 2:
        count = rng.randint(1, 2)
        tiles.extend([val] * count)
        val //= 2
    # Fill remaining with 2s to reach 8-12 total tiles
    target = rng.randint(8, 12)
    while len(tiles) < target:
        tiles.append(2)
    # Trim if too many
    tiles = tiles[:target]
    tiles.sort(reverse=True)
    return tiles


def _place_corner(board: Board, tiles: list[int], rng: random.Random) -> None:
    """Place tiles with large values clustered toward bottom-left corner."""
    # Pick a corner: 0=bottom-left, 1=bottom-right, 2=top-left, 3=top-right
    corner = rng.randint(0, 3)
    corners = [(3, 0), (3, 3), (0, 0), (0, 3)]
    cr, cc = corners[corner]
    # Assign positions sorted by Manhattan distance from corner (closest first),
    # with row-major tie-breaking for deterministic placement.
    positions = sorted(
        [(r, c) for r in range(SIZE) for c in range(SIZE)],
        key=lambda rc: (abs(rc[0] - cr) + abs(rc[1] - cc), rc[0], rc[1]),
    )
    for i, tile in enumerate(tiles):
        r, c = positions[i]
        board[r][c] = tile


def _place_scattered(board: Board, tiles: list[int], rng: random.Random) -> None:
    """Place tiles at random positions on the board."""
    positions = [(r, c) for r in range(SIZE) for c in range(SIZE)]
    rng.shuffle(positions)
    for i, tile in enumerate(tiles):
        r, c = positions[i]
        board[r][c] = tile


def normalize_board(board: Iterable[Iterable[int]]) -> Board:
    """Return a deep-copied 4x4 list-of-lists board."""

    return [list(row) for row in board]


def _transpose(grid: list[list[int]]) -> list[list[int]]:
    return [list(col) for col in zip(*grid)]