"""Small 6x6 Othello helpers for agentic examples.

Pure-Python rules: legal-move enumeration, placement with 8-direction line
flipping, forced-pass handling, two-consecutive-pass terminal detection, and
terminal scoring. No AReno imports, no registration.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

SIZE = 6
Board = list[list[str]]
PLAYERS = ("B", "W")
EMPTY = "."

# All eight line directions (dr, dc).
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_XML_MOVE_RE = re.compile(r"<move>\s*(\d{1,2})\s*,\s*(\d{1,2})\s*</move>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_CHAT_SPECIAL_RE = re.compile(r"<\|[^>]+?\|>|</?s>", re.IGNORECASE)


def new_board() -> Board:
    """Return the standard 6x6 Othello opening position (4 center discs)."""

    board = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    mid = SIZE // 2
    board[mid - 1][mid - 1] = "W"
    board[mid - 1][mid] = "B"
    board[mid][mid - 1] = "B"
    board[mid][mid] = "W"
    return board


def normalize_board(board: Iterable[Iterable[str]]) -> Board:
    """Return a validated 6x6 Othello board."""

    rows = [[str(cell).upper() if str(cell) != EMPTY else EMPTY for cell in row] for row in board]
    if len(rows) != SIZE or any(len(row) != SIZE for row in rows):
        raise ValueError(f"Othello board must be {SIZE}x{SIZE}")
    allowed = {"B", "W", EMPTY}
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell not in allowed:
                raise ValueError(f"Othello cells must be B, W, or .; got {cell!r} at ({r},{c})")
    return rows


def opponent(player: str) -> str:
    """Return the other player."""

    player = player.upper()
    if player == "B":
        return "W"
    if player == "W":
        return "B"
    raise ValueError(f"player must be B or W, got {player!r}")


def count_disks(board: Board) -> dict[str, int]:
    """Return {"B": n, "W": m, ".": empty} counts."""

    board = normalize_board(board)
    counts = {"B": 0, "W": 0, EMPTY: 0}
    for row in board:
        for cell in row:
            counts[cell] += 1
    return counts


def _in_bounds(row: int, col: int) -> bool:
    return 0 <= row < SIZE and 0 <= col < SIZE


def flips_for(board: Board, row: int, col: int, player: str) -> list[tuple[int, int]]:
    """Return the opponent disks that would flip if ``player`` plays (row, col).

    Empty (and the move is illegal) if the cell is non-empty, out of bounds, or
    no direction sandwiches opponent discs.
    """

    board = normalize_board(board)
    player = player.upper()
    if player not in PLAYERS:
        raise ValueError(f"player must be B or W, got {player!r}")
    if not _in_bounds(row, col):
        return []
    if board[row][col] != EMPTY:
        return []
    other = opponent(player)
    flipped: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        line: list[tuple[int, int]] = []
        r, c = row + dr, col + dc
        while _in_bounds(r, c) and board[r][c] == other:
            line.append((r, c))
            r += dr
            c += dc
        # A flip closes the bracket only if the far end is the mover's own disc.
        if line and _in_bounds(r, c) and board[r][c] == player:
            flipped.extend(line)
    return flipped


def legal_moves(board: Board, player: str) -> list[tuple[int, int]]:
    """Return all legal (row, col) moves for ``player``. Empty list forces a pass."""

    board = normalize_board(board)
    player = player.upper()
    moves = [
        (r, c)
        for r in range(SIZE)
        for c in range(SIZE)
        if board[r][c] == EMPTY and flips_for(board, r, c, player)
    ]
    return moves


def has_legal_move(board: Board, player: str) -> bool:
    """Cheap check for whether ``player`` has any legal move."""

    return bool(legal_moves(board, player))


def apply_move(board: Board, row: int, col: int, player: str) -> Board:
    """Apply a legal move, flipping all captured discs. Returns a new board."""

    board = normalize_board(board)
    player = player.upper()
    if player not in PLAYERS:
        raise ValueError(f"player must be B or W, got {player!r}")
    if not _in_bounds(row, col):
        raise ValueError(f"Othello move out of bounds: ({row},{col})")
    if (row, col) not in legal_moves(board, player):
        raise ValueError(f"illegal Othello move: ({row},{col}) for {player}")
    flipped = flips_for(board, row, col, player)
    next_board = [list(row_values) for row_values in board]
    next_board[row][col] = player
    for r, c in flipped:
        next_board[r][c] = player
    return next_board


def is_terminal(board: Board) -> bool:
    """Terminal when neither player has a legal move (two consecutive passes)."""

    board = normalize_board(board)
    return not has_legal_move(board, "B") and not has_legal_move(board, "W")


def score_board(board: Board) -> dict[str, Any]:
    """Return terminal counts and winner (count-majority, draw on tie)."""

    counts = count_disks(board)
    black = counts["B"]
    white = counts["W"]
    if black > white:
        winner = "B"
    elif white > black:
        winner = "W"
    else:
        winner = "draw"
    return {"black": black, "white": white, "winner": winner}


def next_player(board: Board) -> str | None:
    """Infer the side to move from disc counts (Black moves first).

    Returns ``None`` when the counts are not consistent with a reachable game
    (e.g. equal counts) - the caller is then expected to track turns explicitly.
    """

    counts = count_disks(board)
    black = counts["B"]
    white = counts["W"]
    total = black + white
    if total % 2 == 0:
        # After an even number of discs it is Black's turn in a normal game.
        return "B" if black == white else None
    # Odd disc count (after a pass imbalance) generally means White to move.
    return "W" if white == black - 1 else None


def strip_think_tags(text: str) -> str:
    """Remove reasoning spans before parsing the policy action."""

    return _THINK_RE.sub(" ", text)


def strip_chat_special_tokens(text: str) -> str:
    """Remove chat-template sentinels that may trail generated text."""

    return _CHAT_SPECIAL_RE.sub(" ", text)


def parse_move(text: str) -> tuple[int, int] | None:
    """Extract the final XML ``<move>r,c</move>`` from a model response.

    Returns ``None`` on any malformed input (never raises).
    """

    text = strip_chat_special_tokens(strip_think_tags(text)).strip()
    matches = list(_XML_MOVE_RE.finditer(text))
    if not matches:
        return None
    row = int(matches[-1].group(1))
    col = int(matches[-1].group(2))
    if not _in_bounds(row, col):
        return None
    return row, col


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_tool_move(tool_calls: list[dict]) -> tuple[int, int] | None:
    """Extract the (row, col) from ``choose_move`` tool calls.

    Accepts ``arguments`` as a dict or JSON string with ``row``/``col``. Returns
    ``None`` on any malformed or missing tool call (never raises).
    """

    parsed = parse_tool_move_raw(tool_calls)
    if parsed is None:
        return None
    row, col = parsed
    if not _in_bounds(row, col):
        return None
    return row, col


def parse_tool_move_raw(tool_calls: list[dict]) -> tuple[int, int] | None:
    """Like :func:`parse_tool_move` but keeps out-of-bounds coordinates.

    Returns the raw ``(row, col)`` for the first ``choose_move`` call whose
    ``arguments`` parse to two integers, even when the coordinates are out of
    range. Used by the reward function to distinguish "the model emitted a
    ``choose_move`` tool call with a bad coordinate" from "the model emitted no
    ``choose_move`` call at all" — the two cases need different reward tiers so
    that penalizing an illegal move does not also suppress tool-calling behavior
    and lock the policy into producing no tool calls (a cold-start collapse).
    Returns ``None`` when no ``choose_move`` call is present, its arguments are
    not a dict/JSON, or the coordinates are not integers. Never raises.
    """

    import json

    for call in tool_calls or []:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "choose_move":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        row = _coerce_int(arguments.get("row"))
        col = _coerce_int(arguments.get("col"))
        if row is None or col is None:
            return None
        return row, col
    return None


def board_to_text(board: Board) -> str:
    """Render the board; empty cells show their ``(row,col)`` label."""

    board = normalize_board(board)
    rows = []
    for r, row in enumerate(board):
        cells = [f"({r},{c})" if cell == EMPTY else cell for c, cell in enumerate(row)]
        rows.append(" ".join(f"{cell:>5}" for cell in cells))
    return "\n".join(rows)


def format_prompt(board: Board, player: str) -> str:
    """Build the one-step prompt for the tool-call agent."""

    player = player.upper()
    moves = legal_moves(board, player)
    moves_text = ", ".join(f"({r},{c})" for r, c in moves) if moves else "pass (no legal move)"
    side = "Black (B)" if player == "B" else "White (W)"
    return (
        f"You are playing 6x6 Othello as {side}. Choose the best legal next move.\n\n"
        "Rules:\n"
        "- B and W are already-placed discs.\n"
        "- Labels like (row,col) on empty cells are coordinate references, not discs.\n"
        "- A legal move must flank one or more opponent discs in a straight line "
        "(horizontal, vertical, or diagonal) between the new disc and another of yours.\n"
        "- Legal moves flip the flanked opponent discs to your color.\n"
        "- Choose exactly one legal move by calling the choose_move tool with row and col.\n"
        "- Avoid passing (a pass only happens when you have no legal move).\n\n"
        f"Legal moves for you: {moves_text}\n\n"
        f"Board:\n{board_to_text(board)}\n\nMove:"
    )


def format_xml_prompt(board: Board, player: str) -> str:
    """Build the one-step prompt for the XML no-tool agent."""

    player = player.upper()
    moves = legal_moves(board, player)
    moves_text = ", ".join(f"({r},{c})" for r, c in moves) if moves else "pass (no legal move)"
    side = "Black (B)" if player == "B" else "White (W)"
    return (
        f"You are playing 6x6 Othello as {side}. Choose the best legal next move.\n\n"
        "Rules:\n"
        "- B and W are already-placed discs.\n"
        "- Labels like (row,col) on empty cells are coordinate references, not discs.\n"
        "- A legal move must flank opponent discs in a straight line between the new "
        "disc and another of yours; the flanked discs flip to your color.\n"
        "- Choose only one of the legal moves shown on the board.\n"
        "- Answer with exactly one XML tag such as <move>2,3</move>.\n\n"
        f"Legal moves for you: {moves_text}\n\n"
        f"Board:\n{board_to_text(board)}\n\nMove:"
    )


def score_move(board: Board, move: tuple[int, int] | None, player: str) -> float:
    """Score one move using a *tiered* reward kernel.

    The tiers are deliberately distinct so that, in a GSPO group, samples with
    different outcomes get different advantages and thus a non-zero gradient.
    Critically, the tiers separate *what the model did*:

    - ``None`` (no ``choose_move`` tool call was emitted, or its arguments
      could not be parsed) -> ``-1.0``. This is the worst tier: the model did
      not even produce an actionable tool call.
    - The coordinates were emitted but out of range -> ``-0.5``. Still an
      illegal action, but the model at least produced a ``choose_move`` call
      with integers, so it is better than emitting nothing.
    - The coordinates are in range but the cell is not a legal Othello move
      (occupied, or no flanked line) -> ``-0.3``. Better than out-of-range:
      the model targeted a real cell.
    - A legal, non-terminal move -> ``+0.4``. The model took a real, legal,
      in-play action.
    - A legal move that ends the game leaving ``player`` ahead -> ``+1.0``.

    Separating the illegal tiers from the ``None`` tier is what prevents the
    cold-start collapse seen with a flat ``-1.0`` penalty: when every illegal
    action and every absent tool call share the same reward, the group-relative
    advantage for "called the tool with a bad cell" equals that of "called no
    tool", so the gradient that suppresses bad cells also suppresses tool
    calling itself, driving ``tool_calls`` to zero and freezing reward at
    ``-1.0`` with no gradient to recover. The reward function never raises.
    """

    player = player.upper()
    if move is None:
        return -1.0
    row, col = move
    if player not in PLAYERS:
        return -1.0
    if not _in_bounds(row, col):
        return -0.5
    board = normalize_board(board)
    if board[row][col] != EMPTY or not flips_for(board, row, col, player):
        return -0.3
    next_board = apply_move(board, row, col, player)
    if not is_terminal(next_board):
        return 0.4
    result = score_board(next_board)
    if result["winner"] == player:
        return 1.0
    return 0.0