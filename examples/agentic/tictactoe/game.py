"""Small Tic-Tac-Toe helpers for agentic examples."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import cache

Board = list[list[str]]
PLAYERS = ("X", "O")
EMPTY = "."

_XML_MOVE_RE = re.compile(r"<move>\s*([1-9])\s*</move>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_CHAT_SPECIAL_RE = re.compile(r"<\|[^>]+?\|>|</?s>", re.IGNORECASE)


def normalize_board(board: Iterable[Iterable[str]]) -> Board:
    """Return a validated 3x3 Tic-Tac-Toe board."""

    rows = [[str(cell).upper() if str(cell) != EMPTY else EMPTY for cell in row] for row in board]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("Tic-Tac-Toe board must be 3x3")
    allowed = {"X", "O", EMPTY}
    if any(cell not in allowed for row in rows for cell in row):
        raise ValueError("Tic-Tac-Toe cells must be X, O, or .")
    return rows


def board_to_text(board: Board) -> str:
    """Render the board with square numbers for empty cells."""

    rows = []
    for row_idx, row in enumerate(board):
        cells = []
        for col_idx, cell in enumerate(row):
            square = row_idx * 3 + col_idx + 1
            cells.append(str(square) if cell == EMPTY else cell)
        rows.append(" ".join(cells))
    return "\n".join(rows)


def format_prompt(board: Board) -> str:
    """Build the one-step prompt for the tool-call agent."""

    return (
        "You are playing Tic-Tac-Toe as X. Choose the best next square.\n"
        "Empty squares are numbered 1 through 9. Call the choose_square tool with the selected square.\n\n"
        f"Board:\n{board_to_text(board)}\n\nMove:"
    )


def format_xml_prompt(board: Board) -> str:
    """Build the one-step prompt for the XML no-tool agent."""

    return (
        "You are playing Tic-Tac-Toe as X. Choose the best next square.\n"
        "Empty squares are numbered 1 through 9.\n"
        "Answer with exactly one XML tag such as <move>5</move>.\n\n"
        f"Board:\n{board_to_text(board)}\n\nMove:"
    )


def parse_xml_move(text: str) -> int | None:
    """Extract the final XML square from a model response."""

    text = strip_chat_special_tokens(strip_think_tags(text)).strip()
    matches = list(_XML_MOVE_RE.finditer(text))
    if not matches:
        return None
    return int(matches[-1].group(1))


def strip_think_tags(text: str) -> str:
    """Remove reasoning spans before parsing the policy action."""

    return _THINK_RE.sub(" ", text)


def strip_chat_special_tokens(text: str) -> str:
    """Remove chat-template sentinels that may trail generated text."""

    return _CHAT_SPECIAL_RE.sub(" ", text)


def legal_moves(board: Board) -> list[int]:
    """Return legal square numbers."""

    board = normalize_board(board)
    return [idx + 1 for idx, cell in enumerate(_flat(board)) if cell == EMPTY]


def apply_move(board: Board, square: int, player: str = "X") -> Board:
    """Apply a move and return a new board."""

    board = normalize_board(board)
    player = player.upper()
    if player not in PLAYERS:
        raise ValueError("player must be X or O")
    if square not in legal_moves(board):
        raise ValueError(f"illegal Tic-Tac-Toe move: {square}")
    row, col = divmod(square - 1, 3)
    next_board = [list(row_values) for row_values in board]
    next_board[row][col] = player
    return next_board


def winner(board: Board) -> str | None:
    """Return X/O winner or None."""

    board = normalize_board(board)
    lines = []
    lines.extend(board)
    lines.extend([[board[0][col], board[1][col], board[2][col]] for col in range(3)])
    lines.append([board[0][0], board[1][1], board[2][2]])
    lines.append([board[0][2], board[1][1], board[2][0]])
    for line in lines:
        if line[0] != EMPTY and line[0] == line[1] == line[2]:
            return line[0]
    return None


def best_moves(board: Board) -> list[int]:
    """Return minimax-optimal moves for X."""

    moves = legal_moves(board)
    if not moves:
        return []
    scored = [(_minimax(_to_key(apply_move(board, move, "X")), "O"), move) for move in moves]
    best = max(score for score, _move in scored)
    return [move for score, move in scored if score == best]


def score_move(board: Board, square: int | None) -> float:
    """Score one X move."""

    if square is None:
        return -1.0
    try:
        next_board = apply_move(board, square, "X")
    except ValueError:
        return -1.0
    if winner(next_board) == "X":
        return 1.0
    return 0.8 if square in best_moves(board) else 0.0


def next_player(board: Board) -> str:
    """Infer the next player from counts."""

    flat = _flat(normalize_board(board))
    return "X" if flat.count("X") <= flat.count("O") else "O"


def is_terminal(board: Board) -> bool:
    """Return whether the game is finished."""

    return winner(board) is not None or not legal_moves(board)


@cache
def _minimax(key: tuple[str, ...], player: str) -> int:
    board = _from_key(key)
    won = winner(board)
    if won == "X":
        return 1
    if won == "O":
        return -1
    moves = legal_moves(board)
    if not moves:
        return 0
    next_player_value = "O" if player == "X" else "X"
    scores = [_minimax(_to_key(apply_move(board, move, player)), next_player_value) for move in moves]
    return max(scores) if player == "X" else min(scores)


def _flat(board: Board) -> list[str]:
    return [cell for row in board for cell in row]


def _to_key(board: Board) -> tuple[str, ...]:
    return tuple(_flat(normalize_board(board)))


def _from_key(key: tuple[str, ...]) -> Board:
    return [list(key[idx : idx + 3]) for idx in range(0, 9, 3)]
