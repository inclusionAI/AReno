"""Tic-tac-toe rules shared by the VL image example."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

Board = list[list[str]]
EMPTY = "."
PLAYERS = ("X", "O")


def normalize_board(board: Iterable[Iterable[str]]) -> Board:
    """Return a validated 3x3 board."""

    rows = [[str(cell).upper() if str(cell) != EMPTY else EMPTY for cell in row] for row in board]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("Tic-Tac-Toe board must be 3x3")
    allowed = {"X", "O", EMPTY}
    if any(cell not in allowed for row in rows for cell in row):
        raise ValueError("Tic-Tac-Toe cells must be X, O, or .")
    return rows


def legal_moves(board: Board) -> list[int]:
    """Return legal square numbers."""

    return [idx + 1 for idx, cell in enumerate(_flat(normalize_board(board))) if cell == EMPTY]


def board_to_text(board: Board) -> str:
    """Render the board with square numbers for empty cells."""

    rows = []
    for row_idx, row in enumerate(normalize_board(board)):
        cells = []
        for col_idx, cell in enumerate(row):
            square = row_idx * 3 + col_idx + 1
            cells.append(str(square) if cell == EMPTY else cell)
        rows.append(" ".join(cells))
    return "\n".join(rows)


def apply_move(board: Board, square: int, player: str = "X") -> Board:
    """Apply a legal move and return a new board."""

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


def next_player(board: Board) -> str:
    """Infer the next player from counts."""

    flat = _flat(normalize_board(board))
    return "X" if flat.count("X") <= flat.count("O") else "O"


def is_terminal(board: Board) -> bool:
    """Return whether the game is finished."""

    return winner(board) is not None or not legal_moves(board)


def best_moves(board: Board) -> list[int]:
    """Return optimal X moves using BFS over the reachable game tree."""

    board = normalize_board(board)
    moves = legal_moves(board)
    if not moves:
        return []
    scored = [(_bfs_value(apply_move(board, move, "X"), "O"), move) for move in moves]
    best = max(score for score, _move in scored)
    return [move for score, move in scored if score == best]


def score_move(board: Board, square: int | None) -> float:
    """Score one X move using the same policy as the agentic example."""

    if square is None:
        return -1.0
    try:
        next_board = apply_move(board, square, "X")
    except ValueError:
        return -1.0
    if winner(next_board) == "X":
        return 1.0
    return 0.8 if square in best_moves(board) else 0.0


def square_name(square: int) -> str:
    """Return a human-readable square name."""

    names = {
        1: "top-left",
        2: "top-middle",
        3: "top-right",
        4: "middle-left",
        5: "center",
        6: "middle-right",
        7: "bottom-left",
        8: "bottom-middle",
        9: "bottom-right",
    }
    return names[int(square)]


def _bfs_value(board: Board, player: str) -> int:
    root = (_to_key(board), player)
    queue = deque([root])
    parents: dict[tuple[tuple[str, ...], str], list[tuple[tuple[str, ...], str]]] = {}
    children_left: dict[tuple[tuple[str, ...], str], int] = {}
    values: dict[tuple[tuple[str, ...], str], int] = {}

    while queue:
        node = queue.popleft()
        key, turn = node
        node_board = _from_key(key)
        won = winner(node_board)
        if won == "X":
            values[node] = 1
            continue
        if won == "O":
            values[node] = -1
            continue
        moves = legal_moves(node_board)
        if not moves:
            values[node] = 0
            continue
        next_turn = "O" if turn == "X" else "X"
        child_nodes = [(_to_key(apply_move(node_board, move, turn)), next_turn) for move in moves]
        children_left[node] = len(child_nodes)
        for child in child_nodes:
            parents.setdefault(child, []).append(node)
            queue.append(child)

    pending_values: dict[tuple[tuple[str, ...], str], list[int]] = {}
    ready = deque(values)
    while ready:
        child = ready.popleft()
        for parent in parents.get(child, []):
            pending_values.setdefault(parent, []).append(values[child])
            children_left[parent] -= 1
            if children_left[parent] == 0:
                parent_turn = parent[1]
                child_values = pending_values[parent]
                values[parent] = max(child_values) if parent_turn == "X" else min(child_values)
                ready.append(parent)
    return values[root]


def _flat(board: Board) -> list[str]:
    return [cell for row in board for cell in row]


def _to_key(board: Board) -> tuple[str, ...]:
    return tuple(_flat(normalize_board(board)))


def _from_key(key: tuple[str, ...]) -> Board:
    return [list(key[idx : idx + 3]) for idx in range(0, 9, 3)]
