"""Small 6x6 Othello helpers for agentic examples."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

Board = list[list[str]]
SIZE = 6
PLAYERS = ("B", "W")
EMPTY = "."
DIRECTIONS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def normalize_board(board: Iterable[Iterable[str]]) -> Board:
    """Return a validated 6x6 Othello board."""

    rows = [[str(cell).upper() if str(cell) != EMPTY else EMPTY for cell in row] for row in board]
    if len(rows) != SIZE or any(len(row) != SIZE for row in rows):
        raise ValueError(f"Othello board must be {SIZE}x{SIZE}")
    allowed = {"B", "W", EMPTY}
    if any(cell not in allowed for row in rows for cell in row):
        raise ValueError("Othello cells must be B, W, or .")
    return rows


def initial_board() -> Board:
    """Return the standard 6x6 Othello starting position."""

    board = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    mid = SIZE // 2
    board[mid - 1][mid - 1] = "W"
    board[mid - 1][mid] = "B"
    board[mid][mid - 1] = "B"
    board[mid][mid] = "W"
    return board


def board_to_text(board: Board) -> str:
    """Render the board as 6 rows of characters."""

    board = normalize_board(board)
    return "\n".join(" ".join(row) for row in board)


def format_prompt(board: Board, player: str = "B") -> str:
    """Build the one-step prompt for the tool-call agent."""

    player = player.upper()
    opponent = "W" if player == "B" else "B"
    moves = legal_moves(board, player)
    move_str = ", ".join(f"({r},{c})" for r, c in moves) if moves else "(none - must pass)"
    return (
        f"You are playing 6x6 Othello as {player} ({'Black' if player == 'B' else 'White'}). "
        "Choose the best legal next move.\n\n"
        "Rules:\n"
        f"- {player} and {opponent} are already-placed discs; . is empty.\n"
        "- A legal move must flank at least one opponent disc in a straight line "
        "(horizontal, vertical, or diagonal), with your disc at the far end.\n"
        "- All flanked opponent discs flip to your colour.\n"
        "- If you have no legal move you must pass.\n"
        "- The game ends when neither player can move; the player with more discs wins.\n\n"
        f"Legal moves for {player}: {move_str}\n\n"
        f"Board:\n{board_to_text(board)}\n\n"
        f"Call the choose_move tool with row and col (0-indexed, 0-5) to place {player}."
    )


def _flips_for_move(board: Board, row: int, col: int, player: str) -> list[tuple[int, int]]:
    """Return all discs that would flip if *player* places at (row, col).

    Returns an empty list if the cell is occupied or no discs would flip.
    """

    board = normalize_board(board)
    player = player.upper()
    if board[row][col] != EMPTY:
        return []
    opponent = "W" if player == "B" else "B"
    all_flips: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        flips: list[tuple[int, int]] = []
        r, c = row + dr, col + dc
        while 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == opponent:
            flips.append((r, c))
            r += dr
            c += dc
        if flips and 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == player:
            all_flips.extend(flips)
    return all_flips


def legal_moves(board: Board, player: str) -> list[tuple[int, int]]:
    """Return all legal moves for *player* as (row, col) tuples."""

    board = normalize_board(board)
    player = player.upper()
    if player not in PLAYERS:
        raise ValueError("player must be B or W")
    return [
        (r, c)
        for r in range(SIZE)
        for c in range(SIZE)
        if board[r][c] == EMPTY and _flips_for_move(board, r, c, player)
    ]


def has_legal_move(board: Board, player: str) -> bool:
    """Return whether *player* has at least one legal move."""

    return len(legal_moves(board, player)) > 0


def apply_move(board: Board, row: int, col: int, player: str) -> Board:
    """Apply a move and return a new board with discs flipped.

    Raises ValueError if the move is illegal.
    """

    board = normalize_board(board)
    player = player.upper()
    if player not in PLAYERS:
        raise ValueError("player must be B or W")
    if not (0 <= row < SIZE and 0 <= col < SIZE):
        raise ValueError(f"move out of bounds: ({row}, {col})")
    flips = _flips_for_move(board, row, col, player)
    if not flips:
        raise ValueError(f"illegal Othello move for {player}: ({row}, {col})")
    next_board = [list(row_values) for row_values in board]
    next_board[row][col] = player
    for fr, fc in flips:
        next_board[fr][fc] = player
    return next_board


def next_player(board: Board) -> str | None:
    """Infer the next player from disc counts.

    Returns "B" when black disc count <= white disc count (black moves first
    and alternates).  Returns None when the board is full.
    """

    board = normalize_board(board)
    flat = _flat(board)
    if EMPTY not in flat:
        return None
    return "B" if flat.count("B") <= flat.count("W") else "W"


def is_terminal(board: Board) -> bool:
    """Return whether the game is finished.

    The game ends when the board is full or neither player has a legal move.
    """

    board = normalize_board(board)
    flat = _flat(board)
    if EMPTY not in flat:
        return True
    return not has_legal_move(board, "B") and not has_legal_move(board, "W")


def score(board: Board) -> dict[str, int]:
    """Count discs for each player."""

    board = normalize_board(board)
    flat = _flat(board)
    return {"B": flat.count("B"), "W": flat.count("W")}


def winner(board: Board) -> str | None:
    """Return "B" or "W" if the game has a winner, None for a tie or ongoing game."""

    board = normalize_board(board)
    if not is_terminal(board):
        return None
    counts = score(board)
    if counts["B"] > counts["W"]:
        return "B"
    if counts["W"] > counts["B"]:
        return "W"
    return None


def score_move(board: Board, row: int | None, col: int | None, player: str = "B") -> float:
    """Score one move for *player*.

    Returns -1.0 for an illegal move, 0.0 for a legal non-terminal move, and
    1.0 if the move results in a terminal board where *player* has more discs.
    """

    if row is None or col is None:
        return -1.0
    try:
        next_board = apply_move(board, row, col, player)
    except ValueError:
        return -1.0
    if is_terminal(next_board):
        counts = score(next_board)
        player_count = counts[player.upper()]
        opponent = "W" if player.upper() == "B" else "B"
        return 1.0 if player_count > counts[opponent] else 0.0
    return 0.0


def play_episode(
    board: Board,
    policy_fn: Callable[[Board, str], tuple[int, int] | None],
    opponent_fn: Callable[[Board, str], tuple[int, int] | None],
    *,
    first_player: str = "B",
    max_moves: int = 40,
) -> dict:
    """Play a full episode alternating *policy_fn* and *opponent_fn*.

    Handles forced passes and double-pass termination.  Returns a dict with
    the final board, winner, move history, and statistics.
    """

    board = normalize_board(board)
    history: list[dict] = []
    current = first_player.upper()
    passes = 0
    illegal_moves = 0
    total_moves = 0

    for _step in range(max_moves):
        if is_terminal(board):
            break
        if not has_legal_move(board, current):
            passes += 1
            history.append({"player": current, "move": None, "pass": True})
            current = "W" if current == "B" else "B"
            if passes >= 2:
                break
            continue
        passes = 0
        fn = policy_fn if current == first_player.upper() else opponent_fn
        move = fn(board, current)
        total_moves += 1
        if move is None:
            illegal_moves += 1
            current = "W" if current == "B" else "B"
            continue
        try:
            row, col = move
        except (TypeError, ValueError):
            illegal_moves += 1
            current = "W" if current == "B" else "B"
            continue
        flips = _flips_for_move(board, row, col, current)
        if not flips:
            illegal_moves += 1
            current = "W" if current == "B" else "B"
            continue
        board = apply_move(board, row, col, current)
        history.append({"player": current, "move": (row, col), "pass": False})
        current = "W" if current == "B" else "B"

    return {
        "board": board,
        "winner": winner(board),
        "score": score(board),
        "history": history,
        "illegal_moves": illegal_moves,
        "total_moves": total_moves,
    }


def random_policy(rng: random.Random) -> Callable[[Board, str], tuple[int, int] | None]:
    """Return a policy function that picks a random legal move using *rng*."""

    def _policy(board: Board, player: str) -> tuple[int, int] | None:
        moves = legal_moves(board, player)
        if not moves:
            return None
        return rng.choice(moves)

    return _policy


def _flat(board: Board) -> list[str]:
    return [cell for row in board for cell in row]
