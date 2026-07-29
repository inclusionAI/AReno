"""Reward function for the 2048 agentic example.

Replays the agent's move sequence on the seeded board and returns the
final merge score plus per-step monotonicity and empty-cell bonuses.
Invalid moves are penalised. Directions are parsed from model response
text (no tool calls).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

MONOTONICITY_LAMBDA = 0.01
EMPTY_LAMBDA = 0.02
INVALID_MOVE_PENALTY = -1.0


def reward_fn(record) -> float:
    """Replay the agent's moves and return the shaped episode reward.

    Reward = final merge score
           + Σ(per-step monotonicity bonus)
           + Σ(per-step empty-cell bonus)
           + invalid-move penalties
    """

    source = record.source_record
    board = game.normalize_board(source["board"])
    seed = int(source["seed"])
    max_moves = int(source.get("max_moves", game.DEFAULT_MAX_MOVES))

    directions = _extract_directions(record)

    rng = random.Random(seed)
    total = 0.0
    total_score = 0.0
    move_count = 0
    terminal = False

    for direction in directions:
        if move_count >= max_moves:
            break
        board, score, valid, terminal = game.move(board, direction, rng)
        if valid:
            total += _monotonicity_bonus(board)
            total += EMPTY_LAMBDA * game.empty_count(board)
            total_score += score
        else:
            total += INVALID_MOVE_PENALTY
        move_count += 1
        if terminal:
            break

    return total + total_score


def _monotonicity_bonus(board) -> float:
    """Reward rows/cols where values decrease monotonically toward a corner."""
    best = 0.0
    for corner_row, corner_col, row_dir, col_dir in [
        (0, 0, 1, 1),     # top-left
        (0, 3, 1, -1),    # top-right
        (3, 0, -1, 1),    # bottom-left
        (3, 3, -1, -1),   # bottom-right
    ]:
        score = 0.0
        for i in range(game.SIZE):
            for j in range(game.SIZE):
                r = corner_row + i * row_dir
                c = corner_col + j * col_dir
                if r < 0 or r >= game.SIZE or c < 0 or c >= game.SIZE:
                    continue
                if board[r][c] == game.EMPTY:
                    continue
                if i > 0:
                    prev_r = corner_row + (i - 1) * row_dir
                    if board[prev_r][c] >= board[r][c]:
                        score += MONOTONICITY_LAMBDA
                if j > 0:
                    prev_c = corner_col + (j - 1) * col_dir
                    if board[r][prev_c] >= board[r][c]:
                        score += MONOTONICITY_LAMBDA
        if score > best:
            best = score
    return best


def _extract_directions(record) -> list[str]:
    """Parse move directions from the model's response text.

    In single-turn mode, each step produces an independent response.
    Multiple steps from one episode are concatenated in record.completion
    (separated by newlines). We parse each direction keyword in order.
    """

    text = record.completion or ""
    directions: list[str] = []
    for line in text.split("\n"):
        direction = game.parse_action(line)
        if direction:
            directions.append(direction)
    return directions