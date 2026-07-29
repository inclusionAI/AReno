"""Reward function for the 2048 agentic example.

Replays the agent's move sequence on the seeded board and returns the
final merge score (end-of-game total) plus a per-step monotonicity bonus
that encourages corner-locking structure. Invalid moves are penalised.
"""

from __future__ import annotations

import json
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

    rng = random.Random(seed)
    total = 0.0
    move_count = 0
    terminal = False

    for direction in _extract_directions(record):
        if move_count >= max_moves:
            break
        board, score, valid, terminal = game.move(board, direction, rng)
        if valid:
            total += _monotonicity_bonus(board)
            total += EMPTY_LAMBDA * game.empty_count(board)
        else:
            total += INVALID_MOVE_PENALTY
        move_count += 1
        if terminal:
            break

    # Add the final merge score (replay from scratch to get it)
    final_score = _replay_final_score(source, record)
    return total + final_score


def _replay_final_score(source: dict, record) -> float:
    """Replay the full episode and return the total merge score."""
    board = game.normalize_board(source["board"])
    seed = int(source["seed"])
    max_moves = int(source.get("max_moves", game.DEFAULT_MAX_MOVES))
    rng = random.Random(seed)

    total_score = 0.0
    move_count = 0
    for direction in _extract_directions(record):
        if move_count >= max_moves:
            break
        board, score, valid, terminal = game.move(board, direction, rng)
        if valid:
            total_score += score
        move_count += 1
        if terminal:
            break
    return total_score


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
    """Pull move directions from the agent's tool calls, in order."""

    directions = []
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "move":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(arguments, dict):
            direction = arguments.get("direction")
            if isinstance(direction, str) and direction.upper() in game.DIRECTIONS:
                directions.append(direction.upper())
    return directions