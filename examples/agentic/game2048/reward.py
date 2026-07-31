"""Reward function for the 2048 agentic example.

Replays the agent's move sequence on the seeded board and computes a
normalised reward from merge score, empty-cell ratio, snake-layout
alignment, and invalid-move ratio.  Directions are parsed from tool
calls with a text fallback.

Reward formula:
    reward = 0.65 * merge_reward       # log2 compressed merge score gain
           + 0.20 * empty_reward        # average empty-cell ratio during episode
           + 0.15 * layout_reward       # snake-layout positional alignment
           - 0.80 * invalid_rate        # fraction of illegal moves
           - 0.40 if died_early         # terminal penalty for early game-over
    Final output clipped to [-1.0, 1.0].

The snake-layout heuristic encourages the model to keep large tiles in
corner-adjacent cells following the classic "snake" pattern, which is a
well-known sub-optimal but effective strategy for 2048.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


# ------------------------------------------------------------------
# Snake-layout weight matrices (8 orientations)
#
# The base matrix encodes a snake pattern: the highest weight (1.0) is
# at the top-left corner, decreasing along the first row, then continuing
# in a serpentine path through the board.  Large tiles should occupy
# high-weight cells for optimal play.
#
# We generate 8 orientations (4 rotations of the base + 4 rotations of
# its transpose) so the layout score is rotation/reflection invariant —
# the model is rewarded for ANY corner-anchored snake pattern, not just
# one specific orientation.
# ------------------------------------------------------------------

_BASE_WEIGHTS = np.array(
    [
        [1.00, 0.90, 0.80, 0.70],
        [0.40, 0.50, 0.60, 0.65],
        [0.35, 0.30, 0.25, 0.20],
        [0.05, 0.08, 0.12, 0.16],
    ],
    dtype=np.float32,
)

# 4 rotations of base + 4 rotations of transpose = 8 total orientations
_POSITION_WEIGHTS = [np.rot90(_BASE_WEIGHTS, k) for k in range(4)] + [
    np.rot90(_BASE_WEIGHTS.T, k) for k in range(4)
]


def _layout_score(board) -> float:
    """Return how well *board* matches the best snake layout (0~1).

    Tile values are log2-transformed (2→1, 4→2, 8→3, ...) so that larger
    tiles dominate the positional score.  The transformed levels are
    L1-normalised, then dot-producted with each of the 8 weight matrices;
    the maximum across all orientations is returned.
    """

    arr = np.asarray(board, dtype=np.float32)
    occupied = arr > 0
    levels = np.zeros_like(arr)
    levels[occupied] = np.log2(arr[occupied])  # tile 2→1, 4→2, 8→3, ...

    total = levels.sum()
    if total == 0:
        return 0.0

    levels /= total  # L1 normalise so score is scale-invariant
    # Pick the best-matching orientation
    return max(float(np.sum(levels * w)) for w in _POSITION_WEIGHTS)


# ------------------------------------------------------------------
# Reward computation
# ------------------------------------------------------------------


def reward_fn(record) -> float:
    """Replay the agent's moves and return a normalised episode reward.

    The function (1) replays all parsed directions on a fresh seeded board,
    (2) collects per-step statistics (score gain, empty cells, invalid
    moves, consecutive invalids, terminal state), then (3) combines four
    normalised sub-scores into a single reward clipped to [-1, 1].
    """

    source = record.source_record
    board = game.normalize_board(source["board"])
    seed = int(source["seed"])
    max_moves = min(
        max(int(source.get("max_moves", game.DEFAULT_MAX_MOVES)), 1),
        game.DEFAULT_MAX_MOVES,
    )

    directions = _extract_directions(record)

    # --- Phase 1: replay agent moves on the seeded board ---
    rng = random.Random(seed)
    score_gain = 0
    invalid_count = 0
    empty_history: list[int] = []
    move_count = 0
    consecutive_invalid = 0
    terminal = False

    for direction in directions:
        if move_count >= max_moves:
            break
        board, score, valid, terminal = game.move(board, direction, rng)
        empty_history.append(game.empty_count(board))
        if valid:
            score_gain += score
            consecutive_invalid = 0
        else:
            invalid_count += 1
            consecutive_invalid += 1
        move_count += 1

        # Stop early if the agent is stuck repeating invalid moves
        if consecutive_invalid >= 3:
            break
        if terminal:
            break

    # Died before using all allowed moves — penalise early game-over
    died_early = terminal and move_count < max_moves

    # --- Phase 2: compute normalised sub-scores (each in [0, 1]) ---

    # Merge reward: log2 compression prevents large merges from dominating.
    # A score gain of 4095 (merging up to 2048) maps to 1.0.
    merge_reward = float(
        np.clip(np.log2(1 + max(score_gain, 0)) / 12.0, 0.0, 1.0)
    )

    # Empty reward: average empty-cell count / 16, encourages keeping space open
    if empty_history:
        empty_reward = float(
            np.clip(np.mean(empty_history) / 16.0, 0.0, 1.0)
        )
    else:
        empty_reward = 0.0

    # Layout reward: snake-pattern alignment on the final board state
    layout_reward = _layout_score(board)

    # Invalid rate: fraction of moves that were illegal.
    invalid_rate = float(
        np.clip(invalid_count / max(move_count, 1), 0.0, 1.0)
    )

    # --- Phase 3: weighted sum, clip to [-1, 1] ---
    reward = (
        0.65 * merge_reward
        + 0.20 * empty_reward
        + 0.15 * layout_reward
        - 0.80 * invalid_rate
    )

    if died_early:
        reward -= 0.40

    return float(np.clip(reward, -1.0, 1.0))


# ------------------------------------------------------------------
# Direction extraction (tool call + text fallback)
# ------------------------------------------------------------------


def _extract_directions(record) -> list[str]:
    """Pull move directions from tool calls or assistant text, in order.

    Tries tool-call arguments first (the preferred path).  If no valid
    tool calls are found, falls back to parsing direction keywords from
    assistant message content via game.parse_action.
    """

    # --- Primary: extract from structured tool calls ---
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
    if directions:
        return directions

    # --- Fallback: parse from assistant text content ---
    for message in record.messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not content:
            continue
        direction = game.parse_action(content)
        if direction is not None:
            directions.append(direction)
    return directions