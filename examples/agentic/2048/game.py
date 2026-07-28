"""Pure-Python 2048 engine for the agentic example.

The board is a 4x4 grid of non-negative ints (0 == empty). Tile placement is
driven by an explicit ``random.Random`` instance so that replaying the same
move sequence under the same seed reproduces the episode exactly. Public
functions return a *new* board and never mutate their input.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterable

Board = list[list[int]]
ACTIONS = ("up", "down", "left", "right")

SIZE = 4
DEFAULT_EPISODE_CAP = 32
SPAWN_4_PROB = 0.1
INVALID_PENALTY = 0.5
WIN_TILE = 2048

logger = logging.getLogger(__name__)

_MOVE_TOKEN_RE = re.compile(r"\b(up|down|left|right)\b", re.IGNORECASE)
_XML_MOVES_RE = re.compile(r"<moves>\s*([0-9a-zA-Z,\s]*)\s*</moves>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think\b[^>]*>.*?" + "</think" + r">", re.IGNORECASE | re.DOTALL)
_CHAT_SPECIAL_RE = re.compile(r"<\|[^>]+?\|>|</?s>", re.IGNORECASE)


class EpisodeResult:
    """Outcome of replaying a bounded move sequence on a starting board."""

    __slots__ = (
        "board",
        "score",
        "max_tile",
        "total_moves",
        "invalid_moves",
        "reached_2048",
        "truncated",
        "moves",
    )

    def __init__(
        self,
        board: Board,
        score: int,
        max_tile: int,
        total_moves: int,
        invalid_moves: int,
        reached_2048: bool,
        truncated: bool,
        moves: list[str] | None = None,
    ) -> None:
        self.board = board
        self.score = score
        self.max_tile = max_tile
        self.total_moves = total_moves
        self.invalid_moves = invalid_moves
        self.reached_2048 = reached_2048
        self.truncated = truncated
        self.moves = moves if moves is not None else []

    @property
    def invalid_rate(self) -> float:
        """Fraction of replayed moves that did not change the board."""

        if self.total_moves <= 0:
            return 0.0
        return self.invalid_moves / self.total_moves

    def as_dict(self) -> dict:
        """Serializable view used by reward logging and the baseline harness."""

        return {
            "score": self.score,
            "max_tile": self.max_tile,
            "total_moves": self.total_moves,
            "invalid_moves": self.invalid_moves,
            "invalid_rate": self.invalid_rate,
            "reached_2048": self.reached_2048,
            "truncated": self.truncated,
        }


def normalize_board(board: Iterable[Iterable[int]]) -> Board:
    """Return a validated 4x4 2048 board (0 == empty)."""

    rows = [[int(cell) for cell in row] for row in board]
    if len(rows) != SIZE or any(len(row) != SIZE for row in rows):
        raise ValueError("2048 board must be 4x4")
    if any(cell < 0 or cell != int(cell) for row in rows for cell in row):
        raise ValueError("2048 cells must be non-negative integers")
    if any(cell != 0 and (cell & (cell - 1)) != 0 for row in rows for cell in row):
        raise ValueError("2048 tiles must be powers of two (or 0)")
    return rows


def board_to_text(board: Board) -> str:
    """Render the board as a 4x4 grid; empty cells show as a dot."""

    board = normalize_board(board)
    return "\n".join(" ".join(str(cell) if cell != 0 else "." for cell in row) for row in board)


def max_tile(board: Board) -> int:
    """Return the largest tile on the board (0 for an empty board)."""

    return max((max(row, default=0) for row in normalize_board(board)), default=0)


def empty_cells(board: Board) -> list[tuple[int, int]]:
    """Return (row, col) positions of empty cells."""

    board = normalize_board(board)
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]


def format_prompt(board: Board) -> str:
    """Build the prompt asking the agent for a bounded move sequence."""

    board = normalize_board(board)
    return (
        "You are playing 2048. Choose a sequence of directions to play from the "
        "current board by calling the choose_moves tool.\n\n"
        "Rules:\n"
        "- Directions are up, down, left, right.\n"
        "- After every move that changes the board, a 2 (90%) or 4 (10%) tile spawns.\n"
        "- Moves that do not change the board are invalid and penalized; stop playing "
        f"once no direction changes the board or after at most {DEFAULT_EPISODE_CAP} moves.\n"
        "- Prefer merges that build toward larger tiles.\n\n"
        f"Board:\n{board_to_text(board)}\n\nLegal directions: {legal_moves(board)}\n\nMoves:"
    )


def _slide_row_left(row: list[int]) -> tuple[list[int], int, bool]:
    """Compress and merge one row toward the left.

    Returns the new row, the merge score gained, and whether the row changed.
    Each tile may merge at most once per move.
    """

    nonzero = [cell for cell in row if cell != 0]
    merged: list[int] = []
    score = 0
    i = 0
    while i < len(nonzero):
        if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
            value = nonzero[i] * 2
            merged.append(value)
            score += value
            i += 2
        else:
            merged.append(nonzero[i])
            i += 1
    new_row = merged + [0] * (SIZE - len(merged))
    changed = new_row != row
    return new_row, score, changed


def slide(board: Board, direction: str) -> tuple[Board, int, bool]:
    """Apply one direction; return (new_board, merge_score, changed)."""

    board = normalize_board(board)
    direction = str(direction).lower().strip()
    if direction not in ACTIONS:
        raise ValueError(f"illegal 2048 direction: {direction!r}")

    if direction == "left":
        results = [_slide_row_left(list(row)) for row in board]
        new_board = [row for row, _s, _c in results]
    elif direction == "right":
        results = [_slide_row_left(list(reversed(row))) for row in board]
        new_board = [list(reversed(row)) for row, _s, _c in results]
    elif direction == "up":
        cols = [_slide_row_left([board[r][c] for r in range(SIZE)]) for c in range(SIZE)]
        new_board = [[cols[c][0][r] for c in range(SIZE)] for r in range(SIZE)]
        results = cols
    else:  # down
        cols = [
            _slide_row_left([board[SIZE - 1 - r][c] for r in range(SIZE)]) for c in range(SIZE)
        ]
        new_board = [[cols[c][0][SIZE - 1 - r] for c in range(SIZE)] for r in range(SIZE)]
        results = cols

    score = sum(gained for _row, gained, _changed in results)
    changed = any(changed for _row, _gained, changed in results)
    return new_board, score, changed


def spawn_tile(board: Board, rng: random.Random) -> Board:
    """Spawn one 2/4 tile in a random empty cell using ``rng`` (deterministic)."""

    board = normalize_board(board)
    cells = empty_cells(board)
    if not cells:
        return [list(row) for row in board]
    row, col = rng.choice(cells)
    value = 4 if rng.random() < SPAWN_4_PROB else 2
    new_board = [list(row_values) for row_values in board]
    new_board[row][col] = value
    return new_board


def is_terminal(board: Board) -> bool:
    """Return whether no move can change the board."""

    board = normalize_board(board)
    if empty_cells(board):
        return False
    return not legal_moves(board)


def legal_moves(board: Board) -> list[str]:
    """Return directions that change the board."""

    board = normalize_board(board)
    return [direction for direction in ACTIONS if slide(board, direction)[2]]


def play_episode(
    board: Board,
    moves: Iterable[str],
    *,
    seed: int,
    cap: int = DEFAULT_EPISODE_CAP,
) -> EpisodeResult:
    """Replay a bounded move sequence deterministically.

    Tiles spawn from ``random.Random(seed)`` so identical ``(board, moves, seed,
    cap)`` reproduces the episode exactly. Invalid (no-op) moves are counted and
    penalized but do not advance the RNG.
    """

    board = normalize_board(board)
    if cap < 0:
        raise ValueError("cap must be non-negative")
    rng = random.Random(seed)
    score = 0
    total = 0
    invalid = 0
    played: list[str] = []
    truncated = False
    reached_2048 = max_tile(board) >= WIN_TILE

    for raw_move in moves:
        if total >= cap:
            truncated = True
            break
        move = str(raw_move).lower().strip()
        if move not in ACTIONS:
            total += 1
            invalid += 1
            played.append(move)
            continue
        played.append(move)
        new_board, gained, changed = slide(board, move)
        total += 1
        if not changed:
            invalid += 1
            continue
        board = spawn_tile(new_board, rng)
        score += gained
        if max_tile(board) >= WIN_TILE:
            reached_2048 = True
            break
        if is_terminal(board):
            break

    return EpisodeResult(
        board=board,
        score=score,
        max_tile=max_tile(board),
        total_moves=total,
        invalid_moves=invalid,
        reached_2048=reached_2048,
        truncated=truncated,
        moves=played,
    )


def play_episode_frames(
    board: Board,
    moves: Iterable[str],
    *,
    seed: int,
    cap: int = DEFAULT_EPISODE_CAP,
) -> tuple[EpisodeResult, list[dict]]:
    """Replay an episode and also return one frame per move for step-by-step UI.

    Frames capture the board *after* each attempted move (and after the spawn
    for a changed move), so a UI can animate the policy's plan. Shares the same
    determinism as ``play_episode``.
    """

    board = normalize_board(board)
    if cap < 0:
        raise ValueError("cap must be non-negative")
    rng = random.Random(seed)
    score = 0
    total = 0
    invalid = 0
    truncated = False
    reached_2048 = max_tile(board) >= WIN_TILE
    frames: list[dict] = []

    for raw_move in moves:
        if total >= cap:
            truncated = True
            break
        move = str(raw_move).lower().strip()
        if move not in ACTIONS:
            total += 1
            invalid += 1
            frames.append({"move": move, "board": board, "score": score, "gained": 0, "changed": False})
            continue
        new_board, gained, changed = slide(board, move)
        total += 1
        if not changed:
            invalid += 1
            frames.append({"move": move, "board": board, "score": score, "gained": 0, "changed": False})
            continue
        board = spawn_tile(new_board, rng)
        score += gained
        frames.append({"move": move, "board": board, "score": score, "gained": gained, "changed": True})
        if max_tile(board) >= WIN_TILE:
            reached_2048 = True
            break
        if is_terminal(board):
            break

    result = EpisodeResult(
        board=board,
        score=score,
        max_tile=max_tile(board),
        total_moves=total,
        invalid_moves=invalid,
        reached_2048=reached_2048,
        truncated=truncated,
    )
    return result, frames


def score_episode(result: EpisodeResult) -> float:
    """Raw episode quality: merge score minus the no-op penalty."""

    return float(result.score) - INVALID_PENALTY * result.invalid_moves


def score_moves(
    board: Board,
    moves: Iterable[str],
    *,
    seed: int,
    baseline_score: float,
    cap: int = DEFAULT_EPISODE_CAP,
    record_id: object = "?",
) -> float:
    """Replay ``moves`` and return one rollout reward scalar.

    Used by both the tool-call and XML no-tool reward functions: the caller only
    differs in how it parses ``moves``. Episode score, max tile, invalid-move
    rate, and trained-vs-baseline improvement are logged for observability.
    """

    board = normalize_board(board)
    result = play_episode(board, moves, seed=seed, cap=cap)
    improvement = float(result.score) - float(baseline_score)
    reward = float(result.score) - float(baseline_score) - INVALID_PENALTY * result.invalid_moves
    logger.info(
        "2048 episode id=%s score=%d max_tile=%d invalid_rate=%.3f improvement=%.3f reward=%.3f moves=%d",
        record_id,
        result.score,
        result.max_tile,
        result.invalid_rate,
        improvement,
        reward,
        result.total_moves,
    )
    return reward


def random_episode(
    board: Board,
    *,
    seed: int,
    cap: int = DEFAULT_EPISODE_CAP,
    trials: int = 8,
) -> dict:
    """Random-action baseline: mean episode metrics over ``trials`` rollouts.

    Each rollout picks a uniform-random direction from **all four** actions
    every step (a true random policy, not just legal moves). No-op directions
    are counted as invalid moves and do not spawn a tile, so the baseline's
    ``invalid_rate`` reflects how often a random direction wastes a step. Tile
    spawns use an independent ``random.Random`` derived from ``(seed, trial)`` so
    the baseline is reproducible, and the same RNG draws the move choices.
    """

    board = normalize_board(board)
    scores: list[float] = []
    max_tiles: list[int] = []
    invalid_rates: list[float] = []
    for trial in range(trials):
        result = _random_rollout(board, seed=_seeded_spawn_seed(seed, trial), cap=cap)
        scores.append(float(result.score))
        max_tiles.append(result.max_tile)
        invalid_rates.append(result.invalid_rate)
    return {
        "score": sum(scores) / len(scores) if scores else 0.0,
        "max_tile": sum(max_tiles) / len(max_tiles) if max_tiles else 0,
        "invalid_rate": sum(invalid_rates) / len(invalid_rates) if invalid_rates else 0.0,
        "trials": trials,
    }


def _random_rollout(board: Board, *, seed: int, cap: int) -> EpisodeResult:
    """Play one uniform-random-direction episode; ``rng`` drives moves and spawns."""

    rng = random.Random(seed)
    board = normalize_board(board)
    score = 0
    total = 0
    invalid = 0
    reached_2048 = max_tile(board) >= WIN_TILE
    truncated = False

    while total < cap:
        if is_terminal(board):
            break
        move = rng.choice(ACTIONS)  # uniform over all four directions
        new_board, gained, changed = slide(board, move)
        total += 1
        if not changed:
            invalid += 1
            continue
        board = spawn_tile(new_board, rng)
        score += gained
        if max_tile(board) >= WIN_TILE:
            reached_2048 = True
            break
        if is_terminal(board):
            break
    if total >= cap:
        truncated = True

    return EpisodeResult(
        board=board,
        score=score,
        max_tile=max_tile(board),
        total_moves=total,
        invalid_moves=invalid,
        reached_2048=reached_2048,
        truncated=truncated,
        moves=[],
    )


def format_xml_prompt(board: Board) -> str:
    """Build the prompt asking the agent for a single ``<moves>`` XML tag."""

    board = normalize_board(board)
    return (
        "You are playing 2048. Choose a sequence of directions to play from the "
        "current board.\n\n"
        "Rules:\n"
        "- Directions are up, down, left, right.\n"
        "- After every move that changes the board, a 2 (90%) or 4 (10%) tile spawns.\n"
        "- Moves that do not change the board are invalid and penalized; stop playing "
        f"once no direction changes the board or after at most {DEFAULT_EPISODE_CAP} moves.\n"
        "- Prefer merges that build toward larger tiles.\n\n"
        "Answer with exactly one XML tag listing directions in order, e.g. "
        "<moves>up,left,down</moves>.\n\n"
        f"Board:\n{board_to_text(board)}\n\nLegal directions: {legal_moves(board)}\n\nMoves:"
    )


def parse_xml_moves(text: str) -> list[str]:
    """Extract the final ``<moves>`` direction list from a model response."""

    if not isinstance(text, str):
        return []
    text = strip_chat_special_tokens(strip_think_tags(text)).strip()
    matches = list(_XML_MOVES_RE.finditer(text))
    if not matches:
        return []
    tokens = re.split(r"[,\s]+", matches[-1].group(1).strip())
    return [token.lower() for token in tokens if token.lower() in ACTIONS]


def strip_think_tags(text: str) -> str:
    """Remove reasoning spans before parsing the policy action."""

    return _THINK_RE.sub(" ", text)


def strip_chat_special_tokens(text: str) -> str:
    """Remove chat-template sentinels that may trail generated text."""

    return _CHAT_SPECIAL_RE.sub(" ", text)


def parse_moves(payload: object) -> list[str]:
    """Extract a validated direction list from a tool-call payload or text.

    Accepts a ``{"moves": [...]}`` dict, a list of tokens, or a raw string.
    Returns the recognized ``up/down/left/right`` tokens (lowercased), in order;
    unknown tokens are dropped. Never raises.
    """

    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = payload.get("moves")
    if isinstance(payload, (list, tuple)):
        tokens: list[str] = []
        for item in payload:
            token = str(item).lower().strip()
            if token in ACTIONS:
                tokens.append(token)
        return tokens
    if isinstance(payload, str):
        return [match.group(1).lower() for match in _MOVE_TOKEN_RE.finditer(payload)]
    return []


# --- baseline helpers ---------------------------------------------------------


def _seeded_spawn_seed(seed: int, trial: int) -> int:
    """Derive a stable spawn seed for baseline trial ``trial`` from ``seed``."""

    # hash() is randomized across processes; use a deterministic mix instead.
    return (int(seed) * 1_000_003 + int(trial) * 100_003 + 0x9E3779B1) & 0xFFFFFFFF