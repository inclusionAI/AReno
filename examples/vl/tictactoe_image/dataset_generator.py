"""Generate unique legal tic-tac-toe image records for VL training."""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path
from typing import TextIO

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026


def generate_records(
    count: int = DEFAULT_COUNT, *, seed: int = DEFAULT_SEED, include_images: bool = True
) -> list[dict]:
    """Generate reproducible, unique, legal boards where X is to move."""

    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("could not generate enough unique valid Tic-Tac-Toe boards")
        board = _random_board(rng)
        key = tuple(tuple(row) for row in board)
        if key in seen or game.is_terminal(board) or game.next_player(board) != "X":
            continue
        best_moves = game.best_moves(board)
        if not best_moves:
            continue
        seen.add(key)
        record = _record_from_board(board, len(records), best_moves)
        if include_images:
            record["image_base64"] = _image_to_base64(render_board(board))
        records.append(record)
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write records as compact JSONL."""

    for record in records:
        output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def render_board(board: game.Board) -> Image.Image:
    """Render a tic-tac-toe board as a simple board-game PNG."""

    board = game.normalize_board(board)
    size = 384
    margin = 36
    cell = (size - margin * 2) // 3
    image = Image.new("RGB", (size, size), (248, 244, 236))
    draw = ImageDraw.Draw(image)

    line_color = (45, 43, 38)
    for pos in (margin + cell, margin + cell * 2):
        draw.line((pos, margin, pos, size - margin), fill=line_color, width=8)
        draw.line((margin, pos, size - margin, pos), fill=line_color, width=8)

    for row_idx, row in enumerate(board):
        for col_idx, value in enumerate(row):
            cx = margin + col_idx * cell + cell // 2
            cy = margin + row_idx * cell + cell // 2
            radius = cell // 3
            if value == "X":
                color = (36, 91, 186)
                draw.line((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, width=14)
                draw.line((cx + radius, cy - radius, cx - radius, cy + radius), fill=color, width=14)
            elif value == "O":
                color = (210, 77, 62)
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=14)
    return image


def _random_board(rng: random.Random) -> game.Board:
    board = [[game.EMPTY, game.EMPTY, game.EMPTY] for _ in range(3)]
    player = "X"
    for _ in range(rng.randint(0, 6)):
        moves = game.legal_moves(board)
        if not moves or game.is_terminal(board):
            break
        board = game.apply_move(board, rng.choice(moves), player)
        player = "O" if player == "X" else "X"
    if game.next_player(board) == "O" and game.legal_moves(board):
        board = game.apply_move(board, rng.choice(game.legal_moves(board)), "O")
    return board


def _record_from_board(board: game.Board, idx: int, best_moves: list[int]) -> dict:
    move = best_moves[0]
    square = game.square_name(move)
    response = f"X should play square {move}, the {square} square, which is one of the minimax-best moves."
    return {
        "id": f"vl-tictactoe-{idx:05d}",
        "board": board,
        "best_moves": best_moves,
        "valid_moves": game.legal_moves(board),
        "prompt": "Describe the tic-tac-toe board image and name the best next move for X in one sentence.",
        "response": response,
        "reference": response,
        "solutions": [str(move), response],
        "target_keywords": ["X", f"square {move}", square],
    }


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL image rows for the Areno Qwen3.5-VL example.")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--image-dir", help="Optional directory for PNG files. JSONL will store image_path.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of unique boards to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    records = generate_records(args.count, seed=args.seed, include_images=True)
    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser()
        image_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            image_path = image_dir / f"{record['id']}.png"
            render_board(record["board"]).save(image_path)
            record["image_path"] = str(image_path)

    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()
