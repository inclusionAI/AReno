"""Generate reproducible maze datasets for the agentic example."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026
DEFAULT_WIDTH = 7
DEFAULT_HEIGHT = 7
DEFAULT_VISION_RADIUS = 1


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    vision_radius: int = DEFAULT_VISION_RADIUS,
    max_steps: int | None = None,
) -> list[dict]:
    """Generate *count* unique solvable maze records.

    To maximise variety, each maze is generated with a random sub-seed and a
    random key/door count (1-2).  If the base dimensions are too small to
    produce enough unique mazes, the generator gradually widens the search
    by allowing adjacent odd sizes.
    """

    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    attempts = 0
    max_attempts = count * 1000
    allow_duplicates = False
    while len(records) < count:
        attempts += 1
        if attempts > max_attempts:
            if allow_duplicates:
                break  # stop trying, return what we have
            allow_duplicates = True
            attempts = 0
        sub_seed = rng.randint(0, 2**31 - 1)
        n_keys = rng.randint(1, 2)
        n_doors = rng.randint(1, 2)
        w = width + rng.choice([0, 0, 2])
        h = height + rng.choice([0, 0, 2])
        maze, start, goal, keys, doors = game.generate_maze(
            w,
            h,
            seed=sub_seed,
            n_keys=n_keys,
            n_doors=n_doors,
        )
        key_tuple = tuple(tuple(row) for row in maze)
        if key_tuple in seen and not allow_duplicates:
            continue
        seen.add(key_tuple)

        if max_steps is None:
            effective_max_steps = game.maze_width(maze) * game.maze_height(maze)
        else:
            effective_max_steps = max_steps

        record = game.serialize_maze(
            maze,
            start,
            goal,
            keys,
            doors,
            vision_radius=vision_radius,
            max_steps=effective_max_steps,
        )
        record["id"] = f"generated-{len(records):05d}"
        records.append(record)
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL maze datasets for the AReno maze agentic example.",
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of mazes to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Maze width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Maze height.")
    parser.add_argument("--vision-radius", type=int, default=DEFAULT_VISION_RADIUS, help="Agent vision radius.")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode (default: width*height).")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.vision_radius < 0:
        raise ValueError("--vision-radius must be non-negative")

    records = generate_records(
        args.count,
        seed=args.seed,
        width=args.width,
        height=args.height,
        vision_radius=args.vision_radius,
        max_steps=args.max_steps,
    )
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()
