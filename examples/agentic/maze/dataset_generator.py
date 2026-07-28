"""Generate seed-solvable maze states for the agentic maze example."""

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
DEFAULT_ROWS = 4
DEFAULT_COLS = 4
DEFAULT_MAX_STEPS = 30
DEFAULT_VIEW_RADIUS = 1
DEFAULT_NUM_KEYS = 1


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    num_keys: int = DEFAULT_NUM_KEYS,
    max_steps: int = DEFAULT_MAX_STEPS,
    view_radius: int = DEFAULT_VIEW_RADIUS,
) -> list[dict]:
    """Generate reproducible maze prompt states."""

    rng = random.Random(seed)
    records = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 20:
            break
        maze_seed = rng.randint(0, 2**31 - 1)
        nk = rng.randint(0, num_keys)
        grid, agent_pos, goal_pos, key_door_pairs = game.generate_maze(
            rows, cols, seed=maze_seed, num_keys=nk
        )
        if grid in seen:
            continue
        seen.add(grid)
        records.append(_state_to_record(
            grid, agent_pos, goal_pos, key_door_pairs,
            maze_seed, nk, max_steps, view_radius,
            f"maze-{len(records):05d}",
        ))
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def record_to_state(record: dict) -> game.State:
    """Convert a JSONL record back to a :class:`game.State`."""

    return game.make_state_from_record(record)


def _state_to_record(
    grid: tuple[str, ...],
    agent_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    key_door_pairs: list[dict],
    maze_seed: int,
    num_keys: int,
    max_steps: int,
    view_radius: int,
    record_id: str,
) -> dict:
    return {
        "id": record_id,
        "grid": list(grid),
        "agent_pos": list(agent_pos),
        "goal_pos": list(goal_pos),
        "key_door_pairs": [
            {
                "key_id": pair["key_id"],
                "key_pos": list(pair["key_pos"]),
                "door_pos": list(pair["door_pos"]),
            }
            for pair in key_door_pairs
        ],
        "maze_seed": maze_seed,
        "num_keys": num_keys,
        "max_steps": max_steps,
        "view_radius": view_radius,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL maze states for the Areno agentic maze example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of states to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Master random seed.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Maze cell rows (before wall expansion).")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="Maze cell cols (before wall expansion).")
    parser.add_argument("--num-keys", type=int, default=DEFAULT_NUM_KEYS, help="Max key-door pairs per maze.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per episode.")
    parser.add_argument("--view-radius", type=int, default=DEFAULT_VIEW_RADIUS, help="Local view radius.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("--rows and --cols must be positive")
    if args.num_keys < 0:
        raise ValueError("--num-keys must be non-negative")

    records = generate_records(
        args.count,
        seed=args.seed,
        rows=args.rows,
        cols=args.cols,
        num_keys=args.num_keys,
        max_steps=args.max_steps,
        view_radius=args.view_radius,
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
