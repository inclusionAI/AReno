"""Generate elevator-dispatch buildings for the agentic example."""

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
DEFAULT_SEED = game.DEFAULT_SEED
DEFAULT_FLOORS = game.DEFAULT_FLOORS
DEFAULT_CAPACITY = game.DEFAULT_CAPACITY


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    floors: int = DEFAULT_FLOORS,
    capacity: int = DEFAULT_CAPACITY,
    arrivals_per_building: int = 6,
    max_tick: int = 24,
) -> list[dict]:
    """Generate reproducible dispatch buildings.

    Each building starts with the car at floor 0, the door closed, and a seeded
    arrival schedule; two runs with the same ``seed`` are identical.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if floors < 2:
        raise ValueError("floors must be >= 2")
    if arrivals_per_building < 0:
        raise ValueError("arrivals_per_building must be >= 0")
    rng = random.Random(seed)
    records: list[dict] = []
    for idx in range(count):
        arrivals = _seeded_arrivals(rng, floors, arrivals_per_building, max_tick)
        records.append(
            {
                "id": f"generated-{idx:05d}",
                "floors": floors,
                "capacity": capacity,
                "arrivals": arrivals,
                "car": {"floor": 0, "direction": "U", "door_open": False, "passengers": []},
            }
        )
    return records


def _seeded_arrivals(rng: random.Random, floors: int, n: int, max_tick: int) -> list[dict]:
    arrivals: list[dict] = []
    for _ in range(n):
        from_floor = rng.randrange(floors)
        to_floor = rng.randrange(floors - 1)
        if to_floor >= from_floor:
            to_floor += 1  # keep from != to impossible
        tick = rng.randrange(max_tick + 1)
        arrivals.append({"tick": tick, "from_floor": from_floor, "to_floor": to_floor})
    arrivals.sort(key=lambda e: (e["tick"], e["from_floor"]))
    return arrivals


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL buildings for the Areno elevator agentic example.")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of buildings to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--floors", type=int, default=DEFAULT_FLOORS, help="Number of floors per building.")
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY, help="Car passenger capacity.")
    parser.add_argument("--arrivals", type=int, default=6, help="Arrivals per building.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.floors < 2:
        raise ValueError("--floors must be >= 2")
    if args.capacity < 1:
        raise ValueError("--capacity must be >= 1")

    records = generate_records(
        args.count,
        seed=args.seed,
        floors=args.floors,
        capacity=args.capacity,
        arrivals_per_building=args.arrivals,
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
