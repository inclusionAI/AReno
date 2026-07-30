"""Generate deterministic elevator dispatch scenarios for agentic RL training.

Each scenario targets one acceptance area from issue #195:
- ``overload``   cab capacity forces refused boarding
- ``empty_door`` door-state invalid actions
- ``concurrent`` many near-simultaneous requests on different floors
- ``peak``       high arrival density over a long horizon
- ``terminate``  very short horizon forcing early termination
- ``mixed``      default training set blending the above
"""

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

SCENARIOS = ("overload", "empty_door", "concurrent", "peak", "terminate", "mixed")


def generate_records(count: int = DEFAULT_COUNT, *, seed: int = DEFAULT_SEED, scenario: str = "mixed") -> list[dict]:
    """Generate reproducible elevator scenario records.

    ``scenario="mixed"`` samples each of the five focused scenarios plus a few
    generic records so a single training file exercises every acceptance point.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    if count <= 0:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    records: list[dict] = []
    if scenario == "mixed":
        focused = ("overload", "empty_door", "concurrent", "peak", "terminate")
        per_scenario = max(count // (len(focused) + 1), 1)
        for name in focused:
            records.extend(_scenario_records(name, per_scenario, rng))
        # fill the rest with generic mixed traffic
        while len(records) < count:
            records.append(_generic_record(rng, count_id=len(records)))
        return records[:count]
    return _scenario_records(scenario, count, rng)


def _scenario_records(scenario: str, count: int, rng: random.Random) -> list[dict]:
    """Build ``count`` records for one focused scenario."""

    records: list[dict] = []
    for i in range(count):
        if scenario == "overload":
            records.append(_overload_record(rng, i))
        elif scenario == "empty_door":
            records.append(_empty_door_record(rng, i))
        elif scenario == "concurrent":
            records.append(_concurrent_record(rng, i))
        elif scenario == "peak":
            records.append(_peak_record(rng, i))
        elif scenario == "terminate":
            records.append(_terminate_record(rng, i))
        else:
            records.append(_generic_record(rng, count_id=i))
    return records


def _overload_record(rng: random.Random, idx: int) -> dict:
    """Capacity much smaller than demand so boarding is refused at least once."""

    floors = rng.randint(4, 6)
    capacity = 1
    horizon = 48
    n_passengers = rng.randint(4, 6)
    origin = rng.randint(0, floors - 1)
    passengers = []
    for pid in range(n_passengers):
        dest = _distinct_dest(origin, floors, rng)
        passengers.append({"pid": pid, "origin": origin, "dest": dest, "arrive_time": 0})
    return _wrap("overload", idx, floors, capacity, horizon, passengers)


def _empty_door_record(rng: random.Random, idx: int) -> dict:
    """Start with the door open so open_door is immediately invalid."""

    floors = rng.randint(3, 5)
    capacity = rng.randint(2, 3)
    horizon = 24
    passengers = _random_passengers(rng, floors, n=rng.randint(1, 3))
    record = _wrap("empty_door", idx, floors, capacity, horizon, passengers)
    record["door_open"] = True
    return record


def _concurrent_record(rng: random.Random, idx: int) -> dict:
    """Several passengers arrive at the same step on different floors."""

    floors = rng.randint(4, 6)
    capacity = rng.randint(2, 3)
    horizon = 40
    n = rng.randint(4, 6)
    passengers = []
    used_floors = list(range(floors))
    rng.shuffle(used_floors)
    for pid in range(n):
        origin = used_floors[pid % floors]
        dest = _distinct_dest(origin, floors, rng)
        passengers.append({"pid": pid, "origin": origin, "dest": dest, "arrive_time": 0})
    return _wrap("concurrent", idx, floors, capacity, horizon, passengers)


def _peak_record(rng: random.Random, idx: int) -> dict:
    """High density arrivals over a long horizon to stress throughput."""

    floors = rng.randint(5, 7)
    capacity = rng.randint(3, 4)
    horizon = 96
    n = rng.randint(10, 14)
    passengers = []
    for pid in range(n):
        origin = rng.randint(0, floors - 1)
        dest = _distinct_dest(origin, floors, rng)
        arrive_time = rng.randint(0, max(horizon // 2, 1))
        passengers.append({"pid": pid, "origin": origin, "dest": dest, "arrive_time": arrive_time})
    passengers.sort(key=lambda p: p["arrive_time"])
    # re-assign sequential pids after sorting
    for new_pid, passenger in enumerate(passengers):
        passenger["pid"] = new_pid
    return _wrap("peak", idx, floors, capacity, horizon, passengers)


def _terminate_record(rng: random.Random, idx: int) -> dict:
    """Horizon too short to deliver everyone, forcing termination by timeout."""

    floors = rng.randint(3, 5)
    capacity = rng.randint(1, 2)
    horizon = rng.randint(2, 4)
    passengers = _random_passengers(rng, floors, n=rng.randint(2, 3))
    return _wrap("terminate", idx, floors, capacity, horizon, passengers)


def _generic_record(rng: random.Random, *, count_id: int) -> dict:
    """A plain mixed-traffic record not tied to one acceptance point."""

    floors = rng.randint(3, 6)
    capacity = rng.randint(2, 4)
    horizon = rng.randint(32, 64)
    passengers = _random_passengers(rng, floors, n=rng.randint(2, 5))
    return _wrap("mixed", count_id, floors, capacity, horizon, passengers)


def _random_passengers(rng: random.Random, floors: int, *, n: int) -> list[dict]:
    passengers = []
    for pid in range(n):
        origin = rng.randint(0, floors - 1)
        dest = _distinct_dest(origin, floors, rng)
        arrive_time = rng.randint(0, max(floors, 4))
        passengers.append({"pid": pid, "origin": origin, "dest": dest, "arrive_time": arrive_time})
    return passengers


def _distinct_dest(origin: int, floors: int, rng: random.Random) -> int:
    dest = rng.randint(0, floors - 1)
    while dest == origin:
        dest = rng.randint(0, floors - 1)
    return dest


def _wrap(scenario: str, idx: int, floors: int, capacity: int, horizon: int, passengers: list[dict]) -> dict:
    return {
        "id": f"elevator-{scenario}-{idx:05d}",
        "scenario": scenario,
        "floors": floors,
        "capacity": capacity,
        "horizon": horizon,
        "passengers": passengers,
        "door_open": False,
    }


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write records as one JSON object per line."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL elevator scenarios for the AReno agentic example.")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of records to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="mixed",
        help="Focused scenario or 'mixed' (default) for a blended training set.",
    )
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed, scenario=args.scenario)
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()
