"""Generate calendar scheduling scenarios for the agentic RL example."""

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
HELD_OUT_FRACTION = 0.2

# A small set of time zones with fixed offsets (no DST).
TIMEZONES = ["UTC", "UTC+1", "UTC+2", "UTC+3", "UTC+8", "UTC-5", "UTC-8", "UTC+5:30"]

PARTicipant_NAMES = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]


def generate_records(count: int = DEFAULT_COUNT, *, seed: int = DEFAULT_SEED) -> list[dict]:
    """Generate reproducible calendar scheduling scenarios."""
    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[str] = set()

    while len(records) < count:
        num_participants = rng.randint(2, 3)
        chosen_names = rng.sample(PARTicipant_NAMES, num_participants)

        participants = {}
        for name in chosen_names:
            tz = rng.choice(TIMEZONES)
            num_slots = rng.randint(1, 2)
            slots = []
            cursor = rng.randint(0, 8)
            for _ in range(num_slots):
                start = cursor
                duration = rng.randint(2, 6)
                end = min(start + duration, 24)
                if start < end:
                    slots.append({"start_hour": start, "end_hour": end})
                cursor = end + rng.randint(1, 3)
            if not slots:
                slots.append({"start_hour": 9, "end_hour": 17})
            participants[name] = {
                "name": name,
                "timezone": tz,
                "available_slots": slots,
            }

        meeting_id = f"meeting-{len(records):04d}"
        duration_hours = rng.randint(1, 2)
        required = chosen_names

        # Sometimes add an already-confirmed meeting to create conflicts.
        confirmed = {}
        if rng.random() < 0.3:
            other_id = f"meeting-{len(records):04d}-prior"
            other_start = rng.randint(0, 20)
            other_end = min(other_start + rng.randint(1, 3), 24)
            confirmed[other_id] = [other_start, other_end]

        record = {
            "id": meeting_id,
            "participants": participants,
            "meetings": [
                {
                    "id": meeting_id,
                    "duration_hours": duration_hours,
                    "required_participants": required,
                }
            ],
            "confirmed": confirmed,
            "target_meeting_id": meeting_id,
        }

        # Deterministic uniqueness check.
        key = json.dumps(record, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)

    return records


def split_held_out(records: list[dict], *, fraction: float = HELD_OUT_FRACTION) -> tuple[list[dict], list[dict]]:
    """Split records into train and held-out test sets."""
    n_test = max(1, int(len(records) * fraction))
    return records[n_test:], records[:n_test]


def write_jsonl(records: list[dict], output: TextIO) -> None:
    for record in records:
        output.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL calendar scenarios for the AReno calendar agentic example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of scenarios to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--held-out", type=float, default=HELD_OUT_FRACTION, help="Fraction held out for testing.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    records = generate_records(args.count, seed=args.seed)
    train, test = split_held_out(records, fraction=args.held_out)

    if args.output == "-":
        write_jsonl(train, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(train, handle)
        # Also write held-out set.
        test_path = output_path.with_suffix(".held_out.jsonl")
        with test_path.open("w", encoding="utf-8") as handle:
            write_jsonl(test, handle)
        print(f"train: {len(train)} records → {output_path}", file=sys.stderr)
        print(f"held-out: {len(test)} records → {test_path}", file=sys.stderr)


if __name__ == "__main__":
    main()