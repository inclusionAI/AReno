"""Dataset loader for the elevator dispatch agentic example.

The loader normalizes raw JSONL records produced by ``dataset_generator.py``
into AReno prompt records. It validates elevator-specific fields before any
model or worker initialization so malformed inputs fail fast with a clear
error. Records remain tokenizer-independent: the prompt is built lazily from
``game.make_prompt`` and the initial state is reconstructed by ``game.build_state``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load and validate elevator scenario records.

    Uses AReno's ``default_loader`` to read the JSONL rows, then validates each
    record against the elevator contract. Raises ``ValueError`` on the first
    malformed record so callers see the offending field without expensive setup.
    """

    records: list[dict] = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = _normalize_record(row, index)
        records.append(record)
    if not records:
        raise ValueError("elevator dataset is empty; generate records with dataset_generator.py first")
    return records


def _normalize_record(row: dict, index: int) -> dict:
    """Validate and normalize one raw record into a training record."""

    record = dict(row)
    record.setdefault("id", f"elevator-{index:05d}")
    record["id"] = str(record["id"])
    record["scenario"] = str(record.get("scenario", "mixed"))

    floors = int(record.get("floors", game.DEFAULT_FLOORS))
    capacity = int(record.get("capacity", game.DEFAULT_CAPACITY))
    horizon = int(record.get("horizon", game.DEFAULT_HORIZON))
    game.validate_config(floors=floors, capacity=capacity, horizon=horizon)
    record["floors"] = floors
    record["capacity"] = capacity
    record["horizon"] = horizon
    record["door_open"] = bool(record.get("door_open", False))

    raw_passengers = record.get("passengers", [])
    if not isinstance(raw_passengers, list):
        raise ValueError(f"record {record['id']}: passengers must be a list")
    passengers: list[dict] = []
    for pos, raw in enumerate(raw_passengers):
        if not isinstance(raw, dict):
            raise ValueError(f"record {record['id']}: passenger[{pos}] must be an object")
        pid = int(raw.get("pid", pos))
        origin = int(raw["origin"])
        dest = int(raw["dest"])
        arrive_time = int(raw.get("arrive_time", 0))
        if not 0 <= origin < floors:
            raise ValueError(f"record {record['id']}: passenger {pid} origin {origin} out of range 0..{floors - 1}")
        if not 0 <= dest < floors:
            raise ValueError(f"record {record['id']}: passenger {pid} dest {dest} out of range 0..{floors - 1}")
        if origin == dest:
            raise ValueError(f"record {record['id']}: passenger {pid} origin equals dest")
        if arrive_time < 0:
            raise ValueError(f"record {record['id']}: passenger {pid} arrive_time {arrive_time} must be >= 0")
        passengers.append({"pid": pid, "origin": origin, "dest": dest, "arrive_time": arrive_time})
    if not passengers:
        raise ValueError(f"record {record['id']}: at least one passenger is required")
    record["passengers"] = passengers

    # build the initial user prompt and precompute the fresh state snapshot id
    record["prompt"] = game.make_prompt(record)
    # validate the state can be constructed (raises on residual issues)
    game.build_state(record)
    return record
