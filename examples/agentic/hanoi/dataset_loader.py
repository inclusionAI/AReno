"""Dataset loader for the Towers of Hanoi agentic example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL Hanoi tasks and convert them to Areno prompt records."""

    records = _load_records(dataset_path, default_loader=default_loader)
    return [_format_record(raw, index) for index, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str, *, default_loader=None) -> list[dict]:
    if default_loader is not None:
        rows = default_loader(dataset_path)
        if rows:
            return list(rows)
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "tasks.jsonl"
    if not path.exists():
        return dataset_generator.generate_records()
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int) -> dict:
    n = game.validate_disks(int(raw["n"]))
    record = {
        "id": str(raw.get("id", f"hanoi-{index:05d}")),
        "n": n,
        "max_moves": int(raw.get("max_moves", game.default_max_moves(n))),
    }
    if record["max_moves"] <= 0:
        raise ValueError(f"max_moves must be positive for task {record['id']}")
    record["prompt"] = game.make_prompt(record)
    return record
