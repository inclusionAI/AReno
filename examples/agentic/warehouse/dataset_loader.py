"""Dataset loader for the warehouse-picking agentic example.

Loads JSONL warehouse task records and converts them to Areno
prompt-bearing records. Each record specifies a difficulty level,
random seed, and optional explicit order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL warehouse tasks and convert them to Areno prompt records.

    If the file does not exist, falls back to generating records with the
    default seed so tests can run without a pre-built dataset file.
    """

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    """Load raw JSONL records, falling back to generated defaults."""

    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "warehouse_tasks.jsonl"
    if not path.exists():
        return dataset_generator.generate_records()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int) -> dict:
    """Convert a raw JSONL row into a prompt-bearing Areno record."""

    difficulty = str(raw.get("difficulty", "small"))
    seed = int(raw.get("seed", 42))

    # Build the warehouse state to generate the prompt.
    if difficulty == "medium":
        state = game.generate_medium(seed=seed)
    elif difficulty == "hard":
        state = game.generate_hard(seed=seed)
    else:
        state = game.generate_small(seed=seed)

    # Override order if specified in the record.
    if "order" in raw:
        from game import Order
        state.order = Order(order_id="order_1", items=raw["order"])

    return {
        "id": raw.get("id", f"warehouse-{index:05d}"),
        "prompt": game.make_prompt(state),
        "difficulty": difficulty,
        "seed": seed,
        "order": dict(state.order.items) if state.order else {},
        "grid": [list(row) for row in state.grid],
        "shelves": {
            sid: {"row": s.row, "col": s.col, "stock": dict(s.stock)}
            for sid, s in state.shelves.items()
        },
    }