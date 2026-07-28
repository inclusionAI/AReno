"""Dataset loader for the circuit-diagnosis agentic example (issue #193)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL circuit records and convert them to Areno prompt records.

    Args:
        dataset_path: Path to a JSONL file, or a directory containing one.

    Returns:
        A list of records with ``prompt`` and metadata fields.
    """

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw) for raw in records]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "circuits.jsonl"
    if not path.exists():
        return dataset_generator.generate_records()
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict) -> dict:
    """Format a raw circuit record for the trainer."""

    return {
        "id": raw.get("id", "unknown"),
        "prompt": raw.get("prompt", ""),
        "num_inputs": raw.get("num_inputs", 3),
        "num_gates": raw.get("num_gates", 6),
        "gates": raw.get("gates", []),
        "faulty_gate_id": raw.get("faulty_gate_id"),
        "fault_type": raw.get("fault_type", "stuck_at_0"),
    }
