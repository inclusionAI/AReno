"""Dataset loader for the 2048 XML no-tool example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loader import _format_record, _load_records  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL boards and format XML action prompts."""

    del default_loader
    return [_format_record(raw, idx, xml=True) for idx, raw in enumerate(_load_records(dataset_path), start=1)]