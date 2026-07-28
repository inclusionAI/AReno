"""Dataset loader for the maze agentic example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL maze states and convert them to Areno prompt records."""

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "maze_states.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Maze dataset not found: {path}. Generate it with "
            "`python examples/agentic/maze/dataset_generator.py --output <path>`."
        )
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int) -> dict:
    state = dataset_generator.record_to_state(raw)
    radius = raw.get("view_radius", 1)
    return {
        "id": raw.get("id", f"maze-{index:05d}"),
        "prompt": game.format_prompt(state, radius),
        "state": raw,
        "legal_actions": game.legal_actions(state),
    }
