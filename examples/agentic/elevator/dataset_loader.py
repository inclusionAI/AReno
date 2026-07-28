"""Dataset loader for the elevator-dispatch tool-call example.

Each record becomes an Areno prompt that asks the policy for one dispatch
episode (an action string). The building is stored on the record so the reward
function can replay it deterministically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402

DEFAULT_MAX_STEPS = game.DEFAULT_MAX_STEPS


def load_training_dataset(
    dataset_path: str,
    *,
    default_loader=None,
    max_steps: int = DEFAULT_MAX_STEPS,
    **_: object,
) -> list[dict]:
    """Load JSONL buildings and convert them to Areno prompt records."""

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx, max_steps=max_steps, xml=False) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "buildings.jsonl"
    if not path.exists():
        return dataset_generator.generate_records()
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int, *, max_steps: int, xml: bool) -> dict:
    building = game.normalize_building(raw)
    return {
        "id": raw.get("id", f"building-{index:05d}"),
        "prompt": game.format_xml_prompt(building) if xml else game.format_prompt(building),
        "building": building,
        "max_steps": int(raw.get("max_steps", max_steps)),
    }
