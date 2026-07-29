"""Dataset loader for water-jug puzzles.

Implements ``load_training_dataset()``, the function AReno calls to load
puzzle data before training. Reads a JSONL file produced by
``dataset_generator.py`` and converts each line into a training record
with ``prompt`` (str) and ``image`` (dict with puzzle metadata).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    del default_loader
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "water_jug_puzzles.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    items = []
    for idx, raw in enumerate(records, start=1):
        caps = raw["capacities"]
        target = raw["target"]
        prompt = game.build_user_prompt(caps, target)
        items.append({
            "id": raw.get("id", f"puzzle-{idx:05d}"),
            "prompt": prompt,
            "image": {
                "capacities": caps,
                "initial_state": raw.get("initial_state", [0] * len(caps)),
                "target": target,
                "oracle_steps": raw.get("oracle_steps", 0),
            },
        })
    return items