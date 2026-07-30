"""Dataset loader for the odd-ball balance-scale agentic example."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402

logger = logging.getLogger(__name__)


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL puzzles and convert them to Areno prompt records."""

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "balance_scale_puzzles.jsonl"
    if not path.exists():
        logger.warning(
            "Dataset file not found: %s. Falling back to auto-generated records. "
            "If this is unintentional, check your --dataset-path.",
            path,
        )
        return dataset_generator.generate_records()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int) -> dict:
    ball_set = game.BallSet(
        num_balls=raw["num_balls"],
        odd_ball_index=raw["odd_ball_index"],
        direction=raw["direction"],
        max_weighings=raw["max_weighings"],
    )
    return {
        "id": raw.get("id", f"puzzle-{index:05d}"),
        "prompt": game.format_prompt(ball_set),
        "num_balls": ball_set.num_balls,
        "odd_ball_index": ball_set.odd_ball_index,
        "direction": ball_set.direction,
        "max_weighings": ball_set.max_weighings,
    }
