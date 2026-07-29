"""Dataset loader for the Wordle tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """
    Load JSONL Wordle games and convert them to Areno prompt records.

    Args:
        dataset_path: Path to JSONL file or directory containing games.jsonl

    Returns:
        List of formatted prompt records

    Raises:
        DatasetValidationError: If dataset path is invalid before expensive initialization
    """
    # Validate input before expensive operations
    is_valid, error_msg = game.validate_dataset_path(dataset_path)
    if not is_valid:
        raise DatasetValidationError(
            f"Dataset validation failed: {error_msg}. "
            f"AReno validates inputs before expensive model or worker initialization."
        )

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx, xml=False) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "games.jsonl"
    if not path.exists():
        return dataset_generator.generate_records()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int, *, xml: bool) -> dict:
    """Format a raw game record into an Areno prompt record."""
    target = raw["target"]
    max_guesses = raw.get("max_guesses", game.MAX_GUESSES)

    # Create initial game state
    initial_game = game.create_new_game(target)

    return {
        "id": raw.get("id", f"game-{index:05d}"),
        "prompt": game.format_prompt(initial_game),
        "game": initial_game,
        "target": target,
        "max_guesses": max_guesses,
    }