"""Dataset loader for the 2048 agentic example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL boards and convert them to Areno prompt records."""

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _format_record(raw: dict, index: int, *, xml: bool = False) -> dict:
    board = game.normalize_board(raw["board"])
    seed = int(raw["seed"])
    baseline = raw.get("random_baseline") or game.random_episode(board, seed=seed)
    return {
        "id": raw.get("id", f"board-{index:05d}"),
        "prompt": game.format_xml_prompt(board) if xml else game.format_prompt(board),
        "board": board,
        "seed": seed,
        "random_baseline": baseline,
        "legal_moves": game.legal_moves(board),
    }


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "boards.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"2048 dataset not found: {path}. Generate it with "
            "`python examples/agentic/2048/dataset_generator.py --output <path>`."
        )
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records