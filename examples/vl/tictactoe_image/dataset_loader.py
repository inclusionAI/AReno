"""Dataset loader for the Qwen3.5-VL tic-tac-toe image example."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import game  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL examples and return model-agnostic image/text rows.

    The loader intentionally does not load a tokenizer or processor. It returns
    raw ``image_base64`` plus text fields; AReno's trainer encodes them with the
    processor from the current ``--ckpt``.
    """

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "dataset.jsonl"
    if not path.exists():
        return dataset_generator.generate_records()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                record = json.loads(stripped)
                if record.get("image_path"):
                    image_path = Path(str(record["image_path"])).expanduser()
                    if not image_path.is_absolute():
                        image_path = path.parent / image_path
                    record["image_path"] = str(image_path)
                records.append(record)
    return records


def _format_record(raw: dict, index: int) -> dict:
    board = game.normalize_board(raw["board"]) if raw.get("board") is not None else None
    best_moves = [int(move) for move in raw.get("best_moves") or (game.best_moves(board) if board else [])]
    image_base64 = raw.get("image_base64")
    if image_base64 is None:
        image_path = raw.get("image_path")
        if not image_path:
            raise ValueError("VL tic-tac-toe rows must contain image_base64 or image_path")
        image_base64 = _read_image_base64(Path(str(image_path)).expanduser())
    response = str(raw.get("response") or raw.get("reference") or "")
    return {
        "id": raw.get("id", f"vl-tictactoe-{index:05d}"),
        "prompt": str(
            raw.get("prompt") or "Describe the tic-tac-toe board and name the best next move for X in one sentence."
        ),
        "response": response,
        "reference": str(raw.get("reference") or response),
        "solutions": list(raw.get("solutions") or ([response] if response else [])),
        "target_keywords": list(raw.get("target_keywords") or []),
        "board": board,
        "best_moves": best_moves,
        "valid_moves": list(raw.get("valid_moves") or (game.legal_moves(board) if board else [])),
        "image_base64": str(image_base64),
    }


def _read_image_base64(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")
