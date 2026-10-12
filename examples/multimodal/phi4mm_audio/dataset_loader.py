"""JSONL loader for Phi-4 multimodal audio training examples."""

from __future__ import annotations

import json
from pathlib import Path


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load rows containing ``audio_path``, ``prompt``, and ``response``."""

    del default_loader
    manifest = Path(dataset_path).resolve()
    records = []
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                audio_path = Path(row["audio_path"])
                prompt = str(row["prompt"])
                response = str(row["response"])
            except KeyError as exc:
                raise ValueError(f"line {line_number} is missing required field {exc.args[0]!r}") from exc
            if not audio_path.is_absolute():
                audio_path = manifest.parent / audio_path
            records.append(
                {
                    **row,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio_url", "audio_url": {"url": str(audio_path)}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "solutions": [response],
                }
            )
    return records
