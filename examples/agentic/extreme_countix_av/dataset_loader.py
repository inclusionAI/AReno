"""Dataset loader for the Extreme Countix-AV agentic example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import discover_samples, load_manifest, prompt_text  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load an Extreme Countix-AV root or a generated JSONL manifest."""

    del default_loader
    path = Path(dataset_path).expanduser().resolve()
    if path.is_dir():
        records = [sample.as_record(path) for sample in discover_samples(path)]
        for record in records:
            record["video_path"] = str((path / record["video_path"]).resolve())
            record["audio_path"] = str((path / record["audio_path"]).resolve())
    else:
        records = load_manifest(path)
    return [_format_record(record) for record in records]


def _format_record(record: dict) -> dict:
    action = str(record["action_class"])
    count = int(record["repetition_count"])
    response = f"{action}: {count} repetitions"
    return {
        **record,
        "prompt": prompt_text(),
        "response": response,
        "reference": response,
        "solutions": [response],
    }
