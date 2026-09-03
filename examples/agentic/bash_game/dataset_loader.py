"""Dataset loader for the Bash game (巴什博弈) single-turn example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import format_prompt, normalize_record  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load raw position JSONL and attach the single-turn prompt.

    The prompt exposes only the visible state (``n`` stones, ``m`` max-take);
    the oracle move / ``winning`` flag stay hidden in the record for the reward.
    """
    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        try:
            record = normalize_record(dict(row))
            record["prompt"] = format_prompt(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid bash-game record {index}: {exc}") from exc
        records.append(record)
    return records