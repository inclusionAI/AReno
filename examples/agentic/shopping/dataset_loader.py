"""Dataset loader for the multi-turn shopping tool-call example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import make_prompt  # noqa: E402

_TASK_DEFAULTS = {
    "rain commute": {
        "categories": ["jacket", "bottle"],
        "required_features_by_category": {
            "jacket": ["waterproof", "packable"],
            "bottle": ["insulated", "leakproof"],
        },
    },
    "trail day": {
        "categories": ["shoes", "bottle"],
        "required_features_by_category": {
            "shoes": ["trail", "water-resistant"],
            "bottle": ["collapsible", "lightweight"],
        },
    },
    "cold city": {
        "categories": ["jacket", "shoes"],
        "required_features_by_category": {
            "jacket": ["windproof", "warm"],
            "shoes": ["casual", "water-resistant"],
        },
    },
    "full travel": {
        "categories": ["jacket", "shoes", "bottle"],
        "required_features_by_category": {
            "jacket": ["waterproof", "packable"],
            "shoes": ["casual", "water-resistant"],
            "bottle": ["collapsible", "lightweight"],
        },
    },
}


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize JSONL shopping rows into prompt-bearing records."""

    rows = default_loader(dataset_path)
    records = []
    for row in rows:
        record = dict(row)
        defaults = _TASK_DEFAULTS.get(str(record.get("kit_name", "")), {})
        categories = record.get("categories") or defaults.get("categories") or []
        record["categories"] = [str(category) for category in categories]
        raw_required = (
            record.get("required_features_by_category") or defaults.get("required_features_by_category") or {}
        )
        record["required_features_by_category"] = _normalize_required_features(raw_required, record["categories"])
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records


def _normalize_required_features(raw_required: object, categories: list[str]) -> dict[str, list[str]]:
    if not isinstance(raw_required, dict):
        raw_required = {}
    normalized = {}
    for category in categories:
        features = raw_required.get(category) or []
        normalized[category] = [str(feature) for feature in features]
    return normalized
