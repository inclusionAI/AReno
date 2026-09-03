"""Dataset loader for the competition agentic example."""

from __future__ import annotations

import json
from pathlib import Path


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load diary entries and attach user profile to each record."""

    rows = default_loader(dataset_path)
    profile = _load_profile()
    records = []
    for row in rows:
        record = dict(row)
        record["user_profile"] = profile
        record["prompt"] = make_prompt(record, profile)
        records.append(record)
    return records


def make_prompt(record: dict, profile: dict) -> str:
    """Build the prompt combining user profile and diary entry."""

    personality = ", ".join(profile.get("personality", []))
    preferences = "; ".join(profile.get("preferences", []))
    return (
        f"You are a personal daily-summary assistant for {profile['name']}, "
        f"a {profile['age']}-year-old {profile.get('occupation', 'student')}. "
        f"Personality: {personality}. "
        f"Preferences: {preferences}. "
        f"\n\nToday's diary (mood: {record.get('mood', 'unknown')}):\n{record['diary']}\n\n"
        f"Generate a sandwich feedback: affirm today's effort, "
        f"gently point out one area for improvement with a specific suggestion, "
        f"then affirm again. Use tools to fetch the profile and generate content."
    )


def _load_profile() -> dict:
    """Load user profile from user_profile.json next to this file."""

    profile_path = Path(__file__).resolve().parent / "user_profile.json"
    if not profile_path.exists():
        return {"name": "User", "age": 20, "occupation": "student", "personality": [], "preferences": []}
    with profile_path.open(encoding="utf-8") as f:
        return json.load(f)
