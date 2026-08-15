"""Minimal reward module defining both reward_fn and reward_batch.

Run with:
    areno reward inspect --path examples/reward/batch_reward_example.py --fixtures examples/reward/fixtures.jsonl
"""

from areno.api.rewards import RewardRecord


def reward_fn(record: RewardRecord) -> float:
    """Per-example path: 1.0 if the completion matches the gold answer."""
    return 1.0 if record.completion.strip() == str(record.answer).strip() else 0.0


def reward_batch(records: list[RewardRecord]) -> list[float]:
    """Batched path: same contract, one pass over the list."""
    return [1.0 if r.completion.strip() == str(r.answer).strip() else 0.0 for r in records]
