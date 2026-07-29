"""Save and load versioned rollout records for replay training.

A ``RolloutRecord`` captures everything needed to reconstruct a single
``TrainSequence`` without running the model: tokens, masks, logprobs,
advantages, reward, and metadata. Records are stored as JSON Lines (one
JSON object per line) so they stream naturally and are easy to inspect.

.. note::

    Replay is a **debugging and comparison path**, not a general
    checkpoint replacement. It does not restore optimizer state, model
    weights, or critic training — it only reproduces the *batch* that
    was fed to ``Trainer.train()`` so loss functions and hyperparameters
    can be iterated without re-running rollout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Bump when the on-disk schema changes incompatibly.
REPLAY_FORMAT_VERSION = 1


@dataclass(slots=True)
class RolloutRecord:
    """One rollout sequence serialized for offline replay.

    The core fields (``tokens``, ``prompt_mask``, ``loss_mask``, ``logprobs``,
    ``advantages``, ``reward``, ``eos_token_id``) map 1:1 to ``TrainSequence``
    fields. The remaining fields provide provenance for debugging.
    """

    format_version: int
    epoch: int
    step: int
    prompt_index: int
    sample_index: int
    tokens: list[int]
    prompt_mask: list[bool]
    loss_mask: list[bool]
    logprobs: list[float]
    advantages: list[float]
    reward: float
    eos_token_id: int
    metadata: dict[str, Any]


def save_rollout_records(path: str | Path, records: list[RolloutRecord]) -> None:
    """Write rollout records as JSON Lines."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_rollout_records(path: str | Path) -> list[RolloutRecord]:
    """Read and validate rollout records from a JSON Lines file.

    Raises ``ValueError`` on version mismatch, missing fields, or misaligned
    tensor lengths -- never silently coerces.
    """

    path = Path(path)
    if not path.exists():
        raise ValueError(f"replay file not found: {path}")
    records: list[RolloutRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record_dict = json.loads(line)
            records.append(_validate_and_build(record_dict, lineno))
    if not records:
        raise ValueError(f"replay file is empty: {path}")
    return records


_REQUIRED_FIELDS = (
    "tokens",
    "prompt_mask",
    "loss_mask",
    "logprobs",
    "advantages",
    "reward",
    "eos_token_id",
)

_LENGTH_FIELDS = ("prompt_mask", "loss_mask", "logprobs", "advantages")


def _validate_and_build(d: dict, lineno: int) -> RolloutRecord:
    """Validate one JSON object and build a :class:`RolloutRecord`."""

    # 1. Version check -- reject, don't coerce.
    version = d.get("format_version")
    if version is None:
        raise ValueError(f"line {lineno}: missing format_version")
    if version != REPLAY_FORMAT_VERSION:
        raise ValueError(
            f"line {lineno}: format_version {version} is incompatible "
            f"(expected {REPLAY_FORMAT_VERSION})"
        )

    # 2. Required fields.
    for field in _REQUIRED_FIELDS:
        if field not in d:
            raise ValueError(f"line {lineno}: missing required field '{field}'")

    # 3. Length alignment -- tokens, masks, logprobs, advantages must match.
    n = len(d["tokens"])
    for field in _LENGTH_FIELDS:
        if len(d[field]) != n:
            raise ValueError(
                f"line {lineno}: '{field}' length {len(d[field])} "
                f"does not match tokens length {n}"
            )

    return RolloutRecord(
        format_version=int(d["format_version"]),
        epoch=int(d.get("epoch", 0)),
        step=int(d.get("step", 0)),
        prompt_index=int(d.get("prompt_index", 0)),
        sample_index=int(d.get("sample_index", 0)),
        tokens=list(d["tokens"]),
        prompt_mask=list(d["prompt_mask"]),
        loss_mask=list(d["loss_mask"]),
        logprobs=list(d["logprobs"]),
        advantages=list(d["advantages"]),
        reward=float(d["reward"]),
        eos_token_id=int(d["eos_token_id"]),
        metadata=dict(d.get("metadata", {})),
    )