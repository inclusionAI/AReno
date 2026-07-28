"""Reward function for the circuit-diagnosis agentic example (issue #193)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def reward_fn(record: Any) -> float:
    """Score one diagnosis completion by extracting the submit tool call.

    Args:
        record: A record with ``source_record``, ``tool_calls``, and
            ``completion`` fields from the agentic rollout.

    Returns:
        1.0 if the agent correctly identified the faulty gate,
        0.0 otherwise. Partial credit for fewer probes is handled
        by :func:`circuit.score_diagnosis` when used programmatically.
    """

    source = record.source_record
    faulty_gate_id = source["faulty_gate_id"]
    guessed = _tool_gate_id(record)
    if guessed is None:
        return 0.0
    return 1.0 if guessed == faulty_gate_id else 0.0


def _tool_gate_id(record: Any) -> int | None:
    """Extract the gate_id from the last submit tool call."""

    for call in reversed(record.tool_calls):
        name = call.get("name") if isinstance(call, dict) else None
        if name != "submit":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict):
            gate_id = arguments.get("gate_id")
            try:
                return int(gate_id)
            except (TypeError, ValueError):
                return None
    return None
