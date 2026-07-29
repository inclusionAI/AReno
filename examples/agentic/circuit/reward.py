"""Reward function for the circuit-diagnosis agentic example (issue #193).

The agent makes multiple turns of probe/submit calls. The reward function
scans all tool calls across all turns for the last `submit` call and checks
whether the submitted gate_id matches the faulty gate.

Returns a reward dict with:
- ``reward``: 1.0 for correct diagnosis, 0.0 otherwise.
- ``probes_used``: Number of probe calls made before submit (for metrics).
- ``submitted``: Whether the agent made a submit call at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def reward_fn(record: Any) -> dict[str, float]:
    """Score one diagnosis completion by extracting tool calls across all turns.

    Args:
        record: A record with ``source_record``, ``tool_calls``, and
            ``completion`` fields from the agentic rollout.

    Returns:
        A dict with ``reward`` (1.0 correct / 0.0 incorrect),
        ``probes_used`` (int), and ``submitted`` (0.0 or 1.0).
    """

    source = record.source_record
    faulty_gate_id = source["faulty_gate_id"]
    tool_calls = getattr(record, "tool_calls", None) or []

    # Count probes and find the last submit.
    probes_used = 0
    guessed = None
    submitted = False
    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name == "probe":
            probes_used += 1
        elif name == "submit":
            submitted = True
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if isinstance(arguments, dict):
                gate_id = arguments.get("gate_id")
                try:
                    guessed = int(gate_id)
                except (TypeError, ValueError):
                    guessed = None

    correct = guessed is not None and guessed == faulty_gate_id
    return {
        "reward": 1.0 if correct else 0.0,
        "probes_used": float(probes_used),
        "submitted": 1.0 if submitted else 0.0,
    }
