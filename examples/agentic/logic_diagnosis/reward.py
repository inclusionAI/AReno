"""Outcome and efficiency reward for logic-circuit diagnosis trajectories.

Gives partial credit for probing the faulty gate so the model has
a gradient signal even before it learns to submit a diagnosis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import MAX_PROBES, score_episode  # noqa: E402


def reward_fn(record) -> float:
    """Process + outcome reward: partial credit for probing the faulty gate."""

    tool_calls = list(record.tool_calls or [])
    source = dict(record.source_record) if record.source_record is not None else {}
    fault = source.get("fault", {})

    interactions = 0  # any valid tool call counts
    probes_used = 0
    submitted = False
    correct_diagnosis = False

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")

        if name == "set_input_vector":
            interactions += 1
        elif name == "inspect_node":
            interactions += 1
            probes_used += 1
            args = _parse_args(call.get("arguments"))
            node_id = args.get("node_id")
            if isinstance(node_id, int) and node_id == fault.get("node"):
                interactions += 1  # bonus for finding the right gate

        elif name == "submit_diagnosis":
            interactions += 1
            submitted = True
            args = _parse_args(call.get("arguments"))
            node_id = args.get("node_id")
            fault_type = args.get("fault_type")
            if isinstance(node_id, int) and isinstance(fault_type, str):
                correct_diagnosis = (
                    node_id == fault.get("node")
                    and (
                        (fault_type == "stuck_at_0" and fault.get("stuck_value") == 0)
                        or (fault_type == "stuck_at_1" and fault.get("stuck_value") == 1)
                    )
                )

    if not submitted:
        if interactions > 0:
            return -0.2  # interacted but didn't submit
        # No tool calls parsed. Use raw completion text to create per-sample
        # variance so advantages are never all-zero (prevents deadlock).
        completion = getattr(record, "completion", "") or ""
        return _no_interaction_penalty(completion)

    return score_episode(
        correct_diagnosis=correct_diagnosis,
        probes_used=probes_used,
        max_probes=MAX_PROBES,
        submitted=True,
    )


def _no_interaction_penalty(text: str) -> float:
    """Penalty for producing zero valid tool calls. Range [-1.0, -0.3].

    Two samples with different completion text get different penalties →
    advantages never all-zero → no deadlock.
    """
    t = text.strip() if text else ""
    if not t:
        return -1.0  # empty output — worst
    p = -0.5  # baseline: produced text, but no recognizable JSON
    if "{" in t:
        p += 0.05
    if "}" in t:
        p += 0.05
    if ":" in t:
        p += 0.03
    if '"' in t:
        p += 0.03
    if "true" in t.lower() or "false" in t.lower():
        p += 0.02
    if any(ch.isdigit() for ch in t):
        p += 0.02
    return max(-1.0, min(-0.3, p))


def _parse_args(arguments):
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    if isinstance(arguments, dict):
        return arguments
    return {}