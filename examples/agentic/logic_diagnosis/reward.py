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

    probes_used = 0
    probed_faulty = False
    submitted = False
    correct_diagnosis = False

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")

        if name == "inspect_node":
            probes_used += 1
            args = _parse_args(call.get("arguments"))
            node_id = args.get("node_id")
            if isinstance(node_id, int) and node_id == fault.get("node"):
                probed_faulty = True

        elif name == "submit_diagnosis":
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
        if probes_used > 0:
            return -0.3  # interacted but didn't submit — better than nothing
        return -1.0  # did nothing at all

    return score_episode(
        correct_diagnosis=correct_diagnosis,
        probes_used=probes_used,
        max_probes=MAX_PROBES,
        submitted=True,
    )


def _parse_args(arguments):
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    if isinstance(arguments, dict):
        return arguments
    return {}