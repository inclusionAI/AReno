"""Outcome and efficiency reward for logic-circuit diagnosis trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import MAX_PROBES, score_episode  # noqa: E402


def reward_fn(record) -> float:
    """Reward efficient, correct diagnosis and penalize invalid or missing ones.

    Extracts tool calls from ``record.tool_calls``, counts probes, parses the
    final ``submit_diagnosis``, and delegates to ``score_episode``.
    """

    tool_calls = list(record.tool_calls or [])

    probes_used = sum(
        1 for call in tool_calls
        if isinstance(call, dict) and call.get("name") == "inspect_node"
    )

    submitted = False
    correct_diagnosis = False

    for call in tool_calls:
        if not isinstance(call, dict) or call.get("name") != "submit_diagnosis":
            continue
        submitted = True
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        node_id = args.get("node_id")
        fault_type = args.get("fault_type")

        source = dict(record.source_record) if record.source_record is not None else {}
        fault = source.get("fault", {})

        if isinstance(node_id, int) and isinstance(fault_type, str) and isinstance(fault, dict):
            correct_diagnosis = (
                node_id == fault.get("node")
                and (
                    (fault_type == "stuck_at_0" and fault.get("stuck_value") == 0)
                    or (fault_type == "stuck_at_1" and fault.get("stuck_value") == 1)
                )
            )

    if not submitted:
        return -1.0

    return score_episode(
        correct_diagnosis=correct_diagnosis,
        probes_used=probes_used,
        max_probes=MAX_PROBES,
        submitted=True,
    )