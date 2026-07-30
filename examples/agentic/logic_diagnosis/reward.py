"""Three-layer reward for logic-circuit diagnosis trajectories.

Layer 1 (format): scores raw completion text for JSON proximity — guarantees
    per-sample variance even when tool-call parsing fails.
Layer 2 (interaction): rewards valid tool calls and probing the faulty gate.
Layer 3 (outcome): rewards correct diagnosis with efficiency bonus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import MAX_PROBES  # noqa: E402


def reward_fn(record) -> float:
    tool_calls = list(record.tool_calls or [])
    completion = getattr(record, "completion", "") or ""
    source = dict(record.source_record) if getattr(record, "source_record", None) else {}
    fault = source.get("fault", {})

    # ---- layer 1: format (raw completion text, always provides variance) ----
    fmt = _format_score(completion)

    # ---- layer 2: interaction (parsed tool calls) ----
    probes = 0
    hit_fault = False
    submitted = False
    correct = False

    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name", "")
        if name == "inspect_node":
            probes += 1
            args = _parse_args(call.get("arguments"))
            if isinstance(args.get("node_id"), int) and args["node_id"] == fault.get("node"):
                hit_fault = True
        elif name == "submit_diagnosis":
            submitted = True
            args = _parse_args(call.get("arguments"))
            node_id = args.get("node_id")
            fault_type_val = args.get("fault_type", "")
            if isinstance(node_id, int):
                correct = (
                    node_id == fault.get("node")
                    and (
                        (fault_type_val == "stuck_at_0" and fault.get("stuck_value") == 0)
                        or (fault_type_val == "stuck_at_1" and fault.get("stuck_value") == 1)
                    )
                )

    interaction = 0.0
    has_any_call = any(
        isinstance(c, dict)
        and c.get("name") in ("set_input_vector", "inspect_node", "submit_diagnosis")
        for c in tool_calls
    )
    if has_any_call:
        interaction += 0.1
    if hit_fault:
        interaction += 0.2
    if submitted and not correct:
        interaction += 0.05

    # ---- layer 3: outcome ----
    outcome = 0.0
    if submitted and correct:
        eff = max(0.0, 1.0 - probes / max(MAX_PROBES, 1))
        outcome = 0.5 + 0.2 * eff

    return fmt + interaction + outcome


def _format_score(text: str) -> float:
    """Score raw completion for JSON format proximity. Range [-0.05, 0.05].

    Different samples produce different text → format scores almost always
    differ → rewards have variance → advantages never all-zero.
    """
    if not text or not text.strip():
        return -0.05
    t = text.strip()
    s = 0.0
    if "{" in t:
        s += 0.015
    if "}" in t:
        s += 0.015
    if ":" in t:
        s += 0.005
    # Quoted strings (JSON keys or values)
    if '"' in t:
        s += 0.005
    # Boolean values
    if "true" in t.lower() or "false" in t.lower():
        s += 0.005
    # Integer in output (e.g. node_id)
    if any(ch.isdigit() for ch in t):
        s += 0.005
    if len(t) > 200:
        s -= 0.02  # long unstructured output is not compact JSON
    return max(-0.05, min(0.05, s))


def _parse_args(arguments):
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    if isinstance(arguments, dict):
        return arguments
    return {}