"""Outcome-dominant reward with observable diagnostic-progress shaping.

The process term measures how much the agent's *executed* observations shrink
the set of possible stuck-at faults.  It never rewards access to the hidden
fault directly: every candidate is filtered only by the output or node value
returned by the environment for a chosen tool call.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import MAX_PROBES, evaluate, verify_diagnosis  # noqa: E402


_GATE_TYPES = frozenset(("and", "or", "not"))
_MAX_PROCESS_REWARD = 0.30
_PROBE_PENALTY = 0.02


def reward_fn(record) -> float:
    """Score diagnosis correctness and information gained from valid tools.

    Correct diagnoses are always positive.  Every incorrect or unfinished
    trajectory is non-positive, preserving ``rollout/accuracy`` as the exact
    diagnosis pass rate.  Within those failing trajectories, observations
    that eliminate more valid fault hypotheses receive a less-negative reward
    so GSPO can rank otherwise unsuccessful rollouts.
    """

    source = dict(record.source_record) if record.source_record is not None else {}
    nodes = list(source.get("nodes") or [])
    fault = source.get("fault") or {}
    tool_calls = list(record.tool_calls or [])
    max_probes = max(int(source.get("max_probes", MAX_PROBES)), 1)

    progress, probes_used, submitted, correct_diagnosis, valid_actions = _replay_observations(
        tool_calls,
        nodes,
        fault,
    )
    if not valid_actions:
        return -1.0

    if correct_diagnosis:
        efficiency = max(0.0, 1.0 - probes_used / max_probes)
        return 0.8 + 0.2 * efficiency

    process_reward = max(0.0, _MAX_PROCESS_REWARD * progress - _PROBE_PENALTY * probes_used)
    if submitted:
        # Even a fully informative but incorrect final answer stays negative.
        return min(process_reward - 0.35, -0.05)
    return min(process_reward - 0.55, -0.25)


def _replay_observations(
    tool_calls: list[dict],
    nodes: list[dict],
    fault: dict,
) -> tuple[float, int, bool, bool, bool]:
    """Replay valid calls and return progress, outcome, and validity state."""

    candidates = _candidate_faults(nodes)
    if not candidates or not isinstance(fault.get("node"), int):
        return 0.0, 0, False, False, False

    output_node = next((node for node in nodes if node.get("type") == "output"), None)
    if output_node is None or not isinstance(output_node.get("id"), int):
        return 0.0, 0, False, False, False

    initial_candidate_count = len(candidates)
    current_inputs: list[bool] | None = None
    probes_used = 0
    valid_actions = False
    submitted = False
    correct_diagnosis = False
    node_by_id = {node.get("id"): node for node in nodes if isinstance(node.get("id"), int)}

    for call in tool_calls:
        if submitted or not isinstance(call, dict):
            continue
        name = call.get("name")
        args = _parse_args(call.get("arguments"))

        if name == "set_input_vector":
            inputs = _input_vector(args, nodes)
            if inputs is None:
                continue
            current_inputs = inputs
            observed = evaluate(nodes, inputs, fault)[output_node["id"]]
            candidates = _filter_candidates(nodes, candidates, inputs, output_node["id"], observed)
            valid_actions = True

        elif name == "inspect_node":
            node_id = args.get("node_id")
            node = node_by_id.get(node_id)
            if current_inputs is None or not isinstance(node_id, int) or node is None or node.get("type") not in _GATE_TYPES:
                continue
            observed = evaluate(nodes, current_inputs, fault)[node_id]
            candidates = _filter_candidates(nodes, candidates, current_inputs, node_id, observed)
            probes_used += 1
            valid_actions = True

        elif name == "submit_diagnosis":
            node_id = args.get("node_id")
            fault_type = args.get("fault_type")
            if not isinstance(node_id, int) or fault_type not in ("stuck_at_0", "stuck_at_1"):
                continue
            submitted = True
            valid_actions = True
            correct_diagnosis = verify_diagnosis(nodes, fault, node_id, fault_type)

    progress = _information_progress(initial_candidate_count, len(candidates))
    return progress, probes_used, submitted, correct_diagnosis, valid_actions


def _candidate_faults(nodes: list[dict]) -> list[dict[str, int]]:
    """Enumerate the fault hypotheses available to the diagnostician."""

    return [
        {"node": node["id"], "stuck_value": stuck_value}
        for node in nodes
        if node.get("type") in _GATE_TYPES and isinstance(node.get("id"), int)
        for stuck_value in (0, 1)
    ]


def _filter_candidates(
    nodes: list[dict],
    candidates: list[dict[str, int]],
    inputs: list[bool],
    observed_node_id: int,
    observed_value: bool,
) -> list[dict[str, int]]:
    """Keep hypotheses that predict the actual observed value."""

    return [
        candidate
        for candidate in candidates
        if evaluate(nodes, inputs, candidate)[observed_node_id] == observed_value
    ]


def _information_progress(initial_candidate_count: int, candidate_count: int) -> float:
    """Return normalized information gain from the remaining hypothesis count."""

    if initial_candidate_count <= 1 or candidate_count >= initial_candidate_count or candidate_count <= 0:
        return 0.0
    return math.log(initial_candidate_count / candidate_count) / math.log(initial_candidate_count)


def _input_vector(args: dict, nodes: list[dict]) -> list[bool] | None:
    """Match ``set_input_vector`` validation and coercion in ``run_agent``."""

    inputs_raw = args.get("inputs")
    if not isinstance(inputs_raw, list):
        return None
    n_inputs = sum(node.get("type") == "input" for node in nodes)
    inputs = [bool(value) for value in inputs_raw[:n_inputs]]
    inputs.extend([False] * (n_inputs - len(inputs)))
    return inputs


def _parse_args(arguments) -> dict:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return arguments if isinstance(arguments, dict) else {}
