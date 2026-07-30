"""Reward function for the circuit-diagnosis agentic example (issue #193).

The agent makes multiple turns of probe/submit calls. The reward function
scores only the first, valid ``submit`` call in a turn and checks whether its
gate ID matches the hidden faulty gate.

Returns a float: 1.0 for correct diagnosis, 0.0 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def reward_fn(record: Any) -> float:
    """Score one diagnosis completion from an executed submit action.

    Raw model-emitted calls are not enough: unexecuted extra calls and invalid
    submissions never receive credit.

    Args:
        record: A reward record with ``source_record`` and a multi-turn
            ``trace`` from the agentic rollout.

    Returns:
        1.0 if the agent correctly identified the faulty gate,
        0.0 otherwise.
    """

    source = record.source_record
    faulty_gate_id = source["faulty_gate_id"]
    guessed = _executed_submit_gate_id(record, source)
    if guessed is None:
        return 0.0
    return 1.0 if guessed == faulty_gate_id else 0.0


def _event_value(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _first_tool_actions(record: Any) -> list[tuple[str, dict[str, Any]]]:
    """Return the first model-emitted tool call from each trajectory turn."""

    actions: list[tuple[str, dict[str, Any]]] = []
    first_call_seen = False
    for event in getattr(record, "trace", None) or []:
        event_type = _event_value(event, "type")
        if event_type == "request":
            first_call_seen = False
            continue
        if event_type != "assistant_tool_call" or first_call_seen:
            continue
        first_call_seen = True
        name = _event_value(event, "name")
        arguments = _event_value(event, "arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(name, str) and isinstance(arguments, dict):
            actions.append((name, arguments))
    return actions


def _valid_gate_id(arguments: dict[str, Any], source: dict[str, Any] | None) -> int | None:
    gate_id = arguments.get("gate_id")
    if isinstance(gate_id, bool) or not isinstance(gate_id, int):
        return None
    if source is not None:
        num_inputs = source.get("num_inputs")
        num_gates = source.get("num_gates")
        if isinstance(num_inputs, int) and gate_id < num_inputs:
            return None
        if isinstance(num_gates, int) and gate_id >= num_gates:
            return None
    return gate_id


def _executed_submit_gate_id(record: Any, source: dict[str, Any] | None = None) -> int | None:
    """Extract the last valid submit that was first in its model turn."""

    guessed: int | None = None
    for name, arguments in _first_tool_actions(record):
        if name != "submit":
            continue
        gate_id = _valid_gate_id(arguments, source)
        if gate_id is not None:
            guessed = gate_id
    return guessed


def _accepted_probe_count(record: Any) -> int:
    """Count bounded probe actions (one possible execution per turn)."""

    return sum(1 for name, _arguments in _first_tool_actions(record) if name == "probe")


def analyze_tool_calls(record: Any) -> dict[str, Any]:
    """Standalone helper to analyze tool call patterns for debugging.

    This function does NOT affect training.  It returns structured info
    about the agent's behaviour for offline inspection.

    Returns:
        Dict with ``probes_used``, ``submitted``, ``guessed_gate_id``.
    """

    probes_used = _accepted_probe_count(record)
    guessed = _executed_submit_gate_id(record, getattr(record, "source_record", None))
    submitted = guessed is not None
    return {"probes_used": probes_used, "submitted": submitted, "guessed_gate_id": guessed}


def summarize_diagnoses(records: list[Any]) -> dict[str, float]:
    """Aggregate stable evaluation metrics from completed reward records.

    This stays outside ``reward_fn`` because AReno's reward contract requires
    one scalar per trajectory. The returned values are suitable for logging or
    comparing evaluation runs.
    """

    if not records:
        return {"diagnosis_accuracy": 0.0, "average_probes": 0.0, "submission_rate": 0.0}
    analyses = [analyze_tool_calls(record) for record in records]
    return {
        "diagnosis_accuracy": fmean(reward_fn(record) for record in records),
        "average_probes": fmean(item["probes_used"] for item in analyses),
        "submission_rate": fmean(float(item["submitted"]) for item in analyses),
    }
