"""Reward function for the elevator dispatch agentic example.

The reward replays the agent's ``tool_calls`` against the deterministic
``game.build_state`` environment to recover episode metrics, then combines
delivered passengers, normalized waiting time, and invalid-action penalty.
This mirrors how DuelGrid/Codebreaker derive outcome scores from trajectories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

# reward weights (kept module-level so tests and tuning scripts can read them)
DELIVER_WEIGHT = 1.0
WAIT_WEIGHT = 0.3
INVALID_WEIGHT = 0.5


def reward_fn(record: Any) -> float:
    """Score one elevator episode: delivery rate minus waiting and invalid penalties."""

    source = dict(record.source_record)
    state = game.build_state(source)
    actions = _tool_actions(record)
    for action in actions:
        if game.is_terminal(state):
            break
        game.step(state, action)
    metrics = game.episode_metrics(state)
    return _score(metrics)


def _score(metrics: dict[str, Any]) -> float:
    """Map episode metrics to a scalar reward in roughly [-1, 1]."""

    total = max(metrics["total_passengers"], 1)
    delivery_rate = metrics["delivered"] / total
    wait_norm = metrics["mean_wait"] / max(metrics["horizon"], 1)
    invalid_rate = metrics["invalid_actions"] / total
    return DELIVER_WEIGHT * delivery_rate - WAIT_WEIGHT * wait_norm - INVALID_WEIGHT * invalid_rate


def _tool_actions(record: Any) -> list[dict[str, Any]]:
    """Extract ordered elevator actions from the trajectory's tool calls."""

    actions: list[dict[str, Any]] = []
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name not in ("move", "open_door", "close_door", "done"):
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # malformed arguments -> treat as an invalid action that errors out
                actions.append({"name": name, "direction": None} if name == "move" else {"name": name})
                continue
        if not isinstance(arguments, dict):
            arguments = {}
        action = {"name": name, **arguments}
        actions.append(action)
    return actions
