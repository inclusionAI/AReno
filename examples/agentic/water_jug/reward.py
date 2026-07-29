"""Reward function for water-jug Agentic RL.

Implements ``reward_fn(record)``, called by AReno after each rollout to
score the model's trajectory.

Reward design:
  - Solved optimally:     1.0 + 0.1 efficiency bonus = 1.1
  - Solved with excess:   1.0 - 0.1 * (extra steps), floor 0.1
  - Not solved, closer:   0.5 * (1 - dist / initial_dist), range [0, 0.5]
  - Not solved, no progress / unsolvable: 0.0

The ``record`` argument is AReno's ``RewardRecord`` with:
  - ``source_record``: the original dataset item (dict with ``image``)
  - ``tool_calls``: list of ``{"name": ..., "arguments": ...}`` dicts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game


def reward_fn(record: Any) -> float:
    source = record.source_record
    image = source.get("image", source)
    caps = tuple(image.get("capacities", (3, 5)))
    target = int(image.get("target", 4))
    initial = tuple(image.get("initial_state", [0] * len(caps)))
    oracle_steps = int(image.get("oracle_steps", 0))

    state = initial
    action_count = 0
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "water_jug_action":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        action = arguments.get("action", "")
        if not action:
            continue
        try:
            state = game.apply_action(caps, state, action)
            action_count += 1
        except Exception:
            pass

    if game.is_goal(state, target):
        reward = 1.0
        if oracle_steps > 0 and action_count > oracle_steps:
            reward -= 0.1 * (action_count - oracle_steps)
            reward = max(reward, 0.1)
        elif oracle_steps > 0 and action_count <= oracle_steps:
            reward += 0.1
        return min(reward, 1.5)

    dist = game.bfs_distance(caps, state, target)
    if dist is None:
        return 0.0
    init_dist = game.bfs_distance(caps, initial, target)
    if init_dist is None or init_dist == 0:
        return 0.0
    proximity = 1.0 - (dist / init_dist)
    proximity = max(0.0, min(proximity, 1.0))
    return 0.5 * proximity