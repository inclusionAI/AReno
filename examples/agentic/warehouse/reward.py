"""Reward function for the warehouse-picking agentic RL example.

Rewards are based on order completion, picking accuracy, and path
efficiency relative to the BFS baseline. A small bonus is given for
using tools in a sensible order (query → move → pick → submit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import (  # noqa: E402
    generate_small,
    generate_medium,
    generate_hard,
    solve_baseline,
    compute_metrics,
    Order,
    WarehouseState,
)


def reward_fn(record: Any) -> float:
    """Score one warehouse-picking episode.

    Reward structure:
    - Base reward: 1.0 if order completed, -0.5 if not.
    - Penalty: -0.1 per picking mistake, -0.05 per invalid action.
    - Efficiency bonus: +0.2 if distance_ratio <= 1.5.
    - Tool-order bonus: +0.1 if tools used in query → move → pick → submit order.
    """

    source = dict(record.source_record)
    tool_calls = list(record.tool_calls)

    # Reconstruct warehouse state from the record.
    difficulty = source.get("difficulty", "small")
    seed = source.get("seed", 42)

    if difficulty == "medium":
        state = generate_medium(seed=seed)
    elif difficulty == "hard":
        state = generate_hard(seed=seed)
    else:
        state = generate_small(seed=seed)

    if "order" in source:
        state.order = Order(order_id="order_1", items=source["order"])

    # Replay tool calls to reconstruct state.
    _replay_tool_calls(state, tool_calls)

    # Compute metrics.
    baseline = solve_baseline(state)
    metrics = compute_metrics(state, baseline)

    # Base reward.
    reward = 1.0 if metrics["complete_orders"] else -0.5

    # Penalties.
    reward -= 0.1 * metrics["picking_mistakes"]
    reward -= 0.05 * metrics["invalid_actions"]

    # Efficiency bonus.
    ratio = metrics.get("distance_ratio")
    if ratio is not None and ratio <= 1.5:
        reward += 0.2

    # Tool-order bonus: check if tools were used in a sensible sequence.
    names = [call.get("name", "") for call in tool_calls]
    if _has_sensible_order(names) and metrics["complete_orders"]:
        reward += 0.1

    return reward


def _replay_tool_calls(state: WarehouseState, tool_calls: list[dict[str, Any]]) -> None:
    """Replay tool calls on a fresh state to reconstruct the final state."""

    from game import query_inventory, move, pick, submit

    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        if name == "query_inventory":
            query_inventory(state, args.get("shelf_id", ""))
        elif name == "move":
            move(state, args.get("direction", ""))
        elif name == "pick":
            pick(state, args.get("item", ""), args.get("quantity", 1))
        elif name == "submit":
            submit(state)


def _has_sensible_order(names: list[str]) -> bool:
    """Check if the tool sequence follows a query → move → pick → submit pattern."""

    if not names:
        return False
    # The first tool should be query or move, and submit should be last.
    if names[-1] != "submit":
        return False
    if names[0] not in ("query_inventory", "move"):
        return False
    # At least one pick should appear before submit.
    if "pick" not in names:
        return False
    return True