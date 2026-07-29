"""Reward function for the warehouse-picking agentic RL example.

Design:
- Each action (move, check, pick, submit) gets immediate feedback
- Partial success gives positive reward (even if not completed)
- Follows codebreaker/duelgrid pattern: incremental rewards per action
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import baseline_distance, build_state  # noqa: E402


def reward_fn(record) -> float:
    """Calculate reward with incremental rewards per action.

    Similar to codebreaker/duelgrid pattern:
    - Each move gives small reward based on progress
    - Each pick gives reward based on item correctness
    - Final submit gives completion bonus/penalty
    """

    source = dict(record.source_record)
    tool_calls = list(record.tool_calls)

    # Build state to compute baseline
    state = build_state(source)
    baseline = baseline_distance(state)
    order_dict = {item["sku"]: item["qty"] for item in source.get("order", [])}
    total_needed = sum(order_dict.values()) if order_dict else 1

    # Calculate incremental rewards from each action
    total_reward = 0.0
    cart = {}
    completed = False
    distance = 0
    action_count = {"move_to": 0, "check_shelf": 0, "pick_item": 0, "submit_order": 0}

    for msg in record.messages:
        if msg.get("role") != "tool":
            continue

        try:
            content = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue

        success = content.get("success", False)
        data = content.get("data", {})
        message = content.get("message", "")

        # Start with small negative for taking action (cost of exploration)
        action_reward = -0.01

        # Track which action was taken
        tool_name = msg.get("name", "")
        if tool_name in action_count:
            action_count[tool_name] += 1

        # Move actions: reward moving closer, penalize moving away or invalid
        if tool_name == "move_to":
            if success:
                # Check distance change if we have previous position
                if "distance" in data:
                    distance = data["distance"]
                    action_reward = 0.02  # Small positive for valid move
            else:
                action_reward = -0.05  # Penalty for invalid move

        # Check shelf: small reward for gathering information
        elif tool_name == "check_shelf":
            if success:
                action_reward = 0.01  # Small reward for checking inventory

        # Pick actions: reward correct picks, penalize mistakes
        elif tool_name == "pick_item":
            if success:
                # Update cart
                if "cart" in data:
                    cart = dict(data["cart"])

                # Calculate how much of the order is fulfilled
                fulfilled = sum(min(cart.get(sku, 0), qty) for sku, qty in order_dict.items())
                progress = fulfilled / total_needed if total_needed > 0 else 0

                if progress > 0:
                    action_reward = 0.1 * progress  # Scale by progress
                else:
                    action_reward = 0.05  # Picked something not in order
            else:
                # Penalty for picking wrong item, invalid qty, out of stock
                if "not on shelf" in message or "insufficient" in message:
                    action_reward = -0.1
                elif "invalid qty" in message:
                    action_reward = -0.05

        # Submit: big reward for completion, penalty for incomplete
        elif tool_name == "submit_order":
            if data.get("completed"):
                completed = True
                # Calculate efficiency bonus
                if distance > 0 and baseline > 0:
                    efficiency = min(baseline / max(distance, 1), 1.0)
                    # Full completion: 0.8 base + up to 0.2 for efficiency
                    action_reward = 0.8 + 0.2 * efficiency
                else:
                    action_reward = 1.0
            else:
                # Incomplete submission
                fulfilled = sum(min(cart.get(sku, 0), qty) for sku, qty in order_dict.items())
                progress = fulfilled / total_needed if total_needed > 0 else 0
                # Partial credit but negative
                action_reward = -0.3 + 0.2 * progress

        total_reward += action_reward

    # If no actions taken at all, give penalty
    if sum(action_count.values()) == 0:
        return -0.5

    # Ensure reward is in valid range
    return max(-1.0, min(1.0, total_reward))