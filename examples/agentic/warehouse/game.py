"""Core environment logic for the warehouse-navigation agentic RL example."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

DIFFICULTY_CONFIGS: dict[str, dict[str, Any]] = {
    "small": {
        "rows": 2,
        "cols": 2,
        "sku_pool": ["S1", "S2", "S3", "S4"],
        "max_stock": 5,
        "order_size": 1,
    },
    "medium": {
        "rows": 3,
        "cols": 3,
        "sku_pool": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "max_stock": 4,
        "order_size": 1,
    },
    "hard": {
        "rows": 4,
        "cols": 3,
        "sku_pool": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
        "max_stock": 3,
        "order_size": 1,
    },
}


@dataclass
class ShelfInfo:
    """Static information for one shelf."""

    shelf_id: str
    row: int
    col: int
    stock: dict[str, int]


@dataclass
class WarehouseState:
    """Mutable runtime state for one independent navigation episode."""

    shelves: dict[str, ShelfInfo]
    adjacency: dict[str, list[str]]
    order: list[dict[str, Any]]
    agent_pos: str
    target_shelf: str = ""
    total_distance: int = 0
    invalid_actions: int = 0
    completed: bool = False


@dataclass
class ActionResult:
    """Structured result from one validated environment action."""

    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any, field_name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def validate_record(record: dict[str, Any]) -> None:
    """Validate one dataset record before any model request is created."""

    if not isinstance(record, dict):
        raise ValueError("warehouse record must be an object")

    rows = _positive_int(record.get("rows"), "rows")
    cols = _positive_int(record.get("cols"), "cols")
    if rows > 26:
        raise ValueError("rows must be at most 26 so shelf IDs remain alphabetic")

    sku_pool = record.get("sku_pool")
    if not isinstance(sku_pool, list) or not sku_pool or any(not isinstance(sku, str) or not sku for sku in sku_pool):
        raise ValueError("sku_pool must be a non-empty list of non-empty strings")
    if len(set(sku_pool)) != len(sku_pool):
        raise ValueError("sku_pool must not contain duplicate SKUs")

    _positive_int(record.get("max_stock"), "max_stock")
    if not _is_int(record.get("seed")):
        raise ValueError("seed must be an integer")

    difficulty = record.get("difficulty")
    if difficulty is not None and difficulty not in DIFFICULTY_CONFIGS:
        choices = ", ".join(DIFFICULTY_CONFIGS)
        raise ValueError(f"difficulty must be one of: {choices}")

    start_shelf = record.get("start_shelf")
    shelf_ids = {f"{chr(65 + row)}{col + 1}" for row in range(rows) for col in range(cols)}
    if not isinstance(start_shelf, str) or start_shelf not in shelf_ids:
        raise ValueError(f"start_shelf must identify a shelf in the {rows}x{cols} layout")

    order = record.get("order")
    if not isinstance(order, list) or not order:
        raise ValueError("order must be a non-empty list")
    seen_skus: set[str] = set()
    for index, item in enumerate(order):
        if not isinstance(item, dict):
            raise ValueError(f"order[{index}] must be an object")
        sku = item.get("sku")
        if not isinstance(sku, str) or sku not in sku_pool:
            raise ValueError(f"order[{index}].sku must be present in sku_pool")
        if sku in seen_skus:
            raise ValueError(f"order must not contain duplicate SKU {sku}")
        seen_skus.add(sku)
        _positive_int(item.get("qty"), f"order[{index}].qty")


def generate_layout(
    rows: int,
    cols: int,
    sku_pool: list[str],
    max_stock_per_sku: int,
    rng: random.Random,
) -> dict[str, Any]:
    """Deterministically generate shelf stock and a connected grid graph."""

    _positive_int(rows, "rows")
    _positive_int(cols, "cols")
    _positive_int(max_stock_per_sku, "max_stock_per_sku")
    if rows > 26:
        raise ValueError("rows must be at most 26")
    if not isinstance(sku_pool, list) or not sku_pool:
        raise ValueError("sku_pool must be non-empty")

    shelves: dict[str, ShelfInfo] = {}
    adjacency: dict[str, list[str]] = {}
    for row in range(rows):
        for col in range(cols):
            shelf_id = f"{chr(65 + row)}{col + 1}"
            sku_count = rng.randint(1, min(3, len(sku_pool)))
            skus = rng.sample(sku_pool, k=sku_count)
            stock = {sku: rng.randint(1, max_stock_per_sku) for sku in skus}
            shelves[shelf_id] = ShelfInfo(shelf_id, row, col, stock)

            neighbors: list[str] = []
            if col > 0:
                neighbors.append(f"{chr(65 + row)}{col}")
            if col < cols - 1:
                neighbors.append(f"{chr(65 + row)}{col + 2}")
            if row > 0:
                neighbors.append(f"{chr(64 + row)}{col + 1}")
            if row < rows - 1:
                neighbors.append(f"{chr(66 + row)}{col + 1}")
            adjacency[shelf_id] = neighbors

    shelf_ids = list(shelves)
    present_skus = {sku for shelf in shelves.values() for sku in shelf.stock}
    for sku in (candidate for candidate in sku_pool if candidate not in present_skus):
        shelf = shelves[rng.choice(shelf_ids)]
        shelf.stock[sku] = rng.randint(1, max_stock_per_sku)

    return {"shelves": shelves, "adjacency": adjacency}


def generate_order(
    shelves: dict[str, ShelfInfo],
    order_size: int,
    rng: random.Random,
    *,
    exclude_shelf: str = "",
) -> list[dict[str, Any]]:
    """Generate an order with qty=1, preferring SKUs not on the exclude_shelf."""

    _positive_int(order_size, "order_size")
    excluded = set(shelves.get(exclude_shelf, ShelfInfo("", 0, 0, {})).stock.keys()) if exclude_shelf else set()
    remote_stock: dict[str, int] = {}
    all_stock: dict[str, int] = {}
    for shelf in shelves.values():
        for sku in shelf.stock:
            all_stock[sku] = all_stock.get(sku, 0) + 1
            if sku not in excluded:
                remote_stock[sku] = remote_stock.get(sku, 0) + 1

    pool = remote_stock if len(remote_stock) >= order_size else all_stock
    if order_size > len(pool):
        raise ValueError(f"order_size {order_size} exceeds the {len(pool)} stocked SKUs")

    chosen = rng.sample(sorted(pool), k=order_size)
    return [{"sku": sku, "qty": 1} for sku in chosen]


def _bfs_distance(adjacency: dict[str, list[str]], start: str, target: str) -> int:
    if start == target:
        return 0
    visited = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, distance = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor == target:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return -1


def _find_target_shelf(
    shelves: dict[str, ShelfInfo],
    adjacency: dict[str, list[str]],
    start: str,
    sku: str,
) -> str:
    """Find the closest shelf with the given SKU."""

    candidates = sorted(
        shelf_id for shelf_id, shelf in shelves.items()
        if shelf.stock.get(sku, 0) > 0
    )
    if not candidates:
        raise ValueError(f"no shelf has SKU {sku}")
    best = candidates[0]
    best_dist = _bfs_distance(adjacency, start, best)
    for shelf_id in candidates[1:]:
        dist = _bfs_distance(adjacency, start, shelf_id)
        if dist >= 0 and (best_dist < 0 or dist < best_dist):
            best = shelf_id
            best_dist = dist
    return best


def build_state(record: dict[str, Any]) -> WarehouseState:
    """Build a fresh independent runtime state from a validated record."""

    validate_record(record)
    rng = random.Random(record["seed"])
    layout = generate_layout(
        rows=record["rows"],
        cols=record["cols"],
        sku_pool=list(record["sku_pool"]),
        max_stock_per_sku=record["max_stock"],
        rng=rng,
    )
    for item in record["order"]:
        available = sum(shelf.stock.get(item["sku"], 0) for shelf in layout["shelves"].values())
        if available < item["qty"]:
            raise ValueError(f"order SKU {item['sku']} requests {item['qty']} but total stock is {available}")

    sku = record["order"][0]["sku"]
    target = _find_target_shelf(
        layout["shelves"], layout["adjacency"], record["start_shelf"], sku
    )

    return WarehouseState(
        shelves=layout["shelves"],
        adjacency=layout["adjacency"],
        order=[dict(item) for item in record["order"]],
        agent_pos=record["start_shelf"],
        target_shelf=target,
    )


def _inactive_result(state: WarehouseState) -> ActionResult | None:
    if not state.completed:
        return None
    state.invalid_actions += 1
    return ActionResult(
        False,
        "order already completed",
        {"stage": "action_validation"},
    )


def move(state: WarehouseState, target_shelf_id: str) -> ActionResult:
    """Move one step to an adjacent shelf."""

    inactive = _inactive_result(state)
    if inactive is not None:
        return inactive
    if target_shelf_id not in state.shelves:
        state.invalid_actions += 1
        return ActionResult(
            False,
            f"unknown shelf: {target_shelf_id}",
            {"stage": "action_validation", "input": "shelf_id"},
        )
    if target_shelf_id not in state.adjacency.get(state.agent_pos, []):
        state.invalid_actions += 1
        return ActionResult(
            False,
            f"unreachable: {state.agent_pos} -> {target_shelf_id}",
            {
                "stage": "reachability_validation",
                "from": state.agent_pos,
                "to": target_shelf_id,
            },
        )

    previous = state.agent_pos
    state.agent_pos = target_shelf_id
    state.total_distance += 1
    return ActionResult(
        True,
        "moved",
        {
            "from": previous,
            "to": target_shelf_id,
            "distance": state.total_distance,
        },
    )


def submit_order(state: WarehouseState) -> ActionResult:
    """Validate position and complete the order."""

    inactive = _inactive_result(state)
    if inactive is not None:
        return inactive
    if state.agent_pos != state.target_shelf:
        state.invalid_actions += 1
        return ActionResult(
            False,
            f"not at target: at {state.agent_pos}, need {state.target_shelf}",
            {
                "stage": "completion_validation",
                "current": state.agent_pos,
                "target": state.target_shelf,
            },
        )

    state.completed = True
    return ActionResult(
        True,
        "order completed",
        {"completed": True, "distance": state.total_distance},
    )


def _tool_validation_error(
    state: WarehouseState,
    message: str,
    *,
    input_name: str | None = None,
) -> ActionResult:
    state.invalid_actions += 1
    data: dict[str, Any] = {"stage": "tool_validation"}
    if input_name is not None:
        data["input"] = input_name
    return ActionResult(False, message, data)


def execute_action(
    state: WarehouseState,
    tool_name: str,
    arguments: Any,
) -> ActionResult:
    """Validate tool arguments and execute exactly one environment action."""

    if not isinstance(arguments, dict):
        return _tool_validation_error(
            state,
            "tool arguments must be a JSON object",
            input_name="arguments",
        )

    if tool_name == "move_to":
        if set(arguments) != {"shelf_id"} or not isinstance(arguments.get("shelf_id"), str):
            return _tool_validation_error(
                state,
                "move_to requires exactly one string shelf_id",
                input_name="shelf_id",
            )
        return move(state, arguments["shelf_id"])

    if tool_name == "submit_order":
        if arguments:
            return _tool_validation_error(
                state,
                "submit_order does not accept arguments",
                input_name="arguments",
            )
        return submit_order(state)

    return _tool_validation_error(
        state,
        f"unknown tool: {tool_name}",
        input_name="tool_name",
    )


def baseline_distance(state: WarehouseState) -> int:
    """Return the BFS distance from start to target shelf."""

    return _bfs_distance(state.adjacency, state.agent_pos, state.target_shelf)


def baseline_action_count(state: WarehouseState) -> int:
    """Return the minimum actions needed: one move_to per step + one submit."""

    return baseline_distance(state) + 1


def remaining_distance(state: WarehouseState) -> int:
    """Return BFS distance from current position to target shelf."""

    return _bfs_distance(state.adjacency, state.agent_pos, state.target_shelf)


def state_metrics(
    state: WarehouseState,
    *,
    baseline: int,
) -> dict[str, Any]:
    """Return observable structured metrics for one episode state."""

    remaining = remaining_distance(state)
    if baseline > 0:
        progress: float = max(0.0, 1.0 - remaining / baseline)
    else:
        progress = 1.0 if remaining == 0 else 0.0
    return {
        "complete_orders": int(state.completed),
        "invalid_actions": state.invalid_actions,
        "distance": state.total_distance,
        "baseline_distance": baseline,
        "remaining_distance": remaining,
        "progress": progress,
    }


def make_prompt(record: dict[str, Any]) -> str:
    """Build a task prompt that directs the agent to navigate to the target."""

    order = ", ".join(f"{item['sku']} x{item['qty']}" for item in record["order"])
    target = record.get("target_shelf", "")
    target_hint = f" The item is located on shelf {target}." if target else ""
    return (
        f"You are in a warehouse with a {record['rows']}x{record['cols']} grid of shelves. "
        "Shelf IDs follow the pattern A1, A2, ..., B1, B2, etc. "
        f"Your order is: {order}. You start at shelf {record['start_shelf']}.{target_hint} "
        "Use move_to to navigate one adjacent shelf at a time toward the target shelf, "
        "then submit_order to complete the order when you arrive."
    )