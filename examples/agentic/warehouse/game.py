"""Core environment logic for the warehouse-picking agentic RL example."""

from __future__ import annotations

import heapq
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
        "order_size": 2,
    },
    "medium": {
        "rows": 3,
        "cols": 3,
        "sku_pool": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "max_stock": 4,
        "order_size": 3,
    },
    "hard": {
        "rows": 4,
        "cols": 3,
        "sku_pool": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"],
        "max_stock": 3,
        "order_size": 5,
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
    """Mutable runtime state for one independent picking episode."""

    shelves: dict[str, ShelfInfo]
    adjacency: dict[str, list[str]]
    order: list[dict[str, Any]]
    agent_pos: str
    cart: dict[str, int] = field(default_factory=dict)
    total_distance: int = 0
    invalid_actions: int = 0
    picking_errors: int = 0
    empty_shelf_checks: int = 0
    checked_shelves: set[str] = field(default_factory=set)
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
) -> list[dict[str, Any]]:
    """Generate an order whose quantities each fit on at least one shelf."""

    _positive_int(order_size, "order_size")
    total_stock: dict[str, int] = {}
    max_shelf_stock: dict[str, int] = {}
    for shelf in shelves.values():
        for sku, quantity in shelf.stock.items():
            total_stock[sku] = total_stock.get(sku, 0) + quantity
            max_shelf_stock[sku] = max(max_shelf_stock.get(sku, 0), quantity)
    if order_size > len(total_stock):
        raise ValueError(f"order_size {order_size} exceeds the {len(total_stock)} stocked SKUs")

    chosen = rng.sample(sorted(total_stock), k=order_size)
    return [{"sku": sku, "qty": rng.randint(1, min(3, max_shelf_stock[sku]))} for sku in chosen]


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

    return WarehouseState(
        shelves=layout["shelves"],
        adjacency=layout["adjacency"],
        order=[dict(item) for item in record["order"]],
        agent_pos=record["start_shelf"],
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


def query_inventory(state: WarehouseState, sku: str) -> ActionResult:
    """Return every current storage location for one SKU."""

    inactive = _inactive_result(state)
    if inactive is not None:
        return inactive
    if not isinstance(sku, str) or not sku:
        state.invalid_actions += 1
        return ActionResult(
            False,
            "sku must be a non-empty string",
            {"stage": "action_validation", "input": "sku"},
        )

    locations = [
        {"shelf_id": shelf_id, "qty": shelf.stock[sku]}
        for shelf_id, shelf in sorted(state.shelves.items())
        if shelf.stock.get(sku, 0) > 0
    ]
    return ActionResult(
        True,
        "inventory found" if locations else "sku not in stock",
        {"sku": sku, "locations": locations},
    )


def check_shelf(state: WarehouseState) -> ActionResult:
    """Inspect the current shelf and track checks with no requested stock."""

    inactive = _inactive_result(state)
    if inactive is not None:
        return inactive

    shelf = state.shelves[state.agent_pos]
    required = {item["sku"]: item["qty"] for item in state.order}
    requested_stock = {
        sku: min(quantity, required[sku] - state.cart.get(sku, 0))
        for sku, quantity in shelf.stock.items()
        if sku in required and required[sku] > state.cart.get(sku, 0)
    }
    state.checked_shelves.add(state.agent_pos)
    useful = bool(requested_stock)
    if not useful:
        state.empty_shelf_checks += 1

    return ActionResult(
        True,
        "requested stock found" if useful else "no requested stock on current shelf",
        {
            "shelf_id": state.agent_pos,
            "stock": dict(shelf.stock),
            "requested_stock": requested_stock,
            "useful": useful,
        },
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


def pick(state: WarehouseState, sku: str, qty: int) -> ActionResult:
    """Pick stock from the current shelf and record ordering mistakes."""

    inactive = _inactive_result(state)
    if inactive is not None:
        return inactive
    if not _is_int(qty) or qty <= 0:
        state.picking_errors += 1
        return ActionResult(
            False,
            f"invalid qty: {qty}",
            {"stage": "quantity_validation", "input": "qty"},
        )
    if state.agent_pos not in state.checked_shelves:
        state.invalid_actions += 1
        return ActionResult(
            False,
            f"inspect shelf {state.agent_pos} before picking",
            {
                "stage": "inspection_validation",
                "shelf_id": state.agent_pos,
            },
        )

    shelf = state.shelves[state.agent_pos]
    available = shelf.stock.get(sku, 0)
    if available <= 0:
        state.picking_errors += 1
        return ActionResult(
            False,
            f"sku {sku} not on shelf {state.agent_pos}",
            {"stage": "stock_validation", "sku": sku, "available": 0},
        )
    if qty > available:
        state.picking_errors += 1
        return ActionResult(
            False,
            f"insufficient stock: {sku} need {qty} have {available}",
            {
                "stage": "stock_validation",
                "sku": sku,
                "requested": qty,
                "available": available,
            },
        )

    required = {item["sku"]: item["qty"] for item in state.order}
    remaining_needed = required.get(sku, 0) - state.cart.get(sku, 0)
    mistake = sku not in required or qty > max(remaining_needed, 0)
    if mistake:
        state.picking_errors += 1

    shelf.stock[sku] -= qty
    if shelf.stock[sku] == 0:
        del shelf.stock[sku]
    state.cart[sku] = state.cart.get(sku, 0) + qty
    return ActionResult(
        True,
        "picked with mistake" if mistake else "picked",
        {
            "sku": sku,
            "qty": qty,
            "mistake": mistake,
            "cart": dict(state.cart),
        },
    )


def submit_order(state: WarehouseState) -> ActionResult:
    """Validate the cart and complete an exact order."""

    inactive = _inactive_result(state)
    if inactive is not None:
        return inactive
    required = {item["sku"]: item["qty"] for item in state.order}
    missing = {
        sku: quantity - state.cart.get(sku, 0)
        for sku, quantity in required.items()
        if state.cart.get(sku, 0) < quantity
    }
    extra = {
        sku: quantity - required.get(sku, 0) for sku, quantity in state.cart.items() if quantity > required.get(sku, 0)
    }
    if missing or extra:
        state.invalid_actions += 1
        return ActionResult(
            False,
            "order incomplete",
            {
                "stage": "completion_validation",
                "missing": missing,
                "extra": extra,
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

    if tool_name == "query_inventory":
        if set(arguments) != {"sku"} or not isinstance(arguments.get("sku"), str):
            return _tool_validation_error(
                state,
                "query_inventory requires exactly one string sku",
                input_name="sku",
            )
        return query_inventory(state, arguments["sku"])

    if tool_name == "check_shelf":
        if arguments:
            return _tool_validation_error(
                state,
                "check_shelf does not accept arguments",
                input_name="arguments",
            )
        return check_shelf(state)

    if tool_name == "pick_item":
        if set(arguments) != {"sku", "qty"}:
            return _tool_validation_error(
                state,
                "pick_item requires exactly sku and qty",
                input_name="arguments",
            )
        if not isinstance(arguments.get("sku"), str):
            return _tool_validation_error(
                state,
                "pick_item sku must be a string",
                input_name="sku",
            )
        if not _is_int(arguments.get("qty")):
            return _tool_validation_error(
                state,
                "pick_item qty must be an integer",
                input_name="qty",
            )
        return pick(state, arguments["sku"], arguments["qty"])

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


def _bfs_distance(state: WarehouseState, start: str, target: str) -> int:
    if start == target:
        return 0
    visited = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, distance = queue.popleft()
        for neighbor in state.adjacency.get(node, []):
            if neighbor == target:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return -1


def _route_fulfills_order(
    state: WarehouseState,
    shelf_ids: list[str],
    visited_mask: int,
) -> bool:
    required = {item["sku"]: item["qty"] for item in state.order}
    collected = {sku: 0 for sku in required}
    for index, shelf_id in enumerate(shelf_ids):
        if not visited_mask & (1 << index):
            continue
        shelf = state.shelves[shelf_id]
        for sku in required:
            collected[sku] += shelf.stock.get(sku, 0)
    return all(collected[sku] >= quantity for sku, quantity in required.items())


def baseline_route(state: WarehouseState) -> list[str]:
    """Return a deterministic minimum-distance route over sufficient stock."""

    required_skus = {item["sku"] for item in state.order}
    shelf_ids = sorted(
        shelf_id
        for shelf_id, shelf in state.shelves.items()
        if any(shelf.stock.get(sku, 0) > 0 for sku in required_skus)
    )
    if not shelf_ids:
        raise ValueError("no stocked shelf can satisfy the order")

    start_mask = 0
    start_path: tuple[str, ...] = ()
    if state.agent_pos in shelf_ids:
        start_index = shelf_ids.index(state.agent_pos)
        start_mask = 1 << start_index
        start_path = (state.agent_pos,)

    start_rank = (0, len(start_path), start_path)
    best: dict[tuple[str, int], tuple[int, int, tuple[str, ...]]] = {(state.agent_pos, start_mask): start_rank}
    queue: list[tuple[int, int, tuple[str, ...], str, int]] = [(*start_rank, state.agent_pos, start_mask)]

    while queue:
        distance, stop_count, path, position, visited_mask = heapq.heappop(queue)
        if best.get((position, visited_mask)) != (distance, stop_count, path):
            continue
        if _route_fulfills_order(state, shelf_ids, visited_mask):
            return list(path)

        for index, shelf_id in enumerate(shelf_ids):
            bit = 1 << index
            if visited_mask & bit:
                continue
            step_distance = _bfs_distance(state, position, shelf_id)
            if step_distance < 0:
                continue
            new_path = (*path, shelf_id)
            rank = (distance + step_distance, len(new_path), new_path)
            key = (shelf_id, visited_mask | bit)
            if key in best and best[key] <= rank:
                continue
            best[key] = rank
            heapq.heappush(
                queue,
                (*rank, shelf_id, visited_mask | bit),
            )

    raise ValueError("order stock is unreachable")


def baseline_distance(state: WarehouseState) -> int:
    """Return the exact minimum movement distance needed to reach enough stock."""

    position = state.agent_pos
    distance = 0
    for target in baseline_route(state):
        step_distance = _bfs_distance(state, position, target)
        if step_distance < 0:
            raise ValueError(f"order stock on shelf {target} is unreachable")
        distance += step_distance
        position = target
    return distance


def baseline_action_count(state: WarehouseState) -> int:
    """Return actions in a shortest-route query, inspect, pick, and submit plan."""

    remaining = {item["sku"]: item["qty"] for item in state.order}
    pick_actions = 0
    inspected_shelves = 0
    for shelf_id in baseline_route(state):
        used_shelf = False
        for sku in remaining:
            quantity = min(state.shelves[shelf_id].stock.get(sku, 0), remaining[sku])
            if quantity <= 0:
                continue
            remaining[sku] -= quantity
            pick_actions += 1
            used_shelf = True
        inspected_shelves += int(used_shelf)

    if any(quantity > 0 for quantity in remaining.values()):
        raise ValueError("baseline route does not satisfy the order")
    return len(state.order) + baseline_distance(state) + inspected_shelves + pick_actions + 1


def cart_progress(state: WarehouseState) -> float:
    """Return the fulfilled fraction of required order quantity."""

    required = {item["sku"]: item["qty"] for item in state.order}
    total = sum(required.values())
    if total <= 0:
        return 0.0
    fulfilled = sum(min(state.cart.get(sku, 0), quantity) for sku, quantity in required.items())
    return fulfilled / total


def state_metrics(
    state: WarehouseState,
    *,
    baseline: int,
) -> dict[str, Any]:
    """Return observable structured metrics for one episode state."""

    if baseline > 0:
        distance_ratio: float | None = state.total_distance / baseline
        efficiency = min(baseline / max(state.total_distance, 1), 1.0)
    else:
        distance_ratio = 1.0 if state.total_distance == 0 else None
        efficiency = 1.0 if state.total_distance == 0 else 0.0
    return {
        "complete_orders": int(state.completed),
        "picking_mistakes": state.picking_errors,
        "invalid_actions": state.invalid_actions,
        "empty_shelf_checks": state.empty_shelf_checks,
        "distance": state.total_distance,
        "baseline_distance": baseline,
        "distance_ratio": distance_ratio,
        "distance_efficiency": efficiency,
        "cart_progress": cart_progress(state),
    }


def make_prompt(record: dict[str, Any]) -> str:
    """Build a task prompt that directs the agent to query SKU locations."""

    order = ", ".join(f"{item['sku']} x{item['qty']}" for item in record["order"])
    return (
        f"You are in a warehouse with a {record['rows']}x{record['cols']} grid of shelves. "
        "Shelf IDs follow the pattern A1, A2, ..., B1, B2, etc. "
        f"Your order is: {order}. You start at shelf {record['start_shelf']}. "
        "Use query_inventory to find storage locations for each required SKU, move_to to move one "
        "adjacent shelf, check_shelf to verify stock after arrival, pick_item to collect verified "
        "stock, and submit_order only when the cart exactly matches the order."
    )
