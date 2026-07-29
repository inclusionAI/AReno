"""Core environment logic for the warehouse-picking agentic RL example."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShelfInfo:
    """Static information for one shelf."""

    shelf_id: str
    row: int
    col: int
    stock: dict[str, int]


@dataclass
class WarehouseState:
    """Full runtime state for one picking task."""

    shelves: dict[str, ShelfInfo]
    adjacency: dict[str, list[str]]
    order: list[dict[str, Any]]
    agent_pos: str
    cart: dict[str, int] = field(default_factory=dict)
    total_distance: int = 0
    invalid_actions: int = 0
    picking_errors: int = 0
    completed: bool = False


@dataclass
class ActionResult:
    """Return value of one environment action."""

    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


def generate_layout(
    rows: int, cols: int, sku_pool: list[str], max_stock_per_sku: int, rng: random.Random
) -> dict[str, Any]:
    """Deterministically generate shelf layout, stock, and adjacency graph."""

    shelves: dict[str, ShelfInfo] = {}
    adjacency: dict[str, list[str]] = {}
    for r in range(rows):
        for c in range(cols):
            shelf_id = f"{chr(65 + r)}{c + 1}"
            k = rng.randint(1, min(3, len(sku_pool)))
            skus = rng.sample(sku_pool, k=k)
            stock = {sku: rng.randint(1, max_stock_per_sku) for sku in skus}
            shelves[shelf_id] = ShelfInfo(shelf_id, r, c, stock)
            neighbors: list[str] = []
            if c > 0:
                neighbors.append(f"{chr(65 + r)}{c}")
            if c < cols - 1:
                neighbors.append(f"{chr(65 + r)}{c + 2}")
            if r > 0:
                neighbors.append(f"{chr(64 + r)}{c + 1}")
            if r < rows - 1:
                neighbors.append(f"{chr(66 + r)}{c + 1}")
            adjacency[shelf_id] = neighbors
    return {"shelves": shelves, "adjacency": adjacency}


def generate_order(
    shelves: dict[str, ShelfInfo], order_size: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Randomly pick SKUs from all shelves to form an order."""

    all_skus: set[str] = set()
    for shelf in shelves.values():
        all_skus.update(shelf.stock.keys())
    chosen = rng.sample(sorted(all_skus), k=min(order_size, len(all_skus)))
    return [{"sku": sku, "qty": rng.randint(1, 3)} for sku in chosen]


def build_state(record: dict[str, Any]) -> WarehouseState:
    """Reconstruct runtime state from a JSONL record."""

    rng = random.Random(record.get("seed", 0))
    layout = generate_layout(
        rows=record["rows"],
        cols=record["cols"],
        sku_pool=record["sku_pool"],
        max_stock_per_sku=record["max_stock"],
        rng=rng,
    )
    return WarehouseState(
        shelves=layout["shelves"],
        adjacency=layout["adjacency"],
        order=record["order"],
        agent_pos=record["start_shelf"],
    )


def query_inventory(state: WarehouseState, shelf_id: str) -> ActionResult:
    """Query stock information for a specific shelf."""

    shelf = state.shelves.get(shelf_id)
    if shelf is None:
        return ActionResult(False, f"unknown shelf: {shelf_id}", {"shelf_id": shelf_id, "stock": {}})
    return ActionResult(True, "ok", {"shelf_id": shelf_id, "stock": dict(shelf.stock)})


def move(state: WarehouseState, target_shelf_id: str) -> ActionResult:
    """Move to an adjacent shelf."""

    if target_shelf_id not in state.shelves:
        state.invalid_actions += 1
        return ActionResult(False, f"unknown shelf: {target_shelf_id}", {})
    neighbors = state.adjacency.get(state.agent_pos, [])
    if target_shelf_id not in neighbors:
        state.invalid_actions += 1
        return ActionResult(
            False,
            f"unreachable: {state.agent_pos} -> {target_shelf_id}",
            {"from": state.agent_pos, "to": target_shelf_id},
        )
    prev = state.agent_pos
    state.agent_pos = target_shelf_id
    state.total_distance += 1
    return ActionResult(True, "moved", {"from": prev, "to": target_shelf_id, "distance": state.total_distance})


def pick(state: WarehouseState, sku: str, qty: int) -> ActionResult:
    """Pick a quantity of a SKU from the current shelf."""

    if qty <= 0:
        state.picking_errors += 1
        return ActionResult(False, f"invalid qty: {qty}", {})
    shelf = state.shelves[state.agent_pos]
    available = shelf.stock.get(sku, 0)
    if available <= 0:
        state.picking_errors += 1
        return ActionResult(False, f"sku {sku} not on shelf {state.agent_pos}", {})
    if qty > available:
        state.picking_errors += 1
        return ActionResult(False, f"insufficient stock: {sku} need {qty} have {available}", {})
    shelf.stock[sku] -= qty
    if shelf.stock[sku] == 0:
        del shelf.stock[sku]
    state.cart[sku] = state.cart.get(sku, 0) + qty
    return ActionResult(True, "picked", {"sku": sku, "qty": qty, "cart": dict(state.cart)})


def submit_order(state: WarehouseState) -> ActionResult:
    """Submit the order for validation against cart contents."""

    order_dict = {item["sku"]: item["qty"] for item in state.order}
    cart = dict(state.cart)
    missing = {sku: qty for sku, qty in order_dict.items() if cart.get(sku, 0) < qty}
    extra = {sku: qty for sku, qty in cart.items() if sku not in order_dict or qty > order_dict[sku]}
    if missing or extra:
        return ActionResult(False, "order incomplete", {"missing": missing, "extra": extra})
    state.completed = True
    return ActionResult(True, "order completed", {"completed": True, "distance": state.total_distance})


def pick_from_shelf(state: WarehouseState, shelf_id: str, sku: str, qty: int) -> ActionResult:
    """Move to a shelf (must be adjacent) and pick an item in one action."""

    if shelf_id not in state.shelves:
        state.invalid_actions += 1
        return ActionResult(False, f"unknown shelf: {shelf_id}", {})
    if state.agent_pos != shelf_id:
        neighbors = state.adjacency.get(state.agent_pos, [])
        if shelf_id not in neighbors:
            state.invalid_actions += 1
            return ActionResult(
                False,
                f"unreachable: {state.agent_pos} -> {shelf_id}",
                {"from": state.agent_pos, "to": shelf_id},
            )
        prev = state.agent_pos
        state.agent_pos = shelf_id
        state.total_distance += 1
        move_info = {"from": prev, "to": shelf_id, "distance": state.total_distance}
    else:
        move_info = {"from": shelf_id, "to": shelf_id, "distance": state.total_distance}
    if qty <= 0:
        state.picking_errors += 1
        return ActionResult(False, f"invalid qty: {qty}", move_info)
    shelf = state.shelves[state.agent_pos]
    available = shelf.stock.get(sku, 0)
    if available <= 0:
        state.picking_errors += 1
        return ActionResult(False, f"sku {sku} not on shelf {state.agent_pos}", move_info)
    if qty > available:
        state.picking_errors += 1
        return ActionResult(False, f"insufficient stock: {sku} need {qty} have {available}", move_info)
    shelf.stock[sku] -= qty
    if shelf.stock[sku] == 0:
        del shelf.stock[sku]
    state.cart[sku] = state.cart.get(sku, 0) + qty
    return ActionResult(
        True,
        "picked",
        {"sku": sku, "qty": qty, "cart": dict(state.cart), **move_info},
    )


def _bfs_distance(state: WarehouseState, start: str, target: str) -> int:
    """BFS shortest distance between two shelves on the unweighted graph."""

    if start == target:
        return 0
    visited = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in state.adjacency.get(node, []):
            if neighbor == target:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1


def _nearest_shelf_with_sku(
    state: WarehouseState, pos: str, sku: str, qty: int
) -> str | None:
    """BFS to find the nearest shelf with at least ``qty`` of ``sku``."""

    if state.shelves[pos].stock.get(sku, 0) >= qty:
        return pos
    visited = {pos}
    queue: deque[str] = deque([pos])
    while queue:
        node = queue.popleft()
        for neighbor in state.adjacency.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            shelf = state.shelves.get(neighbor)
            if shelf and shelf.stock.get(sku, 0) >= qty:
                return neighbor
            queue.append(neighbor)
    return None


def baseline_distance(state: WarehouseState) -> int:
    """Greedy shortest-route baseline visiting shelves for each order item."""

    pos = state.agent_pos
    distance = 0
    for item in state.order:
        sku, qty = item["sku"], item["qty"]
        target = _nearest_shelf_with_sku(state, pos, sku, qty)
        if target is None:
            continue
        distance += _bfs_distance(state, pos, target)
        pos = target
    return distance


def score_task(record: dict[str, Any], trajectory_data: dict[str, Any]) -> float:
    """Score a warehouse picking trajectory.

    Multi-dimensional scoring ensures different failure modes get different
    rewards, producing non-zero group-relative advantages in GSPO/GRPO.

    Dimensions for incomplete tasks:
    - Tool usage: did the agent call pick_from_shelf at all? (+0.15)
    - Cart progress: fraction of order items correctly picked (×0.4)
    - Valid moves: did the agent move to a real adjacent shelf? (+0.1)
    - Error penalties: picking errors (-0.15 each), invalid actions (-0.1 each)
    - No tool calls at all: extra penalty (-0.2)
    """

    names = trajectory_data.get("tool_names", [])
    cart = trajectory_data.get("cart", {})
    order_dict = {item["sku"]: item["qty"] for item in record.get("order", [])}
    total_needed = sum(order_dict.values()) if order_dict else 1
    total_fulfilled = sum(min(cart.get(sku, 0), qty) for sku, qty in order_dict.items())
    has_pick = "pick_from_shelf" in names
    has_submit = "submit_order" in names

    if not trajectory_data.get("completed"):
        score = -0.5
        if has_pick:
            score += 0.15
        else:
            score -= 0.2
        if has_submit:
            score += 0.05
        if total_needed > 0:
            score += 0.4 * (total_fulfilled / total_needed)
        if trajectory_data.get("distance", 0) > 0:
            score += 0.1
        score -= 0.15 * trajectory_data.get("picking_errors", 0)
        score -= 0.1 * trajectory_data.get("invalid_actions", 0)
        return max(score, -1.0)
    score = 1.0
    score -= 0.1 * trajectory_data.get("picking_errors", 0)
    score -= 0.05 * trajectory_data.get("invalid_actions", 0)
    actual = trajectory_data.get("distance", 0)
    baseline = trajectory_data.get("baseline_distance", 1)
    if actual > 0 and baseline > 0:
        efficiency = min(baseline / max(actual, 1), 1.0)
        score *= 0.7 + 0.3 * efficiency
    if names[:2] == ["pick_from_shelf", "submit_order"]:
        score += 0.2
    return max(score, -1.0)


def make_prompt(record: dict[str, Any]) -> str:
    """Build the user prompt with order details but not shelf layout."""

    order_str = ", ".join(f"{item['sku']} x{item['qty']}" for item in record["order"])
    return (
        f"You are in a warehouse with a {record['rows']}x{record['cols']} grid of shelves. "
        f"Shelf IDs follow the pattern A1, A2, ..., B1, B2, etc. "
        f"Your order: {order_str}. "
        f"You start at shelf {record['start_shelf']}. "
        "Use pick_from_shelf to move to a shelf and pick an item, then submit_order. "
        "You can only move to adjacent shelves."
    )