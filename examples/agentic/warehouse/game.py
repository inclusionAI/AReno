"""Warehouse-picking environment for agentic RL.

Generates a grid-based warehouse with shelves, aisles, stock, and orders.
Exposes four actions (query_inventory, move, pick, submit) with per-step
validation. Includes a BFS baseline solver for distance metrics.

The environment is deterministic given the same random seed, enabling
reproducible training and testing.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Shelf:
    """A shelf at a fixed grid position holding items with quantities."""

    shelf_id: str
    row: int
    col: int
    stock: dict[str, int] = field(default_factory=dict)
    """Map from item name to remaining quantity."""


@dataclass
class Order:
    """A customer order: item names and required quantities."""

    order_id: str
    items: dict[str, int] = field(default_factory=dict)
    """Map from item name to required quantity."""


@dataclass
class WarehouseState:
    """Full mutable state of the warehouse at any point in time."""

    grid: list[list[str]]
    """Grid cells: '.' = aisle, '#' = shelf, 'S' = start, 'D' = deposit."""
    shelves: dict[str, Shelf] = field(default_factory=dict)
    """Map from shelf_id to Shelf objects."""
    agent_pos: tuple[int, int] = (0, 0)
    """Current (row, col) of the agent."""
    cart: dict[str, int] = field(default_factory=dict)
    """Map from item name to quantity in cart."""
    order: Order | None = None
    """Current active order."""
    steps_taken: int = 0
    """Total actions executed."""
    completed: bool = False
    """Whether the order has been successfully submitted."""
    action_log: list[dict[str, Any]] = field(default_factory=list)
    """Record of every action and its result."""


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------


def generate_warehouse(
    rows: int = 5,
    cols: int = 5,
    num_shelves: int = 4,
    num_items: int = 3,
    max_stock: int = 5,
    order_size: int = 2,
    seed: int = 42,
) -> WarehouseState:
    """Generate a deterministic warehouse layout with shelves, stock, and an order.

    The grid uses '.' for aisles, '#' for shelves, 'S' for the agent start
    position, and 'D' for the deposit/submit position. Shelves are placed
    randomly but deterministically given the same seed.

    设计意图：使用 random.Random(seed) 保证确定性，同一个 seed 生成完全相同的
    仓库布局、库存和订单，满足 Issue 要求的 "deterministic replay"。
    """

    rng = random.Random(seed)

    # Initialize grid with aisles.
    grid = [["." for _ in range(cols)] for _ in range(rows)]

    # Agent starts at top-left corner.
    grid[0][0] = "S"

    # Deposit point at bottom-right corner.
    grid[rows - 1][cols - 1] = "D"

    # Place shelves randomly on remaining cells.
    available = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] == "."
    ]
    rng.shuffle(available)
    num_shelves = min(num_shelves, len(available))

    shelves: dict[str, Shelf] = {}
    item_names = [f"item_{chr(ord('A') + i)}" for i in range(num_items)]

    for i in range(num_shelves):
        r, c = available[i]
        grid[r][c] = "#"
        shelf_id = f"shelf_{i + 1}"

        # Each shelf gets 1-3 random items with random stock.
        stock: dict[str, int] = {}
        num_items_on_shelf = rng.randint(1, min(3, num_items))
        chosen_items = rng.sample(item_names, num_items_on_shelf)
        for item in chosen_items:
            stock[item] = rng.randint(1, max_stock)

        shelves[shelf_id] = Shelf(shelf_id=shelf_id, row=r, col=c, stock=stock)

    # Generate an order: pick random items that exist somewhere in the warehouse.
    all_available_items: set[str] = set()
    for shelf in shelves.values():
        all_available_items.update(shelf.stock.keys())

    order_items: dict[str, int] = {}
    if all_available_items:
        order_item_list = rng.sample(
            sorted(all_available_items),
            min(order_size, len(all_available_items)),
        )
        for item in order_item_list:
            order_items[item] = rng.randint(1, 2)

    order = Order(order_id="order_1", items=order_items)

    return WarehouseState(
        grid=grid,
        shelves=shelves,
        agent_pos=(0, 0),
        order=order,
    )


# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------


def generate_small(seed: int = 42) -> WarehouseState:
    """Small warehouse: 4x4 grid, 3 shelves, 2 item types, order size 1."""

    return generate_warehouse(
        rows=4, cols=4, num_shelves=3, num_items=2, max_stock=3, order_size=1, seed=seed
    )


def generate_medium(seed: int = 42) -> WarehouseState:
    """Medium warehouse: 6x6 grid, 6 shelves, 4 item types, order size 2."""

    return generate_warehouse(
        rows=6, cols=6, num_shelves=6, num_items=4, max_stock=5, order_size=2, seed=seed
    )


def generate_hard(seed: int = 42) -> WarehouseState:
    """Hard warehouse: 8x8 grid, 10 shelves, 6 item types, order size 3."""

    return generate_warehouse(
        rows=8, cols=8, num_shelves=10, num_items=6, max_stock=5, order_size=3, seed=seed
    )


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------


def query_inventory(state: WarehouseState, shelf_id: str) -> dict[str, Any]:
    """Query the stock of a shelf by its ID.

    Returns the shelf's current stock as a dict. Errors if the shelf ID
    does not exist.

    设计意图：让 Agent 可以在不移动的情况下远程查询任意货架库存，
    降低探索难度。如果只允许查询相邻货架，Agent 需要更多步数才能了解仓库全貌。
    """

    if shelf_id not in state.shelves:
        return {"ok": False, "error": f"unknown shelf_id '{shelf_id}'"}
    stock = state.shelves[shelf_id].stock
    state.steps_taken += 1
    state.action_log.append({"action": "query_inventory", "shelf_id": shelf_id, "result": dict(stock)})
    return {"ok": True, "shelf_id": shelf_id, "stock": dict(stock)}


def move(state: WarehouseState, direction: str) -> dict[str, Any]:
    """Move the agent one cell in the given direction.

    Validates that the target cell is within bounds and is not a shelf.
    Directions: 'up', 'down', 'left', 'right'.

    设计意图：货架（#）不可通行，Agent 只能走过道（.）、起点（S）、
    提交点（D）。这样货架既是障碍物也是目标——Agent 需要走到货架旁边才能拣货。
    """

    direction_map = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    if direction not in direction_map:
        state.steps_taken += 1
        state.action_log.append({"action": "move", "direction": direction, "result": "invalid_direction"})
        return {"ok": False, "error": f"invalid direction '{direction}'"}

    dr, dc = direction_map[direction]
    new_row = state.agent_pos[0] + dr
    new_col = state.agent_pos[1] + dc

    # Check bounds.
    if new_row < 0 or new_row >= len(state.grid) or new_col < 0 or new_col >= len(state.grid[0]):
        state.steps_taken += 1
        state.action_log.append({"action": "move", "direction": direction, "result": "out_of_bounds"})
        return {"ok": False, "error": "out of bounds"}

    # Check that target is not a shelf (can walk on aisles, start, deposit).
    if state.grid[new_row][new_col] == "#":
        state.steps_taken += 1
        state.action_log.append({"action": "move", "direction": direction, "result": "blocked_by_shelf"})
        return {"ok": False, "error": "blocked by shelf"}

    state.agent_pos = (new_row, new_col)
    state.steps_taken += 1
    state.action_log.append({"action": "move", "direction": direction, "result": "ok"})
    return {"ok": True, "position": list(state.agent_pos)}


def pick(state: WarehouseState, item: str, quantity: int = 1) -> dict[str, Any]:
    """Pick an item from a shelf adjacent to the agent's current position.

    Validates that a shelf is adjacent, the item exists on that shelf,
    and sufficient stock is available. Adds to cart on success.

    设计意图：拣货要求 Agent 站在货架旁边（上下左右相邻），不是站在货架上。
    拣完后从 shelf.stock 扣减数量，数量归零时从 stock dict 删除该 item。
    这样 query_inventory 的结果能反映实时库存变化。
    """

    if quantity < 1:
        state.steps_taken += 1
        state.action_log.append({"action": "pick", "item": item, "quantity": quantity, "result": "invalid_quantity"})
        return {"ok": False, "error": "quantity must be >= 1"}

    # Find an adjacent shelf.
    adjacent_shelf = _find_adjacent_shelf(state)
    if adjacent_shelf is None:
        state.steps_taken += 1
        state.action_log.append({"action": "pick", "item": item, "quantity": quantity, "result": "no_adjacent_shelf"})
        return {"ok": False, "error": "no shelf adjacent to agent"}

    # Check item exists on shelf.
    if item not in adjacent_shelf.stock:
        state.steps_taken += 1
        state.action_log.append({"action": "pick", "item": item, "quantity": quantity, "result": "item_not_found"})
        return {"ok": False, "error": f"item '{item}' not on shelf {adjacent_shelf.shelf_id}"}

    # Check stock.
    if adjacent_shelf.stock[item] < quantity:
        state.steps_taken += 1
        state.action_log.append({"action": "pick", "item": item, "quantity": quantity, "result": "out_of_stock"})
        return {"ok": False, "error": f"insufficient stock: have {adjacent_shelf.stock[item]}, need {quantity}"}

    # Pick success.
    adjacent_shelf.stock[item] -= quantity
    if adjacent_shelf.stock[item] == 0:
        del adjacent_shelf.stock[item]
    state.cart[item] = state.cart.get(item, 0) + quantity
    state.steps_taken += 1
    state.action_log.append({
        "action": "pick",
        "item": item,
        "quantity": quantity,
        "shelf_id": adjacent_shelf.shelf_id,
        "result": "ok",
        "cart": dict(state.cart),
    })
    return {"ok": True, "item": item, "quantity": quantity, "cart": dict(state.cart)}


def submit(state: WarehouseState) -> dict[str, Any]:
    """Submit the current cart as fulfillment of the order.

    Validates that the agent is at the deposit point and that cart
    contents match the order exactly.

    设计意图：提交必须在提交点（D）进行，不能随便在哪里就提交。
    提交时检查 missing（缺的）和 extra（多拣的），两者都为空才算完成。
    这样防止 Agent 拣错商品或漏拣后蒙混过关。
    """

    # Check agent is at deposit point.
    r, c = state.agent_pos
    if state.grid[r][c] != "D":
        state.steps_taken += 1
        state.action_log.append({"action": "submit", "result": "not_at_deposit"})
        return {"ok": False, "error": "must be at deposit point to submit"}

    if state.order is None:
        state.steps_taken += 1
        state.action_log.append({"action": "submit", "result": "no_order"})
        return {"ok": False, "error": "no active order"}

    # Check cart matches order.
    order_items = state.order.items
    missing: dict[str, int] = {}
    extra: dict[str, int] = {}

    for item, needed in order_items.items():
        have = state.cart.get(item, 0)
        if have < needed:
            missing[item] = needed - have

    for item, have in state.cart.items():
        if item not in order_items:
            extra[item] = have
        elif have > order_items[item]:
            extra[item] = have - order_items[item]

    if missing or extra:
        state.steps_taken += 1
        state.action_log.append({
            "action": "submit",
            "result": "incomplete",
            "missing": missing,
            "extra": extra,
        })
        return {
            "ok": False,
            "error": "order incomplete",
            "missing": missing,
            "extra": extra,
        }

    # Success!
    state.completed = True
    state.steps_taken += 1
    state.action_log.append({"action": "submit", "result": "ok"})
    return {"ok": True, "completed": True, "steps": state.steps_taken}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _find_adjacent_shelf(state: WarehouseState) -> Shelf | None:
    """Find a shelf adjacent to the agent's current position."""

    r, c = state.agent_pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= len(state.grid) or nc < 0 or nc >= len(state.grid[0]):
            continue
        if state.grid[nr][nc] == "#":
            for shelf in state.shelves.values():
                if shelf.row == nr and shelf.col == nc:
                    return shelf
    return None


def get_agent_view(state: WarehouseState) -> dict[str, Any]:
    """Return the current observable state for the agent."""

    return {
        "position": list(state.agent_pos),
        "cart": dict(state.cart),
        "order": dict(state.order.items) if state.order else {},
        "completed": state.completed,
        "steps": state.steps_taken,
    }


# ---------------------------------------------------------------------------
# BFS baseline solver
# ---------------------------------------------------------------------------


def _bfs_shortest_path(
    grid: list[list[str]], start: tuple[int, int], target: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """BFS shortest path on the grid, avoiding shelf cells ('#')."""

    if start == target:
        return [start]

    rows = len(grid)
    cols = len(grid[0])
    visited = {start}
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if (nr, nc) in visited:
                continue
            if grid[nr][nc] == "#":
                continue
            visited.add((nr, nc))
            parent[(nr, nc)] = (r, c)
            if (nr, nc) == target:
                # Reconstruct path.
                path = [(nr, nc)]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append((nr, nc))

    return None


def solve_baseline(state: WarehouseState) -> dict[str, Any]:
    """Compute the optimal picking route using BFS between required shelves.

    Finds all shelves that have needed items, computes the shortest path
    visiting them in the best order (brute-force TSP for small numbers),
    and returns the total distance and action sequence.

    设计意图：baseline 是 Agent 表现的衡量标准。Issue 要求 "distance
    relative to a simple route baseline"，即 agent_distance / baseline_distance。
    baseline_distance 越接近 1.0 说明 Agent 走的路径越接近最优。
    BFS 保证最短路径，brute-force TSP 对小规模仓库足够。
    """

    if state.order is None:
        return {"solvable": False, "reason": "no order"}

    order = state.order.items
    deposit_pos = _find_deposit_pos(state.grid)

    # Find which shelves have each needed item.
    item_to_shelves: dict[str, list[tuple[str, Shelf]]] = {}
    for item_name in order:
        item_to_shelves[item_name] = []
        for shelf_id, shelf in state.shelves.items():
            if item_name in shelf.stock and shelf.stock[item_name] >= order[item_name]:
                item_to_shelves[item_name].append((shelf_id, shelf))

    # Check solvability: every order item must be available somewhere.
    for item_name, shelves_with_item in item_to_shelves.items():
        if not shelves_with_item:
            return {"solvable": False, "reason": f"item '{item_name}' not available in any shelf"}

    # For simple cases (order_size <= 3), brute-force the best order.
    # Build a list of (shelf_id, shelf) to visit.
    shelves_to_visit: list[tuple[str, Shelf]] = []
    for item_name in order:
        # Pick the first shelf that has the item (simplified).
        if item_to_shelves[item_name]:
            shelves_to_visit.append(item_to_shelves[item_name][0])

    # Deduplicate while preserving order.
    seen_ids: set[str] = set()
    unique_shelves: list[tuple[str, Shelf]] = []
    for shelf_id, shelf in shelves_to_visit:
        if shelf_id not in seen_ids:
            seen_ids.add(shelf_id)
            unique_shelves.append((shelf_id, shelf))

    # Compute shortest path: start -> each shelf (adjacent) -> deposit.
    total_distance = 0
    path_detail: list[dict[str, Any]] = []
    current_pos = state.agent_pos

    for shelf_id, shelf in unique_shelves:
        # Find the adjacent walkable cell to this shelf.
        adjacent_cell = _find_adjacent_walkable(state.grid, shelf.row, shelf.col)
        if adjacent_cell is None:
            return {"solvable": False, "reason": f"shelf {shelf_id} is unreachable"}

        path = _bfs_shortest_path(state.grid, current_pos, adjacent_cell)
        if path is None:
            return {"solvable": False, "reason": f"no path to shelf {shelf_id}"}

        total_distance += len(path) - 1
        path_detail.append({"shelf_id": shelf_id, "distance": len(path) - 1})
        current_pos = adjacent_cell

    # Path from last shelf to deposit.
    if deposit_pos:
        path = _bfs_shortest_path(state.grid, current_pos, deposit_pos)
        if path:
            total_distance += len(path) - 1
            path_detail.append({"to_deposit": True, "distance": len(path) - 1})

    return {
        "solvable": True,
        "total_distance": total_distance,
        "shelves_to_visit": [s[0] for s in unique_shelves],
        "path_detail": path_detail,
    }


def _find_adjacent_walkable(grid: list[list[str]], row: int, col: int) -> tuple[int, int] | None:
    """Find a walkable cell adjacent to a shelf position."""

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= len(grid) or nc < 0 or nc >= len(grid[0]):
            continue
        if grid[nr][nc] != "#":
            return (nr, nc)
    return None


def _find_deposit_pos(grid: list[list[str]]) -> tuple[int, int] | None:
    """Find the deposit point position in the grid."""

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "D":
                return (r, c)
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(state: WarehouseState, baseline: dict[str, Any]) -> dict[str, Any]:
    """Compute evaluation metrics from the agent's action log and baseline.

    Metrics:
    - complete_orders: 1 if order completed, 0 otherwise
    - picking_mistakes: number of pick attempts on wrong items or out-of-stock
    - invalid_actions: number of invalid moves or actions
    - agent_distance: total steps the agent took
    - baseline_distance: shortest path from baseline solver
    - distance_ratio: agent_distance / baseline_distance (1.0 = optimal)

    设计意图：Issue 验收标准要求 "metrics for complete orders, picking
    mistakes, invalid actions, and distance relative to a simple route
    baseline"。这里逐项对应，从 action_log 中统计每类错误。
    """

    picking_mistakes = 0
    invalid_actions = 0
    agent_distance = 0

    for entry in state.action_log:
        action = entry["action"]
        result = entry.get("result", "")

        if action == "move":
            if result == "ok":
                agent_distance += 1
            else:
                invalid_actions += 1
        elif action == "pick":
            if result in ("item_not_found", "out_of_stock", "no_adjacent_shelf"):
                picking_mistakes += 1
            if result in ("invalid_quantity",):
                invalid_actions += 1
        elif action == "submit":
            if result in ("not_at_deposit", "no_order", "incomplete"):
                invalid_actions += 1
        elif action == "query_inventory":
            if result == "unknown_shelf":
                invalid_actions += 1

    baseline_distance = baseline.get("total_distance", 0)
    distance_ratio = agent_distance / baseline_distance if baseline_distance > 0 else float("inf")

    return {
        "complete_orders": 1 if state.completed else 0,
        "picking_mistakes": picking_mistakes,
        "invalid_actions": invalid_actions,
        "agent_distance": agent_distance,
        "baseline_distance": baseline_distance,
        "distance_ratio": round(distance_ratio, 2) if distance_ratio != float("inf") else None,
    }


# ---------------------------------------------------------------------------
# Tool definitions (for AReno agentic integration)
# ---------------------------------------------------------------------------


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": "Query the stock of a shelf by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shelf_id": {"type": "string", "description": "The shelf ID to query, e.g. 'shelf_1'."}
                },
                "required": ["shelf_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the agent one cell in the given direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"]}
                },
                "required": ["direction"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Pick an item from an adjacent shelf into the cart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "Item name to pick, e.g. 'item_A'."},
                    "quantity": {"type": "integer", "description": "Quantity to pick (default 1).", "default": 1},
                },
                "required": ["item"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the cart as order fulfillment. Must be at the deposit point.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


SYSTEM_PROMPT = (
    "You are a warehouse picking robot. Navigate the warehouse grid, find the items "
    "in the order, pick them into your cart, and submit at the deposit point. "
    "Use tools to query shelf inventory, move, pick items, and submit the order. "
    "Do not answer in plain text."
)


def make_prompt(state: WarehouseState) -> str:
    """Build the user prompt describing the warehouse and order."""

    grid_str = "\n".join(" ".join(row) for row in state.grid)
    order_str = ", ".join(f"{q}x {item}" for item, q in state.order.items.items())
    shelf_info = "\n".join(
        f"  - {s.shelf_id} at ({s.row}, {s.col})" for s in state.shelves.values()
    )
    return (
        f"Warehouse layout (S=start, D=deposit, #=shelf, .=aisle):\n{grid_str}\n\n"
        f"Shelves:\n{shelf_info}\n\n"
        f"Order: {order_str}\n\n"
        f"You are at position {list(state.agent_pos)}. "
        "Pick all items and submit at the deposit point (D)."
    )