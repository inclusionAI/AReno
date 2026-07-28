"""Maze helpers for a partially-observable agentic RL example.

The agent sees only a local window around its position and must navigate
to the goal, optionally picking up keys to open doors along the way.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

# ---------------------------------------------------------------------------
# Tile symbols
# ---------------------------------------------------------------------------

EMPTY = "."
WALL = "#"
AGENT = "A"
KEY = "K"
DOOR = "D"
GOAL = "G"
UNKNOWN = "?"

DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class State:
    """Immutable maze state.

    ``grid`` uses the tile symbols above.  Doors that have been opened are
    replaced with ``EMPTY`` on the grid so walkability checks are uniform.
    ``keys`` is the set of key ids the agent currently holds.
    """

    grid: tuple[str, ...]
    agent_row: int
    agent_col: int
    keys: frozenset[str] = frozenset()
    steps: int = 0
    max_steps: int = 30
    done: bool = False
    # bookkeeping for reward / metrics
    invalid_moves: int = 0
    picked_up_keys: int = 0
    opened_doors: int = 0
    reached_goal: bool = False


# ---------------------------------------------------------------------------
# Maze generation
# ---------------------------------------------------------------------------


def generate_maze(
    rows: int = 7,
    cols: int = 7,
    *,
    seed: int = 0,
    num_keys: int = 1,
) -> tuple[tuple[str, ...], tuple[int, int], tuple[int, int], list[dict[str, Any]]]:
    """Generate a solvable maze with keys and doors.

    Returns ``(grid, agent_pos, goal_pos, key_door_pairs)`` where
    ``key_door_pairs`` is a list of ``{"key_id": str, "key_pos": (r,c),
    "door_pos": (r,c)}`` dicts.
    """

    rng = random.Random(seed)
    grid, agent_pos, goal_pos = _carve_maze(rows, cols, rng)
    key_door_pairs = _place_keys_and_doors(grid, agent_pos, goal_pos, num_keys, rng)
    grid_tuple = tuple("".join(row) for row in grid)
    return grid_tuple, agent_pos, goal_pos, key_door_pairs


def _carve_maze(
    rows: int, cols: int, rng: random.Random
) -> tuple[list[list[str]], tuple[int, int], tuple[int, int]]:
    """Carve a perfect maze using iterative DFS backtracking.

    The grid is sized ``2*rows+1`` by ``2*cols+1`` so that walls separate
    every pair of cells.  Returns ``(grid, agent_pos, goal_pos)``.
    """

    h = 2 * rows + 1
    w = 2 * cols + 1
    grid = [[WALL] * w for _ in range(h)]

    # cell (cr, cc) maps to grid position (2*cr+1, 2*cc+1)
    visited = [[False] * cols for _ in range(rows)]
    stack: list[tuple[int, int]] = []
    cr, cc = 0, 0
    visited[cr][cc] = True
    grid[2 * cr + 1][2 * cc + 1] = EMPTY
    stack.append((cr, cc))

    while stack:
        cr, cc = stack[-1]
        neighbours: list[tuple[int, int, int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                neighbours.append((nr, nc, 2 * cr + 1 + dr, 2 * cc + 1 + dc))
        if not neighbours:
            stack.pop()
            continue
        nr, nc, wr, wc = rng.choice(neighbours)
        grid[wr][wc] = EMPTY
        grid[2 * nr + 1][2 * nc + 1] = EMPTY
        visited[nr][nc] = True
        stack.append((nr, nc))

    agent_pos = (1, 1)
    goal_pos = (h - 2, w - 2)
    grid[agent_pos[0]][agent_pos[1]] = AGENT
    grid[goal_pos[0]][goal_pos[1]] = GOAL
    return grid, agent_pos, goal_pos


def _place_keys_and_doors(
    grid: list[list[str]],
    agent_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    num_pairs: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Place key-door pairs on wall positions that lie on the solution path.

    Guarantees the key is reachable without passing through its paired door.
    """

    h = len(grid)
    w = len(grid[0])
    pairs: list[dict[str, Any]] = []
    # Find wall positions that are on the path from agent to goal
    path = _bfs_path(grid, agent_pos, goal_pos, set())
    if not path or len(path) < 4:
        return pairs

    placed_doors: set[tuple[int, int]] = set()
    for i in range(num_pairs):
        # Pick a wall cell on the path to convert into a door
        candidates = []
        for idx in range(2, len(path) - 1):
            r, c = path[idx]
            if (r, c) in placed_doors:
                continue
            if grid[r][c] in (WALL, EMPTY):
                candidates.append((r, c))
        if not candidates:
            break
        door_pos = rng.choice(candidates)
        placed_doors.add(door_pos)
        grid[door_pos[0]][door_pos[1]] = DOOR

        # Find a reachable key position that is before the door (on path)
        key_pos = _find_key_position(grid, agent_pos, door_pos, path, rng)
        if key_pos is None:
            # Fallback: place key right before the door on the path
            door_idx = path.index(door_pos)
            key_pos = path[door_idx - 1]
        grid[key_pos[0]][key_pos[1]] = KEY
        pairs.append({
            "key_id": f"k{i}",
            "key_pos": key_pos,
            "door_pos": door_pos,
        })
    return pairs


def _find_key_position(
    grid: list[list[str]],
    agent_pos: tuple[int, int],
    door_pos: tuple[int, int],
    path: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    """Find an empty cell reachable from agent without passing the door."""

    door_idx = path.index(door_pos)
    # Cells on path before the door are reachable without the key
    before_door = set(path[:door_idx])
    h = len(grid)
    w = len(grid[0])
    empty_cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == EMPTY and (r, c) in before_door:
                empty_cells.append((r, c))
    if not empty_cells:
        return None
    return rng.choice(empty_cells)


def _bfs_path(
    grid: list[list[str]] | tuple[str, ...],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """BFS shortest path from *start* to *goal* avoiding *blocked* cells."""

    h = len(grid)
    w = len(grid[0])
    if start == goal:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        (r, c), path = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in visited or (nr, nc) in blocked:
                continue
            tile = grid[nr][nc] if isinstance(grid, list) else grid[nr][nc]
            if tile == WALL:
                continue
            visited.add((nr, nc))
            new_path = path + [(nr, nc)]
            if (nr, nc) == goal:
                return new_path
            queue.append(((nr, nc), new_path))
    return []


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------


def make_state(
    grid: tuple[str, ...] | list[str],
    agent_row: int = 1,
    agent_col: int = 1,
    *,
    keys: frozenset[str] | set[str] | None = None,
    steps: int = 0,
    max_steps: int = 30,
) -> State:
    """Parse and validate a maze grid into a :class:`State`."""

    rows = tuple(str(row) for row in grid)
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("Maze grid must be a non-empty rectangle")
    allowed = {EMPTY, WALL, AGENT, KEY, DOOR, GOAL}
    for row in rows:
        for cell in row:
            if cell not in allowed:
                raise ValueError(f"Invalid maze tile: {cell!r}")
    return State(
        grid=rows,
        agent_row=agent_row,
        agent_col=agent_col,
        keys=frozenset(keys) if keys else frozenset(),
        steps=steps,
        max_steps=max_steps,
    )


def make_state_from_record(record: dict[str, Any]) -> State:
    """Build a :class:`State` from a JSONL record dict."""

    grid = tuple(record["grid"])
    agent_pos = tuple(record["agent_pos"])
    return make_state(
        grid,
        agent_row=agent_pos[0],
        agent_col=agent_pos[1],
        max_steps=record.get("max_steps", 30),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_full_map(state: State) -> str:
    """Render the full maze grid with the agent overlaid."""

    lines = []
    for r in range(len(state.grid)):
        chars = list(state.grid[r])
        if r == state.agent_row and chars[state.agent_col] != WALL:
            chars[state.agent_col] = AGENT
        lines.append("".join(chars))
    return "\n".join(lines)


def render_local_view(state: State, radius: int = 1) -> str:
    """Render a ``(2*radius+1)`` square window centred on the agent.

    Cells outside the maze boundary are shown as ``UNKNOWN``.
    """

    size = 2 * radius + 1
    h = len(state.grid)
    w = len(state.grid[0])
    lines = []
    for dr in range(-radius, radius + 1):
        r = state.agent_row + dr
        chars = []
        for dc in range(-radius, radius + 1):
            c = state.agent_col + dc
            if 0 <= r < h and 0 <= c < w:
                if r == state.agent_row and c == state.agent_col:
                    chars.append(AGENT)
                else:
                    chars.append(state.grid[r][c])
            else:
                chars.append(UNKNOWN)
        lines.append("".join(chars))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def format_prompt(state: State, radius: int = 1) -> str:
    """Build the initial user prompt describing the visible maze."""

    view = render_local_view(state, radius)
    return (
        "You are in a partially observable maze. You can only see the tiles "
        f"around you ({radius*2+1}x{radius*2+1} window).\n\n"
        "Tiles:\n"
        f"  {WALL} = wall (blocked)   {EMPTY} = open floor   {AGENT} = you\n"
        f"  {KEY} = key (pick up)   {DOOR} = locked door (needs key)   {GOAL} = goal\n"
        f"  {UNKNOWN} = unknown (outside view)\n\n"
        "Actions (one per turn):\n"
        "  move  — step UP/DOWN/LEFT/RIGHT\n"
        "  pickup — pick up a key if standing on one\n"
        "  use_key — open an adjacent door if you hold a matching key\n\n"
        f"Steps taken: {state.steps}/{state.max_steps}\n"
        f"Keys held: {sorted(state.keys) if state.keys else 'none'}\n\n"
        f"Your view:\n{view}\n\n"
        "Call the act tool with one action to proceed."
    )


def format_step_prompt(state: State, radius: int = 1, feedback: str = "") -> str:
    """Build a follow-up prompt after each action."""

    view = render_local_view(state, radius)
    parts = [f"Steps: {state.steps}/{state.max_steps}"]
    if state.keys:
        parts.append(f"Keys: {sorted(state.keys)}")
    else:
        parts.append("Keys: none")
    if feedback:
        parts.append(feedback)
    parts.append(f"\nYour view:\n{view}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def legal_actions(state: State) -> list[dict[str, str]]:
    """Return the list of currently legal actions."""

    actions: list[dict[str, str]] = []
    for direction, (dr, dc) in DIRECTIONS.items():
        nr, nc = state.agent_row + dr, state.agent_col + dc
        tile = _tile_at(state, nr, nc)
        if tile is not None and tile != WALL and tile != DOOR:
            actions.append({"action": "move", "direction": direction})
        elif tile == DOOR:
            # Door can be opened if agent holds any key
            if state.keys:
                actions.append({"action": "use_key", "direction": direction})
    # Pickup if standing on a key
    current_tile = _tile_at(state, state.agent_row, state.agent_col)
    if current_tile == KEY:
        actions.append({"action": "pickup"})
    return actions


def step(state: State, action: dict[str, str] | None) -> tuple[State, float, bool, dict[str, Any]]:
    """Apply one action and return ``(next_state, reward, done, info)``."""

    if state.done:
        return state, 0.0, True, {"reason": "already_done"}

    parsed = _parse_action(action)
    if parsed is None:
        new_state = replace(state, steps=state.steps + 1, invalid_moves=state.invalid_moves + 1)
        return _check_terminal(new_state, -0.1, {"illegal": True, "reason": "invalid_action"})

    act = parsed["action"]
    if act == "move":
        return _do_move(state, parsed)
    if act == "pickup":
        return _do_pickup(state, parsed)
    if act == "use_key":
        return _do_use_key(state, parsed)
    # Unknown action
    new_state = replace(state, steps=state.steps + 1, invalid_moves=state.invalid_moves + 1)
    return _check_terminal(new_state, -0.1, {"illegal": True, "reason": f"unknown_action:{act}"})


def _do_move(state: State, parsed: dict[str, str]) -> tuple[State, float, bool, dict[str, Any]]:
    direction = parsed.get("direction", "")
    delta = DIRECTIONS.get(direction)
    if delta is None:
        new_state = replace(state, steps=state.steps + 1, invalid_moves=state.invalid_moves + 1)
        return _check_terminal(new_state, -0.1, {"illegal": True, "reason": "bad_direction"})

    dr, dc = delta
    nr, nc = state.agent_row + dr, state.agent_col + dc
    tile = _tile_at(state, nr, nc)
    if tile is None or tile == WALL or tile == DOOR:
        new_state = replace(state, steps=state.steps + 1, invalid_moves=state.invalid_moves + 1)
        return _check_terminal(new_state, -0.1, {"illegal": True, "reason": "blocked"})
    new_state = replace(state, agent_row=nr, agent_col=nc, steps=state.steps + 1)
    if tile == GOAL:
        new_state = replace(new_state, done=True, reached_goal=True)
        return new_state, 1.0, True, {"goal": True}
    return _check_terminal(new_state, -0.01, {"moved": True, "direction": direction})


def _do_pickup(state: State, parsed: dict[str, str]) -> tuple[State, float, bool, dict[str, Any]]:
    tile = _tile_at(state, state.agent_row, state.agent_col)
    if tile != KEY:
        new_state = replace(state, steps=state.steps + 1, invalid_moves=state.invalid_moves + 1)
        return _check_terminal(new_state, -0.1, {"illegal": True, "reason": "no_key_here"})
    # Remove key from grid, add to inventory
    new_grid = list(state.grid)
    row_chars = list(new_grid[state.agent_row])
    row_chars[state.agent_col] = EMPTY
    new_grid[state.agent_row] = "".join(row_chars)
    key_id = f"k{state.picked_up_keys}"
    new_state = replace(
        state,
        grid=tuple(new_grid),
        keys=state.keys | {key_id},
        steps=state.steps + 1,
        picked_up_keys=state.picked_up_keys + 1,
    )
    return _check_terminal(new_state, 0.2, {"picked_up": key_id})


def _do_use_key(state: State, parsed: dict[str, str]) -> tuple[State, float, bool, dict[str, Any]]:
    direction = parsed.get("direction", "")
    delta = DIRECTIONS.get(direction)
    if delta is None or not state.keys:
        new_state = replace(state, steps=state.steps + 1, invalid_moves=state.invalid_moves + 1)
        return _check_terminal(new_state, -0.1, {"illegal": True, "reason": "cannot_use_key"})
    dr, dc = delta
    nr, nc = state.agent_row + dr, state.agent_col + dc
    tile = _tile_at(state, nr, nc)
    if tile != DOOR:
        new_state = replace(state, steps=state.steps + 1, invalid_moves=state.invalid_moves + 1)
        return _check_terminal(new_state, -0.1, {"illegal": True, "reason": "no_door"})
    # Open the door: replace with EMPTY
    new_grid = list(state.grid)
    row_chars = list(new_grid[nr])
    row_chars[nc] = EMPTY
    new_grid[nr] = "".join(row_chars)
    new_state = replace(
        state,
        grid=tuple(new_grid),
        steps=state.steps + 1,
        opened_doors=state.opened_doors + 1,
    )
    return _check_terminal(new_state, 0.1, {"opened_door": True, "direction": direction})


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------


def compute_trajectory_reward(
    state: State, actions: list[dict[str, str] | None]
) -> tuple[float, dict[str, Any]]:
    """Simulate a full action sequence and return ``(total_reward, metrics)``."""

    current = state
    total_reward = 0.0
    invalid = 0
    for action in actions:
        current, reward, done, info = step(current, action)
        total_reward += reward
        if info.get("illegal"):
            invalid += 1
        if done:
            break
    metrics = {
        "reached_goal": current.reached_goal,
        "steps": current.steps,
        "invalid_moves": invalid,
        "total_invalid": current.invalid_moves,
        "keys_picked": current.picked_up_keys,
        "doors_opened": current.opened_doors,
    }
    return total_reward, metrics


def shortest_path_length(state: State) -> int | None:
    """Return the BFS shortest path length from agent to goal, or ``None``."""

    h = len(state.grid)
    w = len(state.grid[0])
    start = (state.agent_row, state.agent_col)
    goal = _find_tile(state, GOAL)
    if goal is None:
        return None
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in visited:
                continue
            tile = state.grid[nr][nc]
            if tile == WALL:
                continue
            # Doors are passable for shortest-path (assume key available)
            visited.add((nr, nc))
            queue.append(((nr, nc), dist + 1))
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_action(action: dict[str, str] | None) -> dict[str, str] | None:
    if action is None or not isinstance(action, dict):
        return None
    act = action.get("action")
    if not act:
        return None
    return {"action": str(act), "direction": str(action.get("direction", ""))}


def _tile_at(state: State, row: int, col: int) -> str | None:
    if 0 <= row < len(state.grid) and 0 <= col < len(state.grid[0]):
        return state.grid[row][col]
    return None


def _find_tile(state: State, tile: str) -> tuple[int, int] | None:
    for r in range(len(state.grid)):
        for c in range(len(state.grid[0])):
            if state.grid[r][c] == tile:
                return (r, c)
    return None


def _check_terminal(state: State, reward: float, info: dict[str, Any]) -> tuple[State, float, bool, dict[str, Any]]:
    if state.steps >= state.max_steps and not state.done:
        new_state = replace(state, done=True)
        return new_state, reward, True, {**info, "timeout": True}
    return state, reward, state.done, info
