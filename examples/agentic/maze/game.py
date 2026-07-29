"""Partially-observable maze environment for agentic RL.

Pure-Python maze with walls, keys, doors, and a goal.  The agent sees only
a bounded local view around its position; the full map is never exposed
through any observation.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace
from typing import Literal

CellType = Literal["empty", "wall", "key", "door", "goal"]
Direction = Literal["up", "down", "left", "right"]
Maze = list[list[str]]
Position = tuple[int, int]

EMPTY: str = "empty"
WALL: str = "wall"
KEY: str = "key"
DOOR: str = "door"
GOAL: str = "goal"

_DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Glyphs used in text rendering (observation and prompt).
_GLYPH: dict[str, str] = {
    EMPTY: ".",
    WALL: "#",
    KEY: "k",
    DOOR: "D",
    GOAL: "G",
}
_AGENT_GLYPH = "@"
_UNSEEN = "?"


@dataclass(frozen=True)
class MazeState:
    """Immutable snapshot of an in-progress maze episode."""

    maze: Maze
    agent_pos: Position
    has_key: bool
    steps_taken: int
    max_steps: int
    vision_radius: int = 1


@dataclass(frozen=True)
class MoveResult:
    """Outcome of a single ``apply_move`` call."""

    state: MazeState
    success: bool
    reason: str
    terminal: bool
    observation: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def normalize_maze(raw: list[list[str]]) -> Maze:
    """Validate and normalise a raw maze grid."""

    if not raw or not raw[0]:
        raise ValueError("maze must be a non-empty 2-D grid")
    h = len(raw)
    w = len(raw[0])
    for row in raw:
        if len(row) != w:
            raise ValueError("maze rows must have equal length")
    maze: Maze = []
    for row in raw:
        normalised_row: list[str] = []
        for cell in row:
            if cell not in (EMPTY, WALL, KEY, DOOR, GOAL):
                raise ValueError(f"invalid cell type: {cell!r}")
            normalised_row.append(cell)
        maze.append(normalised_row)
    return maze


def maze_width(maze: Maze) -> int:
    return len(maze[0]) if maze else 0


def maze_height(maze: Maze) -> int:
    return len(maze)


# ---------------------------------------------------------------------------
# Maze generation
# ---------------------------------------------------------------------------

def generate_maze(
    width: int,
    height: int,
    *,
    seed: int = 2026,
    n_keys: int = 1,
    n_doors: int = 1,
) -> tuple[Maze, Position, Position, list[Position], list[Position]]:
    """Generate a solvable maze with keys and doors.

    Returns ``(maze, start, goal, key_positions, door_positions)``.
    The maze uses randomized DFS wall carving on an odd-dimension grid.
    """
    if width < 5 or height < 5:
        raise ValueError("maze dimensions must be at least 5x5")
    if width % 2 == 0:
        width += 1
    if height % 2 == 0:
        height += 1

    rng = random.Random(seed)

    # Start fully walled.
    maze: Maze = [[WALL for _ in range(width)] for _ in range(height)]

    # Carve passages with randomized DFS.
    stack: list[Position] = [(1, 1)]
    maze[1][1] = EMPTY
    while stack:
        r, c = stack[-1]
        neighbours: list[tuple[int, int]] = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and maze[nr][nc] == WALL:
                neighbours.append((nr, nc))
        if not neighbours:
            stack.pop()
            continue
        nr, nc = rng.choice(neighbours)
        maze[(r + nr) // 2][(c + nc) // 2] = EMPTY
        maze[nr][nc] = EMPTY
        stack.append((nr, nc))

    start: Position = (1, 1)
    goal: Position = (height - 2, width - 2)
    maze[goal[0]][goal[1]] = GOAL

    # Collect all passage cells (excluding start and goal).
    passages: list[Position] = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if maze[r][c] == EMPTY and (r, c) != start
    ]
    rng.shuffle(passages)

    # Place door(s) on the shortest path so the key is actually needed.
    # Never place a door adjacent to start — the agent must have room to
    # explore and find the key before encountering the door.
    shortest = _bfs_plain(maze, start, goal)
    door_positions: list[Position] = []
    if shortest and n_doors > 0:
        candidates = [
            pos for pos in shortest
            if maze[pos[0]][pos[1]] == EMPTY
            and abs(pos[0] - start[0]) + abs(pos[1] - start[1]) > 1
        ]
        rng.shuffle(candidates)
        for pos in candidates[:n_doors]:
            maze[pos[0]][pos[1]] = DOOR
            door_positions.append(pos)

    # Place key(s) on passages reachable without a key (before any door).
    key_positions: list[Position] = []
    reachable = _flood_reachable(maze, start)
    key_candidates = [pos for pos in passages if pos in reachable and maze[pos[0]][pos[1]] == EMPTY]
    rng.shuffle(key_candidates)
    for pos in key_candidates[:n_keys]:
        maze[pos[0]][pos[1]] = KEY
        key_positions.append(pos)

    # Verify solvability with key.
    if not solve_shortest_path(maze, start, goal, has_key=True):
        raise RuntimeError("generated maze is unsolvable")

    return maze, start, goal, key_positions, door_positions


def _bfs_plain(maze: Maze, start: Position, goal: Position) -> list[Position]:
    """BFS ignoring doors/keys — just through empty/goal cells."""
    h, w = maze_height(maze), maze_width(maze)
    q: deque[Position] = deque([start])
    prev: dict[Position, Position | None] = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in prev:
                cell = maze[nr][nc]
                if cell in (EMPTY, GOAL):
                    prev[(nr, nc)] = (r, c)
                    q.append((nr, nc))
    if goal not in prev:
        return []
    path: list[Position] = []
    cur: Position | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


def _flood_reachable(maze: Maze, start: Position) -> set[Position]:
    """Flood-fill cells reachable from start, auto-picking up keys en route."""
    h, w = maze_height(maze), maze_width(maze)
    # Track (position, has_key) state for key-gated traversal.
    visited: set[tuple[Position, bool]] = set()
    reachable: set[Position] = {start}
    q: deque[tuple[Position, bool]] = deque([(start, False)])
    visited.add((start, False))
    while q:
        (r, c), key = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cell = maze[nr][nc]
                if cell == WALL:
                    continue
                if cell == DOOR and not key:
                    continue
                new_key = key or (cell == KEY)
                new_state = ((nr, nc), new_key)
                if new_state not in visited:
                    visited.add(new_state)
                    reachable.add((nr, nc))
                    q.append(new_state)
    return reachable


def solve_shortest_path(maze: Maze, start: Position, goal: Position, has_key: bool) -> list[Position]:
    """BFS shortest path.  Doors are passable only when *has_key* is True.

    When *has_key* is False the search will auto-pickup keys found en route,
    enabling a two-phase path (start -> key -> goal through doors).
    """
    h, w = maze_height(maze), maze_width(maze)
    # State = (position, has_key) — tracks key pickup during search.
    visited: set[tuple[Position, bool]] = set()
    q: deque[tuple[Position, bool]] = deque([(start, has_key)])
    prev: dict[tuple[Position, bool], tuple[Position, bool] | None] = {(start, has_key): None}
    visited.add((start, has_key))

    found_state: tuple[Position, bool] | None = None
    while q:
        pos, key = q.popleft()
        if pos == goal:
            found_state = (pos, key)
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                cell = maze[nr][nc]
                if cell == WALL:
                    continue
                if cell == DOOR and not key:
                    continue
                new_key = key or (cell == KEY)
                new_state = ((nr, nc), new_key)
                if new_state not in visited:
                    visited.add(new_state)
                    prev[new_state] = (pos, key)
                    q.append(new_state)
    if found_state is None:
        return []
    path: list[Position] = []
    cur: tuple[Position, bool] | None = found_state
    while cur is not None:
        path.append(cur[0])
        cur = prev.get(cur)
    return list(reversed(path))


def is_solvable(maze: Maze, start: Position | None = None, goal: Position | None = None) -> bool:
    """Check whether the maze is solvable assuming key can be found en route."""
    if start is None:
        start = (1, 1)
    if goal is None:
        goal = (maze_height(maze) - 2, maze_width(maze) - 2)
    # Try without key first, then with key.
    if solve_shortest_path(maze, start, goal, has_key=False):
        return True
    return bool(solve_shortest_path(maze, start, goal, has_key=True))


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def apply_move(state: MazeState, direction: str) -> MoveResult:
    """Execute one move and return the resulting state + observation."""

    if direction not in _DIRECTIONS:
        return MoveResult(
            state=state,
            success=False,
            reason="invalid_direction",
            terminal=state.steps_taken >= state.max_steps,
            observation=local_view(state),
        )

    if state.steps_taken >= state.max_steps:
        return MoveResult(
            state=state,
            success=False,
            reason="max_steps",
            terminal=True,
            observation=local_view(state),
        )

    dr, dc = _DIRECTIONS[direction]
    nr, nc = state.agent_pos[0] + dr, state.agent_pos[1] + dc
    h, w = maze_height(state.maze), maze_width(state.maze)

    if nr < 0 or nr >= h or nc < 0 or nc >= w:
        return MoveResult(
            state=replace(state, steps_taken=state.steps_taken + 1),
            success=False,
            reason="out_of_bounds",
            terminal=state.steps_taken + 1 >= state.max_steps,
            observation=local_view(state),
        )

    cell = state.maze[nr][nc]
    if cell == WALL:
        return MoveResult(
            state=replace(state, steps_taken=state.steps_taken + 1),
            success=False,
            reason="wall",
            terminal=state.steps_taken + 1 >= state.max_steps,
            observation=local_view(state),
        )
    if cell == DOOR and not state.has_key:
        return MoveResult(
            state=replace(state, steps_taken=state.steps_taken + 1),
            success=False,
            reason="locked_door",
            terminal=state.steps_taken + 1 >= state.max_steps,
            observation=local_view(state),
        )

    # Valid move — auto-pickup key.
    has_key = state.has_key
    if cell == KEY:
        has_key = True

    new_state = MazeState(
        maze=state.maze,
        agent_pos=(nr, nc),
        has_key=has_key,
        steps_taken=state.steps_taken + 1,
        max_steps=state.max_steps,
        vision_radius=state.vision_radius,
    )

    terminal = cell == GOAL or new_state.steps_taken >= new_state.max_steps
    reason = "goal" if cell == GOAL else "ok"

    return MoveResult(
        state=new_state,
        success=True,
        reason=reason,
        terminal=terminal,
        observation=local_view(new_state),
    )


# ---------------------------------------------------------------------------
# Observation (partial observability)
# ---------------------------------------------------------------------------

def local_view(state: MazeState) -> str:
    """Render only the cells within *vision_radius* of the agent.

    Cells outside the view are shown as ``?``.  The agent's own position
    is marked with ``@``.
    """
    r, c = state.agent_pos
    rad = state.vision_radius
    h, w = maze_height(state.maze), maze_width(state.maze)

    lines: list[str] = []
    for dr in range(-rad, rad + 1):
        row_chars: list[str] = []
        for dc in range(-rad, rad + 1):
            ir, ic = r + dr, c + dc
            if ir < 0 or ir >= h or ic < 0 or ic >= w:
                row_chars.append(_UNSEEN)
            elif (ir, ic) == (r, c):
                row_chars.append(_AGENT_GLYPH)
            else:
                cell = state.maze[ir][ic]
                if state.has_key and cell == KEY:
                    # Key already picked up — show as empty for accuracy.
                    row_chars.append(_GLYPH[EMPTY])
                else:
                    row_chars.append(_GLYPH[cell])
        lines.append(" ".join(row_chars))
    return "\n".join(lines)


def format_prompt(state: MazeState) -> str:
    """Build the user-facing prompt for the current observation."""

    view = local_view(state)
    legend = (
        "Legend: @ = you, # = wall, . = empty, k = key, "
        "D = locked door, G = goal, ? = unseen"
    )
    return (
        f"You are at step {state.steps_taken}/{state.max_steps} in a maze.\n"
        f"{'You have the key.' if state.has_key else 'You do not have the key.'}\n"
        f"{legend}\n\n"
        f"Your local view:\n{view}\n\n"
        f"Choose a direction to move (up, down, left, right)."
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _find_goal(maze: Maze) -> Position:
    """Scan the maze for the goal cell."""
    for r, row in enumerate(maze):
        for c, cell in enumerate(row):
            if cell == GOAL:
                return (r, c)
    raise ValueError("maze has no goal cell")


def bfs_distance(maze: Maze, start: Position, goal: Position, has_key: bool) -> int:
    """Return BFS distance from *start* to *goal* (edge count), or a large number if unreachable."""
    path = solve_shortest_path(maze, start, goal, has_key)
    if not path:
        return maze_width(maze) * maze_height(maze)
    return len(path) - 1


def score_episode(
    results: list[MoveResult],
    shortest_path_len: int,
) -> float:
    """Score a completed episode using BFS closest-approach shaping.

    - Goal reached: ``1.0 - 0.05 * excess_steps``, clamped to ``[0.3, 1.0]``.
    - Goal not reached: ``-0.5 + 0.3 * (1 - min_dist / maze_size)``.
    - Invalid moves: ``-0.1`` per invalid move.
    - Final result clamped to ``[-1.0, 1.0]``.
    """
    invalid_count = sum(1 for r in results if not r.success)
    goal_reached = any(r.reason == "goal" for r in results)

    valid_steps = [r for r in results if r.success]
    if goal_reached:
        excess = max(0, len(valid_steps) - shortest_path_len)
        reward = 1.0 - 0.05 * excess
        reward = max(0.3, min(1.0, reward))
    else:
        maze = results[0].state.maze if results else None
        if maze is not None and valid_steps:
            goal_pos = _find_goal(maze)
            maze_size = maze_width(maze) * maze_height(maze)
            min_dist = min(
                bfs_distance(r.state.maze, r.state.agent_pos, goal_pos, r.state.has_key)
                for r in valid_steps
            )
            reward = -0.5 + 0.3 * (1.0 - min_dist / maze_size)
        else:
            reward = -0.5

    reward -= 0.1 * invalid_count
    return max(-1.0, min(1.0, reward))


def score_episode_pbrs(
    results: list[MoveResult],
    shortest_path_len: int,
    source: dict,
) -> float:
    """Score with Potential-Based Reward Shaping (PBRS).

    Uses ``Phi(s) = -bfs_distance(agent_pos, goal, has_key)`` as the potential.
    Shaping reward per transition: ``gamma * Phi(s') - Phi(s)``.
    Total = base ``score_episode`` + ``alpha * sum(shaping)``.
    """
    base = score_episode(results, shortest_path_len)

    maze = normalize_maze(source["maze"])
    goal_pos: Position = (source["goal"][0], source["goal"][1])
    start_pos: Position = (source["start"][0], source["start"][1])

    gamma = 0.95
    alpha = 0.1

    prev_phi = -bfs_distance(maze, start_pos, goal_pos, has_key=False)
    shaped = 0.0
    for r in results:
        if r.success:
            cur_phi = -bfs_distance(r.state.maze, r.state.agent_pos, goal_pos, r.state.has_key)
            shaped += gamma * cur_phi - prev_phi
            prev_phi = cur_phi

    return max(-1.0, min(1.0, base + alpha * shaped))


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def serialize_maze(
    maze: Maze,
    start: Position,
    goal: Position,
    key_positions: list[Position],
    door_positions: list[Position],
    *,
    vision_radius: int = 1,
    max_steps: int | None = None,
) -> dict:
    """Serialise maze + config into a JSON-safe dict."""
    if max_steps is None:
        max_steps = maze_width(maze) * maze_height(maze)
    shortest = solve_shortest_path(maze, start, goal, has_key=False)
    return {
        "maze": [list(row) for row in maze],
        "width": maze_width(maze),
        "height": maze_height(maze),
        "start": list(start),
        "goal": list(goal),
        "keys": [list(p) for p in key_positions],
        "doors": [list(p) for p in door_positions],
        "vision_radius": vision_radius,
        "max_steps": max_steps,
        "shortest_path_len": len(shortest),
    }


def deserialize_maze(record: dict) -> MazeState:
    """Reconstruct the initial ``MazeState`` from a serialised record."""
    maze = normalize_maze(record["maze"])
    start: Position = (record["start"][0], record["start"][1])
    return MazeState(
        maze=maze,
        agent_pos=start,
        has_key=False,
        steps_taken=0,
        max_steps=record["max_steps"],
        vision_radius=record.get("vision_radius", 1),
    )