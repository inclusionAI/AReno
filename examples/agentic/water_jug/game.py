"""Water-jug game logic: state transitions, BFS solver, and prompt formatting."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

State = tuple[int, ...]


def normalize_state(capacities: Iterable[int], state: Iterable[int]) -> State:
    """Validate and return a water-jug state."""
    caps = [int(c) for c in capacities]
    lst = [int(s) for s in state]
    if len(lst) != len(caps):
        raise ValueError(f"State length {len(lst)} != capacities length {len(caps)}")
    for i, (s, c) in enumerate(zip(lst, caps)):
        if not 0 <= s <= c:
            raise ValueError(f"Jug {i} has {s} litres, capacity {c}")
    return tuple(lst)


def apply_action(capacities: Iterable[int], state: Iterable[int], action: str) -> State:
    """Apply a single action string to *state* and return the new state.

    Supported actions:
        fill(i)     - fill jug i to its capacity
        empty(i)    - empty jug i
        pour(i,j)   - pour from jug i into jug j until j is full or i is empty
    """
    caps = tuple(int(c) for c in capacities)
    st = list(int(s) for s in state)
    n = len(caps)

    action = action.strip()
    if action.startswith("fill("):
        idx = _parse_index(action, "fill", n)
        st[idx] = caps[idx]
    elif action.startswith("empty("):
        idx = _parse_index(action, "empty", n)
        st[idx] = 0
    elif action.startswith("pour("):
        src, dst = _parse_pair(action, n)
        space = caps[dst] - st[dst]
        amount = min(st[src], space)
        st[src] -= amount
        st[dst] += amount
    else:
        raise ValueError(f"Unknown action: {action!r}")

    return normalize_state(caps, st)


def _parse_index(action: str, name: str, n: int) -> int:
    inner = action[len(name) + 1: -1].strip()
    idx = int(inner)
    if not 0 <= idx < n:
        raise ValueError(f"Jug index {idx} out of range (0-{n - 1})")
    return idx


def _parse_pair(action: str, n: int) -> tuple[int, int]:
    # "pour(" is 5 chars; strip prefix and trailing ")"
    inner = action[5:-1].strip()
    parts = [x.strip() for x in inner.split(",")]
    if len(parts) != 2:
        raise ValueError(f"pour expects two indices, got: {action!r}")
    src, dst = int(parts[0]), int(parts[1])
    for idx in (src, dst):
        if not 0 <= idx < n:
            raise ValueError(f"Jug index {idx} out of range (0-{n - 1})")
    if src == dst:
        raise ValueError("Cannot pour from a jug into itself")
    return src, dst


def is_goal(state: Iterable[int], target: int) -> bool:
    """True when any jug holds exactly *target* litres."""
    return any(int(s) == int(target) for s in state)


def all_actions(n: int) -> list[str]:
    """Return the full action list for *n* jugs."""
    acts: list[str] = []
    for i in range(n):
        acts.append(f"fill({i})")
        acts.append(f"empty({i})")
    for i in range(n):
        for j in range(n):
            if i != j:
                acts.append(f"pour({i},{j})")
    return acts


def neighbours(capacities: Iterable[int], state: Iterable[int]):
    """Yield (action, next_state) pairs reachable from *state* in one step."""
    caps = tuple(int(c) for c in capacities)
    st = tuple(int(s) for s in state)
    n = len(caps)
    seen: set[State] = set()

    def _emit(action: str) -> tuple[str, State] | None:
        """Try an action; return (action, new_state) if it changes the state."""
        ns = apply_action(caps, st, action)
        if ns != st and ns not in seen:
            seen.add(ns)
            return (action, ns)
        return None

    for i in range(n):
        r = _emit(f"fill({i})")
        if r:
            yield r
    for i in range(n):
        r = _emit(f"empty({i})")
        if r:
            yield r
    for i in range(n):
        for j in range(n):
            if i != j:
                r = _emit(f"pour({i},{j})")
                if r:
                    yield r


def shortest_path(capacities: Iterable[int], state: Iterable[int], target: int) -> list[str] | None:
    """BFS shortest action sequence to reach *target*; ``None`` if impossible."""
    caps = tuple(int(c) for c in capacities)
    start = normalize_state(caps, state)
    tgt = int(target)
    if is_goal(start, tgt):
        return []
    visited: set[State] = {start}
    queue: deque[tuple[State, list[str]]] = deque([(start, [])])

    while queue:
        cur, path = queue.popleft()
        for action, nxt in neighbours(caps, cur):
            if nxt in visited:
                continue
            new_path = path + [action]
            if is_goal(nxt, tgt):
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def bfs_distance(capacities: Iterable[int], state: Iterable[int], target: int) -> int | None:
    """Minimum number of actions to reach *target*; ``None`` if impossible."""
    sp = shortest_path(capacities, state, target)
    return None if sp is None else len(sp)


def format_board(capacities: Iterable[int], state: Iterable[int], target: int) -> str:
    """Human-readable board description for the LLM prompt."""
    caps = [int(c) for c in capacities]
    st = [int(s) for s in state]
    lines = [
        f"Target: measure exactly {int(target)} litres in any jug.",
        "Jugs:",
    ]
    for i, (c, s) in enumerate(zip(caps, st)):
        lines.append(f"  Jug {i}: {s}/{c} litres")
    lines.append(f"Available actions: {', '.join(all_actions(len(caps)))}")
    return "\n".join(lines)


def build_user_prompt(capacities: Iterable[int], target: int) -> str:
    """Build the initial system+user prompt for the puzzle."""
    caps = [int(c) for c in capacities]
    n = len(caps)
    board = format_board(caps, [0] * n, target)
    actions_list = ", ".join(all_actions(n))
    prompt = (
        f"You are solving a water-jug puzzle.\n\n"
        f"{board}\n\n"
        f"Call the `water_jug_action` tool with one of: {actions_list}.\n"
        f"Keep calling actions until a jug contains exactly {int(target)} litres."
    )
    return prompt


def parse_trajectory(messages: list[dict[str, Any]]) -> tuple[State, list[str]]:
    """Extract the sequence of actions from chat messages (debugging only).

    This function is NOT called by the AReno training pipeline. It exists for
    manual inspection of trajectories. The reward function uses its own
    replay logic via ``record.tool_calls``.

    Returns (final_state, action_list).
    """
    caps: tuple[int, ...] | None = None
    target: int | None = None
    state: State = ()
    actions: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "system" and "Jug" in str(content):
            for line in str(content).splitlines():
                line = line.strip()
                if line.startswith("Jug ") and "/" in line:
                    parts = line.split("/")
                    if caps is None:
                        caps = ()
                    caps = caps + (int(parts[0].split()[-1]),)

        if role == "assistant" and tool_calls:
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                if fn_name == "water_jug_action":
                    import json
                    args = json.loads(tc["function"]["arguments"])
                    action = args.get("action", "")
                    if action:
                        actions.append(action)

    if caps is None:
        caps = (3, 5)
    state = tuple(0 for _ in caps)
    for a in actions:
        try:
            state = apply_action(caps, state, a)
        except Exception:
            pass
    return state, actions
