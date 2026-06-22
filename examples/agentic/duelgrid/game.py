"""DuelGrid helpers for an agentic grid-tactics example.

The example is intentionally small: one user-controlled player (U), one
LLM-controlled player (A), a text map, strict JSON/tool actions, and rewards
computed directly from the rules engine for RLVR-style training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

EMPTY = "."
WALL = "#"
AGENT = "A"
USER = "U"
HEALTH = "H"
ENERGY = "E"
TRAP = "X"
DIRECTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
ACTION_NAMES = {"MOVE", "ATTACK", "RANGED_ATTACK", "SHIELD", "PICKUP", "WAIT"}
MAX_ENERGY = 3
MAX_ENERGY_CAP = 6
ACTION_COSTS = {
    "MOVE": 1,
    "ATTACK": 1,
    "RANGED_ATTACK": 2,
    "SHIELD": 1,
    "PICKUP": 0,
    "WAIT": 0,
}

DEFAULT_MAP = (
    "###########",
    "#A..#....E#",
    "#.#.#.###.#",
    "#.#...#...#",
    "#...X...#.#",
    "###.#.#...#",
    "#H..#.#.#U#",
    "#...#...#.#",
    "#.#...X...#",
    "#E....#..H#",
    "###########",
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Player:
    row: int
    col: int
    hp: int = 10
    energy: int = 2
    max_energy: int = MAX_ENERGY
    shield: int = 0


@dataclass(frozen=True)
class State:
    grid: tuple[str, ...]
    agent: Player
    user: Player
    turn: int = 0
    max_turns: int = 40


def make_state(
    grid: tuple[str, ...] | list[str] = DEFAULT_MAP,
    *,
    agent_hp: int = 10,
    user_hp: int = 10,
    agent_energy: int = 2,
    user_energy: int = 2,
    agent_max_energy: int = MAX_ENERGY,
    user_max_energy: int = MAX_ENERGY,
    turn: int = 0,
    max_turns: int = 40,
) -> State:
    """Parse a text map into a validated DuelGrid state."""

    rows = tuple(str(row) for row in grid)
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("DuelGrid map must be a non-empty rectangle")
    allowed = {EMPTY, WALL, AGENT, USER, HEALTH, ENERGY, TRAP}
    agent_pos = None
    user_pos = None
    clean_rows = []
    for row_idx, row in enumerate(rows):
        clean = []
        for col_idx, cell in enumerate(row):
            if cell not in allowed:
                raise ValueError(f"unsupported DuelGrid cell: {cell!r}")
            if cell == AGENT:
                if agent_pos is not None:
                    raise ValueError("DuelGrid map must contain exactly one A")
                agent_pos = (row_idx, col_idx)
                clean.append(EMPTY)
            elif cell == USER:
                if user_pos is not None:
                    raise ValueError("DuelGrid map must contain exactly one U")
                user_pos = (row_idx, col_idx)
                clean.append(EMPTY)
            else:
                clean.append(cell)
        clean_rows.append("".join(clean))
    if agent_pos is None or user_pos is None:
        raise ValueError("DuelGrid map must contain A and U")
    return State(
        grid=tuple(clean_rows),
        agent=Player(*agent_pos, hp=agent_hp, energy=min(agent_energy, agent_max_energy), max_energy=agent_max_energy),
        user=Player(*user_pos, hp=user_hp, energy=min(user_energy, user_max_energy), max_energy=user_max_energy),
        turn=turn,
        max_turns=max_turns,
    )


def render_map(state: State) -> str:
    """Render a state as a text map with A and U overlays."""

    rows = [list(row) for row in state.grid]
    if state.agent.hp > 0:
        rows[state.agent.row][state.agent.col] = AGENT
    if state.user.hp > 0:
        rows[state.user.row][state.user.col] = USER
    return "\n".join("".join(row) for row in rows)


def format_prompt(state: State) -> str:
    """Build a turn prompt for the LLM-controlled player."""

    legal = legal_actions(state)
    return (
        "You are player A in a turn-based grid tactics game.\n\n"
        "Goal:\nDefeat player U.\n\n"
        "Rules:\n"
        "- Choose a sequence of legal action objects from the Legal actions list.\n"
        "- You may use multiple actions in one turn until A energy is spent.\n"
        "- Energy costs: MOVE 1, ATTACK 1, RANGED_ATTACK 2, SHIELD 1, PICKUP 0, WAIT 0.\n"
        "- If using tool calls, the function name is choose_action; action names are arguments, not tool names.\n"
        "- MOVE moves one tile up/down/left/right.\n"
        "- ATTACK hits U for 2 damage if adjacent.\n"
        "- RANGED_ATTACK hits U for 1 damage up to 3 tiles away in a straight line if no wall blocks.\n"
        "- MOVE, ATTACK, and RANGED_ATTACK require the direction shown in the legal action object.\n"
        "- SHIELD, PICKUP, and WAIT do not use direction.\n"
        "- SHIELD reduces the next incoming damage.\n"
        "- PICKUP collects H or E when standing on that tile.\n"
        "- WAIT does nothing.\n"
        "- You cannot move into walls.\n\n"
        f"Current state:\nA hp: {state.agent.hp}\nU hp: {state.user.hp}\n"
        f"A energy: {state.agent.energy}/{state.agent.max_energy}\nTurn: {state.turn}\n\n"
        f"Map:\n{render_map(state)}\n\n"
        f"Current legal first actions:\n{json.dumps(legal, separators=(',', ':'))}\n\n"
        'Return JSON shaped like {"actions":[...]}.'
    )


def legal_actions(state: State, actor: str = AGENT) -> list[dict[str, str]]:
    """Return legal actions for A or U."""

    player = _player(state, actor)
    opponent = _opponent(state, actor)
    actions: list[dict[str, str]] = []
    for direction, (dr, dc) in DIRECTIONS.items():
        row, col = player.row + dr, player.col + dc
        if (
            player.energy >= ACTION_COSTS["MOVE"]
            and _walkable(state, row, col)
            and (row, col) != (opponent.row, opponent.col)
        ):
            actions.append({"action": "MOVE", "direction": direction})
        if player.energy >= ACTION_COSTS["ATTACK"] and (row, col) == (opponent.row, opponent.col):
            actions.append({"action": "ATTACK", "direction": direction})
    for direction in (
        _ranged_directions(state, player, opponent) if player.energy >= ACTION_COSTS["RANGED_ATTACK"] else []
    ):
        actions.append({"action": "RANGED_ATTACK", "direction": direction})
    if _tile(state, player.row, player.col) in {HEALTH, ENERGY}:
        actions.append({"action": "PICKUP"})
    if player.energy >= ACTION_COSTS["SHIELD"]:
        actions.append({"action": "SHIELD"})
    actions.append({"action": "WAIT"})
    return _dedupe_actions(actions)


def parse_action(text: str | dict[str, Any] | None) -> dict[str, str] | None:
    """Parse a JSON action from model text or an existing dict."""

    if isinstance(text, dict):
        raw = text
    elif isinstance(text, str):
        match = _JSON_RE.search(text)
        if not match:
            return None
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    else:
        return None
    action = str(raw.get("action", "")).upper()
    direction = raw.get("direction")
    parsed: dict[str, str] = {"action": action}
    if direction is not None:
        parsed["direction"] = str(direction).upper()
    return parsed if action in ACTION_NAMES else None


def parse_actions(text: str | dict[str, Any] | list[Any] | None) -> list[dict[str, str]]:
    """Parse a sequence of action objects from JSON/tool arguments."""

    raw: Any
    if isinstance(text, list):
        raw = {"actions": text}
    elif isinstance(text, dict):
        raw = text
    elif isinstance(text, str):
        match = _JSON_RE.search(text)
        if not match:
            return []
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    else:
        return []
    if isinstance(raw, dict) and isinstance(raw.get("actions"), list):
        parsed = [parse_action(item) for item in raw["actions"]]
        return [action for action in parsed if action is not None]
    single = parse_action(raw)
    return [single] if single is not None else []


def step(
    state: State, action: dict[str, str] | None, actor: str = AGENT, *, advance_turn: bool = True
) -> tuple[State, float, bool, dict[str, Any]]:
    """Apply one action and return next_state, reward, done, info."""

    parsed = parse_action(action)
    legal = legal_actions(state, actor)
    if parsed not in legal:
        return state, -0.3, is_terminal(state), {"illegal": True, "legal_actions": legal}

    before_agent_hp = state.agent.hp
    before_user_hp = state.user.hp
    before_distance = _manhattan(_player(state, actor), _opponent(state, actor))
    next_state = _apply_legal_action(state, parsed, actor, advance_turn=advance_turn)
    after_distance = _manhattan(_player(next_state, actor), _opponent(next_state, actor))
    damage_dealt = (
        max(0, before_user_hp - next_state.user.hp) if actor == AGENT else max(0, before_agent_hp - next_state.agent.hp)
    )
    damage_taken = (
        max(0, before_agent_hp - next_state.agent.hp) if actor == AGENT else max(0, before_user_hp - next_state.user.hp)
    )
    reward = 0.10 * damage_dealt - 0.08 * damage_taken
    if parsed["action"] == "MOVE":
        if after_distance < before_distance:
            reward += 0.03
        elif after_distance > before_distance:
            reward -= 0.02
    if parsed["action"] == "PICKUP":
        reward += 0.10
    if parsed["action"] == "WAIT":
        reward -= 0.02
    done = is_terminal(next_state)
    if done:
        reward += terminal_score(next_state)
    return next_state, reward, done, {"illegal": False, "damage_dealt": damage_dealt, "damage_taken": damage_taken}


def step_turn(
    state: State, actions: list[dict[str, str]] | dict[str, str] | None, actor: str = AGENT
) -> tuple[State, float, bool, dict[str, Any]]:
    """Apply an energy-budgeted turn and refresh the actor's energy after it."""

    action_list = actions if isinstance(actions, list) else ([actions] if actions is not None else [])
    current = state
    total_reward = 0.0
    applied = []
    spent_energy = 0
    for action in action_list:
        before_actor = _player(current, actor)
        parsed = parse_action(action)
        action_cost = ACTION_COSTS.get(parsed["action"], 0) if parsed is not None else 0
        if spent_energy + action_cost > _player(state, actor).energy:
            return (
                _end_turn(current, actor, state.turn),
                total_reward - 0.3,
                is_terminal(current),
                {
                    "illegal": True,
                    "reason": "energy_exceeded",
                    "spent_energy": spent_energy,
                    "energy_budget": _player(state, actor).energy,
                    "applied_actions": applied,
                    "failed_action": action,
                    "legal_actions": legal_actions(current, actor),
                },
            )
        next_state, reward, done, info = step(current, action, actor, advance_turn=False)
        total_reward += reward
        if info.get("illegal"):
            return (
                _end_turn(current, actor, state.turn),
                total_reward,
                done,
                {
                    "illegal": True,
                    "applied_actions": applied,
                    "failed_action": action,
                    "legal_actions": info.get("legal_actions", []),
                },
            )
        applied.append(action)
        spent_energy += action_cost
        current = next_state
        if done or _player(current, actor).energy <= 0 or before_actor.energy == _player(current, actor).energy == 0:
            break
    if not applied:
        current, reward, done, info = step(current, {"action": "WAIT"}, actor, advance_turn=False)
        total_reward += reward
        applied.append({"action": "WAIT"})
        if info.get("illegal"):
            return current, total_reward, done, {"illegal": True, "applied_actions": []}
    refreshed = _end_turn(current, actor, state.turn)
    return (
        refreshed,
        total_reward,
        is_terminal(refreshed),
        {
            "illegal": False,
            "spent_energy": spent_energy,
            "energy_budget": _player(state, actor).energy,
            "applied_actions": applied,
        },
    )


def end_turn(state: State, actor: str) -> State:
    """Advance one turn and refresh the actor's energy."""

    return _end_turn(state, actor, state.turn)


def spend_energy(state: State, actor: str, amount: int) -> State:
    """Spend actor energy without otherwise changing the state."""

    amount = max(0, int(amount))
    if actor == AGENT:
        return replace(state, agent=replace(state.agent, energy=max(0, state.agent.energy - amount)))
    return replace(state, user=replace(state.user, energy=max(0, state.user.energy - amount)))


def score_action(state: State, action: dict[str, str] | None) -> float:
    """Score a single LLM action for reward functions."""

    _next_state, reward, _done, _info = step(state, action, AGENT)
    return reward


def score_actions(state: State, actions: list[dict[str, str]] | dict[str, str] | None) -> float:
    """Score one LLM turn."""

    _next_state, reward, _done, _info = step_turn(state, actions, AGENT)
    return reward


def heuristic_actions(state: State, actor: str = AGENT) -> list[dict[str, str]]:
    """Return a deterministic baseline action sequence for one energy-budgeted turn."""

    actions = []
    current = state
    while _player(current, actor).energy > 0:
        action = heuristic_action(current, actor)
        if action["action"] == "WAIT":
            break
        next_state, _reward, _done, info = step(current, action, actor, advance_turn=False)
        if info.get("illegal"):
            break
        actions.append(action)
        current = next_state
        if is_terminal(current):
            break
    return actions or [{"action": "WAIT"}]


def heuristic_action(state: State, actor: str = AGENT) -> dict[str, str]:
    """Return a deterministic baseline single action."""

    legal = legal_actions(state, actor)
    for preferred in ("ATTACK", "RANGED_ATTACK", "PICKUP"):
        for action in legal:
            if action["action"] == preferred:
                return action
    player = _player(state, actor)
    opponent = _opponent(state, actor)
    moves = [action for action in legal if action["action"] == "MOVE"]
    if moves:
        return min(moves, key=lambda item: _distance_after_move(player, opponent, item["direction"]))
    return {"action": "SHIELD"} if {"action": "SHIELD"} in legal else {"action": "WAIT"}


def is_terminal(state: State) -> bool:
    return state.agent.hp <= 0 or state.user.hp <= 0 or state.turn >= state.max_turns


def terminal_score(state: State) -> float:
    if state.user.hp <= 0 and state.agent.hp > 0:
        return 1.0
    if state.agent.hp <= 0 and state.user.hp > 0:
        return -1.0
    return (state.agent.hp - state.user.hp) / 10.0


def _apply_legal_action(state: State, action: dict[str, str], actor: str, *, advance_turn: bool = True) -> State:
    player = _player(state, actor)
    opponent = _opponent(state, actor)
    grid = [list(row) for row in state.grid]
    next_player = player
    next_opponent = opponent
    name = action["action"]
    direction = action.get("direction")
    cost = ACTION_COSTS[name]
    next_player = replace(next_player, energy=max(0, next_player.energy - cost))
    if name == "MOVE" and direction in DIRECTIONS:
        dr, dc = DIRECTIONS[direction]
        next_player = replace(next_player, row=player.row + dr, col=player.col + dc)
        if _tile(state, next_player.row, next_player.col) == TRAP:
            next_player = replace(next_player, hp=max(0, next_player.hp - 1))
    elif name in {"ATTACK", "RANGED_ATTACK"}:
        damage = 2 if name == "ATTACK" else 1
        blocked = min(opponent.shield, damage)
        next_opponent = replace(
            opponent, hp=max(0, opponent.hp - damage + blocked), shield=max(0, opponent.shield - blocked)
        )
    elif name == "SHIELD":
        next_player = replace(next_player, shield=min(3, player.shield + 1))
    elif name == "PICKUP":
        tile = _tile(state, player.row, player.col)
        if tile == HEALTH:
            next_player = replace(next_player, hp=min(10, player.hp + 2))
        elif tile == ENERGY:
            upgraded_max = min(MAX_ENERGY_CAP, next_player.max_energy + 1)
            next_player = replace(next_player, max_energy=upgraded_max, energy=upgraded_max)
        grid[player.row][player.col] = EMPTY
    next_turn = state.turn + 1 if advance_turn else state.turn
    if actor == AGENT:
        return State(tuple("".join(row) for row in grid), next_player, next_opponent, next_turn, state.max_turns)
    return State(tuple("".join(row) for row in grid), next_opponent, next_player, next_turn, state.max_turns)


def _refresh_actor_energy(state: State, actor: str) -> State:
    if actor == AGENT:
        return replace(state, agent=replace(state.agent, energy=state.agent.max_energy))
    return replace(state, user=replace(state.user, energy=state.user.max_energy))


def _end_turn(state: State, actor: str, start_turn: int) -> State:
    return replace(_refresh_actor_energy(state, actor), turn=start_turn + 1)


def _ranged_directions(state: State, player: Player, opponent: Player) -> list[str]:
    directions = []
    for direction, (dr, dc) in DIRECTIONS.items():
        for distance in range(1, 4):
            row, col = player.row + dr * distance, player.col + dc * distance
            if not _in_bounds(state, row, col) or _tile(state, row, col) == WALL:
                break
            if (row, col) == (opponent.row, opponent.col):
                directions.append(direction)
                break
    return directions


def _dedupe_actions(actions: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for action in actions:
        key = tuple(sorted(action.items()))
        if key not in seen:
            seen.add(key)
            result.append(action)
    return result


def _distance_after_move(player: Player, opponent: Player, direction: str) -> int:
    dr, dc = DIRECTIONS[direction]
    return abs(player.row + dr - opponent.row) + abs(player.col + dc - opponent.col)


def _manhattan(first: Player, second: Player) -> int:
    return abs(first.row - second.row) + abs(first.col - second.col)


def _player(state: State, actor: str) -> Player:
    return state.agent if actor == AGENT else state.user


def _opponent(state: State, actor: str) -> Player:
    return state.user if actor == AGENT else state.agent


def _walkable(state: State, row: int, col: int) -> bool:
    return _in_bounds(state, row, col) and _tile(state, row, col) != WALL


def _in_bounds(state: State, row: int, col: int) -> bool:
    return 0 <= row < len(state.grid) and 0 <= col < len(state.grid[0])


def _tile(state: State, row: int, col: int) -> str:
    return state.grid[row][col]
