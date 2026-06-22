"""Generate DuelGrid states for the agentic grid-tactics example."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026
MAP_SIZE = 11


def generate_records(count: int = DEFAULT_COUNT, *, seed: int = DEFAULT_SEED) -> list[dict]:
    """Generate reproducible DuelGrid prompt states."""

    rng = random.Random(seed)
    records = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError("could not generate enough unique DuelGrid states")
        state = _random_state(rng)
        key = (
            tuple(game.render_map(state).splitlines()),
            state.agent.hp,
            state.user.hp,
            state.agent.energy,
            state.user.energy,
            state.turn,
        )
        if key in seen or not game.legal_actions(state):
            continue
        seen.add(key)
        records.append(_state_to_record(state, f"generated-{len(records):05d}"))
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def record_to_state(record: dict) -> game.State:
    """Rebuild a State from a generated JSON record."""

    return game.make_state(
        record["map"],
        agent_hp=int(record.get("agent_hp", 10)),
        user_hp=int(record.get("user_hp", 10)),
        agent_energy=int(record.get("agent_energy", 2)),
        user_energy=int(record.get("user_energy", 2)),
        agent_max_energy=int(record.get("agent_max_energy", game.MAX_ENERGY)),
        user_max_energy=int(record.get("user_max_energy", game.MAX_ENERGY)),
        turn=int(record.get("turn", 0)),
        max_turns=int(record.get("max_turns", 40)),
    )


def _random_state(rng: random.Random) -> game.State:
    scenario = rng.choice(["adjacent", "ranged", "pickup_energy", "pickup_health", "approach", "random", "random"])
    if scenario != "random":
        return _scenario_state(rng, scenario)
    return _random_exploration_state(rng)


def _random_exploration_state(rng: random.Random) -> game.State:
    rows = [
        [game.WALL if row in {0, MAP_SIZE - 1} or col in {0, MAP_SIZE - 1} else game.EMPTY for col in range(MAP_SIZE)]
        for row in range(MAP_SIZE)
    ]
    for row in range(2, MAP_SIZE - 2, 2):
        for col in range(2, MAP_SIZE - 2, 3):
            if rng.random() < 0.65:
                rows[row][col] = game.WALL
    for _ in range(10):
        row = rng.randrange(1, MAP_SIZE - 1)
        col = rng.randrange(1, MAP_SIZE - 1)
        if rows[row][col] == game.EMPTY and rng.random() < 0.45:
            rows[row][col] = game.WALL
    open_cells = [(r, c) for r in range(1, MAP_SIZE - 1) for c in range(1, MAP_SIZE - 1) if rows[r][c] == game.EMPTY]
    agent_pos, user_pos = _choose_distant_pair(rng, open_cells)
    rows[agent_pos[0]][agent_pos[1]] = game.AGENT
    rows[user_pos[0]][user_pos[1]] = game.USER
    for tile, count in ((game.HEALTH, 2), (game.ENERGY, 2), (game.TRAP, 2)):
        for _ in range(count):
            _place_tile(rng, rows, tile)
    return game.make_state(
        tuple("".join(row) for row in rows),
        agent_hp=rng.randint(5, 10),
        user_hp=rng.randint(5, 10),
        agent_energy=rng.randint(1, 3),
        user_energy=rng.randint(1, 3),
        turn=rng.randint(0, 10),
        max_turns=40,
    )


def _scenario_state(rng: random.Random, scenario: str) -> game.State:
    rows = _base_open_map()
    if scenario == "adjacent":
        agent_pos = (5, 5)
        user_pos = rng.choice([(5, 6), (5, 4), (4, 5), (6, 5)])
    elif scenario == "ranged":
        agent_pos = (5, 3)
        user_pos = (5, 6)
    elif scenario == "pickup_energy":
        agent_pos = (5, 5)
        user_pos = (2, 8)
        rows[agent_pos[0]][agent_pos[1]] = game.ENERGY
    elif scenario == "pickup_health":
        agent_pos = (5, 5)
        user_pos = (2, 8)
        rows[agent_pos[0]][agent_pos[1]] = game.HEALTH
    else:
        agent_pos = (7, 2)
        user_pos = (3, 7)
        for wall in [(5, 4), (5, 5), (5, 6), (4, 5), (6, 5)]:
            rows[wall[0]][wall[1]] = game.WALL
        rows[7][3] = game.ENERGY
        rows[4][7] = game.HEALTH
    _place_actors(rows, agent_pos, user_pos)
    return game.make_state(
        tuple("".join(row) for row in rows),
        agent_hp=rng.randint(6, 10),
        user_hp=rng.randint(4, 8),
        agent_energy=rng.randint(2, 3),
        user_energy=rng.randint(1, 3),
        turn=rng.randint(0, 5),
        max_turns=40,
    )


def _base_open_map() -> list[list[str]]:
    rows = [
        [game.WALL if row in {0, MAP_SIZE - 1} or col in {0, MAP_SIZE - 1} else game.EMPTY for col in range(MAP_SIZE)]
        for row in range(MAP_SIZE)
    ]
    for row, col in [(2, 2), (2, 8), (4, 2), (6, 8), (8, 2), (8, 8)]:
        rows[row][col] = game.WALL
    rows[3][3] = game.TRAP
    rows[7][7] = game.TRAP
    rows[2][5] = game.ENERGY
    rows[8][5] = game.HEALTH
    return rows


def _place_actors(rows: list[list[str]], agent_pos: tuple[int, int], user_pos: tuple[int, int]) -> None:
    rows[agent_pos[0]][agent_pos[1]] = game.AGENT
    rows[user_pos[0]][user_pos[1]] = game.USER


def _choose_distant_pair(rng: random.Random, cells: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    for _ in range(100):
        first, second = rng.sample(cells, 2)
        if abs(first[0] - second[0]) + abs(first[1] - second[1]) >= 6:
            return first, second
    return rng.sample(cells, 2)


def _place_tile(rng: random.Random, rows: list[list[str]], tile: str) -> None:
    empty = [(r, c) for r in range(1, MAP_SIZE - 1) for c in range(1, MAP_SIZE - 1) if rows[r][c] == game.EMPTY]
    if empty:
        row, col = rng.choice(empty)
        rows[row][col] = tile


def _state_to_record(state: game.State, record_id: str) -> dict:
    best_actions = game.heuristic_actions(state)
    return {
        "id": record_id,
        "map": game.render_map(state).splitlines(),
        "agent_hp": state.agent.hp,
        "user_hp": state.user.hp,
        "agent_energy": state.agent.energy,
        "user_energy": state.user.energy,
        "agent_max_energy": state.agent.max_energy,
        "user_max_energy": state.user.max_energy,
        "turn": state.turn,
        "max_turns": state.max_turns,
        "best_action": best_actions[0],
        "best_actions": best_actions,
        "legal_actions": game.legal_actions(state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL states for the Areno DuelGrid agentic example.")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of states to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    records = generate_records(args.count, seed=args.seed)
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()
