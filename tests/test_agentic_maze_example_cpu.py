"""CPU tests for the maze agentic example — no GPU required."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "maze" / f"{name}.py"
    mod_name = f"agentic_maze_{name}_for_tests"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Maze generation
# ---------------------------------------------------------------------------


def test_generate_maze_is_solvable():
    game = _load_module("game")

    grid, agent_pos, goal_pos, pairs = game.generate_maze(4, 4, seed=42, num_keys=1)

    assert grid[agent_pos[0]][agent_pos[1]] == game.AGENT
    assert grid[goal_pos[0]][goal_pos[1]] == game.GOAL
    # BFS should find a path (doors treated as passable for solvability check)
    state = game.make_state(grid, agent_pos[0], agent_pos[1])
    path_len = game.shortest_path_length(state)
    assert path_len is not None and path_len > 0


def test_generate_maze_deterministic_with_same_seed():
    game = _load_module("game")

    g1, a1, g1p, p1 = game.generate_maze(4, 4, seed=123, num_keys=1)
    g2, a2, g2p, p2 = game.generate_maze(4, 4, seed=123, num_keys=1)

    assert g1 == g2
    assert a1 == a2
    assert g1p == g2p
    assert p1 == p2


def test_generate_maze_different_seeds_produce_different_mazes():
    game = _load_module("game")

    g1, _, _, _ = game.generate_maze(5, 5, seed=1, num_keys=0)
    g2, _, _, _ = game.generate_maze(5, 5, seed=2, num_keys=0)

    assert g1 != g2


def test_generate_maze_key_reachable_before_door():
    game = _load_module("game")

    for seed in range(20):
        grid, agent_pos, goal_pos, pairs = game.generate_maze(4, 4, seed=seed, num_keys=1)
        for pair in pairs:
            key_pos = tuple(pair["key_pos"])
            door_pos = tuple(pair["door_pos"])
            # Key must be reachable from agent without passing through the door
            state = game.make_state(grid, agent_pos[0], agent_pos[1])
            path = game._bfs_path(grid, agent_pos, key_pos, {door_pos})
            assert path, f"seed={seed}: key at {key_pos} unreachable without passing door at {door_pos}"


def test_generate_maze_no_keys_when_num_keys_zero():
    game = _load_module("game")

    grid, agent_pos, goal_pos, pairs = game.generate_maze(4, 4, seed=0, num_keys=0)

    assert pairs == []
    # No K or D tiles in the grid
    for row in grid:
        assert game.KEY not in row
        assert game.DOOR not in row


# ---------------------------------------------------------------------------
# Local view rendering
# ---------------------------------------------------------------------------


def test_render_local_view_radius_one():
    game = _load_module("game")

    grid = (
        "#####",
        "#A..#",
        "#.#.#",
        "#..G#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    view = game.render_local_view(state, radius=1)

    lines = view.split("\n")
    assert len(lines) == 3
    assert all(len(line) == 3 for line in lines)
    # Centre is the agent
    assert lines[1][1] == game.AGENT
    # Top-left is wall, top is wall
    assert lines[0][0] == game.WALL
    assert lines[0][1] == game.WALL


def test_render_local_view_corner_agent():
    game = _load_module("game")

    grid = (
        "#####",
        "#A..#",
        "#.#.#",
        "#..G#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    view = game.render_local_view(state, radius=2)

    lines = view.split("\n")
    assert len(lines) == 5
    assert all(len(line) == 5 for line in lines)
    # Top-left corner is unknown (outside maze)
    assert lines[0][0] == game.UNKNOWN


def test_render_local_view_shows_goal_when_in_range():
    game = _load_module("game")

    grid = (
        "#####",
        "#A.G#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    view = game.render_local_view(state, radius=2)

    lines = view.split("\n")
    # Goal is at (1,3), agent at (1,1), radius=2 covers it
    # lines[2] is the agent row (dr=0), index 4 is dc=2 -> col 3 -> G
    assert lines[2][4] == game.GOAL


# ---------------------------------------------------------------------------
# Actions and step
# ---------------------------------------------------------------------------


def test_move_into_wall_is_invalid():
    game = _load_module("game")

    grid = (
        "#####",
        "#A..#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    next_state, reward, done, info = game.step(state, {"action": "move", "direction": "UP"})

    assert info["illegal"] is True
    assert reward < 0
    assert next_state.invalid_moves == 1


def test_move_into_open_floor():
    game = _load_module("game")

    grid = (
        "#####",
        "#A..#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    next_state, reward, done, info = game.step(state, {"action": "move", "direction": "RIGHT"})

    assert info.get("moved") is True
    assert next_state.agent_row == 1
    assert next_state.agent_col == 2
    assert not done


def test_move_to_goal_gives_positive_reward():
    game = _load_module("game")

    grid = (
        "#####",
        "#AG.#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    next_state, reward, done, info = game.step(state, {"action": "move", "direction": "RIGHT"})

    assert done is True
    assert reward == 1.0
    assert next_state.reached_goal is True


def test_move_into_door_blocked_without_key():
    game = _load_module("game")

    grid = (
        "#####",
        "#ADG#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    next_state, reward, done, info = game.step(state, {"action": "move", "direction": "RIGHT"})

    assert info["illegal"] is True
    assert next_state.agent_col == 1  # did not move


def test_pickup_key_when_standing_on_key():
    game = _load_module("game")

    grid = (
        "#####",
        "#K.G#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    next_state, reward, done, info = game.step(state, {"action": "pickup"})

    assert info.get("picked_up") is not None
    assert len(next_state.keys) == 1
    assert reward > 0


def test_pickup_without_key_is_invalid():
    game = _load_module("game")

    grid = (
        "#####",
        "#A.G#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    next_state, reward, done, info = game.step(state, {"action": "pickup"})

    assert info["illegal"] is True
    assert reward < 0


def test_use_key_opens_door():
    game = _load_module("game")

    grid = (
        "#####",
        "#KDG#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    # Pickup key first
    state, _, _, _ = game.step(state, {"action": "pickup"})
    # Use key on door to the right
    next_state, reward, done, info = game.step(state, {"action": "use_key", "direction": "RIGHT"})

    assert info.get("opened_door") is True
    assert reward > 0
    # Door tile is now empty
    assert next_state.grid[1][2] == game.EMPTY


def test_use_key_without_key_is_invalid():
    game = _load_module("game")

    grid = (
        "#####",
        "#ADG#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    next_state, reward, done, info = game.step(state, {"action": "use_key", "direction": "RIGHT"})

    assert info["illegal"] is True


def test_use_key_on_non_door_is_invalid():
    game = _load_module("game")

    grid = (
        "#####",
        "#A.G#",
        "#####",
    )
    state = game.make_state(grid, 1, 1, keys=frozenset({"k0"}))
    next_state, reward, done, info = game.step(state, {"action": "use_key", "direction": "RIGHT"})

    assert info["illegal"] is True


def test_invalid_action_returns_penalty():
    game = _load_module("game")

    grid = (
        "#####",
        "#A..#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    next_state, reward, done, info = game.step(state, None)

    assert info["illegal"] is True
    assert reward < 0


def test_step_after_done_returns_zero_reward():
    game = _load_module("game")

    grid = (
        "#####",
        "#AG.#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    state, _, done, _ = game.step(state, {"action": "move", "direction": "RIGHT"})
    assert done
    next_state, reward, done2, _ = game.step(state, {"action": "move", "direction": "RIGHT"})

    assert done2 is True
    assert reward == 0.0


def test_max_steps_triggers_terminal():
    game = _load_module("game")

    grid = (
        "#######",
        "#A....#",
        "#######",
    )
    state = game.make_state(grid, 1, 1, max_steps=2)
    state, _, done, _ = game.step(state, {"action": "move", "direction": "RIGHT"})
    assert not done
    state, _, done, info = game.step(state, {"action": "move", "direction": "RIGHT"})
    assert done
    assert info.get("timeout") is True


# ---------------------------------------------------------------------------
# Legal actions
# ---------------------------------------------------------------------------


def test_legal_actions_includes_move_options():
    game = _load_module("game")

    grid = (
        "#####",
        "#A..#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    actions = game.legal_actions(state)

    move_dirs = {a["direction"] for a in actions if a["action"] == "move"}
    assert "RIGHT" in move_dirs
    assert "UP" not in move_dirs  # wall


def test_legal_actions_includes_pickup_on_key():
    game = _load_module("game")

    grid = (
        "#####",
        "#K.G#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    actions = game.legal_actions(state)

    assert any(a["action"] == "pickup" for a in actions)


def test_legal_actions_includes_use_key_adjacent_to_door():
    game = _load_module("game")

    grid = (
        "#####",
        "#ADG#",
        "#####",
    )
    state = game.make_state(grid, 1, 1, keys=frozenset({"k0"}))
    actions = game.legal_actions(state)

    use_dirs = {a["direction"] for a in actions if a["action"] == "use_key"}
    assert "RIGHT" in use_dirs


# ---------------------------------------------------------------------------
# Trajectory reward
# ---------------------------------------------------------------------------


def test_compute_trajectory_reward_reaching_goal():
    game = _load_module("game")

    grid = (
        "#####",
        "#AG.#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    actions = [{"action": "move", "direction": "RIGHT"}]
    reward, metrics = game.compute_trajectory_reward(state, actions)

    assert reward == 1.0
    assert metrics["reached_goal"] is True


def test_compute_trajectory_reward_invalid_moves():
    game = _load_module("game")

    grid = (
        "#####",
        "#A..#",
        "#####",
    )
    state = game.make_state(grid, 1, 1, max_steps=5)
    actions = [
        {"action": "move", "direction": "UP"},  # wall
        {"action": "move", "direction": "LEFT"},  # wall
    ]
    reward, metrics = game.compute_trajectory_reward(state, actions)

    assert reward < 0
    assert metrics["invalid_moves"] >= 2


def test_compute_trajectory_reward_key_door_goal():
    game = _load_module("game")

    # A at (1,1), K at (1,2), D at (1,3), G at (1,4)
    grid = (
        "#######",
        "#AKDG.#",
        "#######",
    )
    state = game.make_state(grid, 1, 1)
    actions = [
        {"action": "move", "direction": "RIGHT"},  # step onto key
        {"action": "pickup"},                       # pick up key
        {"action": "use_key", "direction": "RIGHT"},  # open door
        {"action": "move", "direction": "RIGHT"},  # through opened door
        {"action": "move", "direction": "RIGHT"},  # to goal
    ]
    reward, metrics = game.compute_trajectory_reward(state, actions)

    assert metrics["reached_goal"] is True
    assert metrics["keys_picked"] == 1
    assert metrics["doors_opened"] == 1


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------


def test_dataset_generator_produces_valid_records():
    generator = _load_module("dataset_generator")
    game = _load_module("game")

    records = generator.generate_records(16, seed=7, rows=3, cols=3, num_keys=1)

    assert len(records) == 16
    for record in records:
        state = generator.record_to_state(record)
        # Every generated maze should be solvable
        path_len = game.shortest_path_length(state)
        assert path_len is not None and path_len > 0


def test_dataset_generator_deterministic():
    generator = _load_module("dataset_generator")

    r1 = generator.generate_records(8, seed=99, rows=3, cols=3, num_keys=0)
    r2 = generator.generate_records(8, seed=99, rows=3, cols=3, num_keys=0)

    assert r1 == r2


def test_dataset_generator_write_jsonl(tmp_path):
    generator = _load_module("dataset_generator")

    records = generator.generate_records(4, seed=1, rows=3, cols=3)
    out = tmp_path / "maze.jsonl"
    with out.open("w") as f:
        generator.write_jsonl(records, f)

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 4
    for line in lines:
        record = json.loads(line)
        assert "grid" in record
        assert "agent_pos" in record
        assert "goal_pos" in record


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def test_dataset_loader_loads_and_formats(tmp_path):
    generator = _load_module("dataset_generator")
    loader = _load_module("dataset_loader")
    game = _load_module("game")

    records = generator.generate_records(4, seed=1, rows=3, cols=3)
    out = tmp_path / "maze.jsonl"
    with out.open("w") as f:
        generator.write_jsonl(records, f)

    loaded = loader.load_training_dataset(str(out))

    assert len(loaded) == 4
    for item in loaded:
        assert "prompt" in item
        assert "state" in item
        assert "legal_actions" in item
        assert game.AGENT in item["prompt"]


def test_dataset_loader_missing_file_raises(tmp_path):
    loader = _load_module("dataset_loader")

    import pytest

    with pytest.raises(FileNotFoundError):
        loader.load_training_dataset(str(tmp_path / "nonexistent.jsonl"))


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def test_reward_fn_reaching_goal():
    reward = _load_module("reward")
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    grid = (
        "#####",
        "#AG.#",
        "#####",
    )
    record = {
        "grid": list(grid),
        "agent_pos": [1, 1],
        "max_steps": 5,
    }
    state = generator.record_to_state(record)
    # Simulate tool calls
    record_obj = SimpleNamespace(
        source_record=record,
        tool_calls=[{"name": "act", "arguments": {"action": "move", "direction": "RIGHT"}}],
    )

    result = reward.reward_fn(record_obj)
    assert result == 1.0


def test_reward_fn_invalid_moves():
    reward = _load_module("reward")
    generator = _load_module("dataset_generator")

    grid = (
        "#####",
        "#A..#",
        "#####",
    )
    record = {
        "grid": list(grid),
        "agent_pos": [1, 1],
        "max_steps": 10,
    }
    record_obj = SimpleNamespace(
        source_record=record,
        tool_calls=[
            {"name": "act", "arguments": {"action": "move", "direction": "UP"}},
            {"name": "act", "arguments": {"action": "move", "direction": "LEFT"}},
        ],
    )

    result = reward.reward_fn(record_obj)
    assert result < 0


def test_reward_fn_ignores_non_act_tool_calls():
    reward = _load_module("reward")
    generator = _load_module("dataset_generator")

    grid = (
        "#####",
        "#AG.#",
        "#####",
    )
    record = {
        "grid": list(grid),
        "agent_pos": [1, 1],
        "max_steps": 5,
    }
    record_obj = SimpleNamespace(
        source_record=record,
        tool_calls=[
            {"name": "look", "arguments": {}},
            {"name": "act", "arguments": {"action": "move", "direction": "RIGHT"}},
        ],
    )

    result = reward.reward_fn(record_obj)
    assert result == 1.0


# ---------------------------------------------------------------------------
# Shortest path
# ---------------------------------------------------------------------------


def test_shortest_path_length_simple():
    game = _load_module("game")

    grid = (
        "#####",
        "#A.G#",
        "#####",
    )
    state = game.make_state(grid, 1, 1)
    length = game.shortest_path_length(state)

    assert length == 2


def test_shortest_path_length_none_if_unreachable():
    game = _load_module("game")

    grid = (
        "#######",
        "#A....#",
        "#######",
        "#G....#",
        "#######",
    )
    state = game.make_state(grid, 1, 1)
    length = game.shortest_path_length(state)

    assert length is None
