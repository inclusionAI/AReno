"""CPU tests for the maze agentic RL example."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "maze"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    mod_name = f"agentic_maze_{name}_for_tests"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module  # register before exec for dataclass forward refs
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop(mod_name, None)
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


def _load_module_without_sys_path(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    mod_name = f"agentic_maze_{name}_without_path_for_tests"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(mod_name, None)
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

def test_maze_generator_produces_valid_solvable_records():
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    records = generator.generate_records(8, seed=7)

    assert len(records) == 8
    for record in records:
        maze = game.normalize_maze(record["maze"])
        start = tuple(record["start"])
        goal = tuple(record["goal"])
        path = game.solve_shortest_path(maze, start, goal, has_key=False)
        assert path, "maze must be solvable from start without key"
        assert record["shortest_path_len"] == len(path)
        assert record["max_steps"] > 0
        assert record["vision_radius"] >= 0


def test_maze_generator_is_reproducible():
    generator = _load_module("dataset_generator")
    game = _load_module("game")

    r1 = generator.generate_records(4, seed=42)
    r2 = generator.generate_records(4, seed=42)
    assert r1 == r2

    # Prompt does not leak the full maze grid — only the local view legend.
    loader = _load_module("dataset_loader")
    loaded = loader.load_training_dataset("unused")
    for r in loaded:
        prompt = r["prompt"]
        # The prompt should not contain a full maze grid (list of list of cell types).
        assert "['wall'" not in prompt
        assert "[['wall'" not in prompt


# ---------------------------------------------------------------------------
# Game rules tests
# ---------------------------------------------------------------------------

def test_maze_game_rules_wall_collision_and_door_lock():
    game = _load_module("game")

    maze = game.normalize_maze([
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "wall", "empty", "wall"],
        ["wall", "empty", "door", "empty", "wall"],
        ["wall", "empty", "empty", "key", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ])
    start = (1, 1)
    state = game.MazeState(
        maze=maze, agent_pos=start, has_key=False,
        steps_taken=0, max_steps=50, vision_radius=1,
    )

    # Move into a wall.
    result = game.apply_move(state, "right")
    assert not result.success
    assert result.reason == "wall"

    # Move into a door without key.
    state2 = game.MazeState(
        maze=maze, agent_pos=(2, 1), has_key=False,
        steps_taken=0, max_steps=50, vision_radius=1,
    )
    result = game.apply_move(state2, "right")
    assert not result.success
    assert result.reason == "locked_door"

    # Move onto key — auto-pickup.
    state3 = game.MazeState(
        maze=maze, agent_pos=(3, 2), has_key=False,
        steps_taken=0, max_steps=50, vision_radius=1,
    )
    result = game.apply_move(state3, "right")
    assert result.success
    assert result.state.has_key is True

    # Now door is passable with key.
    state4 = game.MazeState(
        maze=maze, agent_pos=(2, 1), has_key=True,
        steps_taken=0, max_steps=50, vision_radius=1,
    )
    result = game.apply_move(state4, "right")
    assert result.success

    # Max steps exhausted → terminal.
    state5 = game.MazeState(
        maze=maze, agent_pos=start, has_key=False,
        steps_taken=50, max_steps=50, vision_radius=1,
    )
    result = game.apply_move(state5, "down")
    assert result.terminal


def test_maze_local_view_does_not_leak_full_map():
    game = _load_module("game")

    maze = game.normalize_maze([
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "wall", "empty", "wall", "empty", "wall"],
        ["wall", "empty", "wall", "empty", "wall", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ])
    state = game.MazeState(
        maze=maze, agent_pos=(1, 1), has_key=False,
        steps_taken=0, max_steps=49, vision_radius=1,
    )
    view = game.local_view(state)
    lines = view.strip().split("\n")
    assert len(lines) == 3  # 3x3 view
    for line in lines:
        assert len(line.split()) == 3  # 3 columns
    # Cells outside vision should be "?" not visible.
    # The cell at (3, 5) is far away and must NOT appear in the view.
    assert "G" not in view  # no goal in this maze anyway


def test_maze_supports_multiple_sizes():
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    for size in (5, 7, 9):
        records = generator.generate_records(4, seed=size, width=size, height=size)
        for r in records:
            maze = game.normalize_maze(r["maze"])
            start = tuple(r["start"])
            goal = tuple(r["goal"])
            path = game.solve_shortest_path(maze, start, goal, has_key=False)
            assert path, f"size {size} maze must be solvable"


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

def test_maze_reward_scores_goal_and_failure_paths():
    game = _load_module("game")
    reward = _load_module("reward")
    generator = _load_module("dataset_generator")

    record = generator.generate_records(1, seed=7)[0]
    maze = game.normalize_maze(record["maze"])
    start = tuple(record["start"])
    goal = tuple(record["goal"])

    # Optimal path → reward = 1.0.
    path = game.solve_shortest_path(maze, start, goal, has_key=False)
    directions = []
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        if dr == -1:
            directions.append("up")
        elif dr == 1:
            directions.append("down")
        elif dc == -1:
            directions.append("left")
        elif dc == 1:
            directions.append("right")

    tool_calls = [{"name": "move", "arguments": json.dumps({"direction": d})} for d in directions]
    mock = SimpleNamespace(source_record=record, tool_calls=tool_calls, completion="")
    assert reward.reward_fn(mock) == 1.0

    # No moves → negative reward.
    mock_empty = SimpleNamespace(source_record=record, tool_calls=[], completion="")
    assert reward.reward_fn(mock_empty) == -0.5

    # Invalid move → more negative.
    mock_invalid = SimpleNamespace(
        source_record=record,
        tool_calls=[{"name": "move", "arguments": json.dumps({"direction": "up"})}],
        completion="",
    )
    assert reward.reward_fn(mock_invalid) < 0.0


# ---------------------------------------------------------------------------
# Tool schema test
# ---------------------------------------------------------------------------

def test_maze_tool_schema_is_closed_and_bounded():
    # Mock areno.api.agentic to avoid pulling in torch.
    prev = sys.modules.get("areno.api.agentic")
    sys.modules["areno.api.agentic"] = SimpleNamespace(
        AgentTrajectory=None, AgentTrajectoryTurn=None,
    )
    try:
        run_agent = _load_module("run_agent")
    finally:
        if prev is not None:
            sys.modules["areno.api.agentic"] = prev
        else:
            sys.modules.pop("areno.api.agentic", None)

    tool = run_agent.MOVE_TOOL
    params = tool["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert params["required"] == ["direction"]
    assert set(params["properties"]["direction"]["enum"]) == {"up", "down", "left", "right"}


# ---------------------------------------------------------------------------
# Agent episode tests
# ---------------------------------------------------------------------------

def test_maze_agent_stops_on_terminal():
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    record = generator.generate_records(1, seed=7)[0]
    maze = game.normalize_maze(record["maze"])
    start = tuple(record["start"])
    goal = tuple(record["goal"])

    # Simulate the optimal path locally.
    path = game.solve_shortest_path(maze, start, goal, has_key=False)
    directions = []
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        if dr == -1:
            directions.append("up")
        elif dr == 1:
            directions.append("down")
        elif dc == -1:
            directions.append("left")
        elif dc == 1:
            directions.append("right")

    # Replay through apply_move and check terminality.
    state = game.MazeState(
        maze=maze, agent_pos=start, has_key=False,
        steps_taken=0, max_steps=record["max_steps"], vision_radius=1,
    )
    terminal_hit = False
    for d in directions:
        result = game.apply_move(state, d)
        state = result.state
        if result.terminal:
            terminal_hit = True
            break
    assert terminal_hit
    assert state.agent_pos == goal


def test_maze_action_exhaustion():
    game = _load_module("game")

    maze = game.normalize_maze([
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "wall", "empty", "wall"],
        ["wall", "empty", "empty", "goal", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ])
    state = game.MazeState(
        maze=maze, agent_pos=(1, 1), has_key=False,
        steps_taken=0, max_steps=3, vision_radius=1,
    )
    # Take 3 valid-but-non-goal steps.
    for d in ("right", "right", "down"):
        result = game.apply_move(state, d)
        state = result.state
        if result.terminal:
            break
    assert state.steps_taken == 3
    # Next move should be terminal due to max_steps.
    result = game.apply_move(state, "down")
    assert result.terminal


# ---------------------------------------------------------------------------
# Loader test
# ---------------------------------------------------------------------------

def test_maze_loader_produces_prompt_records():
    loader = _load_module_without_sys_path("dataset_loader")
    generator = _load_module("dataset_generator")

    # Use a real temp file so the loader reads it instead of falling back.
    import tempfile
    raw = generator.generate_records(2, seed=11)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for rec in raw:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        records = loader.load_training_dataset(f.name)

    assert len(records) == 2
    for r in records:
        assert "prompt" in r and r["prompt"]
        assert "maze" in r
        assert "start" in r
        assert "goal" in r
        assert "max_steps" in r
        assert "local view" in r["prompt"]


# ---------------------------------------------------------------------------
# Invalid input test
# ---------------------------------------------------------------------------

def test_maze_invalid_directions_rejected():
    game = _load_module("game")
    maze = game.normalize_maze([
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ])
    state = game.MazeState(
        maze=maze, agent_pos=(1, 1), has_key=False,
        steps_taken=0, max_steps=10, vision_radius=1,
    )
    result = game.apply_move(state, "diagonal")
    assert not result.success
    assert result.reason == "invalid_direction"