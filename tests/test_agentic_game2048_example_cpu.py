"""CPU tests for the agentic 2048 example.

Every test runs without GPU, torch, or a live model server.  The ``run_agent``
module is loaded with a stub ``areno.api.agentic`` so that its import does not
pull in the full training stack.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "game2048"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    previous_agentic = sys.modules.get("areno.api.agentic")
    if name == "run_agent":
        sys.modules["areno.api.agentic"] = SimpleNamespace(
            AgentTrajectory=type("AgentTrajectory", (), {}),
            AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
        )
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_game2048_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game
        if name == "run_agent":
            sys.modules.pop("areno.api.agentic", None)
            if previous_agentic is not None:
                sys.modules["areno.api.agentic"] = previous_agentic


# ------------------------------------------------------------------
# 1. Merge edge cases
# ------------------------------------------------------------------


def test_move_merge_edge_cases():
    game = _load_module("game")
    rng = random.Random(0)

    # [2,2,2,2] → [4,4,0,0] score=8
    board = [[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    result, score, valid, _ = game.move(board, "LEFT", rng)
    assert result[0] == [4, 4, 0, 0]
    assert score == 8
    assert valid is True

    # [4,4,2,2] → [8,4,0,0] score=8  (4+4=8, then 2+2=4 separate merge)
    board = [[4, 4, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    result, score, valid, _ = game.move(board, "LEFT", rng)
    assert result[0] == [8, 4, 0, 0]
    assert score == 12

    # No chain merge: [2,2,4] → [4,4,0] (the merged 4 does NOT merge with existing 4)
    board = [[2, 2, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    result, score, valid, _ = game.move(board, "LEFT", rng)
    assert result[0] == [4, 4, 0, 0]
    assert score == 4


# ------------------------------------------------------------------
# 2. Seeded replay determinism
# ------------------------------------------------------------------


def test_seeded_replay_deterministic():
    game = _load_module("game")

    def run_episode(seed):
        rng = random.Random(seed)
        board = game.new_board(seed)
        boards = [game.normalize_board(board)]
        for _ in range(10):
            direction = game.random_action(board, rng)
            board, _, valid, terminal = game.move(board, direction, rng)
            if terminal:
                break
            boards.append(game.normalize_board(board))
        return boards

    seq1 = run_episode(99)
    seq2 = run_episode(99)
    assert seq1 == seq2
    assert len(seq1) >= 2


# ------------------------------------------------------------------
# 3. Invalid move does not change board
# ------------------------------------------------------------------


def test_invalid_move_no_change():
    game = _load_module("game")
    rng = random.Random(0)

    # All tiles already on the left, no merges possible
    board = [[2, 0, 0, 0], [4, 0, 0, 0], [2, 0, 0, 0], [8, 0, 0, 0]]
    result, score, valid, _ = game.move(board, "LEFT", rng)
    assert valid is False
    assert score == 0
    assert result == board


# ------------------------------------------------------------------
# 4. Terminal detection
# ------------------------------------------------------------------


def test_terminal_detection():
    game = _load_module("game")

    # Full board, no adjacent equals
    full = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    assert game.is_terminal(full) is True

    # Full board but has adjacent equals
    mergeable = [[2, 2, 4, 8], [4, 8, 16, 2], [2, 4, 8, 16], [4, 2, 4, 8]]
    assert game.is_terminal(mergeable) is False

    # Has empty cell
    sparse = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 0], [4, 2, 4, 2]]
    assert game.is_terminal(sparse) is False


# ------------------------------------------------------------------
# 5. Episode length cap in _run_episode
# ------------------------------------------------------------------


def test_episode_length_cap():
    run_agent = _load_module("run_agent")

    class FakeCompletions:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.messages = []

        async def create(self, **kwargs):
            self.messages.append(kwargs["messages"])
            return next(self.responses)

    def response(direction):
        message = SimpleNamespace(content=direction, tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    item = SimpleNamespace(
        prompt="play 2048",
        record={"board": [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], "seed": 1, "max_moves": 3},
    )
    responses = [response("LEFT"), response("UP"), response("RIGHT")]
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    turns = asyncio.run(run_agent._run_episode(item, client))

    assert len(turns) == 3


# ------------------------------------------------------------------
# 6. Random action baseline produces complete metrics
# ------------------------------------------------------------------


def test_random_action_baseline():
    game = _load_module("game")
    rng = random.Random(42)
    board = game.new_board(42)
    total_score = 0

    for _ in range(game.DEFAULT_MAX_MOVES):
        direction = game.random_action(board, rng)
        board, score, valid, terminal = game.move(board, direction, rng)
        if valid:
            total_score += score
        if terminal:
            break

    assert total_score >= 0
    assert game.max_tile(board) >= 2


# ------------------------------------------------------------------
# 7. score_episode returns all required fields
# ------------------------------------------------------------------


def test_episode_metrics_fields():
    game = _load_module("game")
    metrics = game.score_episode(
        total_merge_score=500,
        max_tile_value=256,
        valid_moves=20,
        invalid_moves=5,
    )
    assert set(metrics.keys()) == {"reward", "total_score", "max_tile", "invalid_rate", "valid_moves", "invalid_moves"}
    assert metrics["total_score"] == 500
    assert metrics["max_tile"] == 256
    assert metrics["invalid_rate"] == 5 / 25
    assert metrics["valid_moves"] == 20
    assert metrics["invalid_moves"] == 5


# ------------------------------------------------------------------
# 8. Reward: valid moves outscore invalid moves
# ------------------------------------------------------------------


def test_reward_scores_valid_vs_invalid():
    reward = _load_module("reward")
    game = _load_module("game")
    board = game.new_board(10)

    # All valid distinct directions
    valid_record = SimpleNamespace(
        source_record={"board": board, "seed": 10, "max_moves": 50},
        completion="MOVE: LEFT\nMOVE: DOWN\nMOVE: RIGHT",
    )
    valid_reward = reward.reward_fn(valid_record)

    # All same direction (likely invalid after first)
    invalid_record = SimpleNamespace(
        source_record={"board": board, "seed": 10, "max_moves": 50},
        completion="MOVE: LEFT\nMOVE: LEFT\nMOVE: LEFT",
    )
    invalid_reward = reward.reward_fn(invalid_record)

    # Pure outcome reward: valid moves should accumulate more score
    assert valid_reward >= invalid_reward
    assert isinstance(valid_reward, float)
    assert isinstance(invalid_reward, float)


# ------------------------------------------------------------------
# 9. Malformed direction rejected by parse_action
# ------------------------------------------------------------------


def test_malformed_direction_rejected():
    game = _load_module("game")

    assert game.parse_action("UP") == "UP"
    assert game.parse_action("move DOWN please") == "DOWN"
    assert game.parse_action("no direction here") is None
    assert game.parse_action("") is None
    assert game.parse_action("SOUTH") == "DOWN"
    assert game.parse_action("move WEST") == "LEFT"
    assert game.parse_action("EAST is best") == "RIGHT"
    assert game.parse_action("NORTH") == "UP"


# ------------------------------------------------------------------
# 10. Generator is reproducible
# ------------------------------------------------------------------


def test_generator_reproducible():
    generator = _load_module("dataset_generator")

    records1 = generator.generate_records(16, seed=7)
    records2 = generator.generate_records(16, seed=7)

    assert len(records1) == 16
    assert records1 == records2
    for record in records1:
        assert "id" in record
        assert "seed" in record
        assert "board" in record


# ------------------------------------------------------------------
# 11. Loader normalizes records
# ------------------------------------------------------------------


def test_loader_normalizes_records():
    loader = _load_module("dataset_loader")
    game = _load_module("game")

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        board = game.new_board(1)
        f.write(json.dumps({"id": "test-001", "seed": 1, "board": board}) + "\n")
        f.flush()
        path = f.name

    records = loader.load_training_dataset(path)
    assert len(records) == 1
    record = records[0]
    assert record["id"] == "test-001"
    assert record["seed"] == 1
    assert record["board"] == board
    assert "prompt" in record
    assert "max_moves" in record
    assert record["max_moves"] == game.DEFAULT_MAX_MOVES

    Path(path).unlink()


# ------------------------------------------------------------------
# 12. Agent episode with fake client preserves tool order
# ------------------------------------------------------------------


def test_agent_episode_with_fake_client():
    run_agent = _load_module("run_agent")

    class FakeCompletions:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.messages = []

        async def create(self, **kwargs):
            self.messages.append(kwargs["messages"])
            return next(self.responses)

    def response(direction):
        message = SimpleNamespace(content=f"I should merge tiles.\nMOVE: {direction}", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    item = SimpleNamespace(
        prompt="play 2048",
        record={"board": [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], "seed": 5, "max_moves": 2},
    )
    completions = FakeCompletions([response("LEFT"), response("UP")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    turns = asyncio.run(run_agent._run_episode(item, client))

    assert len(turns) >= 1
    first_messages = completions.messages[0]
    assert first_messages[0]["role"] == "system"
    assert first_messages[1]["role"] == "user"

    # Single-turn: each step has independent messages (system + user only, no history)
    for step_messages in completions.messages:
        assert len(step_messages) == 2
        assert step_messages[0]["role"] == "system"
        assert step_messages[1]["role"] == "user"

    # No direction parsed → episode still runs for max_moves
    no_dir_completions = FakeCompletions([
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="I give up", tool_calls=[]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="I give up", tool_calls=[]))]),
    ])
    no_dir_turns = asyncio.run(
        run_agent._run_episode(item, SimpleNamespace(chat=SimpleNamespace(completions=no_dir_completions)))
    )
    assert len(no_dir_turns) == 2  # max_moves=2, fallback to random for each


# ------------------------------------------------------------------
# 13. System prompt and parse_action are well-formed
# ------------------------------------------------------------------


def test_tool_schema_is_closed_and_bounded():
    game = _load_module("game")

    assert "UP" in game.SYSTEM_PROMPT
    assert "DOWN" in game.SYSTEM_PROMPT
    assert "LEFT" in game.SYSTEM_PROMPT
    assert "RIGHT" in game.SYSTEM_PROMPT
    assert "one word" in game.SYSTEM_PROMPT

    assert game.parse_action("LEFT") == "LEFT"
    assert game.parse_action("I think UP is best") == "UP"
    assert game.parse_action("no direction here") is None
    assert not hasattr(game, "MOVE_TOOL")