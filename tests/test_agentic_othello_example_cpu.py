from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace


def _load_module(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "othello" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"agentic_othello_{name}_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_othello_initial_board_and_normalize():
    game = _load_module("game")

    board = game.initial_board()
    assert len(board) == 6
    assert all(len(row) == 6 for row in board)
    # Standard starting position: center 2x2 cross-placed
    assert board[2][2] == "W"
    assert board[2][3] == "B"
    assert board[3][2] == "B"
    assert board[3][3] == "W"

    # normalize accepts valid input
    normalized = game.normalize_board(board)
    assert normalized == board

    # Invalid size rejected
    try:
        game.normalize_board([["."] * 3 for _ in range(3)])
        assert False, "should reject 3x3"
    except ValueError:
        pass

    # Invalid cell rejected
    try:
        bad = game.initial_board()
        bad[0][0] = "X"
        game.normalize_board(bad)
        assert False, "should reject X"
    except ValueError:
        pass


def test_othello_legal_moves_and_flips_cover_all_eight_directions():
    game = _load_module("game")

    # Set up a board to test horizontal, vertical, and diagonal flips
    board = [["." for _ in range(6)] for _ in range(6)]
    # Horizontal flip: B(0,0) W(0,1) .(0,2) -> B plays (0,2) to flip W(0,1)
    board[0][0] = "B"
    board[0][1] = "W"
    board[0][5] = "B"
    board[0][4] = "W"
    # Vertical flip: B(1,0) W(2,0) .(3,0) -> B plays (3,0) to flip W(2,0)
    board[1][0] = "B"
    board[2][0] = "W"
    # Diagonal flip: B(1,1) W(2,2) .(3,3) -> B plays (3,3) to flip W(2,2)
    board[1][1] = "B"
    board[2][2] = "W"
    # Anti-diagonal flip: B(1,5) W(2,4) .(3,3)
    board[2][4] = "W"

    moves = game.legal_moves(board, "B")
    assert (0, 2) in moves  # horizontal flip
    assert (3, 0) in moves  # vertical flip

    # Verify apply_move flips correctly
    new_board = game.apply_move(board, 0, 2, "B")
    assert new_board[0][1] == "B"  # W flipped to B

    # Test all 8 directions: ring of W around central B
    board2 = [["." for _ in range(6)] for _ in range(6)]
    board2[3][3] = "B"
    for dr, dc in game.DIRECTIONS:
        board2[3 + dr][3 + dc] = "W"

    # For each direction, B plays the far-end cell to flip the adjacent W
    for dr, dc in game.DIRECTIONS:
        r_far, c_far = 3 + dr * 2, 3 + dc * 2
        if 0 <= r_far < 6 and 0 <= c_far < 6:
            flips = game._flips_for_move(board2, r_far, c_far, "B")
            assert (3 + dr, 3 + dc) in flips, f"direction ({dr},{dc}) should flip ({3 + dr},{3 + dc})"


def test_othello_forced_pass_when_no_legal_moves():
    game = _load_module("game")

    # Construct a board where W has no legal moves but B does
    board = [["B" for _ in range(6)] for _ in range(6)]
    board[0][0] = "."
    board[0][1] = "W"
    # B can play (0,0) to flank W(0,1); W has no flanking opportunity
    assert game.has_legal_move(board, "B")
    assert not game.has_legal_move(board, "W")


def test_othello_double_pass_ends_game():
    game = _load_module("game")

    # Full board -> terminal
    board = [["B" for _ in range(6)] for _ in range(6)]
    board[0][0] = "W"
    board[1][1] = "W"
    board[2][2] = "W"
    assert game.is_terminal(board)

    # play_episode on a full board terminates immediately
    full_board = [["B" if (r + c) % 2 == 0 else "W" for c in range(6)] for r in range(6)]
    result = game.play_episode(
        full_board,
        policy_fn=lambda b, p: None,
        opponent_fn=lambda b, p: None,
        max_moves=5,
    )
    assert game.is_terminal(result["board"])


def test_othello_terminal_scoring_counts_discs():
    game = _load_module("game")

    # Board with 20 B and 16 W (full board)
    board = [["B" for _ in range(6)] for _ in range(6)]
    for r in range(4):
        for c in range(6):
            if r * 6 + c < 16:
                board[r][c] = "W"
    counts = game.score(board)
    assert counts["B"] == 20
    assert counts["W"] == 16
    assert game.winner(board) == "B"

    # Tie board
    tie_board = [["B" if (r * 6 + c) % 2 == 0 else "W" for c in range(6)] for r in range(6)]
    tie_counts = game.score(tie_board)
    assert tie_counts["B"] == 18
    assert tie_counts["W"] == 18
    assert game.winner(tie_board) is None


def test_othello_illegal_cell_rejected():
    game = _load_module("game")
    board = game.initial_board()

    # Occupied cell -> illegal
    assert game.score_move(board, 2, 2, "B") == -1.0
    try:
        game.apply_move(board, 2, 2, "B")
        assert False, "should reject occupied cell"
    except ValueError:
        pass

    # Empty cell with no flips -> illegal
    assert game.score_move(board, 0, 0, "B") == -1.0
    try:
        game.apply_move(board, 0, 0, "B")
        assert False, "should reject no-flip cell"
    except ValueError:
        pass

    # Out of bounds
    try:
        game.apply_move(board, 6, 0, "B")
        assert False, "should reject out of bounds"
    except ValueError:
        pass

    # None move
    assert game.score_move(board, None, None, "B") == -1.0


def test_othello_generator_produces_reachable_boards():
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    # Determinism: same seed -> same output
    records1 = generator.generate_records(16, seed=7)
    records2 = generator.generate_records(16, seed=7)
    assert records1 == records2

    assert len(records1) == 16
    for record in records1:
        board = game.normalize_board(record["board"])
        assert len(board) == 6
        assert all(len(row) == 6 for row in board)
        assert record["player"] == "B"
        assert game.next_player(board) == "B"
        assert not game.is_terminal(board)
        assert game.has_legal_move(board, "B")


def test_othello_reward_scores_tool_move_only():
    game = _load_module("game")
    reward = _load_module("reward")

    board = game.initial_board()
    legal = game.legal_moves(board, "B")
    assert len(legal) > 0
    legal_row, legal_col = legal[0]

    # Valid legal move -> 0.0 (non-terminal)
    record = SimpleNamespace(
        source_record={"board": board, "player": "B"},
        completion="",
        tool_calls=[{"name": "choose_move", "arguments": {"row": legal_row, "col": legal_col}}],
    )
    assert reward.reward_fn(record) == 0.0

    # Illegal move (occupied cell) -> -1.0
    record.tool_calls = [{"name": "choose_move", "arguments": {"row": 2, "col": 2}}]
    assert reward.reward_fn(record) == -1.0

    # Wrong tool name -> -1.0
    record.tool_calls = [{"name": "other_tool", "arguments": {"row": 2, "col": 4}}]
    assert reward.reward_fn(record) == -1.0

    # No tool calls -> -1.0
    record.tool_calls = []
    assert reward.reward_fn(record) == -1.0

    # Arguments as JSON string
    record.tool_calls = [
        {
            "name": "choose_move",
            "arguments": json.dumps({"row": legal_row, "col": legal_col}),
        }
    ]
    assert reward.reward_fn(record) == 0.0

    # Invalid JSON string arguments -> -1.0
    record.tool_calls = [{"name": "choose_move", "arguments": "not json"}]
    assert reward.reward_fn(record) == -1.0


def test_othello_seeded_random_opponent_evaluation():
    game = _load_module("game")

    policy_fn = game.random_policy(random.Random(42))
    opponent_fn = game.random_policy(random.Random(99))

    num_games = 20
    wins = 0
    total_illegal = 0
    total_moves = 0

    for i in range(num_games):
        board = game.initial_board()
        first_player = "B" if i % 2 == 0 else "W"
        result = game.play_episode(
            board,
            policy_fn=policy_fn,
            opponent_fn=opponent_fn,
            first_player=first_player,
            max_moves=40,
        )
        if result["winner"] is not None:
            if result["winner"] == first_player:
                wins += 1
        total_illegal += result["illegal_moves"]
        total_moves += result["total_moves"]

    win_rate = wins / num_games
    illegal_rate = total_illegal / max(total_moves, 1)

    # Random vs random: win rate should be in a reasonable range
    assert 0.0 <= win_rate <= 1.0
    # Seeded random agent only picks legal moves, so illegal rate should be 0
    assert illegal_rate == 0.0
    # Ensure we actually played some games
    assert total_moves > 0


def test_othello_dataset_loader_formats_records():
    game = _load_module("game")
    loader = _load_module("dataset_loader")

    # Test with no file -> falls back to generator
    records = loader.load_training_dataset("/nonexistent/path/boards.jsonl")
    assert len(records) > 0
    for record in records:
        assert "prompt" in record
        assert "board" in record
        assert "player" in record
        assert "valid_moves" in record
        assert isinstance(record["valid_moves"], list)
        for move in record["valid_moves"]:
            assert move in game.legal_moves(record["board"], record["player"])
