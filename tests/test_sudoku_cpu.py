"""CPU tests for the Sudoku agentic example.

Run from the repository root:

    pytest tests/test_sudoku_cpu.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "agentic" / "sudoku"))
import game  # noqa: E402


# --- Board validation ------------------------------------------------------

def test_normalize_board_valid():
    board = [[0] * 9 for _ in range(9)]
    result = game.normalize_board(board)
    assert len(result) == 9
    assert all(len(row) == 9 for row in result)


def test_normalize_board_wrong_size():
    with pytest.raises(ValueError, match="9x9"):
        game.normalize_board([[1, 2, 3]])


def test_normalize_board_invalid_cell():
    board = [[0] * 9 for _ in range(9)]
    board[0][0] = 10
    with pytest.raises(ValueError, match="0 or 1-9"):
        game.normalize_board(board)


# --- Constraint validation -------------------------------------------------

def test_validate_placement_valid():
    board = [[0] * 9 for _ in range(9)]
    result = game.validate_placement(board, 0, 0, 5)
    assert result["valid"] is True


def test_validate_placement_row_conflict():
    board = [[5, 0, 0, 0, 0, 0, 0, 0, 0]] + [[0] * 9 for _ in range(8)]
    result = game.validate_placement(board, 0, 1, 5)
    assert result["valid"] is False
    assert "row" in result["error"]


def test_validate_placement_col_conflict():
    board = [[5] + [0] * 8] + [[0] * 9 for _ in range(8)]
    result = game.validate_placement(board, 1, 0, 5)
    assert result["valid"] is False
    assert "column" in result["error"]


def test_validate_placement_box_conflict():
    board = [[5, 0, 0, 0, 0, 0, 0, 0, 0]] + [[0] * 9 for _ in range(8)]
    result = game.validate_placement(board, 1, 1, 5)
    assert result["valid"] is False
    assert "box" in result["error"]


def test_validate_placement_filled_cell():
    board = [[5] + [0] * 8] + [[0] * 9 for _ in range(8)]
    result = game.validate_placement(board, 0, 0, 3)
    assert result["valid"] is False
    assert "already filled" in result["error"]


def test_validate_placement_out_of_range():
    board = [[0] * 9 for _ in range(9)]
    result = game.validate_placement(board, 9, 0, 1)
    assert result["valid"] is False
    assert "out of range" in result["error"]


def test_validate_placement_invalid_digit():
    board = [[0] * 9 for _ in range(9)]
    result = game.validate_placement(board, 0, 0, 0)
    assert result["valid"] is False
    assert "1-9" in result["error"]


# --- Inspect candidates ----------------------------------------------------

def test_inspect_candidates_empty_cell():
    board = [[5, 0, 0, 0, 0, 0, 0, 0, 0]] + [[0] * 9 for _ in range(8)]
    result = game.inspect_candidates(board, 0, 1)
    assert result["valid"] is True
    assert 5 not in result["candidates"]


def test_inspect_candidates_filled_cell():
    board = [[5] + [0] * 8] + [[0] * 9 for _ in range(8)]
    result = game.inspect_candidates(board, 0, 0)
    assert result["valid"] is False


def test_inspect_candidates_out_of_range():
    board = [[0] * 9 for _ in range(9)]
    result = game.inspect_candidates(board, -1, 0)
    assert result["valid"] is False


# --- Episode undo ----------------------------------------------------------

def test_undo_reverts_placement():
    board = [[0] * 9 for _ in range(9)]
    episode = game.SudokuEpisode(board, max_actions=10)
    episode.place(0, 0, 5)
    assert episode.board[0][0] == 5
    result = episode.undo()
    assert result["valid"] is True
    assert episode.board[0][0] == 0


def test_undo_no_history():
    board = [[0] * 9 for _ in range(9)]
    episode = game.SudokuEpisode(board, max_actions=10)
    result = episode.undo()
    assert result["valid"] is False
    assert "no moves" in result["error"]


# --- Episode termination ---------------------------------------------------

def test_action_budget_exhaustion():
    board = [[0] * 9 for _ in range(9)]
    episode = game.SudokuEpisode(board, max_actions=3)
    assert episode.is_done() is False
    episode.place(0, 0, 1)
    episode.place(0, 1, 2)
    episode.place(0, 2, 3)
    assert episode.is_done() is True


def test_is_solved_false_on_empty():
    board = [[0] * 9 for _ in range(9)]
    episode = game.SudokuEpisode(board, max_actions=10)
    assert episode.is_solved() is False


def test_is_done_when_board_full_but_invalid():
    """Board is full but has conflicts — is_done should still return True."""
    # Row 0 has two 1s — invalid but complete.
    board = [
        [1, 1, 2, 3, 4, 5, 6, 7, 8],
        [9, 2, 3, 4, 5, 6, 7, 8, 1],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [2, 3, 4, 5, 6, 7, 8, 9, 1],
        [5, 6, 7, 8, 9, 1, 2, 3, 4],
        [6, 7, 8, 9, 1, 2, 3, 4, 5],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [8, 9, 1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 1, 2],
    ]
    episode = game.SudokuEpisode(board, max_actions=100)
    assert game.is_complete(episode.board) is True
    assert game.is_valid_board(episode.board) is False
    assert episode.is_done() is True
    assert episode.is_solved() is False


# --- Puzzle generation -----------------------------------------------------

def test_generate_puzzle_easy():
    result = game.generate_puzzle("easy", seed=2026)
    assert "puzzle" in result
    assert result["difficulty"] == "easy"
    empty_count = sum(1 for r in range(9) for c in range(9) if result["puzzle"][r][c] == 0)
    assert empty_count >= 30  # easy should have a reasonable number of empty cells


def test_generate_puzzle_medium():
    result = game.generate_puzzle("medium", seed=2026)
    empty_count = sum(1 for r in range(9) for c in range(9) if result["puzzle"][r][c] == 0)
    assert empty_count >= 40


def test_generate_puzzle_hard():
    result = game.generate_puzzle("hard", seed=2026)
    empty_count = sum(1 for r in range(9) for c in range(9) if result["puzzle"][r][c] == 0)
    assert empty_count >= 48


def test_generate_puzzle_invalid_difficulty():
    with pytest.raises(ValueError, match="difficulty"):
        game.generate_puzzle("impossible", seed=2026)


def test_generate_puzzle_unique_solution():
    """The generated puzzle must have exactly one solution."""
    result = game.generate_puzzle("easy", seed=42)
    count = game._count_solutions(result["puzzle"], limit=2)
    assert count == 1


def test_generate_puzzle_reproducible():
    a = game.generate_puzzle("easy", seed=2026)
    b = game.generate_puzzle("easy", seed=2026)
    assert a["puzzle"] == b["puzzle"]


# --- Board completeness ----------------------------------------------------

def test_is_complete_false_empty():
    board = [[0] * 9 for _ in range(9)]
    assert game.is_complete(board) is False


def test_is_complete_true_full():
    board = [[1] * 9 for _ in range(9)]
    assert game.is_complete(board) is True


# --- Scoring ---------------------------------------------------------------

def test_score_episode_solved():
    puzzle = game.generate_puzzle("easy", seed=2026)["puzzle"]
    # Fill the puzzle correctly by generating a solution from the same seed.
    solution = game._fill_board(random.Random(2026))
    actions = []
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                actions.append({"name": "place_digit", "arguments": {"row": r, "col": c, "digit": solution[r][c]}})
    result = game.score_episode(puzzle, actions, max_actions=200)
    assert result["solved"] is True
    assert result["reward"] > 0.8


def test_score_episode_empty_actions():
    puzzle = game.generate_puzzle("easy", seed=2026)["puzzle"]
    result = game.score_episode(puzzle, [], max_actions=120)
    assert result["solved"] is False
    assert result["reward"] == -1.0


def test_score_episode_invalid_action():
    board = [[0] * 9 for _ in range(9)]
    actions = [{"name": "place_digit", "arguments": {"row": 0, "col": 0, "digit": 1}},
               {"name": "place_digit", "arguments": {"row": 0, "col": 1, "digit": 1}}]
    result = game.score_episode(board, actions, max_actions=10)
    assert result["invalid_actions"] == 1
    assert result["valid_actions"] == 1


def test_score_episode_undo():
    board = [[0] * 9 for _ in range(9)]
    actions = [
        {"name": "place_digit", "arguments": {"row": 0, "col": 0, "digit": 5}},
        {"name": "undo", "arguments": {}},
    ]
    result = game.score_episode(board, actions, max_actions=10)
    assert result["undos"] == 1
    assert result["valid_actions"] == 2


def test_score_episode_undo_no_history():
    board = [[0] * 9 for _ in range(9)]
    actions = [{"name": "undo", "arguments": {}}]
    result = game.score_episode(board, actions, max_actions=10)
    assert result["invalid_actions"] == 1


def test_score_episode_partial_fill():
    puzzle = game.generate_puzzle("easy", seed=2026)["puzzle"]
    # Place one legal digit.
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                for d in range(1, 10):
                    if game.validate_placement(puzzle, r, c, d)["valid"]:
                        actions = [{"name": "place_digit", "arguments": {"row": r, "col": c, "digit": d}}]
                        result = game.score_episode(puzzle, actions, max_actions=120)
                        assert 0 < result["reward"] < 0.8
                        return
    pytest.fail("no empty cell found")


def test_score_episode_unknown_tool():
    board = [[0] * 9 for _ in range(9)]
    actions = [{"name": "bad_tool", "arguments": {}}]
    result = game.score_episode(board, actions, max_actions=10)
    assert result["invalid_actions"] == 1


# --- Prompt building -------------------------------------------------------

def test_make_prompt_contains_board():
    puzzle = game.generate_puzzle("easy", seed=2026)["puzzle"]
    record = {"puzzle": puzzle, "difficulty": "easy", "max_actions": 120}
    prompt = game.make_prompt(record)
    assert "Sudoku" in prompt
    assert "." in prompt  # empty cells rendered as .


def test_make_prompt_does_not_leak_solution():
    puzzle = game.generate_puzzle("easy", seed=2026)["puzzle"]
    solution = game._fill_board(random.Random(2026))
    record = {"puzzle": puzzle, "difficulty": "easy", "max_actions": 120}
    prompt = game.make_prompt(record)
    # Verify that every empty cell in the puzzle is rendered as "." in the
    # prompt, not as its solution digit. We check the board rendering section
    # by comparing that the prompt's board matches board_to_text output.
    assert game.board_to_text(puzzle) in prompt


# --- Dataset generator -----------------------------------------------------

def test_dataset_generator_basic():
    import dataset_generator

    records = dataset_generator.generate_records(5, "easy", seed=100)
    assert len(records) == 5
    assert all("puzzle" in r and "difficulty" in r for r in records)
    assert records[0]["difficulty"] == "easy"


def test_dataset_generator_ids_unique():
    import dataset_generator

    records = dataset_generator.generate_records(10, "medium", seed=200)
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids))


# --- Dataset loader --------------------------------------------------------

def test_dataset_loader_loads_jsonl(tmp_path):
    import dataset_generator
    import dataset_loader

    jsonl_path = tmp_path / "puzzles.jsonl"
    records = dataset_generator.generate_records(3, "easy", seed=300)
    with jsonl_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    def mock_default_loader(path):
        import json
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    loaded = dataset_loader.load_training_dataset(str(jsonl_path), default_loader=mock_default_loader)
    assert len(loaded) == 3
    assert all("prompt" in r for r in loaded)
    assert all("puzzle" in r for r in loaded)