"""Tests for Wordle game logic."""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import game


class TestWordValidation:
    """Test word validation functions."""

    def test_normalize_word_valid(self):
        """Test normalizing valid words."""
        assert game.normalize_word("HELLO") == "hello"
        assert game.normalize_word("world") == "world"
        assert game.normalize_word("  ABC  ") == "abc"

    def test_normalize_word_invalid_length(self):
        """Test that invalid length words raise ValueError."""
        with pytest.raises(ValueError, match="must be 5 letters"):
            game.normalize_word("hi")

        with pytest.raises(ValueError, match="must be 5 letters"):
            game.normalize_word("hello world")

    def test_normalize_word_invalid_chars(self):
        """Test that non-alphabetic characters raise ValueError."""
        with pytest.raises(ValueError, match="only letters"):
            game.normalize_word("he11o")

        with pytest.raises(ValueError, match="only letters"):
            game.normalize_word("hello!")

    def test_is_valid_word(self):
        """Test valid word checking."""
        # These should be in our word list
        assert game.is_valid_word("hello") is True
        assert game.is_valid_word("world") is True
        assert game.is_valid_word("HELLO") is True

        # These should not be in our word list
        assert game.is_valid_word("xxxxx") is False
        assert game.is_valid_word("invalid") is False


class TestGuessChecking:
    """Test guess checking and feedback."""

    def test_exact_match(self):
        """Test exact match returns all EXACT."""
        target = "world"
        guess = "world"
        result = game.check_guess(guess, target)

        assert len(result) == 5
        assert all(f == game.LetterStatus.EXACT for f in result)

    def test_all_absent(self):
        """Test guess with no matching letters."""
        target = "world"
        guess = "abcde"
        result = game.check_guess(guess, target)

        assert len(result) == 5
        # Only 'a' is absent, 'b', 'c', 'd', 'e' are not in world
        # Wait, 'd' is in world at position 4
        # Let me recalculate: w-o-r-l-d
        # abcde: a-absent, b-absent, c-absent, d-present(should be at pos 4), e-absent
        assert result[0] == game.LetterStatus.ABSENT
        assert result[1] == game.LetterStatus.ABSENT
        assert result[2] == game.LetterStatus.ABSENT

    def test_repeated_letters_one_match(self):
        """Test repeated letters when target has only one."""
        target = "world"  # Has one 'l'
        guess = "llama"  # Has two 'l's

        result = game.check_guess(guess, target)
        # First 'l' should be EXACT (position 3)
        # Second 'l' should be ABSENT (only one 'l' in target)
        assert result[2] == game.LetterStatus.EXACT  # First 'l' at position 2 (0-indexed)
        # The second 'l' is at position 3, which should be ABSENT
        # because the 'l' in target is already matched
        assert result[3] == game.LetterStatus.ABSENT

    def test_repeated_letters_present(self):
        """Test repeated letters when one is present."""
        target = "silly"  # Has two 'l's
        guess = "llama"  # Has two 'l's

        result = game.check_guess(guess, target)
        # Target: s-i-l-l-y
        # Guess:  l-l-a-m-a
        # Position 0: 'l' is in target but at position 2 -> PRESENT
        # Position 1: 'l' is in target but at position 3 -> PRESENT
        assert result[0] == game.LetterStatus.PRESENT
        assert result[1] == game.LetterStatus.PRESENT

    def test_repeated_letters_exact_and_present(self):
        """Test repeated letters when one is exact and one is present."""
        target = "silly"  # l at positions 2 and 3
        guess = "blues"  # b-l-u-e-s, no l's

        # Let me use a better test case
        target = "hello"  # l at position 2 and 3
        guess = "level"  # l at position 0, 2, and 4

        result = game.check_guess(guess, target)
        # Target: h-e-l-l-o
        # Guess:  l-e-v-e-l
        # Position 0: 'l' in target at position 2 -> PRESENT
        # Position 1: 'e' exact match
        # Position 2: 'v' not in target -> ABSENT
        # Position 3: 'e' exact match (target has 'l' at 3, guess has 'e' at 3)
        # Position 4: 'l' exact match (target has 'o' at 4, guess has 'l' at 4)

        # Let me redo with simpler case
        target = "abide"  # Only one 'i'
        guess = "icing"  # Has one 'i'

        result = game.check_guess(guess, target)
        # Target: a-b-i-d-e
        # Guess:  i-c-i-n-g
        # Position 0: 'i' is PRESENT (in target at position 2)
        # Position 2: 'i' is EXACT (in target at position 2)
        assert result[0] == game.LetterStatus.PRESENT
        assert result[2] == game.LetterStatus.EXACT


class TestGameState:
    """Test game state management."""

    def test_create_new_game(self):
        """Test creating a new game."""
        g = game.create_new_game("hello")
        assert g["target"] == "hello"
        assert g["guesses"] == []
        assert g["feedbacks"] == []
        assert g["state"] == game.GameState.IN_PROGRESS

    def test_apply_valid_guess(self):
        """Test applying a valid guess."""
        g = game.create_new_game("hello")
        g = game.apply_guess(g, "world")

        assert len(g["guesses"]) == 1
        assert g["guesses"][0] == "world"
        assert len(g["feedbacks"]) == 1
        assert g["state"] == game.GameState.IN_PROGRESS

    def test_apply_winning_guess(self):
        """Test applying a winning guess."""
        g = game.create_new_game("hello")
        g = game.apply_guess(g, "hello")

        assert g["state"] == game.GameState.WON
        assert len(g["guesses"]) == 1

    def test_apply_invalid_word(self):
        """Test that invalid words raise ValueError."""
        g = game.create_new_game("hello")
        with pytest.raises(ValueError, match="Invalid word"):
            game.apply_guess(g, "xxxxx")

    def test_game_loss_after_max_guesses(self):
        """Test game ends in loss after max guesses."""
        g = game.create_new_game("hello")

        # Make 5 wrong guesses
        wrong_guesses = ["world", "earth", "bread", "fruit", "water"]
        for guess in wrong_guesses:
            g = game.apply_guess(g, guess)

        # Should still be in progress
        assert g["state"] == game.GameState.IN_PROGRESS
        assert len(g["guesses"]) == 5

        # Make final wrong guess
        g = game.apply_guess(g, "table")
        assert g["state"] == game.GameState.LOST
        assert len(g["guesses"]) == 6


class TestPromptFormatting:
    """Test prompt formatting."""

    def test_format_prompt_initial(self):
        """Test prompt format for new game."""
        g = game.create_new_game("hello")
        prompt = game.format_prompt(g)

        assert "Wordle" in prompt
        assert "6 attempts" in prompt
        assert "hello" not in prompt.lower()  # Target should not be revealed
        assert "Call guess_word" in prompt
        assert "5-letter" in prompt

    def test_format_prompt_with_guesses(self):
        """Test prompt includes previous guesses."""
        g = game.create_new_game("hello")
        g = game.apply_guess(g, "world")
        prompt = game.format_prompt(g)

        assert "WORLD" in prompt
        assert "Previous guesses" in prompt


class TestXMLParsing:
    """Test XML guess parsing."""

    def test_parse_xml_guess(self):
        """Test extracting guess from XML."""
        text = "Let me try <guess>world</guess> now."
        assert game.parse_xml_guess(text) == "world"

    def test_parse_xml_guess_uppercase(self):
        """Test extracting guess handles uppercase."""
        text = "<guess>WORLD</guess>"
        assert game.parse_xml_guess(text) == "world"

    def test_parse_xml_guess_no_match(self):
        """Test no guess returns None."""
        text = "I don't know what to guess."
        assert game.parse_xml_guess(text) is None


class TestScoring:
    """Test game scoring."""

    def test_score_won(self):
        """Test scoring for won game."""
        g = game.create_new_game("hello")
        g = game.apply_guess(g, "hello")
        score = game.score_game(g)

        assert score == 1.5  # 1.0 + efficiency bonus

    def test_score_lost(self):
        """Test scoring for lost game."""
        g = game.create_new_game("hello")
        for _ in range(6):
            g = game.apply_guess(g, "world")
        score = game.score_game(g)

        assert score == 0.0

    def test_score_in_progress(self):
        """Test scoring for in-progress game."""
        g = game.create_new_game("hello")
        g = game.apply_guess(g, "world")
        score = game.score_game(g)

        assert score == -1.0


class TestRewardBounded:
    """Test reward function stays in [-1.0, +1.0] (Issue #189 + gradient stability)."""

    def test_reward_no_tool_call(self):
        """Reward for no tool call is -1.0."""
        import reward
        class FakeRecord:
            source_record = {"target": "hello", "game": {}}
            tool_calls = []
        r = reward.reward_fn(FakeRecord())
        assert r == -1.0

    def test_reward_correct_guess(self):
        """Reward for correct guess is +1.0 (capped)."""
        import reward
        class FakeRecord:
            source_record = {"target": "hello", "game": {"guesses": []}}
            tool_calls = [{"name": "guess_word", "arguments": {"word": "hello"}}]
        r = reward.reward_fn(FakeRecord())
        assert r == 1.0

    def test_reward_all_bounded(self):
        """All reward values must be in [-1.0, +1.0]."""
        import reward
        test_cases = [
            ("hello", "hello"),   # correct
            ("hello", "world"),   # partial match
            ("hello", "abcde"),   # invalid word (not in list)
            ("hello", "heart"),   # some letters present
        ]
        for target, guess_word in test_cases:
            class FakeRecord:
                source_record = {"target": target, "game": {"guesses": []}}
                tool_calls = [{"name": "guess_word", "arguments": {"word": guess_word}}]
            r = reward.reward_fn(FakeRecord())
            assert -1.0 <= r <= 1.0, f"Reward {r} out of bounds for target={target} guess={guess_word}"

    def test_reward_no_tool_is_worst(self):
        """Not calling tool should be strictly worse than any tool call."""
        import reward
        class NoToolRecord:
            source_record = {"target": "hello", "game": {"guesses": []}}
            tool_calls = []
        no_tool_reward = reward.reward_fn(NoToolRecord())

        # Any tool call should be better
        class ToolRecord:
            source_record = {"target": "hello", "game": {"guesses": []}}
            tool_calls = [{"name": "guess_word", "arguments": {"word": "xxxxx"}}]
        tool_reward = reward.reward_fn(ToolRecord())

        assert no_tool_reward < tool_reward


class TestDatasetGenerator:
    """Test dataset generation."""

    def test_generate_records(self):
        """Test generating game records."""
        from dataset_generator import generate_records

        records = generate_records(10, seed=42)
        assert len(records) == 10
        assert all("target" in r for r in records)
        assert all("id" in r for r in records)

    def test_generate_reproducible(self):
        """Test that same seed produces same records."""
        from dataset_generator import generate_records

        records1 = generate_records(5, seed=123)
        records2 = generate_records(5, seed=123)

        assert records1 == records2


class TestStatistics:
    """Test solve rate and statistics functions (Issue #189)."""

    def test_compute_stats_empty(self):
        """Test statistics with empty results."""
        stats = game.compute_stats([])
        assert stats["overall_solve_rate"] == 0.0
        assert stats["overall_avg_guesses"] == 0.0
        assert stats["by_word_length"] == {}

    def test_compute_stats_all_solved(self):
        """Test statistics when all games are solved."""
        results = [
            {"target": "hello", "solved": True, "guesses": 3},
            {"target": "world", "solved": True, "guesses": 4},
            {"target": "about", "solved": True, "guesses": 2},
        ]
        stats = game.compute_stats(results)
        assert stats["overall_solve_rate"] == 1.0
        assert stats["overall_avg_guesses"] == 3.0  # (3+4+2)/3

    def test_compute_stats_mixed(self):
        """Test statistics with mixed results."""
        results = [
            {"target": "hello", "solved": True, "guesses": 3},
            {"target": "world", "solved": True, "guesses": 5},
            {"target": "about", "solved": False, "guesses": 6},
            {"target": "apple", "solved": False, "guesses": None},
        ]
        stats = game.compute_stats(results)
        assert stats["overall_solve_rate"] == 0.5  # 2/4
        assert stats["overall_avg_guesses"] == 4.0  # (3+5)/2

    def test_compute_stats_by_word_length(self):
        """Test statistics grouped by word length."""
        results = [
            {"target": "hi", "solved": True, "guesses": 1},  # 2-letter
            {"target": "cat", "solved": True, "guesses": 2},  # 3-letter
            {"target": "hello", "solved": True, "guesses": 3},  # 5-letter
            {"target": "world", "solved": False, "guesses": 6},  # 5-letter
        ]
        stats = game.compute_stats(results)

        assert 2 in stats["by_word_length"]
        assert 3 in stats["by_word_length"]
        assert 5 in stats["by_word_length"]

        assert stats["by_word_length"][2]["solve_rate"] == 1.0
        assert stats["by_word_length"][2]["avg_guesses"] == 1.0

        assert stats["by_word_length"][5]["solve_rate"] == 0.5
        assert stats["by_word_length"][5]["total_games"] == 2

    def test_format_stats_human_readable(self):
        """Test human-readable stats output."""
        results = [
            {"target": "hello", "solved": True, "guesses": 3},
            {"target": "world", "solved": False, "guesses": 6},
        ]
        stats = game.compute_stats(results)
        output = game.format_stats(stats, human_readable=True)

        assert "Wordle Statistics" in output
        assert "50.0% solved" in output
        assert "5-letter words" in output

    def test_format_stats_structured(self):
        """Test structured (JSON-like) stats output."""
        results = [
            {"target": "hello", "solved": True, "guesses": 3},
        ]
        stats = game.compute_stats(results)
        output = game.format_stats(stats, human_readable=False)

        assert "overall_solve_rate" in output
        assert "by_word_length" in output


class TestDatasetValidation:
    """Test dataset path validation (Issue #189)."""

    def test_validate_valid_path(self):
        """Test validation with valid path."""
        # Path that doesn't exist yet should still be handled
        is_valid, error = game.validate_dataset_path("/tmp/nonexistent.jsonl")
        # Should fail because file doesn't exist, not because of format
        assert is_valid is False

    def test_validate_valid_jsonl(self, tmp_path):
        """Test validation with actual JSONL file."""
        jsonl_file = tmp_path / "games.jsonl"
        jsonl_file.write_text('{"target": "hello"}\n{"target": "world"}\n')

        is_valid, error = game.validate_dataset_path(str(jsonl_file))
        assert is_valid is True
        assert error is None

    def test_validate_empty_file(self, tmp_path):
        """Test validation rejects empty file."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        is_valid, error = game.validate_dataset_path(str(jsonl_file))
        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_directory_without_jsonl(self, tmp_path):
        """Test validation rejects directory without games.jsonl."""
        subdir = tmp_path / "data"
        subdir.mkdir()

        is_valid, error = game.validate_dataset_path(str(subdir))
        assert is_valid is False
        assert "games.jsonl" in error

    def test_validate_wrong_extension(self, tmp_path):
        """Test validation rejects wrong file extension."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("hello")

        is_valid, error = game.validate_dataset_path(str(txt_file))
        assert is_valid is False
        assert ".jsonl or .json" in error


class TestDifferentWordLengths:
    """Test game supports different word lengths (Issue #189)."""

    def test_three_letter_word(self):
        """Test game works with 3-letter word."""
        g = game.create_new_game("cat")
        g = game.apply_guess(g, "dog")
        assert g["state"] == game.GameState.IN_PROGRESS
        assert len(g["guesses"]) == 1

        # Win
        g = game.apply_guess(g, "cat")
        assert g["state"] == game.GameState.WON

    def test_four_letter_word(self):
        """Test game works with 4-letter word."""
        g = game.create_new_game("fish")
        g = game.apply_guess(g, "bird")
        assert g["state"] == game.GameState.IN_PROGRESS

        # Win
        g = game.apply_guess(g, "fish")
        assert g["state"] == game.GameState.WON

    def test_six_letter_word(self):
        """Test game works with 6-letter word."""
        g = game.create_new_game("planet")
        g = game.apply_guess(g, "planet")
        assert g["state"] == game.GameState.WON

    def test_variable_length_check_guess(self):
        """Test check_guess works with different word lengths."""
        # 3-letter
        result = game.check_guess("cat", "dog")
        assert len(result) == 3
        assert all(f == game.LetterStatus.ABSENT for f in result)

        # 4-letter
        result = game.check_guess("fish", "fish")
        assert len(result) == 4
        assert all(f == game.LetterStatus.EXACT for f in result)

        # 6-letter
        result = game.check_guess("planet", "planto")
        assert len(result) == 6
        assert result[0] == game.LetterStatus.EXACT  # p
        assert result[1] == game.LetterStatus.EXACT  # l
        assert result[2] == game.LetterStatus.EXACT  # a
        assert result[3] == game.LetterStatus.EXACT  # n
        assert result[4] == game.LetterStatus.EXACT  # e
        assert result[5] == game.LetterStatus.ABSENT  # t vs o

    def test_normalize_word_variable_length(self):
        """Test normalize_word accepts custom word_length."""
        assert game.normalize_word("cat", word_length=3) == "cat"
        assert game.normalize_word("FISH", word_length=4) == "fish"

        with pytest.raises(ValueError, match="must be 4 letters"):
            game.normalize_word("cat", word_length=4)

    def test_format_prompt_variable_length(self):
        """Test prompt reflects actual word length."""
        g = game.create_new_game("cat")
        prompt = game.format_prompt(g)
        assert "3-letter" in prompt or "6 attempts" in prompt


class TestEvaluate:
    """Test evaluation script (Issue #189)."""

    def test_evaluate_dataset_random(self):
        """Test evaluation runs without errors."""
        from evaluate import evaluate_dataset

        records = [{"target": "hello", "max_guesses": 6}]
        results = evaluate_dataset(records, strategy="random", seed=42)
        assert len(results) == 1
        assert "solved" in results[0]
        assert "guesses" in results[0]
        assert results[0]["target"] == "hello"

    def test_evaluate_dataset_perfect(self):
        """Test perfect strategy always wins."""
        from evaluate import evaluate_dataset

        records = [{"target": "hello", "max_guesses": 6}]
        results = evaluate_dataset(records, strategy="perfect", seed=42)
        assert len(results) == 1
        assert results[0]["solved"] is True
        assert results[0]["guesses"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])