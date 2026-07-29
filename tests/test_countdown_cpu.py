"""CPU tests for the Countdown agentic RL demo.

These tests cover game logic, scoring, random baseline, trace replay,
evaluation metrics, malformed input, boundary values, and fixtures.
All tests run on CPU without GPU, torch, or network access.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "agentic" / "countdown"))
import game  # noqa: E402


class CalculateTest(unittest.TestCase):
    """Tests for basic arithmetic."""

    def test_addition(self):
        self.assertEqual(game.calculate(3, 5, "+"), 8)

    def test_subtraction(self):
        self.assertEqual(game.calculate(10, 4, "-"), 6)

    def test_multiplication(self):
        self.assertEqual(game.calculate(6, 7, "*"), 42)

    def test_integer_division(self):
        self.assertEqual(game.calculate(20, 5, "/"), 4)

    def test_non_integer_division_returns_none(self):
        self.assertIsNone(game.calculate(10, 3, "/"))

    def test_division_by_zero_returns_none(self):
        self.assertIsNone(game.calculate(5, 0, "/"))

    def test_invalid_operator_returns_none(self):
        self.assertIsNone(game.calculate(1, 2, "%"))


class ScoreMoveTest(unittest.TestCase):
    """Tests for the scoring function."""

    def test_exact_match_scores_one(self):
        self.assertEqual(game.score_move([25, 10], 250, 25, 10, "*"), 1.0)

    def test_close_match_scores_proportional(self):
        score = game.score_move([25, 10], 525, 25, 10, "*")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.8)

    def test_invalid_number_returns_negative(self):
        self.assertEqual(game.score_move([1, 5], 10, 100, 5, "*"), -1.0)

    def test_duplicate_number_returns_negative(self):
        self.assertEqual(game.score_move([1, 5, 10], 20, 5, 5, "+"), -1.0)

    def test_duplicate_number_allowed_if_present_twice(self):
        self.assertEqual(game.score_move([5, 5, 10], 10, 5, 5, "+"), 1.0)

    def test_none_arguments_return_negative(self):
        self.assertEqual(game.score_move([1, 5], 10, None, 5, "+"), -1.0)

    def test_result_zero_with_nonzero_target(self):
        """5 - 5 = 0, target 10: legal move, proximity = max(0, 1 - 10/10) = 0."""
        score = game.score_move([5, 5], 10, 5, 5, "-")
        self.assertEqual(score, 0.0)


class NormalizeNumbersTest(unittest.TestCase):
    """Tests for input validation."""

    def test_valid_numbers(self):
        self.assertEqual(game.normalize_numbers([1, 5, 10]), [1, 5, 10])

    def test_single_number_rejected(self):
        with self.assertRaises(ValueError):
            game.normalize_numbers([5])

    def test_zero_rejected(self):
        with self.assertRaises(ValueError):
            game.normalize_numbers([0, 5])

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            game.normalize_numbers([-1, 5])


class RandomBaselineTest(unittest.TestCase):
    """Tests for random-policy baseline."""

    def test_returns_valid_move(self):
        a, b, op = game.random_baseline([1, 5, 10, 25], 250, seed=42)
        self.assertIn(a, [1, 5, 10, 25])
        self.assertIn(b, [1, 5, 10, 25])
        self.assertNotEqual(a, b)
        self.assertIn(op, game.OPERATIONS)

    def test_deterministic_with_seed(self):
        m1 = game.random_baseline([1, 5, 10, 25], 250, seed=42)
        m2 = game.random_baseline([1, 5, 10, 25], 250, seed=42)
        self.assertEqual(m1, m2)

    def test_baseline_score_is_float(self):
        score = game.random_baseline_score([1, 5, 10, 25], 250, seed=42, trials=10)
        self.assertIsInstance(score, float)


class TraceReplayTest(unittest.TestCase):
    """Tests for readable trace replay."""

    def test_trace_contains_puzzle_info(self):
        trace = game.format_trace([1, 5, 10, 25], 525, 25, 10, "*")
        self.assertIn("target=525", trace)
        self.assertIn("25 * 10", trace)
        self.assertIn("250", trace)
        self.assertIn("Score:", trace)

    def test_trace_invalid_move(self):
        """Non-integer division should show 'invalid' in trace."""
        trace = game.format_trace([3, 10], 5, 10, 3, "/")
        self.assertIn("invalid", trace)


class EvaluateMovesTest(unittest.TestCase):
    """Tests for evaluation metrics."""

    def test_all_exact_solves(self):
        moves = [(25, 10, "+"), (25, 10, "-"), (25, 10, "*"), (25, 10, "/")]
        metrics = game.evaluate_moves([25, 10], 35, moves)
        self.assertEqual(metrics["exact_solves"], 1)  # 25 + 10 = 35
        self.assertEqual(metrics["exact_solve_rate"], 0.25)
        self.assertEqual(metrics["invalid_actions"], 1)  # 25/10 is non-integer

    def test_all_invalid(self):
        moves = [(100, 5, "*"), (200, 1, "+")]
        metrics = game.evaluate_moves([1, 5], 10, moves)
        self.assertEqual(metrics["invalid_actions"], 2)
        self.assertEqual(metrics["invalid_action_rate"], 1.0)

    def test_empty_moves(self):
        metrics = game.evaluate_moves([1, 5], 10, [])
        self.assertEqual(metrics["total"], 0)
        self.assertEqual(metrics["mean_reward"], 0.0)

    def test_metrics_fields_present(self):
        moves = [(1, 5, "+")]
        metrics = game.evaluate_moves([1, 5], 6, moves)
        for field in ["total", "exact_solves", "invalid_actions", "valid_actions",
                       "exact_solve_rate", "invalid_action_rate", "mean_reward", "best_reward"]:
            self.assertIn(field, metrics)


class OracleSolverTest(unittest.TestCase):
    """Tests for the oracle solver."""

    def test_oracle_finds_exact_solution(self):
        self.assertEqual(game.oracle_solve([25, 10], 250), 1.0)

    def test_oracle_finds_best_when_no_exact(self):
        score = game.oracle_solve([1, 2], 100)
        self.assertGreater(score, -1.0)
        self.assertLess(score, 1.0)


class FixtureTest(unittest.TestCase):
    """Integration tests using the deterministic easy/medium/hard fixtures."""

    FIXTURE_DIR = Path(__file__).resolve().parent.parent / "examples" / "agentic" / "countdown" / "fixtures"

    def test_easy_fixtures_load_and_are_solvable(self):
        records = self._load_fixture("easy.jsonl")
        self.assertGreater(len(records), 0)
        for record in records:
            score = game.oracle_solve(record["numbers"], record["target"])
            self.assertEqual(score, 1.0, f"Easy fixture {record['id']} should be exactly solvable")

    def test_medium_fixtures_load_and_are_solvable(self):
        records = self._load_fixture("medium.jsonl")
        self.assertGreater(len(records), 0)
        for record in records:
            score = game.oracle_solve(record["numbers"], record["target"])
            self.assertEqual(score, 1.0, f"Medium fixture {record['id']} should be exactly solvable")

    def test_hard_fixtures_load(self):
        records = self._load_fixture("hard.jsonl")
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIn("numbers", record)
            self.assertIn("target", record)
            self.assertGreater(len(record["numbers"]), 0)

    def test_easy_beats_random_baseline(self):
        """Oracle score should be >= random baseline on easy fixtures."""
        records = self._load_fixture("easy.jsonl")
        for record in records:
            oracle = game.oracle_solve(record["numbers"], record["target"])
            random_score = game.random_baseline_score(
                record["numbers"], record["target"], seed=42, trials=50
            )
            self.assertGreaterEqual(oracle, random_score)

    def _load_fixture(self, name: str) -> list[dict]:
        path = self.FIXTURE_DIR / name
        if not path.exists():
            self.skipTest(f"Fixture {name} not found at {path}")
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records


class PromptFormatTest(unittest.TestCase):
    """Tests for prompt formatting."""

    def test_prompt_contains_numbers_and_target(self):
        prompt = game.format_prompt([1, 5, 10, 25], 525)
        self.assertIn("1", prompt)
        self.assertIn("25", prompt)
        self.assertIn("525", prompt)
        self.assertIn("calculate", prompt)


if __name__ == "__main__":
    unittest.main()