"""CPU-only tests for the balance-scale agentic example.

No GPU, network, or model weights required.  These tests validate the game
logic, dataset generation, dataset loading, and reward function.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Load example modules from their directory.
example_dir = Path(__file__).resolve().parent.parent / "examples" / "agentic" / "balance_scale"
sys.path.insert(0, str(example_dir))

game = importlib.import_module("game")
dataset_generator = importlib.import_module("dataset_generator")
dataset_loader = importlib.import_module("dataset_loader")
reward = importlib.import_module("reward")


def _make_record(
    *,
    odd_ball_index: int = 3,
    odd_ball_direction: str = "heavier",
    tool_calls: list[dict] | None = None,
    num_balls: int = 9,
    max_weighings: int = 3,
) -> SimpleNamespace:
    """Build a minimal RewardRecord-like object for reward_fn tests."""

    return SimpleNamespace(
        source_record={
            "id": "test-001",
            "prompt": "test prompt",
            "num_balls": num_balls,
            "odd_ball_index": odd_ball_index,
            "odd_ball_direction": odd_ball_direction,
            "max_weighings": max_weighings,
        },
        tool_calls=tool_calls or [],
    )


class GameLogicTest(unittest.TestCase):
    """Tests for BalanceGame and generate_game."""

    def test_heavier_ball_on_left(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=2, odd_ball_direction="heavier", max_weighings=3)
        self.assertEqual(g.weigh([2], [0]), "left_heavy")

    def test_lighter_ball_on_left(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=2, odd_ball_direction="lighter", max_weighings=3)
        self.assertEqual(g.weigh([2], [0]), "right_heavy")

    def test_heavier_ball_on_right(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=5, odd_ball_direction="heavier", max_weighings=3)
        self.assertEqual(g.weigh([0], [5]), "right_heavy")

    def test_odd_ball_not_in_groups(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=4, odd_ball_direction="heavier", max_weighings=3)
        self.assertEqual(g.weigh([0, 1], [2, 3]), "balanced")

    def test_multi_ball_groups_balanced(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=8, odd_ball_direction="heavier", max_weighings=3)
        self.assertEqual(g.weigh([0, 1, 2], [3, 4, 5]), "balanced")

    def test_multi_ball_groups_heavier(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=1, odd_ball_direction="heavier", max_weighings=3)
        self.assertEqual(g.weigh([0, 1, 2], [3, 4, 5]), "left_heavy")

    def test_unequal_group_size_raises(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=0, odd_ball_direction="heavier", max_weighings=3)
        with self.assertRaisesRegex(ValueError, "equal size"):
            g.weigh([0, 1], [2])

    def test_overlapping_groups_raises(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=0, odd_ball_direction="heavier", max_weighings=3)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            g.weigh([0, 1], [1, 2])

    def test_index_out_of_range_raises(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=0, odd_ball_direction="heavier", max_weighings=3)
        with self.assertRaisesRegex(ValueError, "out of range"):
            g.weigh([0, 9], [1, 2])

    def test_empty_groups_raises(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=0, odd_ball_direction="heavier", max_weighings=3)
        with self.assertRaisesRegex(ValueError, "not be empty"):
            g.weigh([], [])

    def test_duplicate_index_in_group_raises(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=0, odd_ball_direction="heavier", max_weighings=3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            g.weigh([0, 0], [1, 2])

    def test_weighing_budget_exhausted_raises(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=0, odd_ball_direction="heavier", max_weighings=1)
        g.weigh([0], [1])
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            g.weigh([2], [3])

    def test_weighings_remaining_decrements(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=0, odd_ball_direction="heavier", max_weighings=3)
        self.assertEqual(g.weighings_remaining, 3)
        g.weigh([0], [1])
        self.assertEqual(g.weighings_remaining, 2)
        self.assertEqual(g.weighings_used, 1)

    def test_check_answer_full_correct(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=3, odd_ball_direction="heavier", max_weighings=3)
        identity, direction = g.check_answer(3, "heavier")
        self.assertTrue(identity)
        self.assertTrue(direction)

    def test_check_answer_identity_only(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=3, odd_ball_direction="heavier", max_weighings=3)
        identity, direction = g.check_answer(3, "lighter")
        self.assertTrue(identity)
        self.assertFalse(direction)

    def test_check_answer_both_wrong(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=3, odd_ball_direction="heavier", max_weighings=3)
        identity, direction = g.check_answer(5, "lighter")
        self.assertFalse(identity)
        self.assertFalse(direction)

    def test_check_answer_invalid_index_raises(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=3, odd_ball_direction="heavier", max_weighings=3)
        with self.assertRaisesRegex(ValueError, "out of range"):
            g.check_answer(99, "heavier")

    def test_check_answer_invalid_direction_raises(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=3, odd_ball_direction="heavier", max_weighings=3)
        with self.assertRaisesRegex(ValueError, "direction"):
            g.check_answer(3, "same")

    def test_invalid_num_balls_raises(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            game.BalanceGame(num_balls=2, odd_ball_index=0, odd_ball_direction="heavier", max_weighings=3)

    def test_invalid_odd_index_raises(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            game.BalanceGame(num_balls=5, odd_ball_index=10, odd_ball_direction="heavier", max_weighings=3)

    def test_generate_game_reproducible(self):
        g1 = game.generate_game(num_balls=9, seed=42, max_weighings=3)
        g2 = game.generate_game(num_balls=9, seed=42, max_weighings=3)
        self.assertEqual(g1.odd_ball_index, g2.odd_ball_index)
        self.assertEqual(g1.odd_ball_direction, g2.odd_ball_direction)

    def test_generate_game_different_seeds_differ(self):
        g1 = game.generate_game(num_balls=9, seed=1, max_weighings=3)
        g2 = game.generate_game(num_balls=9, seed=999, max_weighings=3)
        # Extremely unlikely to be identical with 9 balls × 2 directions.
        self.assertTrue(
            g1.odd_ball_index != g2.odd_ball_index or g1.odd_ball_direction != g2.odd_ball_direction
        )

    def test_format_prompt_contains_ball_count(self):
        text = game.format_prompt(12, 4)
        self.assertIn("12", text)
        self.assertIn("0 to 11", text)
        self.assertIn("4", text)

    def test_lighter_ball_on_right(self):
        g = game.BalanceGame(num_balls=9, odd_ball_index=7, odd_ball_direction="lighter", max_weighings=3)
        self.assertEqual(g.weigh([0], [7]), "left_heavy")


class DatasetGeneratorTest(unittest.TestCase):
    """Tests for dataset_generator."""

    def test_generates_correct_count(self):
        records = dataset_generator.generate_records(50, seed=2026, num_balls=9, max_weighings=3)
        self.assertEqual(len(records), 50)

    def test_records_have_valid_fields(self):
        records = dataset_generator.generate_records(18, seed=2026, num_balls=9, max_weighings=3)
        keys = {(r["odd_ball_index"], r["odd_ball_direction"]) for r in records}
        # With 9 balls × 2 directions = 18 possible combinations, 18 records
        # should cover all of them with high probability.
        self.assertGreaterEqual(len(keys), 10)

    def test_seeded_reproducibility(self):
        r1 = dataset_generator.generate_records(10, seed=123, num_balls=9, max_weighings=3)
        r2 = dataset_generator.generate_records(10, seed=123, num_balls=9, max_weighings=3)
        self.assertEqual(r1, r2)

    def test_different_seeds_differ(self):
        r1 = dataset_generator.generate_records(10, seed=1, num_balls=9, max_weighings=3)
        r2 = dataset_generator.generate_records(10, seed=2, num_balls=9, max_weighings=3)
        self.assertNotEqual(r1, r2)

    def test_record_fields(self):
        records = dataset_generator.generate_records(5, seed=2026, num_balls=9, max_weighings=2)
        for r in records:
            self.assertIn("id", r)
            self.assertIn("num_balls", r)
            self.assertIn("odd_ball_index", r)
            self.assertIn("odd_ball_direction", r)
            self.assertIn("max_weighings", r)
            self.assertEqual(r["num_balls"], 9)
            self.assertEqual(r["max_weighings"], 2)

    def test_zero_count_raises(self):
        with self.assertRaises(ValueError):
            dataset_generator.generate_records(0)

    def test_too_few_balls_raises(self):
        with self.assertRaises(ValueError):
            dataset_generator.generate_records(5, seed=2026, num_balls=2, max_weighings=3)


class DatasetLoaderTest(unittest.TestCase):
    """Tests for dataset_loader."""

    def test_loads_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "puzzles.jsonl"
            records = dataset_generator.generate_records(10, seed=2026, num_balls=9, max_weighings=3)
            with path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            loaded = dataset_loader.load_training_dataset(str(path))
            self.assertEqual(len(loaded), 10)
            for record in loaded:
                self.assertIn("prompt", record)
                self.assertIn("num_balls", record)
                self.assertIn("odd_ball_index", record)
                self.assertIn("odd_ball_direction", record)
                self.assertIn("max_weighings", record)

    def test_fallback_to_generator_when_file_missing(self):
        loaded = dataset_loader.load_training_dataset("/nonexistent/path/to/puzzles.jsonl")
        self.assertTrue(len(loaded) > 0)
        self.assertIn("prompt", loaded[0])


class RewardTest(unittest.TestCase):
    """Tests for reward_fn."""

    def test_full_correct_answer(self):
        record = _make_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            tool_calls=[
                {"name": "answer", "arguments": json.dumps({"ball_index": 3, "direction": "heavier"})}
            ],
        )
        self.assertEqual(reward.reward_fn(record), 1.0)

    def test_identity_only_wrong_direction(self):
        record = _make_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            tool_calls=[
                {"name": "answer", "arguments": json.dumps({"ball_index": 3, "direction": "lighter"})}
            ],
        )
        self.assertEqual(reward.reward_fn(record), 0.5)

    def test_wrong_ball(self):
        record = _make_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            tool_calls=[
                {"name": "answer", "arguments": json.dumps({"ball_index": 5, "direction": "heavier"})}
            ],
        )
        self.assertEqual(reward.reward_fn(record), 0.0)

    def test_no_answer_tool_call(self):
        record = _make_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            tool_calls=[
                {"name": "weigh", "arguments": json.dumps({"left_group": [0], "right_group": [1]})},
            ],
        )
        self.assertEqual(reward.reward_fn(record), 0.0)

    def test_no_tool_calls_at_all(self):
        record = _make_record(odd_ball_index=3, odd_ball_direction="heavier", tool_calls=[])
        self.assertEqual(reward.reward_fn(record), 0.0)

    def test_uses_last_answer_call(self):
        """When multiple answer calls exist, the last one is used."""
        record = _make_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            tool_calls=[
                {"name": "answer", "arguments": json.dumps({"ball_index": 5, "direction": "heavier"})},
                {"name": "answer", "arguments": json.dumps({"ball_index": 3, "direction": "heavier"})},
            ],
        )
        self.assertEqual(reward.reward_fn(record), 1.0)

    def test_answer_with_dict_arguments(self):
        record = _make_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            tool_calls=[
                {"name": "answer", "arguments": {"ball_index": 3, "direction": "heavier"}},
            ],
        )
        self.assertEqual(reward.reward_fn(record), 1.0)

    def test_malformed_arguments(self):
        record = _make_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            tool_calls=[
                {"name": "answer", "arguments": "not json"},
            ],
        )
        self.assertEqual(reward.reward_fn(record), 0.0)


class ImportBoundaryTest(unittest.TestCase):
    """Verify example modules import without AReno engine dependencies."""

    def test_game_module_imports_cleanly(self):
        # game.py should only use stdlib.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "game_check", example_dir / "game.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "BalanceGame"))

    def test_reward_constants(self):
        self.assertEqual(reward.FULL_ANSWER_REWARD, 1.0)
        self.assertEqual(reward.IDENTITY_ONLY_REWARD, 0.5)
        self.assertEqual(reward.WRONG_REWARD, 0.0)


# ---------------------------------------------------------------------------
# XML no-tool variant tests
# ---------------------------------------------------------------------------

reward_no_tool = importlib.import_module("reward_no_tool")
dataset_loader_no_tool = importlib.import_module("dataset_loader_no_tool")


def _make_no_tool_record(
    *,
    odd_ball_index: int = 3,
    odd_ball_direction: str = "heavier",
    completion: str = "",
    num_balls: int = 9,
    max_weighings: int = 3,
) -> SimpleNamespace:
    """Build a minimal record for the no-tool reward_fn tests."""

    return SimpleNamespace(
        source_record={
            "id": "test-001",
            "prompt": "test prompt",
            "num_balls": num_balls,
            "odd_ball_index": odd_ball_index,
            "odd_ball_direction": odd_ball_direction,
            "max_weighings": max_weighings,
        },
        completion=completion,
    )


class XmlParseTest(unittest.TestCase):
    """Tests for parse_xml_weigh and parse_xml_answer in game.py."""

    def test_parse_weigh_basic(self):
        result = game.parse_xml_weigh('<weigh left="0,1" right="2,3"/>')
        self.assertEqual(result, ([0, 1], [2, 3]))

    def test_parse_weigh_single_ball(self):
        result = game.parse_xml_weigh('<weigh left="0" right="1"/>')
        self.assertEqual(result, ([0], [1]))

    def test_parse_weigh_with_spaces(self):
        result = game.parse_xml_weigh('<weigh left=" 0 , 1 " right=" 2 , 3 "/>')
        self.assertEqual(result, ([0, 1], [2, 3]))

    def test_parse_weigh_case_insensitive(self):
        result = game.parse_xml_weigh('<WEIGH LEFT="0,1" RIGHT="2,3"/>')
        self.assertEqual(result, ([0, 1], [2, 3]))

    def test_parse_weigh_non_self_closing(self):
        result = game.parse_xml_weigh('<weigh left="0,1" right="2,3"></weigh>')
        self.assertEqual(result, ([0, 1], [2, 3]))

    def test_parse_weigh_takes_last_match(self):
        text = '<weigh left="0" right="1"/>\n<weigh left="2" right="3"/>'
        result = game.parse_xml_weigh(text)
        self.assertEqual(result, ([2], [3]))

    def test_parse_weigh_no_match(self):
        self.assertIsNone(game.parse_xml_weigh("no weigh tag here"))

    def test_parse_weigh_invalid_ints(self):
        self.assertIsNone(game.parse_xml_weigh('<weigh left="a,b" right="2,3"/>'))

    def test_parse_answer_basic(self):
        result = game.parse_xml_answer('<answer ball="3" direction="heavier"/>')
        self.assertEqual(result, (3, "heavier"))

    def test_parse_answer_case_insensitive_direction(self):
        result = game.parse_xml_answer('<answer ball="3" direction="HEAVIER"/>')
        self.assertEqual(result, (3, "heavier"))

    def test_parse_answer_non_self_closing(self):
        result = game.parse_xml_answer('<answer ball="5" direction="lighter"></answer>')
        self.assertEqual(result, (5, "lighter"))

    def test_parse_answer_takes_last_match(self):
        text = '<answer ball="1" direction="heavier"/>\n<answer ball="3" direction="lighter"/>'
        result = game.parse_xml_answer(text)
        self.assertEqual(result, (3, "lighter"))

    def test_parse_answer_with_reasoning_text(self):
        text = "Let me think...\nThe odd ball is 3, it feels heavier.\n<answer ball=\"3\" direction=\"heavier\"/>"
        result = game.parse_xml_answer(text)
        self.assertEqual(result, (3, "heavier"))

    def test_parse_answer_no_match(self):
        self.assertIsNone(game.parse_xml_answer("no answer tag"))

    def test_parse_answer_invalid_direction(self):
        self.assertIsNone(game.parse_xml_answer('<answer ball="3" direction="same"/>'))


class FormatXmlPromptTest(unittest.TestCase):
    """Tests for format_xml_prompt."""

    def test_contains_ball_count(self):
        text = game.format_xml_prompt(12, 4)
        self.assertIn("12", text)
        self.assertIn("0 to 11", text)
        self.assertIn("4", text)

    def test_contains_weigh_example(self):
        text = game.format_xml_prompt(9, 3)
        self.assertIn("<weigh", text)
        self.assertIn("left=", text)
        self.assertIn("right=", text)

    def test_contains_answer_example(self):
        text = game.format_xml_prompt(9, 3)
        self.assertIn("<answer", text)
        self.assertIn("ball=", text)
        self.assertIn("direction=", text)


class NoToolRewardTest(unittest.TestCase):
    """Tests for reward_no_tool.reward_fn."""

    def test_full_correct(self):
        record = _make_no_tool_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            completion='<answer ball="3" direction="heavier"/>',
        )
        self.assertEqual(reward_no_tool.reward_fn(record), 1.0)

    def test_identity_only(self):
        record = _make_no_tool_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            completion='<answer ball="3" direction="lighter"/>',
        )
        self.assertEqual(reward_no_tool.reward_fn(record), 0.5)

    def test_wrong_ball(self):
        record = _make_no_tool_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            completion='<answer ball="5" direction="heavier"/>',
        )
        self.assertEqual(reward_no_tool.reward_fn(record), 0.0)

    def test_no_answer_tag(self):
        record = _make_no_tool_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            completion="I think the odd ball is 3 but I forgot to use the tag.",
        )
        self.assertEqual(reward_no_tool.reward_fn(record), 0.0)

    def test_answer_with_reasoning(self):
        record = _make_no_tool_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            completion=(
                "After weighing, ball 3 is heavier.\n"
                '<answer ball="3" direction="heavier"/>'
            ),
        )
        self.assertEqual(reward_no_tool.reward_fn(record), 1.0)

    def test_takes_last_answer(self):
        record = _make_no_tool_record(
            odd_ball_index=3,
            odd_ball_direction="heavier",
            completion=(
                '<answer ball="5" direction="heavier"/>\n'
                '<answer ball="3" direction="heavier"/>'
            ),
        )
        self.assertEqual(reward_no_tool.reward_fn(record), 1.0)


class NoToolDatasetLoaderTest(unittest.TestCase):
    """Tests for dataset_loader_no_tool."""

    def test_loads_jsonl_with_xml_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "puzzles.jsonl"
            records = dataset_generator.generate_records(5, seed=2026, num_balls=9, max_weighings=3)
            with path.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            loaded = dataset_loader_no_tool.load_training_dataset(str(path))
            self.assertEqual(len(loaded), 5)
            for record in loaded:
                self.assertIn("prompt", record)
                self.assertIn("<weigh", record["prompt"])
                self.assertIn("<answer", record["prompt"])


if __name__ == "__main__":
    unittest.main()
