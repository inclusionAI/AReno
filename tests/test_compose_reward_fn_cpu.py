"""CPU tests for weighted reward composition.

Covers normal path, boundary values, invalid input, and error-handling
strategies (fail-fast vs mark-invalid) for ``compose_reward_fn``.
"""

from __future__ import annotations

import os
import unittest

from areno.api.rewards import (
    RewardRecord,
    compose_reward_fn,
    compute_group_advantages,
    load_reward_fn,
    make_reward_record,
)


def _const_1(_record) -> float:
    return 1.0


def _const_half(_record) -> float:
    return 0.5


def _const_zero(_record) -> float:
    return 0.0


def _const_seven_tenths(_record) -> float:
    return 0.7


def _nan_fn(_record) -> float:
    return float("nan")


def _inf_fn(_record) -> float:
    return float("inf")


def _keyword_match_fn(record) -> float:
    return 1.0 if "good" in record.completion else 0.0


class ComposeRewardFnTest(unittest.TestCase):
    """Tests for weighted reward composition."""

    def _make_record(self, completion: str = "test") -> RewardRecord:
        return make_reward_record(
            prompt="prompt",
            completion=completion,
            source_record={"answer": "4"},
        )

    # -- Normal path --

    def test_weighted_sum_matches_hand_calculation(self):
        """Weighted total should match hand calculation."""
        composed = compose_reward_fn([("a", _const_1, 0.7), ("b", _const_half, 0.3)])
        record = self._make_record()
        # 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        self.assertAlmostEqual(composed(record), 0.85, places=6)

    def test_component_values_stored_in_metadata(self):
        """Each component score should be stored in record.metadata."""
        composed = compose_reward_fn([("correct", _const_1, 0.8), ("format", _const_zero, 0.2)])
        record = self._make_record()
        composed(record)
        components = record.metadata["reward_components"]
        self.assertEqual(components["correct"], 1.0)
        self.assertEqual(components["format"], 0.0)

    def test_all_components_share_same_record(self):
        """All component functions should receive the same RewardRecord."""
        seen = []

        def spy(record):
            seen.append(record)
            return 0.5

        composed = compose_reward_fn([("a", spy, 0.5), ("b", spy, 0.5)])
        record = self._make_record("hello")
        composed(record)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0], seen[1])
        self.assertEqual(seen[0].completion, "hello")

    def test_single_component_works(self):
        """A single component should work correctly."""
        composed = compose_reward_fn([("only", _const_seven_tenths, 1.0)])
        record = self._make_record()
        self.assertAlmostEqual(composed(record), 0.7, places=6)

    def test_negative_weights_allowed(self):
        """Negative weights should be allowed (e.g. penalty terms)."""
        composed = compose_reward_fn([("reward", _const_1, 1.0), ("penalty", _const_1, -0.5)])
        record = self._make_record()
        # 1.0 * 1.0 + (-0.5) * 1.0 = 0.5
        self.assertAlmostEqual(composed(record), 0.5, places=6)

    # -- Boundary values --

    def test_zero_weight_contributes_nothing(self):
        """A zero-weight component should not affect the total."""
        composed = compose_reward_fn([("a", _const_1, 1.0), ("b", _const_1, 0.0)])
        record = self._make_record()
        self.assertAlmostEqual(composed(record), 1.0, places=6)

    def test_all_zero_weights_produces_zero_total(self):
        """All-zero weights should produce a zero total."""
        composed = compose_reward_fn([("a", _const_1, 0.0)])
        record = self._make_record()
        self.assertAlmostEqual(composed(record), 0.0, places=6)

    # -- Invalid input --

    def test_rejects_empty_component_list(self):
        """An empty component list should raise."""
        with self.assertRaisesRegex(ValueError, "at least one component"):
            compose_reward_fn([])

    def test_rejects_duplicate_names(self):
        """Duplicate component names should raise."""
        with self.assertRaisesRegex(ValueError, "duplicate reward component names"):
            compose_reward_fn([("a", _const_half, 0.5), ("a", _const_half, 0.5)])

    def test_rejects_non_finite_weight(self):
        """Non-finite weights should raise."""
        with self.assertRaisesRegex(ValueError, "must be finite"):
            compose_reward_fn([("a", _const_half, float("nan"))])

    # -- Error handling strategies --

    def test_fail_fast_on_exception(self):
        """on_error='raise' should re-raise component exceptions immediately."""

        def bad_fn(_record):
            raise RuntimeError("boom")

        composed = compose_reward_fn(
            [("good", _const_1, 0.5), ("bad", bad_fn, 0.5)],
            on_error="raise",
        )
        record = self._make_record()
        with self.assertRaises(RuntimeError):
            composed(record)

    def test_mark_invalid_on_exception(self):
        """on_error='mark_invalid' should set the offending score to 0.0 and continue."""

        def bad_fn(_record):
            raise RuntimeError("boom")

        composed = compose_reward_fn(
            [("good", _const_1, 0.8), ("bad", bad_fn, 0.2)],
            on_error="mark_invalid",
        )
        record = self._make_record()
        # 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        total = composed(record)
        self.assertAlmostEqual(total, 0.8, places=6)
        self.assertEqual(record.metadata["reward_components"]["bad"], 0.0)

    def test_fail_fast_on_non_finite_return(self):
        """on_error='raise' should raise when a component returns NaN."""
        composed = compose_reward_fn([("a", _nan_fn, 1.0)], on_error="raise")
        record = self._make_record()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            composed(record)

    def test_mark_invalid_on_non_finite_return(self):
        """on_error='mark_invalid' should set Inf returns to 0.0."""
        composed = compose_reward_fn(
            [("a", _const_1, 0.5), ("b", _inf_fn, 0.5)],
            on_error="mark_invalid",
        )
        record = self._make_record()
        # 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        total = composed(record)
        self.assertAlmostEqual(total, 0.5, places=6)

    # -- Backward compatibility --

    def test_compose_then_compute_advantages(self):
        """Composed reward should work with compute_group_advantages."""
        composed = compose_reward_fn([("correct", _keyword_match_fn, 1.0)])
        records = [
            self._make_record("good"),
            self._make_record("bad"),
            self._make_record("good"),
        ]
        rewards = [composed(r) for r in records]
        advantages = compute_group_advantages(rewards)
        self.assertEqual(len(advantages), 3)
        self.assertAlmostEqual(sum(rewards) / 3, 2 / 3, places=6)

    # -- Cross-coverage: Inf weight, mark_invalid + NaN, attribute, isolation --

    def test_rejects_inf_weight(self):
        """Inf weights should be rejected just like NaN weights."""
        with self.assertRaisesRegex(ValueError, "must be finite"):
            compose_reward_fn([("a", _const_half, float("inf"))])

    def test_mark_invalid_on_nan_return(self):
        """on_error='mark_invalid' should set NaN returns to 0.0."""
        composed = compose_reward_fn(
            [("a", _const_1, 0.6), ("b", _nan_fn, 0.4)],
            on_error="mark_invalid",
        )
        record = self._make_record()
        # 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        total = composed(record)
        self.assertAlmostEqual(total, 0.6, places=6)
        self.assertEqual(record.metadata["reward_components"]["b"], 0.0)

    def test_composed_fn_exposes_component_metadata(self):
        """The composed function should expose component names and weights."""
        composed = compose_reward_fn([("correctness", _const_1, 0.7), ("format", _const_half, 0.3)])
        meta = composed._reward_components  # type: ignore[attr-defined]
        self.assertEqual(meta, [("correctness", 0.7), ("format", 0.3)])

    def test_repeated_calls_do_not_cross_contaminate(self):
        """Calling the same composed_fn on different records must not leak metadata."""
        composed = compose_reward_fn([("a", _const_1, 1.0)])
        r1 = self._make_record("first")
        r2 = self._make_record("second")
        composed(r1)
        composed(r2)
        # r1's metadata should still only reflect its own call.
        self.assertEqual(r1.metadata["reward_components"], {"a": 1.0})
        self.assertEqual(r2.metadata["reward_components"], {"a": 1.0})

    def test_metadata_overwritten_on_repeated_call_same_record(self):
        """Calling composed_fn on the same record twice should overwrite, not append."""
        composed = compose_reward_fn([("a", _const_1, 1.0)])
        record = self._make_record()
        composed(record)
        composed(record)
        # Should still be a flat dict, not nested or duplicated.
        self.assertIsInstance(record.metadata["reward_components"], dict)
        self.assertEqual(record.metadata["reward_components"], {"a": 1.0})

    # -- End-to-end: load_reward_fn with a composed file --

    def test_load_reward_fn_loads_composed_file(self):
        """load_reward_fn should load a file that uses compose_reward_fn."""
        example_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples",
            "math",
            "composed_reward.py",
        )
        fn = load_reward_fn(example_path)
        self.assertTrue(callable(fn))
        # The loaded function should expose composed component metadata.
        meta = fn._reward_components  # type: ignore[attr-defined]
        self.assertEqual(meta, [("correctness", 0.7), ("format", 0.2), ("brevity", 0.1)])

    # -- Explicit backward compatibility --

    def test_record_without_compose_has_no_reward_components(self):
        """A plain reward_fn that does not use compose_reward_fn should not set reward_components."""

        def plain_fn(record) -> float:
            return 0.5

        record = self._make_record()
        result = plain_fn(record)
        self.assertAlmostEqual(result, 0.5, places=6)
        self.assertNotIn("reward_components", record.metadata)

    # -- 3-component weighted sum with distinct return values --

    def test_three_component_weighted_sum(self):
        """Three components with different weights and return values."""
        composed = compose_reward_fn(
            [
                ("correctness", _const_1, 0.7),
                ("format", _const_half, 0.2),
                ("brevity", _const_seven_tenths, 0.1),
            ]
        )
        record = self._make_record()
        total = composed(record)
        # 0.7*1.0 + 0.2*0.5 + 0.1*0.7 = 0.7 + 0.1 + 0.07 = 0.87
        self.assertAlmostEqual(total, 0.87, places=6)
        # Verify each component value in metadata
        components = record.metadata["reward_components"]
        self.assertEqual(components["correctness"], 1.0)
        self.assertEqual(components["format"], 0.5)
        self.assertAlmostEqual(components["brevity"], 0.7, places=6)

    def test_three_component_all_metadata_present(self):
        """All three component names should appear in metadata after composition."""
        composed = compose_reward_fn(
            [
                ("a", _const_1, 0.5),
                ("b", _const_half, 0.3),
                ("c", _const_zero, 0.2),
            ]
        )
        record = self._make_record()
        composed(record)
        components = record.metadata["reward_components"]
        self.assertEqual(len(components), 3)
        self.assertIn("a", components)
        self.assertIn("b", components)
        self.assertIn("c", components)

    # -- Components with input-dependent return values --

    def test_components_with_input_dependent_values(self):
        """Components that return different values based on record content."""

        def correctness_fn(record) -> float:
            """1.0 if answer matches, else 0.0."""
            return 1.0 if str(record.answer) in record.completion else 0.0

        def format_fn(record) -> float:
            """0.5 if completion contains 'boxed', else 0.0."""
            return 0.5 if "boxed" in record.completion else 0.0

        def length_fn(record) -> float:
            """Shorter completions score higher."""
            return max(0.0, 1.0 - len(record.completion) / 100.0)

        composed = compose_reward_fn(
            [
                ("correctness", correctness_fn, 0.7),
                ("format", format_fn, 0.2),
                ("brevity", length_fn, 0.1),
            ]
        )

        # Record with correct answer, boxed format, short completion.
        good_record = make_reward_record(
            prompt="What is 2+2?",
            completion="The answer is 4. \\boxed{4}",
            source_record={},
            answer="4",
        )
        total_good = composed(good_record)
        # correctness=1.0, format=0.5, brevity=max(0, 1 - 26/100)=0.74
        # total = 0.7*1.0 + 0.2*0.5 + 0.1*0.74 = 0.7 + 0.1 + 0.074 = 0.874
        self.assertAlmostEqual(total_good, 0.874, places=5)
        self.assertEqual(good_record.metadata["reward_components"]["correctness"], 1.0)
        self.assertEqual(good_record.metadata["reward_components"]["format"], 0.5)

        # Record with wrong answer, no boxed, long completion.
        bad_record = make_reward_record(
            prompt="What is 2+2?",
            completion="I think the answer might be five." + "x" * 200,
            source_record={},
            answer="4",
        )
        total_bad = composed(bad_record)
        # correctness=0.0, format=0.0, brevity=max(0, 1 - 231/100)=0.0 (clamped)
        # total = 0.7*0.0 + 0.2*0.0 + 0.1*0.0 = 0.0
        self.assertAlmostEqual(total_bad, 0.0, places=6)
        self.assertEqual(bad_record.metadata["reward_components"]["correctness"], 0.0)
        self.assertEqual(bad_record.metadata["reward_components"]["format"], 0.0)

    def test_same_composed_fn_different_records_different_totals(self):
        """The same composed function should produce different totals for different inputs."""

        def keyword_score(record) -> float:
            return 1.0 if "good" in record.completion else 0.0

        def length_penalty(record) -> float:
            return -0.5 if len(record.completion) > 50 else 0.0

        composed = compose_reward_fn(
            [
                ("keyword", keyword_score, 1.0),
                ("length_penalty", length_penalty, 1.0),
            ]
        )

        short_good = self._make_record("good")
        long_good = self._make_record("good" + "x" * 100)

        total_short = composed(short_good)
        total_long = composed(long_good)

        # short_good: keyword=1.0, penalty=0.0 -> 1.0
        self.assertAlmostEqual(total_short, 1.0, places=6)
        # long_good: keyword=1.0, penalty=-0.5 -> 0.5
        self.assertAlmostEqual(total_long, 0.5, places=6)
        # Different totals confirm input-dependency
        self.assertNotAlmostEqual(total_short, total_long, places=6)


if __name__ == "__main__":
    unittest.main()
