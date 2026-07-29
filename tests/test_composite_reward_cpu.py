from __future__ import annotations

import unittest

from areno.api.rewards import CompositeReward, CompositeScore, RewardRecord


def _record(prompt: str = "p", completion: str = "c") -> RewardRecord:
    """Build a minimal reward record for unit tests (no tokenizer needed)."""

    return RewardRecord(prompt=prompt, completion=completion)


class CompositeRewardMathTest(unittest.TestCase):
    """CompositeReward math and validation against hand-computed fixtures."""

    def test_total_matches_hand_calculated_weighted_average(self):
        """total should be the weight-normalised average of component values."""
        # accuracy=1.0, format=0.0, weights 0.7/0.3 -> 0.7/1.0 = 0.7
        cr = CompositeReward(
            [("accuracy", lambda r: 1.0, 0.7), ("format", lambda r: 0.0, 0.3)]
        )
        score = cr.score(_record())
        self.assertAlmostEqual(score.total, 0.7, places=6)
        self.assertEqual(score.components, {"accuracy": 1.0, "format": 0.0})
        self.assertEqual(score.invalid, [])

    def test_total_normalized_not_absolute_sum(self):
        """Weights express ratios, so 0.7/0.3 and 7.0/3.0 give the same total."""
        a = CompositeReward([("x", lambda r: 1.0, 0.7), ("y", lambda r: 1.0, 0.3)])
        b = CompositeReward([("x", lambda r: 1.0, 7.0), ("y", lambda r: 1.0, 3.0)])
        self.assertAlmostEqual(a.score(_record()).total, b.score(_record()).total, places=6)

    def test_call_returns_total(self):
        """__call__ must return the weighted total for trainer compatibility."""
        cr = CompositeReward([("a", lambda r: 1.0, 0.7), ("b", lambda r: 1.0, 0.3)])
        self.assertEqual(cr(_record()), cr.score(_record()).total)
        self.assertIsInstance(cr(_record()), float)

    def test_length_alignment_per_record(self):
        """Scoring N records yields N independent totals (length alignment)."""
        cr = CompositeReward([("a", lambda r: 1.0, 1.0)])
        records = [_record(completion=str(i)) for i in range(5)]
        totals = [cr.score(rec).total for rec in records]
        self.assertEqual(len(totals), len(records))
        self.assertTrue(all(t == 1.0 for t in totals))


class CompositeRewardValidationTest(unittest.TestCase):
    """Constructor validation rejects misconfiguration up front."""

    def test_rejects_duplicate_names(self):
        """Duplicate component names must raise at construction."""
        with self.assertRaisesRegex(ValueError, "duplicate reward component name 'a'"):
            CompositeReward([("a", lambda r: 1.0, 1.0), ("a", lambda r: 1.0, 1.0)])

    def test_rejects_empty_components(self):
        """At least one component is required."""
        with self.assertRaisesRegex(ValueError, "at least one component"):
            CompositeReward([])

    def test_rejects_bad_weight(self):
        """Weights must be finite and non-negative."""
        with self.assertRaisesRegex(ValueError, "non-negative number"):
            CompositeReward([("a", lambda r: 1.0, float("nan"))])
        with self.assertRaisesRegex(ValueError, "non-negative number"):
            CompositeReward([("a", lambda r: 1.0, -0.5)])

    def test_rejects_zero_total_weight(self):
        """All-zero weights make the normalised total undefined."""
        with self.assertRaisesRegex(ValueError, "positive number"):
            CompositeReward([("a", lambda r: 1.0, 0.0), ("b", lambda r: 1.0, 0.0)])

    def test_rejects_bad_on_error(self):
        """on_error must be one of the two supported modes."""
        with self.assertRaisesRegex(ValueError, "on_error must be"):
            CompositeReward([("a", lambda r: 1.0, 1.0)], on_error="bogus")  # type: ignore[arg-type]

    def test_rejects_empty_component_name(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            CompositeReward([("", lambda r: 1.0, 1.0)])


class CompositeRewardErrorModeTest(unittest.TestCase):
    """raise vs mark_invalid behavior when a component fails."""

    def _bad(self):
        def _raise(_record):
            raise RuntimeError("boom")

        return CompositeReward([("ok", lambda r: 1.0, 0.7), ("bad", _raise, 0.3)])

    def test_raise_mode_propagates_with_component_name(self):
        """raise mode re-raises naming the component and preserving the cause."""
        with self.assertRaisesRegex(ValueError, "reward component 'bad' raised"):
            self._bad().score(_record())

    def test_raise_mode_preserves_original_cause(self):
        with self.assertRaises(ValueError) as ctx:
            self._bad().score(_record())
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    def test_mark_invalid_continues_and_records(self):
        """mark_invalid substitutes the failed component and recomputes the total."""

        def _raise(_record):
            raise RuntimeError("boom")

        cr_clean = CompositeReward([("ok", lambda r: 1.0, 0.7), ("bad", _raise, 0.3)], on_error="mark_invalid")
        score = cr_clean.score(_record())
        self.assertIn("bad", score.invalid)
        # Only 'ok' survives with weight 0.7 -> total = 1.0.
        self.assertAlmostEqual(score.total, 1.0, places=6)

    def test_mark_invalid_nonfinite_is_recorded(self):
        """A non-finite return value is treated like a raised exception."""

        cr = CompositeReward(
            [("ok", lambda r: 1.0, 0.7), ("nan", lambda r: float("nan"), 0.3)],
            on_error="mark_invalid",
        )
        score = cr.score(_record())
        self.assertIn("nan", score.invalid)
        self.assertAlmostEqual(score.total, 1.0, places=6)

    def test_raise_mode_nonfinite_names_component(self):
        with self.assertRaisesRegex(ValueError, "reward component 'nan' returned a non-finite value"):
            CompositeReward([("ok", lambda r: 1.0, 1.0), ("nan", lambda r: float("nan"), 1.0)]).score(_record())

    def test_failure_message_does_not_expose_full_sample(self):
        """The raised message should name the component, not echo prompt/completion."""
        record = RewardRecord(prompt="SECRET PROMPT TEXT", completion="SECRET COMPLETION TEXT")

        def _raise(_record):
            raise RuntimeError("boom")

        with self.assertRaises(ValueError) as ctx:
            CompositeReward([("bad", _raise, 1.0)]).score(record)
        self.assertNotIn("SECRET", str(ctx.exception))


class CompositeRewardComponentAccessTest(unittest.TestCase):
    """Components are independently inspectable for metrics output."""

    def test_components_dict_preserves_all_values(self):
        cr = CompositeReward([("a", lambda r: 0.5, 1.0), ("b", lambda r: 0.25, 1.0)])
        score = cr.score(_record())
        self.assertEqual(set(score.components.keys()), {"a", "b"})
        self.assertAlmostEqual(score.components["a"], 0.5, places=6)
        self.assertAlmostEqual(score.components["b"], 0.25, places=6)

    def test_composite_score_is_dataclass(self):
        score = CompositeScore(total=0.5, components={"a": 0.5}, invalid=[])
        self.assertEqual(score.total, 0.5)
        self.assertEqual(score.invalid, [])


class ExampleRewardLoadTest(unittest.TestCase):
    """Load the bundled example reward files and check their semantics."""

    def test_accuracy_and_format_examples_match_expected_semantics(self):
        """The example commands in the docs must produce the documented rewards."""
        import importlib.util
        from pathlib import Path

        def load(name: str):
            path = Path(__file__).resolve().parents[1] / "examples" / "math" / f"{name}_reward.py"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.reward_fn

        accuracy = load("accuracy")
        format_reward = load("format")

        boxed_correct = RewardRecord(prompt="q", completion=r"The answer is \boxed{42}", answer=["42"])
        boxed_wrong = RewardRecord(prompt="q", completion=r"\boxed{7}", answer=["42"])
        plain = RewardRecord(prompt="q", completion="42", answer=["42"])

        self.assertEqual(accuracy(boxed_correct), 1.0)
        self.assertEqual(accuracy(boxed_wrong), 0.0)
        self.assertEqual(accuracy(plain), 0.0)
        self.assertEqual(format_reward(boxed_correct), 1.0)
        self.assertEqual(format_reward(boxed_wrong), 1.0)
        self.assertEqual(format_reward(plain), 0.0)


if __name__ == "__main__":
    unittest.main()
