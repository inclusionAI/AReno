from __future__ import annotations

import unittest

from areno.api.rewards import CompositeReward, CompositeScore, RewardRecord
from areno.cli.train import _parse_reward_fn_paths, _reject_duplicate_reward_component_names


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


class TrainerComponentStatsTest(unittest.TestCase):
    """The trainer turns a CompositeReward's per-component scores into metric keys.

    Drives ``_score_reward`` -> ``_collect_component_stats`` -> ``_augment_train_stats``
    directly on a minimally constructed :class:`PolicyOnlyTrainer` (no backend, no
    tokenizer): these methods only touch the reward function and the per-step
    component accumulators, so stubbing ``instance``/``dataset``/``loss_fn`` is safe
    and Exercises the exact channel that emits ``reward/<name>_mean`` in real runs.
    """

    def _trainer(self, reward_fn):
        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        # config/instance/dataset/loss_fn are unused by the component-stat methods.
        return PolicyOnlyTrainer(None, instance=None, dataset=None, reward_fn=reward_fn, loss_fn=None)

    def test_component_means_collected_and_augmented(self):
        """Scoring N records populates reward/<name>_mean and folds into train_stats."""
        cr = CompositeReward([("accuracy", lambda r: 1.0, 0.7), ("format", lambda r: 0.0, 0.3)])
        trainer = self._trainer(cr)

        for _ in range(3):
            trainer._score_reward(_record())

        stats = trainer._collect_component_stats()
        self.assertAlmostEqual(stats["reward/accuracy_reward_mean"], 1.0, places=6)
        self.assertAlmostEqual(stats["reward/format_reward_mean"], 0.0, places=6)
        self.assertNotIn("reward/accuracy_reward_invalid_count", stats)  # no failures recorded

        augmented = trainer._augment_train_stats({"loss": 0.5})
        self.assertAlmostEqual(augmented["reward/accuracy_reward_mean"], 1.0, places=6)
        self.assertAlmostEqual(augmented["reward/format_reward_mean"], 0.0, places=6)
        self.assertEqual(augmented["loss"], 0.5)

    def test_mark_invalid_records_component_count(self):
        """A failed component in mark_invalid mode surfaces as reward/<name>_invalid_count."""

        def _raise(_record):
            raise RuntimeError("boom")

        cr = CompositeReward(
            [("ok", lambda r: 1.0, 0.7), ("bad", _raise, 0.3)], on_error="mark_invalid"
        )
        trainer = self._trainer(cr)

        totals = [trainer._score_reward(_record())[0] for _ in range(2)]
        self.assertTrue(all(t == 1.0 for t in totals))  # surviving component total

        stats = trainer._collect_component_stats()
        self.assertEqual(stats["reward/bad_invalid_count"], 2.0)
        self.assertIn("reward/ok_mean", stats)

    def test_plain_reward_fn_emits_no_component_keys(self):
        """A plain single-reward function must not leak any reward/<name>_* keys."""
        trainer = self._trainer(lambda r: 1.0)

        for _ in range(4):
            total, score = trainer._score_reward(_record())
            self.assertEqual(total, 1.0)
            self.assertIsNone(score)

        self.assertEqual(trainer._collect_component_stats(), {})
        # Empty component stats must not alter an existing train_stats dict.
        self.assertEqual(trainer._augment_train_stats({"loss": 0.5}), {"loss": 0.5})


class CliRewardParsingTest(unittest.TestCase):
    """CLI --reward-fn-path parsing: reject bad/weights and duplicate names."""

    def test_invalid_weight_raises(self):
        """A non-numeric weight suffix must raise an Invalid reward weight message naming the
        offending value and the full --reward-fn-path, so a colon-containing path can be told
        apart from a genuine weight typo."""
        with self.assertRaisesRegex(ValueError, r"Invalid reward weight 'abc' in --reward-fn-path"):
            _parse_reward_fn_paths(("examples/math/reward.py:abc",))

    def test_negative_weight_raises(self):
        """Negative weights are reported with the same Invalid reward weight prefix."""
        with self.assertRaisesRegex(ValueError, "Invalid reward weight: -0.5"):
            _parse_reward_fn_paths(("examples/math/reward.py:-0.5",))

    def test_duplicate_same_file_raises(self):
        """Registering the same reward file twice collides on the stem name."""
        components = _parse_reward_fn_paths(
            ("examples/math/reward.py:0.5", "examples/math/reward.py:0.5")
        )
        with self.assertRaisesRegex(ValueError, "Duplicate reward component name: reward"):
            _reject_duplicate_reward_component_names(components)

    def test_valid_weighted_components_parse(self):
        """Two distinct files with weights parse to (stem, resolved_path, weight)."""
        components = _parse_reward_fn_paths(
            ("examples/math/accuracy_reward.py:0.7", "examples/math/format_reward.py:0.3")
        )
        self.assertEqual([c[0] for c in components], ["accuracy_reward", "format_reward"])
        self.assertEqual([c[2] for c in components], [0.7, 0.3])

    def test_single_unweighted_is_legacy_path(self):
        """A lone path without :weight parses with default weight 1.0."""
        components = _parse_reward_fn_paths(("examples/math/reward.py",))
        self.assertEqual(components[0][2], 1.0)


if __name__ == "__main__":
    unittest.main()
