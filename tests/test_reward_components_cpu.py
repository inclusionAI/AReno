"""CPU tests for the reward-component analyzer and loader.

These cover the issue's data-shape requirements: constant / sparse /
non-finite / missing / dynamic components, large-batch aggregation, and bounded
history. Stats are checked against numpy references, not just exit status.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from areno.api.dashboard import (
    DEFAULT_HISTORY_LIMIT,
    RewardComponentAnalyzer,
    analyze_reward_components,
    load_reward_component_steps,
)


class RewardComponentAnalyzerTest(unittest.TestCase):
    """Pure-logic analyzer tests run without a GPU or any I/O."""

    def test_constant_components_have_zero_zero_and_outlier_fractions(self):
        analyzer = RewardComponentAnalyzer()
        for step in range(3):
            analyzer.update(step, {"correctness": 0.8, "format": 0.2}, total=1.0)

        snap = analyzer.snapshot()
        by_name = {c["name"]: c for c in snap["components"]}
        for comp in snap["components"]:
            self.assertEqual(comp["zero_fraction"], 0.0)
            self.assertEqual(comp["outlier_fraction"], 0.0)
            self.assertEqual(comp["missing_count"], 0)
            self.assertEqual(comp["non_finite_count"], 0)
        # Contribution fractions partition the total across components.
        total = sum(c["contribution_fraction"] for c in snap["components"])
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertAlmostEqual(by_name["correctness"]["weighted_contribution"], 0.8, places=6)
        self.assertAlmostEqual(by_name["format"]["weighted_contribution"], 0.2, places=6)

    def test_sparse_component_reports_exact_zero_fraction(self):
        analyzer = RewardComponentAnalyzer()
        for value in (0.0, 0.0, 0.0, 1.0):
            analyzer.update(value, {"reward": value})

        comp = analyzer.snapshot()["components"][0]
        self.assertEqual(comp["count"], 4)
        self.assertAlmostEqual(comp["zero_fraction"], 0.75, places=6)

    def test_non_finite_values_are_isolated_from_mean_and_std(self):
        analyzer = RewardComponentAnalyzer()
        analyzer.update(0, {"a": 1.0})
        analyzer.update(1, {"a": float("nan")})
        analyzer.update(2, {"a": float("inf")})
        analyzer.update(3, {"a": 2.0})

        comp = analyzer.snapshot()["components"][0]
        self.assertEqual(comp["count"], 2)  # only the two finite values
        self.assertEqual(comp["non_finite_count"], 2)
        self.assertAlmostEqual(comp["non_finite_fraction"], 2 / 4, places=6)
        # Mean/std match a finite-only numpy reference (sample std, ddof=1).
        finite = np.array([1.0, 2.0], dtype=np.float64)
        self.assertAlmostEqual(comp["mean"], float(finite.mean()), places=6)
        self.assertAlmostEqual(comp["std"], float(finite.std(ddof=1)), places=6)

    def test_missing_component_is_counted_and_not_treated_as_zero(self):
        analyzer = RewardComponentAnalyzer()
        analyzer.update(0, {"a": 1.0, "b": 2.0})
        analyzer.update(1, {"a": 3.0})  # b absent this step
        analyzer.update(2, {"a": None, "b": 4.0})  # a explicit-null

        snap = analyzer.snapshot()
        by_name = {c["name"]: c for c in snap["components"]}
        self.assertEqual(by_name["a"]["missing_count"], 1)
        self.assertEqual(by_name["b"]["missing_count"], 1)
        # b's only present finite value is 2.0; missing must not bias the mean.
        self.assertAlmostEqual(by_name["b"]["mean"], (2.0 + 4.0) / 2, places=6)
        self.assertEqual(by_name["b"]["zero_fraction"], 0.0)

    def test_dynamic_components_do_not_premark_missing(self):
        analyzer = RewardComponentAnalyzer()
        analyzer.update(0, {"a": 1.0})
        analyzer.update(1, {"a": 2.0, "b": 3.0})

        snap = analyzer.snapshot()
        by_name = {c["name"]: c for c in snap["components"]}
        self.assertEqual(set(by_name), {"a", "b"})
        # b first appears at step 1; step 0 is "not yet introduced", not missing.
        self.assertEqual(by_name["b"]["missing_count"], 0)
        self.assertEqual(by_name["b"]["count"], 1)

    def test_large_batch_matches_numpy_and_stays_bounded(self):
        n = 5000
        values = [float(i) - 2500.0 for i in range(n)]
        analyzer = RewardComponentAnalyzer(history_limit=200, outlier_z=3.0)
        for step, value in enumerate(values):
            analyzer.update(step, {"v": value})

        comp = analyzer.snapshot()["components"][0]
        arr = np.asarray(values, dtype=np.float64)
        self.assertAlmostEqual(comp["mean"], float(arr.mean()), places=4)
        self.assertAlmostEqual(comp["std"], float(arr.std(ddof=1)), places=4)
        self.assertEqual(comp["min"], float(arr.min()))
        self.assertEqual(comp["max"], float(arr.max()))
        # History is bounded — no raw per-sample array grows with the batch.
        self.assertEqual(len(comp["history"]), 200)
        self.assertEqual(comp["history"][-1]["step"], n - 1)

    def test_history_is_bounded_to_limit(self):
        analyzer = RewardComponentAnalyzer(history_limit=3)
        for step in range(5):
            analyzer.update(step, {"v": float(step)})

        history = analyzer.snapshot()["components"][0]["history"]
        self.assertEqual(len(history), 3)
        self.assertEqual([h["step"] for h in history], [2, 3, 4])

    def test_weighted_contribution_hand_computed(self):
        analyzer = RewardComponentAnalyzer()
        analyzer.update(0, {"a": 1.0, "b": 1.0}, total=2.0)
        analyzer.update(1, {"a": 2.0, "b": 0.0}, total=2.0)

        snap = analyzer.snapshot()
        by_name = {c["name"]: c for c in snap["components"]}
        # total over 2 steps = 4.0; a sums to 3.0, b sums to 1.0.
        self.assertAlmostEqual(by_name["a"]["weighted_contribution"], 0.75, places=6)
        self.assertAlmostEqual(by_name["b"]["weighted_contribution"], 0.25, places=6)
        self.assertAlmostEqual(by_name["b"]["zero_fraction"], 0.5, places=6)

    def test_outlier_fraction_uses_z_threshold(self):
        # 0..0..0..0..100: the 100 is a clear z-score outlier.
        analyzer = RewardComponentAnalyzer(outlier_z=3.0)
        for _ in range(4):
            analyzer.update(0, {"v": 0.0})
        analyzer.update(1, {"v": 100.0})

        comp = analyzer.snapshot()["components"][0]
        self.assertAlmostEqual(comp["outlier_fraction"], 1 / 5, places=6)

    def test_invalidConstructor_args_raise(self):
        with self.assertRaises(ValueError):
            RewardComponentAnalyzer(history_limit=0)
        with self.assertRaises(ValueError):
            RewardComponentAnalyzer(outlier_z=0)
        with self.assertRaises(ValueError):
            RewardComponentAnalyzer(outlier_z=float("nan"))


class RewardComponentLoaderTest(unittest.TestCase):
    """Loader tests: malformed input, missing dir, and no-sample-leak errors."""

    def _write(self, dirpath: Path, name: str, lines: list[str]) -> Path:
        path = dirpath / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_missing_dir_reports_artifact_resolution_error(self):
        steps, errors = load_reward_component_steps("/tmp/areno-does-not-exist-xyz")
        self.assertEqual(steps, [])
        self.assertTrue(any(e["stage"] == "artifact resolution" for e in errors))

    def test_malformed_rows_collected_without_sample_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(
                d,
                "reward_components.0.jsonl",
                [
                    json.dumps({"step": 0, "name": "a", "value": 1.0}),
                    "{not valid json secret-completion-text}",  # malformed
                    json.dumps({"step": 2, "name": "a", "value": "oops", "prompt": "secret-prompt-text"}),
                    json.dumps({"name": "a", "value": 1.0}),  # missing step
                ],
            )
            steps, errors = load_reward_component_steps(d)

        # Valid rows survive; bad rows are skipped. Step 2's value is non-numeric
        # so the component is recorded as missing but the step still exists.
        self.assertEqual([s["step"] for s in steps], [0, 2])
        # Errors reference file/line/component only — never sample text.
        rendered = json.dumps(errors)
        self.assertIn("artifact parse", rendered)
        self.assertNotIn("secret-completion-text", rendered)
        self.assertNotIn("secret-prompt-text", rendered)

    def test_non_numeric_value_treated_as_missing_with_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(
                d,
                "reward_components.0.jsonl",
                [json.dumps({"step": 0, "name": "a", "value": "not-a-number"})],
            )
            steps, errors = load_reward_component_steps(d)

        self.assertEqual(steps[0]["components"]["a"], None)
        self.assertTrue(any("non-numeric" in e["message"] for e in errors))

    def test_analyze_reward_components_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write(
                d,
                "reward_components.0.jsonl",
                [
                    json.dumps({"step": 0, "name": "a", "value": 1.0}),
                    json.dumps({"step": 0, "name": "b", "value": 0.0}),
                    json.dumps({"step": 1, "name": "a", "value": 0.0}),
                    json.dumps({"step": 1, "name": "b", "value": 1.0}),
                ],
            )
            snapshot, errors = analyze_reward_components(d)

        self.assertEqual(errors, [])
        by_name = {c["name"]: c for c in snapshot["components"]}
        self.assertAlmostEqual(by_name["a"]["zero_fraction"], 0.5, places=6)
        self.assertEqual(len(snapshot["steps"]), 2)
        self.assertEqual(snapshot["history_limit"], DEFAULT_HISTORY_LIMIT)


if __name__ == "__main__":
    unittest.main()
