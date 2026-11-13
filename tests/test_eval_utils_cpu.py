from __future__ import annotations

import unittest

from areno.api.eval_utils import (
    benchmark_due,
    summarize_benchmark,
    summarize_generation_rewards,
)


class BenchmarkScheduleTest(unittest.TestCase):
    def test_before_phase_requires_explicit_opt_in(self):
        self.assertFalse(benchmark_due(0, eval_interval=5, max_steps=None,
                                       eval_before_step=False, phase="before"))

    def test_before_phase_schedule(self):
        opts = dict(eval_interval=5, max_steps=None, eval_before_step=True, phase="before")
        self.assertTrue(benchmark_due(0, **opts))
        self.assertFalse(benchmark_due(1, **opts))
        self.assertTrue(benchmark_due(4, **opts))

    def test_after_phase_schedule_and_final_step(self):
        opts = dict(eval_interval=5, max_steps=5, eval_before_step=True, phase="after")
        self.assertTrue(benchmark_due(0, **opts))
        self.assertFalse(benchmark_due(3, **opts))
        self.assertTrue(benchmark_due(5, **opts))

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "eval_interval"):
            benchmark_due(0, eval_interval=0, max_steps=None,
                          eval_before_step=True, phase="after")
        with self.assertRaisesRegex(ValueError, "unknown benchmark phase"):
            benchmark_due(0, eval_interval=5, max_steps=None,
                          eval_before_step=True, phase="middle")


class BenchmarkSummaryTest(unittest.TestCase):
    def test_aggregates_response_tokens_only(self):
        # prompt positions (index 0 and 1) carry deliberately extreme values;
        # only tokens from index 2 onward may influence the result.
        teacher = [[0.0, 999.0, -1.0, -2.0]]
        student = [[0.0, -999.0, -2.0, -3.0]]

        stats = summarize_benchmark(teacher, student, [2])

        self.assertAlmostEqual(stats["teacher_logprob_mean"], -1.5)
        self.assertAlmostEqual(stats["student_logprob_mean"], -2.5)
        self.assertAlmostEqual(stats["teacher_student_gap"], 1.0)
        self.assertAlmostEqual(stats["teacher_student_gap_abs"], 1.0)
        self.assertAlmostEqual(stats["loss"], 2.5)
        self.assertEqual(stats["target_tokens"], 2.0)
        self.assertEqual(stats["batches"], 1.0)

    def test_optional_reward_mean(self):
        stats = summarize_benchmark(
            [[0.0, -1.0]], [[0.0, -2.0]], [1], rewards=[0.0, 1.0]
        )
        self.assertAlmostEqual(stats["reward_mean"], 0.5)

    def test_mismatched_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            summarize_benchmark([[0.0, -1.0]], [], [1])
        with self.assertRaisesRegex(ValueError, "equal length"):
            summarize_benchmark([[0.0, -1.0]], [[0.0]], [1])

    def test_non_finite_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            summarize_benchmark([[0.0, float("nan")]], [[0.0, -1.0]], [1])

    def test_empty_response_rejected(self):
        with self.assertRaisesRegex(ValueError, "no response tokens"):
            summarize_benchmark([[0.0, -1.0]], [[0.0, -1.0]], [2])


class GenerationSummaryTest(unittest.TestCase):
    def test_binary_rewards_give_task_accuracy(self):
        stats = summarize_generation_rewards([1.0, 0.0, 1.0, 0.0])

        self.assertAlmostEqual(stats["accuracy"], 0.5)
        self.assertAlmostEqual(stats["reward_mean"], 0.5)
        self.assertEqual(stats["samples"], 4.0)

    def test_continuous_rewards_still_report_mean(self):
        stats = summarize_generation_rewards([0.2, 0.4, 0.9])

        self.assertAlmostEqual(stats["reward_mean"], 0.5, places=6)
        self.assertAlmostEqual(stats["accuracy"], 1 / 3, places=6)

    def test_pass_threshold_is_configurable(self):
        stats = summarize_generation_rewards([0.4, 0.6], pass_threshold=0.8)

        self.assertEqual(stats["accuracy"], 0.0)

    def test_empty_and_non_finite_rejected(self):
        with self.assertRaisesRegex(ValueError, "no scored samples"):
            summarize_generation_rewards([])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            summarize_generation_rewards([1.0, float("nan")])


if __name__ == "__main__":
    unittest.main()
