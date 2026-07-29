from __future__ import annotations

import unittest

from areno.api.data import OverlengthPolicy, OverlengthReason, PromptBatch, PromptItem
from areno.api.metrics import init_rollout_stats, record_training_stats
from areno.api.overlength import decide_overlength, truncate_dpo_pair, truncate_sft_response


class OverlengthDecisionTest(unittest.TestCase):
    """Decision matrix, boundaries, malformed input, determinism."""

    def test_within_budget_returns_no_action(self):
        decision = decide_overlength(prompt_len=10, max_prompt_tokens=20, policy=OverlengthPolicy.REJECT)
        self.assertEqual(decision.reason, OverlengthReason.WITHIN_BUDGET)
        self.assertFalse(decision.truncated)
        self.assertIsNone(decision.detail)

    def test_exact_limit_is_not_overlength(self):
        # prompt_len == max_prompt_tokens is the inclusive cap, not a violation.
        decision = decide_overlength(prompt_len=20, max_prompt_tokens=20, policy=OverlengthPolicy.REJECT)
        self.assertEqual(decision.reason, OverlengthReason.EXACT_LIMIT)
        self.assertFalse(decision.truncated)

    def test_prompt_overlength_one_token_over(self):
        decision = decide_overlength(prompt_len=21, max_prompt_tokens=20, policy=OverlengthPolicy.REJECT)
        self.assertEqual(decision.reason, OverlengthReason.SINGLE_MESSAGE_OVERSIZED)
        self.assertEqual(decision.action, OverlengthPolicy.REJECT)
        self.assertFalse(decision.truncated)
        self.assertEqual(decision.detail, {"over_by_tokens": 1})

    def test_response_overlength(self):
        decision = decide_overlength(
            prompt_len=5,
            max_prompt_tokens=20,
            response_len=60,
            max_new_tokens=50,
            policy=OverlengthPolicy.REJECT,
        )
        self.assertEqual(decision.reason, OverlengthReason.RESPONSE_TOO_LONG)
        self.assertEqual(decision.detail, {"over_by_tokens": 10})

    def test_prompt_overlength_takes_precedence_over_response(self):
        # If the prompt itself is already over budget, that is the reported
        # reason even when the response is also too long.
        decision = decide_overlength(
            prompt_len=25,
            max_prompt_tokens=20,
            response_len=60,
            max_new_tokens=50,
            policy=OverlengthPolicy.REJECT,
        )
        self.assertEqual(decision.reason, OverlengthReason.SINGLE_MESSAGE_OVERSIZED)

    def test_truncate_policy_marks_truncated(self):
        decision = decide_overlength(
            prompt_len=5,
            max_prompt_tokens=20,
            response_len=60,
            max_new_tokens=50,
            policy=OverlengthPolicy.TRUNCATE,
        )
        self.assertEqual(decision.action, OverlengthPolicy.TRUNCATE)
        self.assertTrue(decision.truncated)

    def test_warn_policy_keeps_sample_not_truncated(self):
        decision = decide_overlength(
            prompt_len=5,
            max_prompt_tokens=20,
            response_len=60,
            max_new_tokens=50,
            policy=OverlengthPolicy.WARN,
        )
        self.assertEqual(decision.action, OverlengthPolicy.WARN)
        self.assertFalse(decision.truncated)

    def test_invalid_max_prompt_tokens_raises(self):
        with self.assertRaises(ValueError):
            decide_overlength(prompt_len=1, max_prompt_tokens=0, policy=OverlengthPolicy.REJECT)

    def test_invalid_max_new_tokens_raises(self):
        with self.assertRaises(ValueError):
            decide_overlength(
                prompt_len=1, max_prompt_tokens=10, response_len=1, max_new_tokens=0, policy=OverlengthPolicy.REJECT
            )

    def test_invalid_policy_type_raises(self):
        with self.assertRaises(ValueError):
            decide_overlength(prompt_len=1, max_prompt_tokens=10, policy="reject")  # type: ignore[arg-type]

    def test_decision_is_deterministic(self):
        kwargs = dict(prompt_len=25, max_prompt_tokens=20, response_len=60, max_new_tokens=50)
        first = decide_overlength(policy=OverlengthPolicy.REJECT, **kwargs)
        for _ in range(5):
            self.assertEqual(
                decide_overlength(policy=OverlengthPolicy.REJECT, **kwargs),
                first,
            )


class PromptBatchCountersTest(unittest.TestCase):
    """PromptBatch exposes the new per-reason counter field with a safe default."""

    def test_default_overlength_counters_is_empty_dict(self):
        batch = PromptBatch(items=[], scanned=0, skipped_long=0, total_skipped_long=0)
        self.assertEqual(batch.overlength_counters, {})

    def test_each_batch_gets_independent_default_dict(self):
        # default_factory must give each instance its own dict, not a shared one.
        batch_a = PromptBatch(items=[], scanned=0, skipped_long=0, total_skipped_long=0)
        batch_b = PromptBatch(items=[], scanned=0, skipped_long=0, total_skipped_long=0)
        batch_a.overlength_counters["response_too_long/reject"] = 1
        self.assertEqual(batch_b.overlength_counters, {})

    def test_legacy_fields_still_accepted(self):
        item = PromptItem(prompt="hi", solutions=None, input_tokens=[1, 2], record={})
        batch = PromptBatch(items=[item], scanned=1, skipped_long=0, total_skipped_long=0)
        self.assertEqual(batch.skipped_long, 0)
        self.assertEqual(batch.items, [item])


class OverlengthMetricsTest(unittest.TestCase):
    """record_training_stats emits one scalar per (reason, action) counter."""

    def test_overlength_counters_emitted_as_scalars(self):
        writer = _FakeWriter()
        stats = init_rollout_stats(
            skipped_long=2,
            total_skipped_long=5,
            overlength_counters={"response_too_long/reject": 3, "single_message_oversized/warn": 1},
        )
        record_training_stats(writer, stats, step=0, train_res={}, train_batch=[])

        tags = {call[0] for call in writer.scalars}
        self.assertIn("rollout/overlength/response_too_long/reject", tags)
        self.assertIn("rollout/overlength/single_message_oversized/warn", tags)
        self.assertIn("rollout/skipped_long", tags)
        self.assertIn("rollout/total_skipped_long", tags)

    def test_empty_overlength_counters_emits_nothing_extra(self):
        writer = _FakeWriter()
        stats = init_rollout_stats()
        record_training_stats(writer, stats, step=0, train_res={}, train_batch=[])
        overlength_tags = [call[0] for call in writer.scalars if call[0].startswith("rollout/overlength/")]
        self.assertEqual(overlength_tags, [])

    def test_collect_train_batch_stats_forwards_overlength_counters(self):
        # The agentic path populates counters on the trainer and hands them to
        # `collect_train_batch_stats`, which must inject them into the stats
        # accumulator so `record_training_stats` emits the scalars.
        from areno.api.metrics import collect_train_batch_stats

        counters = {"trajectory_too_long/warn": 2}
        stats = collect_train_batch_stats([], overlength_counters=counters)
        self.assertEqual(stats["overlength_counters"], counters)
        writer = _FakeWriter()
        record_training_stats(writer, stats, step=0, train_res={}, train_batch=[])
        tags = {call[0] for call in writer.scalars}
        self.assertIn("rollout/overlength/trajectory_too_long/warn", tags)

    def test_collect_train_batch_stats_without_counters_is_empty(self):
        from areno.api.metrics import collect_train_batch_stats

        stats = collect_train_batch_stats([])
        self.assertEqual(stats["overlength_counters"], {})


class _FakeWriter:
    """Minimal TensorBoard writer substitute that records add_scalar calls."""

    def __init__(self):
        self.scalars: list[tuple] = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass


class TruncateSftResponseTest(unittest.TestCase):
    """SFT response truncation preserves EOS and keeps mask length consistent."""

    def test_short_response_returned_unchanged(self):
        prompt_ids = [1, 2, 3]
        response_ids = [4, 5, 7]  # 7 is EOS
        tokens, mask, truncated = truncate_sft_response(
            prompt_ids=prompt_ids, response_ids=response_ids, max_new_tokens=10, eos_token_ids=(7,)
        )
        self.assertFalse(truncated)
        self.assertEqual(tokens, [1, 2, 3, 4, 5, 7])
        self.assertEqual(mask, [True, True, True, False, False, False])

    def test_long_response_cut_to_limit_with_eos_reappended(self):
        prompt_ids = [1, 2]
        response_ids = [4, 5, 6, 8, 9]  # last token 9 is not EOS
        tokens, mask, truncated = truncate_sft_response(
            prompt_ids=prompt_ids, response_ids=response_ids, max_new_tokens=3, eos_token_ids=(7,)
        )
        self.assertTrue(truncated)
        # cut = [4,5,6] then EOS re-appended -> [4,5,6,7]
        self.assertEqual(tokens, [1, 2, 4, 5, 6, 7])
        self.assertEqual(mask, [True, True, False, False, False, False])
        self.assertEqual(len(tokens), len(mask))

    def test_cut_keeps_existing_eos_at_boundary(self):
        prompt_ids = [1]
        response_ids = [4, 5, 7, 8, 9]  # 7 is EOS at position 2
        tokens, mask, truncated = truncate_sft_response(
            prompt_ids=prompt_ids, response_ids=response_ids, max_new_tokens=3, eos_token_ids=(7,)
        )
        # cut = [4,5,7], last is EOS -> no re-append
        self.assertTrue(truncated)
        self.assertEqual(tokens, [1, 4, 5, 7])
        self.assertEqual(tokens[-1], 7)

    def test_no_eos_ids_no_reappend(self):
        prompt_ids = [1]
        response_ids = [4, 5, 6, 8]
        tokens, mask, truncated = truncate_sft_response(
            prompt_ids=prompt_ids, response_ids=response_ids, max_new_tokens=2, eos_token_ids=None
        )
        self.assertTrue(truncated)
        self.assertEqual(tokens, [1, 4, 5])

    def test_invalid_max_new_tokens_raises(self):
        with self.assertRaises(ValueError):
            truncate_sft_response(prompt_ids=[1], response_ids=[2], max_new_tokens=0, eos_token_ids=(7,))

    def test_prompt_mask_length_matches_tokens(self):
        # mask length must always equal tokens length (backend asserts this).
        prompt_ids = [1, 2, 3, 4]
        response_ids = list(range(10, 40))
        tokens, mask, _ = truncate_sft_response(
            prompt_ids=prompt_ids, response_ids=response_ids, max_new_tokens=5, eos_token_ids=(99,)
        )
        self.assertEqual(len(tokens), len(mask))
        self.assertEqual(sum(1 for m in mask if m), len(prompt_ids))


class TruncateDpoPairTest(unittest.TestCase):
    """DPO pair truncation keeps chosen/rejected comparable over the shared prefix."""

    def test_both_fit_returned_unchanged(self):
        # prefix=[1,2], chosen suffix=[3,7], rejected suffix=[4,7]; 7 is EOS.
        chosen = [1, 2, 3, 7]
        cmask = [True, True, False, False]
        rejected = [1, 2, 4, 7]
        rmask = [True, True, False, False]
        result = truncate_dpo_pair(
            chosen_tokens=chosen,
            chosen_mask=cmask,
            rejected_tokens=rejected,
            rejected_mask=rmask,
            prefix_len=2,
            max_seq_len=10,
            eos_token_ids=(7,),
        )
        self.assertIsNotNone(result)
        ct, cm, rt, rm, truncated = result
        self.assertFalse(truncated)
        self.assertEqual(ct, chosen)
        self.assertEqual(rt, rejected)

    def test_chosen_overlong_both_truncated_to_common_budget(self):
        # prefix=[1,2]; chosen suffix long, rejected short. max_seq_len=5.
        chosen = [1, 2, 3, 4, 5, 6, 7]  # len 7 > 5
        cmask = [True, True, False, False, False, False, False]
        rejected = [1, 2, 8, 7]  # len 4 <= 5
        rmask = [True, True, False, False]
        result = truncate_dpo_pair(
            chosen_tokens=chosen,
            chosen_mask=cmask,
            rejected_tokens=rejected,
            rejected_mask=rmask,
            prefix_len=2,
            max_seq_len=5,
            eos_token_ids=(7,),
        )
        self.assertIsNotNone(result)
        ct, cm, rt, rm, truncated = result
        self.assertTrue(truncated)
        # chosen cut to budget=3 suffix -> [1,2,?,7], last replaced with EOS
        self.assertLessEqual(len(ct), 5)
        self.assertEqual(ct[:2], [1, 2])  # prefix preserved
        self.assertEqual(ct[-1], 7)  # EOS preserved/re-appended
        # rejected already fit; unchanged
        self.assertEqual(rt, rejected)
        # comparability: shared prefix identical
        self.assertEqual(ct[:2], rt[:2])

    def test_both_overlong_truncated_comparable(self):
        chosen = [1, 2, 3, 4, 5, 6, 7]
        cmask = [True, True, False, False, False, False, False]
        rejected = [1, 2, 8, 9, 10, 11, 7]
        rmask = [True, True, False, False, False, False, False]
        result = truncate_dpo_pair(
            chosen_tokens=chosen,
            chosen_mask=cmask,
            rejected_tokens=rejected,
            rejected_mask=rmask,
            prefix_len=2,
            max_seq_len=4,
            eos_token_ids=(7,),
        )
        self.assertIsNotNone(result)
        ct, cm, rt, rm, truncated = result
        self.assertTrue(truncated)
        self.assertLessEqual(len(ct), 4)
        self.assertLessEqual(len(rt), 4)
        self.assertEqual(ct[:2], rt[:2])  # shared prefix identical -> comparable
        self.assertEqual(ct[-1], 7)
        self.assertEqual(rt[-1], 7)
        self.assertEqual(len(ct), len(cm))
        self.assertEqual(len(rt), len(rm))

    def test_too_small_to_fit_returns_none(self):
        # prefix=4, max_seq_len=4 -> budget=0, cannot keep any response token.
        chosen = [1, 2, 3, 4, 5, 6, 7]
        cmask = [True, True, True, True, False, False, False]
        rejected = [1, 2, 3, 4, 8, 7]
        rmask = [True, True, True, True, False, False]
        result = truncate_dpo_pair(
            chosen_tokens=chosen,
            chosen_mask=cmask,
            rejected_tokens=rejected,
            rejected_mask=rmask,
            prefix_len=4,
            max_seq_len=4,
            eos_token_ids=(7,),
        )
        self.assertIsNone(result)

    def test_invalid_max_seq_len_raises(self):
        with self.assertRaises(ValueError):
            truncate_dpo_pair(
                chosen_tokens=[1],
                chosen_mask=[True],
                rejected_tokens=[1],
                rejected_mask=[True],
                prefix_len=0,
                max_seq_len=0,
                eos_token_ids=(7,),
            )

    def test_invalid_prefix_len_raises(self):
        with self.assertRaises(ValueError):
            truncate_dpo_pair(
                chosen_tokens=[1, 2],
                chosen_mask=[True, False],
                rejected_tokens=[1],
                rejected_mask=[True],
                prefix_len=2,
                max_seq_len=5,
                eos_token_ids=(7,),
            )


class AgentOverlengthFilterTest(unittest.TestCase):
    """Agentic overlength filter: reject drops, warn keeps, truncate degrades to reject.

    The real `_filter_overlong_agent_samples` lives on `PolicyOnlyTrainer` and
    needs a backend; here we drive the method on a tiny fake trainer + fake ctx
    so the policy branching is covered without a GPU.
    """

    def _make_fake_sample(self, token_len: int):
        from types import SimpleNamespace

        return SimpleNamespace(
            messages=[{"role": "assistant", "content": "x"}],
            item=SimpleNamespace(prompt="p", prompt_index=0, sample_index=0, input_tokens=[1] * token_len),
            response_tokens=[1] * token_len,
            response_logprobs=[0.0] * token_len,
            trace=[],
            response_kind="assistant_text",
            last_tool_calls=[],
            token_row=[1] * token_len,
            response_mask_row=[True] * token_len,
            loss_mask_row=[True] * token_len,
            rollout_logprobs_row=[0.0] * token_len,
        )

    def _make_fake_trainer(self, policy: str):
        import logging
        from types import SimpleNamespace

        trainer = SimpleNamespace()
        trainer.config = SimpleNamespace(overlength_policy=policy, max_context_len=10)
        trainer.logger = logging.getLogger("test")
        # Bind the real method under test.
        from areno.api.trainers.policy_only import PolicyOnlyTrainer

        trainer._filter_overlong_agent_samples = PolicyOnlyTrainer._filter_overlong_agent_samples.__get__(trainer)
        trainer._agent_sample_filter_detail = PolicyOnlyTrainer._agent_sample_filter_detail.__get__(trainer)
        trainer._agent_filter_diagnostics = PolicyOnlyTrainer._agent_filter_diagnostics.__get__(trainer)
        trainer._format_agent_filter_diagnostics = PolicyOnlyTrainer._format_agent_filter_diagnostics.__get__(trainer)
        trainer._percentile_value = PolicyOnlyTrainer._percentile_value.__get__(trainer)
        trainer._agent_model_context_len = lambda: 10
        return trainer

    class _FakeCtx:
        def _train_rows_from_samples(self, samples):
            from types import SimpleNamespace

            row = samples[0].token_row
            return SimpleNamespace(
                token_rows=[list(row)],
                response_masks=[[]],
                loss_masks=[[]],
                rollout_logprobs=[[]],
                total_tokens=len(row),
            )

    def test_reject_drops_overlong(self):
        trainer = self._make_fake_trainer("reject")
        samples = [self._make_fake_sample(5), self._make_fake_sample(20)]  # 20 > 10
        kept, dropped, diag = trainer._filter_overlong_agent_samples(self._FakeCtx(), samples, None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(diag["overlength_counters"]["trajectory_too_long/reject"], 1)

    def test_warn_keeps_overlong(self):
        trainer = self._make_fake_trainer("warn")
        samples = [self._make_fake_sample(5), self._make_fake_sample(20)]
        kept, dropped, diag = trainer._filter_overlong_agent_samples(self._FakeCtx(), samples, None)
        # warn keeps both; dropped stays 0 to preserve caller invariant.
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)
        self.assertEqual(diag["overlength_counters"]["trajectory_too_long/warn"], 1)

    def test_truncate_degrades_to_reject(self):
        # truncate is not implemented for agentic yet; it must degrade to reject
        # (drop the overlong sample) rather than split a tool call/result pair.
        trainer = self._make_fake_trainer("truncate")
        samples = [self._make_fake_sample(5), self._make_fake_sample(20)]
        kept, dropped, diag = trainer._filter_overlong_agent_samples(self._FakeCtx(), samples, None)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(diag["overlength_counters"]["trajectory_too_long/truncate"], 1)

    def test_all_within_budget_no_counters(self):
        trainer = self._make_fake_trainer("reject")
        samples = [self._make_fake_sample(5), self._make_fake_sample(8)]
        kept, dropped, diag = trainer._filter_overlong_agent_samples(self._FakeCtx(), samples, None)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)
        self.assertNotIn("overlength_counters", diag)


class OverlengthConfigTest(unittest.TestCase):
    """TrainerConfig exposes overlength_policy with a safe default and validation."""

    def test_default_policy_is_reject(self):
        from areno.api.trainer_config import TrainerConfig

        config = TrainerConfig(algo="sft", ckpt="x", dataset_path="x")
        self.assertEqual(config.overlength_policy, "reject")

    def test_valid_policies_accepted(self):
        from areno.api.trainer_config import TrainerConfig

        for policy in ("reject", "warn", "truncate"):
            config = TrainerConfig(algo="sft", ckpt="x", dataset_path="x", overlength_policy=policy)
            self.assertEqual(config.overlength_policy, policy)

    def test_invalid_policy_rejected(self):
        from areno.api.trainer_config import TrainerConfig

        with self.assertRaises(ValueError):
            TrainerConfig(algo="sft", ckpt="x", dataset_path="x", overlength_policy="bogus")


if __name__ == "__main__":
    unittest.main()
