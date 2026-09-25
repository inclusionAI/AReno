"""CPU tests for experimental grouped-softmax classification training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from areno.api.backend.cuda.training import make_train_pack
from areno.api.models import TrainSequence
from areno.engine.config import OptimizerConfig, RuntimeConfig
from areno.engine.runtime.common import split_data_pack_by_dp
from areno.engine.score_head import (
    SCORE_HEAD_FILENAME,
    attach_score_head,
    build_score_head,
    packed_sequence_scores,
    save_score_head,
)
from areno.experimental.classify.config import ClassifyTrainerConfig
from areno.experimental.classify.loss import classify_loss_fn, grouped_softmax_loss
from areno.experimental.classify.trainer import EncodedQuestion, build_step_rows, encode_question


def _reference_question_loss(logits, target, brier_weight):
    """JevForge `question_loss` for one question."""

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = logits.softmax(dim=-1)
    return -(target * log_probs).sum() + brier_weight * ((probs - target) ** 2).sum()


class GroupedSoftmaxLossTest(unittest.TestCase):
    def test_matches_per_question_reference_and_ignores_padding(self):
        torch.manual_seed(0)
        sizes = [3, 2, 4]
        targets = [torch.tensor([0.2, 0.5, 0.3]), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0, 0.0])]
        scores = torch.randn(sum(sizes) + 2, requires_grad=True)
        # Interleave question rows and put two padding rows in the middle.
        group = torch.tensor([0, 1, 0, -1, 2, 2, 1, 0, -1, 2, 2], dtype=torch.float32)
        target = torch.zeros(11)
        cursor = {0: 0, 1: 0, 2: 0}
        for row, gid in enumerate(group.long().tolist()):
            if gid >= 0:
                target[row] = targets[gid][cursor[gid]]
                cursor[gid] += 1
        weight = torch.full((11,), 1.0 / 3.0)

        loss, metrics = grouped_softmax_loss(scores, group, target, weight, brier_weight=0.5)
        loss.backward()
        grad = scores.grad.clone()

        ref_scores = scores.detach().clone().requires_grad_(True)
        reference = 0.0
        for gid in range(3):
            rows = (group == gid).nonzero().flatten()
            reference = reference + _reference_question_loss(ref_scores[rows], targets[gid], 0.5)
        (reference / 3.0).backward()

        torch.testing.assert_close(loss.detach(), reference.detach() / 3.0)
        torch.testing.assert_close(grad, ref_scores.grad)
        self.assertEqual(float(grad[3]), 0.0)
        self.assertEqual(float(grad[8]), 0.0)
        self.assertEqual(float(metrics["classify_questions_per_rank"]), 3.0)

    def test_all_padding_rank_returns_connected_zero(self):
        scores = torch.randn(4, requires_grad=True)
        group = torch.full((4,), -1.0)
        loss, _ = grouped_softmax_loss(scores, group, torch.zeros(4), torch.ones(4), brier_weight=0.5)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(scores.grad, torch.zeros(4)))

    def test_loss_fn_reads_sequence_labels(self):
        pack = {
            "sequence_labels": {
                "group": torch.tensor([0.0, 0.0]),
                "target": torch.tensor([1.0, 0.0]),
                "weight": torch.tensor([1.0, 1.0]),
            }
        }
        loss, metrics = classify_loss_fn(pack, torch.tensor([2.0, 0.0]), brier_weight=0.0)
        torch.testing.assert_close(loss, -torch.log_softmax(torch.tensor([2.0, 0.0]), dim=0)[0])
        self.assertEqual(float(metrics["classify_top1"]), 1.0)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]


class StepLayoutTest(unittest.TestCase):
    def _questions(self):
        return [
            EncodedQuestion(leaves=[[1] * (5 + q), [2] * (3 + q), [3] * 4][: 2 + q % 2], target=[])
            for q in range(7)
        ]

    def test_questions_stay_whole_per_rank_and_microbatch(self):
        questions = self._questions()
        for question in questions:
            question.target = [1.0 / len(question.leaves)] * len(question.leaves)
        dp_size = 2
        rows, mini_bs = build_step_rows(questions, dp_size=dp_size, microbatch_tokens=20, pad_token_id=0)
        self.assertEqual(len(rows) % mini_bs, 0)
        self.assertEqual(mini_bs % dp_size, 0)
        num_micro = len(rows) // mini_bs
        owner = {}
        seen_rows = 0
        for micro in range(num_micro):
            chunk = rows[micro * mini_bs : (micro + 1) * mini_bs]
            for rank in range(dp_size):
                for row in chunk[rank::dp_size]:
                    gid = int(row.sequence_labels["group"])
                    if gid < 0:
                        self.assertEqual(len(row.tokens), 2)
                        continue
                    seen_rows += 1
                    self.assertEqual(owner.setdefault(gid, (micro, rank)), (micro, rank))
                    self.assertAlmostEqual(row.sequence_labels["weight"], dp_size * num_micro / len(questions))
        self.assertEqual(set(owner), set(range(len(questions))))
        self.assertEqual(seen_rows, sum(len(q.leaves) for q in questions))

    def test_pack_labels_follow_dp_split(self):
        questions = self._questions()
        for question in questions:
            question.target = [1.0 / len(question.leaves)] * len(question.leaves)
        rows, mini_bs = build_step_rows(questions, dp_size=2, microbatch_tokens=1000, pad_token_id=0)
        pack = make_train_pack(rows[:mini_bs])
        labels = pack["sequence_labels"]
        self.assertEqual(set(labels), {"group", "target", "weight"})
        self.assertEqual(labels["group"].shape, (mini_bs,))
        for rank, shard in enumerate(split_data_pack_by_dp(pack, 2)):
            expected = [row.sequence_labels["group"] for row in rows[:mini_bs][rank::2]]
            self.assertEqual(shard["sequence_labels"]["group"].tolist(), expected)
            self.assertEqual(shard["input_ids"].shape[0], len(expected))

    def test_mixed_label_keys_are_rejected(self):
        rows = [
            TrainSequence(tokens=[1, 2], prompt_mask=[True, True], logprobs=[0.0, 0.0], advantages=[0.0, 0.0]),
            TrainSequence(
                tokens=[1, 2],
                prompt_mask=[True, True],
                logprobs=[0.0, 0.0],
                advantages=[0.0, 0.0],
                sequence_labels={"group": 0.0},
            ),
        ]
        with self.assertRaisesRegex(ValueError, "sequence_labels"):
            make_train_pack(rows)

    def test_encode_question_prefix_plus_candidate(self):
        question = encode_question({"prompt": "ab", "candidates": ["c", "de"], "target": [0.25, 0.75]}, _Tokenizer())
        self.assertEqual(question.leaves, [[97, 98, 99], [97, 98, 100, 101]])
        with self.assertRaisesRegex(ValueError, "probability"):
            encode_question({"prompt": "a", "candidates": ["b", "c"], "target": [0.5, 0.6]}, _Tokenizer())


class _DeferModel(nn.Module):
    def forward(self, input_ids, defer_lm_head=False):
        del input_ids, defer_lm_head


class ScoreHeadTest(unittest.TestCase):
    def test_head_init_is_seeded_and_roundtrips(self):
        first = build_score_head(8)
        second = build_score_head(8)
        for a, b in zip(first.parameters(), second.parameters(), strict=True):
            self.assertTrue(torch.equal(a, b))
        with tempfile.TemporaryDirectory() as tmp:
            with torch.no_grad():
                first[2].bias.fill_(3.0)
            save_score_head(first, tmp)
            model = _DeferModel()
            head = attach_score_head(
                model, hidden_size=8, dtype=torch.float32, device=torch.device("cpu"), model_path=tmp
            )
            self.assertIs(model.score_head, head)
            self.assertEqual(float(head[2].bias), 3.0)
            self.assertEqual(head[0].weight._areno_lr_group, "score_head")
            self.assertTrue(Path(tmp, SCORE_HEAD_FILENAME).is_file())

    def test_attach_rejects_models_without_defer_lm_head(self):
        class Plain(nn.Module):
            def forward(self, input_ids):
                return input_ids

        with self.assertRaisesRegex(ValueError, "defer_lm_head"):
            attach_score_head(Plain(), hidden_size=4, dtype=torch.float32, device=torch.device("cpu"), model_path=None)

    def test_packed_scores_read_last_token_per_sequence(self):
        head = build_score_head(4)
        hidden = torch.randn(1, 9, 4)
        cu_seqlens = torch.tensor([0, 3, 7, 8, 9], dtype=torch.int32)  # last entry is TP padding
        scores = packed_sequence_scores(head, hidden, cu_seqlens, 3, sequence_parallel=False)
        expected = head(hidden[0, [2, 6, 7]]).squeeze(-1)
        torch.testing.assert_close(scores, expected)


class ClassifyConfigTest(unittest.TestCase):
    def test_backend_config_enables_score_head(self):
        config = ClassifyTrainerConfig(
            algo="classify",
            ckpt="unused",
            dataset_path="unused",
            backend="cuda",
            score_head_lr=1e-3,
            score_head_warmup_steps=5,
        )
        cuda = config.cuda_config()
        self.assertTrue(RuntimeConfig(**cuda.runtime).score_head)
        optimizer = OptimizerConfig(**cuda.optimizer)
        self.assertEqual(optimizer.score_head_lr, 1e-3)
        self.assertEqual(optimizer.score_head_warmup_steps, 5)

    def test_registered_loss_binds_brier_weight(self):
        from areno.api.algorithms import get_algorithm

        config = ClassifyTrainerConfig(
            algo="classify", ckpt="unused", dataset_path="unused", backend="cuda", brier_weight=0.25
        )
        spec = get_algorithm("classify")
        self.assertFalse(spec.requires_rollout)
        loss_fn = spec.make_loss_fn(config)
        self.assertIs(loss_fn.func, classify_loss_fn)
        self.assertEqual(loss_fn.keywords, {"brier_weight": 0.25})


if __name__ == "__main__":
    unittest.main()
