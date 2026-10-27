"""CPU tests for the CUDA JEV decision path (loss, config, score head)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from areno.api.backend.cuda.decision import jev_loss
from areno.api.backend.cuda.training import is_jev_loss_fn, jev_pack_loss, make_train_pack
from areno.api.jev.config import JevCudaConfig, JevTrainerConfig
from areno.api.jev.data import encode_question
from areno.api.models import TrainSequence
from areno.engine.config import OptimizerConfig, RuntimeConfig
from areno.engine.score_head import (
    SCORE_HEAD_FILENAME,
    attach_score_head,
    build_score_head,
    packed_sequence_scores,
    save_score_head,
)


def _question_labels(questions: list[tuple[int, int, list[float]]]) -> dict:
    """Build `sequence_labels` for `(group, size, target)` tuples."""

    groups, targets, weights = [], [], []
    total = len(questions)
    for group_id, size, target in questions:
        for probability in target:
            groups.append(float(group_id))
            targets.append(float(probability))
            weights.append(1.0 / total)
    return {
        "group": torch.tensor(groups, dtype=torch.float32),
        "target": torch.tensor(targets, dtype=torch.float32),
        "weight": torch.tensor(weights, dtype=torch.float32),
    }


def _reference_question_loss(logits, target, brier_weight):
    """JevForge `question_loss` for one question."""

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = logits.softmax(dim=-1)
    return -(target * log_probs).sum() + brier_weight * ((probs - target) ** 2).sum()


class JevLossTest(unittest.TestCase):
    def test_matches_per_question_reference(self):
        torch.manual_seed(0)
        questions = [
            (0, 3, [0.2, 0.5, 0.3]),
            (1, 2, [1.0, 0.0]),
            (2, 4, [0.0, 0.0, 1.0, 0.0]),
        ]
        labels = _question_labels(questions)
        scores = torch.randn(labels["group"].numel(), requires_grad=True)
        pack = {"sequence_labels": labels, "_loss_settings": {"brier_weight": 0.5}}

        loss, metrics = jev_loss(pack, scores)
        loss.backward()
        grad = scores.grad.clone()

        ref_scores = scores.detach().clone().requires_grad_(True)
        reference = 0.0
        cursor = 0
        for _, size, target in questions:
            rows = ref_scores[cursor : cursor + size]
            reference = reference + _reference_question_loss(rows, torch.tensor(target), 0.5)
            cursor += size
        # One weight per question of 1/len(questions) makes the weighted sum the
        # per-question mean.
        (reference / len(questions)).backward()

        torch.testing.assert_close(loss.detach(), reference.detach() / len(questions))
        torch.testing.assert_close(grad, ref_scores.grad)
        self.assertEqual(float(metrics["jev_questions"]), 3.0)

    def test_brier_weight_defaults_and_overrides(self):
        labels = _question_labels([(0, 2, [1.0, 0.0])])
        scores = torch.tensor([2.0, 0.0])
        default_loss, _ = jev_loss({"sequence_labels": labels}, scores)
        weighted_loss, _ = jev_loss(
            {"sequence_labels": labels, "_loss_settings": {"brier_weight": 0.0}}, scores
        )
        manual = _reference_question_loss(scores, torch.tensor([1.0, 0.0]), 0.5)
        torch.testing.assert_close(default_loss, manual)
        torch.testing.assert_close(weighted_loss, _reference_question_loss(scores, torch.tensor([1.0, 0.0]), 0.0))

    def test_rejects_missing_labels(self):
        with self.assertRaisesRegex(ValueError, "sequence_labels"):
            jev_loss({}, torch.randn(2))

    def test_registered_loss_is_recognized_and_bound(self):
        from areno.api.algorithms import get_algorithm

        spec = get_algorithm("jev")
        loss_fn = spec.make_loss_fn(
            JevTrainerConfig(algo="jev", ckpt="unused", dataset_path="unused", backend="cuda", save_path="unused")
        )
        self.assertTrue(is_jev_loss_fn(loss_fn))
        self.assertFalse(is_jev_loss_fn(lambda pack, scores: None))
        pack = {"sequence_labels": _question_labels([(0, 2, [1.0, 0.0])])}
        loss, _ = jev_pack_loss(loss_fn)(pack, torch.tensor([2.0, 0.0]))
        torch.testing.assert_close(loss, _reference_question_loss(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]), 0.5))


class JevPackLayoutTest(unittest.TestCase):
    def test_question_rows_keep_labels(self):
        rows = [
            TrainSequence(
                tokens=[1, 2, 3],
                trajectory_id=0,
                sequence_labels={"group": 0.0, "target": 1.0, "weight": 0.5},
            ),
            TrainSequence(
                tokens=[1, 2, 3, 4],
                trajectory_id=0,
                sequence_labels={"group": 0.0, "target": 0.0, "weight": 0.5},
            ),
            TrainSequence(
                tokens=[1, 2],
                trajectory_id=1,
                sequence_labels={"group": 1.0, "target": 1.0, "weight": 0.5},
            ),
        ]
        pack = make_train_pack(rows)
        self.assertEqual(set(pack["sequence_labels"]), {"group", "target", "weight"})
        self.assertEqual(pack["sequence_labels"]["group"].tolist(), [0.0, 0.0, 1.0])
        self.assertNotIn("features", pack)
        # Candidate rows share their question's trajectory id so the engine's
        # grouped DP split keeps each question's candidates together.
        self.assertEqual(pack["loss_sequence_ids"].tolist(), [0, 0, 1])

    def test_encode_question_stamps_group_labels(self):
        class _Tokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return [ord(char) for char in text]

        question = {"prompt": "ab", "candidates": ["c", "de"], "target": [0.25, 0.75], "question_id": "q"}
        rows = encode_question(question, _Tokenizer(), 64, group=3, weight=0.25)
        self.assertEqual([row.trajectory_id for row in rows], [3, 3])
        self.assertEqual([row.sequence_labels["group"] for row in rows], [3.0, 3.0])
        self.assertEqual([row.sequence_labels["target"] for row in rows], [0.25, 0.75])
        self.assertEqual([row.sequence_labels["weight"] for row in rows], [0.25, 0.25])
        pack = make_train_pack(rows)
        self.assertEqual(pack["sequence_labels"]["target"].tolist(), [0.25, 0.75])
        self.assertEqual(pack["loss_sequence_ids"].tolist(), [0, 0])


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
            self.assertEqual(float(head[2].bias.detach()), 3.0)
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


class JevCudaConfigTest(unittest.TestCase):
    def test_cuda_config_enables_score_head(self):
        config = JevTrainerConfig(
            algo="jev",
            ckpt="unused",
            dataset_path="unused",
            backend="cuda",
            save_path="unused",
            head_lr=1e-3,
            score_head_warmup_steps=5,
        )
        cuda = config.cuda_config()
        self.assertTrue(RuntimeConfig(**cuda.runtime).score_head)
        optimizer = OptimizerConfig(**cuda.optimizer)
        self.assertEqual(optimizer.score_head_lr, 1e-3)
        self.assertEqual(optimizer.score_head_warmup_steps, 5)

    def test_head_lr_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "head_lr"):
            JevCudaConfig(head_lr=0.0)

    def test_direct_construction_runs_base_post_init(self):
        # `JevCudaConfig` subclasses a slots dataclass, whose class-object
        # rewrite breaks a zero-arg `super()`; constructing it directly must
        # still run the base validation and stamp the score-head runtime keys.
        config = JevCudaConfig(head_lr=2e-4)
        self.assertTrue(config.runtime["score_head"])
        self.assertEqual(config.optimizer["score_head_lr"], 2e-4)
        from areno.adapters.config import LoraConfig

        with self.assertRaisesRegex(ValueError, "matching train and rollout TP"):
            JevCudaConfig(lora=LoraConfig(qlora=True), tp_size=2, rollout_tp_size=4)

    def test_cuda_trainer_config_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            JevTrainerConfig(algo="jev", ckpt="unused", dataset_path="unused", backend="npu")


if __name__ == "__main__":
    unittest.main()