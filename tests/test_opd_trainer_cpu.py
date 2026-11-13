from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from areno.api.trainers.opd import OPDTrainer


class _Logger:
    def info(self, *args, **kwargs):
        del args, kwargs


class FakeAreno:
    """Minimal stand-in for areno.api.Trainer; no GPU or model is created."""

    def __init__(self, teacher_rows=None, *, role_rows=None):
        self.teacher_rows = teacher_rows
        self.role_rows = role_rows or {}
        self.calls: list[tuple[str, list, dict]] = []

    def score_logprobs(self, role, token_rows, **kwargs):
        self.calls.append((role, [list(row) for row in token_rows], kwargs))
        rows = self.role_rows.get(role)
        if rows is None and role == "teacher":
            rows = self.teacher_rows
        if rows is None:
            # Auto-aligned fallback: one initial zero plus a constant score.
            rows = [[0.0] + [-1.0] * (len(row) - 1) for row in token_rows]
        return [list(row) for row in rows]

    def get_tokenizer(self):
        return SimpleNamespace(
            eos_token_id=0,
            encode=lambda text, add_special_tokens=True: [ord(char) % 20 for char in text] or [1],
            batch_decode=lambda token_lists: ["answer 42" for _ in token_lists],
        )


def _item(tokens=(1, 2, 3)):
    return SimpleNamespace(
        input_tokens=list(tokens),
        record={"features": None},
        prompt="p",
        solutions=None,
    )


def _seq(tokens=(10, 11), logprobs=(-0.5, -1.0)):
    return SimpleNamespace(
        resp_tokens=list(tokens),
        resp_logprobs=list(logprobs),
        routed_experts=None,
    )


def _prompt_batch():
    return SimpleNamespace(items=[_item()])


def _rollout():
    return [SimpleNamespace(sequences=[_seq()])]


def _tokenizer():
    return SimpleNamespace(eos_token_id=0)


def _trainer(areno):
    trainer = object.__new__(OPDTrainer)
    trainer.areno = areno
    trainer.config = SimpleNamespace(
        opd_teacher_micro_bs=None,
        teacher_ckpt="teacher",
        model_hub="modelscope",
    )
    trainer._last_opd_stats = {}
    trainer.logger = _Logger()
    return trainer


class OpdTrainerTest(unittest.TestCase):
    def _rows(self):
        # prompt (3) + response (2) = 5 positions, including the initial zero.
        return [[0.0, -0.1, -0.2, -0.6, -0.9]]

    def test_teacher_scores_are_written_to_reference_slot(self):
        areno = FakeAreno(self._rows())
        trainer = _trainer(areno)

        batch, rewards, rollout_logprobs = trainer._materialize_train_batch(_tokenizer(), _prompt_batch(), _rollout())

        self.assertEqual(rewards, [])
        self.assertEqual(rollout_logprobs, [-0.5, -1.0])
        self.assertEqual(len(batch), 1)
        row = batch[0]
        self.assertEqual(row.tokens, [1, 2, 3, 10, 11])
        self.assertEqual(row.prompt_len, 3)
        self.assertEqual(row.logprobs, [0.0, 0.0, 0.0, -0.5, -1.0])
        self.assertEqual(row.ref_logprobs, [0.0, -0.1, -0.2, -0.6, -0.9])
        self.assertEqual(row.scalar_advantage, 0.0)
        self.assertEqual(areno.calls[0][0], "teacher")

    def test_teacher_scores_every_step_not_reused(self):
        areno = FakeAreno(self._rows())
        trainer = _trainer(areno)

        trainer._materialize_train_batch(_tokenizer(), _prompt_batch(), _rollout())
        trainer._materialize_train_batch(_tokenizer(), _prompt_batch(), _rollout())

        teacher_calls = [call for call in areno.calls if call[0] == "teacher"]
        self.assertEqual(len(teacher_calls), 2)

    def test_empty_rollout_skips_teacher_forward(self):
        areno = FakeAreno([])
        trainer = _trainer(areno)

        batch, rewards, stats = trainer._materialize_train_batch(
            _tokenizer(), _prompt_batch(), [SimpleNamespace(sequences=[])]
        )

        self.assertEqual((batch, rewards, stats), ([], [], []))
        self.assertEqual(areno.calls, [])

    def test_row_count_mismatch_raises(self):
        areno = FakeAreno([])
        trainer = _trainer(areno)

        with self.assertRaisesRegex(ValueError, "different number of logprob rows"):
            trainer._materialize_train_batch(_tokenizer(), _prompt_batch(), _rollout())

    def test_token_length_mismatch_raises(self):
        areno = FakeAreno([[0.0, -1.0]])
        trainer = _trainer(areno)

        with self.assertRaisesRegex(ValueError, "does not match token length"):
            trainer._materialize_train_batch(_tokenizer(), _prompt_batch(), _rollout())

    def test_non_finite_teacher_score_raises(self):
        areno = FakeAreno([[0.0, -0.1, -0.2, float("nan"), -0.9]])
        trainer = _trainer(areno)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            trainer._materialize_train_batch(_tokenizer(), _prompt_batch(), _rollout())

    def test_non_finite_teacher_prompt_score_raises(self):
        areno = FakeAreno([[0.0, float("nan"), -0.2, -0.3, -0.9]])
        trainer = _trainer(areno)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            trainer._materialize_train_batch(_tokenizer(), _prompt_batch(), _rollout())

    def test_misaligned_rollout_logprobs_raises(self):
        areno = FakeAreno(self._rows())
        trainer = _trainer(areno)
        rollout = [SimpleNamespace(sequences=[_seq(logprobs=(-0.5,))])]

        with self.assertRaisesRegex(ValueError, "rollout logprobs must align"):
            trainer._materialize_train_batch(_tokenizer(), _prompt_batch(), rollout)

    def test_train_stats_include_teacher_timing(self):
        areno = FakeAreno(self._rows())
        trainer = _trainer(areno)
        trainer._materialize_train_batch(_tokenizer(), _prompt_batch(), _rollout())

        result = trainer._augment_train_stats({"total_loss": 1.0})

        self.assertIn("teacher_score_time_s", result)
        self.assertEqual(result["teacher_rows"], 1.0)
        self.assertEqual(result["total_loss"], 1.0)

    def test_startup_rejects_incompatible_teacher_tokenizer(self):
        class Tok:
            eos_token_id = 1
            bos_token_id = 0
            pad_token_id = 2

            def __init__(self, size):
                self.vocab_size = size

            def encode(self, text, **kwargs):
                return [7, 8, 9]

        areno = FakeAreno([])
        areno.get_tokenizer = lambda: Tok(100)
        trainer = _trainer(areno)
        trainer.config.model_hub = "hf"
        trainer.config.teacher_ckpt = "teacher"

        with (
            patch("areno.cli.model_refs.resolve_model_ref", return_value="/tmp/teacher"),
            patch("areno.api.trainers.opd.load_tokenizer", return_value=Tok(128)),
            self.assertRaisesRegex(ValueError, "vocab sizes differ"),
        ):
            trainer._check_tokenizer_compatibility()


if __name__ == "__main__":
    unittest.main()
