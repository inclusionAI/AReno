"""CPU-only Stage 2 OPD model-role tests (no GPU/model weights required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from areno.api.trainer_config import OPDTrainerConfig
from areno.api.trainers.opd import OPDTrainer, align_teacher_scores


class FakeBackend:
    def __init__(self, *, fail_at=None, output=None):
        self.events = []
        self.fail_at = fail_at
        self.output = output
        self.scored = []

    def init(self):
        self.events.append("student_init")
        if self.fail_at == "init":
            raise RuntimeError("student init failed")

    def ensure_roles(self, roles):
        self.events.append("teacher_init")
        assert list(roles) == ["ref"]
        assert roles["ref"].name == "ref"
        assert roles["ref"].path == "teacher"
        assert roles["ref"].trainable is False
        if self.fail_at == "roles":
            raise RuntimeError("teacher init failed")

    def score_logprobs(self, role, token_rows, *, features, microbatch_size):
        self.events.append("score")
        self.scored.append((role, token_rows, features, microbatch_size))
        if self.fail_at == "score":
            raise RuntimeError("teacher scoring failed")
        return self.output if self.output is not None else [
            [-0.25] * len(row) for row in token_rows
        ]

    def close(self):
        self.events.append("close")


def make_trainer(backend, **overrides):
    options = dict(
        algo="opd", ckpt="student", dataset_path="prompts.jsonl",
        teacher_ckpt="teacher", score_micro_bs=2,
        metrics_log_dir=None,
    )
    options.update(overrides)
    config = OPDTrainerConfig(**options)
    return OPDTrainer(
        config,
        instance=backend,
        dataset=[],
        reward_fn=None,
        loss_fn=lambda *args, **kwargs: None,
    )


def test_model_roles_initialized_in_order_and_cleaned_up():
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        assert trainer._roles_ready
        assert backend.events == ["student_init", "teacher_init"]
    assert backend.events == ["student_init", "teacher_init", "close"]
    assert not trainer._roles_ready


def test_teacher_scores_exact_student_token_rows():
    backend = FakeBackend(output=[[-0.1, -0.2, -0.3], [-0.4, -0.5]])
    trainer = make_trainer(backend)
    rows = [[11, 12, 13], [21, 22]]
    features = [None, {"image": "metadata"}]
    with trainer.student_teacher_session():
        scores = trainer._score_teacher_tokens(rows, features=features)
    assert scores == backend.output
    assert backend.scored == [("ref", rows, features, 2)]
    assert backend.events[-1] == "close"


def test_scoring_before_initialization_is_rejected():
    trainer = make_trainer(FakeBackend())
    with pytest.raises(RuntimeError, match="initialized"):
        trainer._score_teacher_tokens([[1, 2]])


@pytest.mark.parametrize(
    "bad_scores, error",
    [
        ([[-0.1, -0.2]], "rows"),
        ([[-0.1], [-0.2]], "row 0"),
        ([[-0.1, float("nan")], [-0.2]], "non-finite"),
        ([[-0.1, float("inf")], [-0.2]], "non-finite"),
    ],
)
def test_teacher_score_alignment_failures(bad_scores, error):
    with pytest.raises(ValueError, match=error):
        align_teacher_scores([[1, 2], [3]], bad_scores)


def test_scores_preserve_input_row_order():
    rows = [[7, 8], [9]]
    scores = [[-0.4, -0.1], [-1.2]]
    assert align_teacher_scores(rows, scores) == scores


def test_reject_features_row_count_mismatch():
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        with pytest.raises(ValueError, match="feature rows"):
            trainer._score_teacher_tokens([[1, 2]], features=[])
    assert "score" not in backend.events


@pytest.mark.parametrize("failure", ["init", "roles", "score"])
def test_backend_always_closed_on_failures(failure):
    backend = FakeBackend(fail_at=failure)
    trainer = make_trainer(backend)
    with pytest.raises(RuntimeError):
        with trainer.student_teacher_session():
            trainer._score_teacher_tokens([[1, 2]])
    assert backend.events[-1] == "close"
    assert not trainer._roles_ready


def test_cleanup_after_user_exception():
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with pytest.raises(RuntimeError, match="interrupted"):
        with trainer.student_teacher_session():
            raise RuntimeError("interrupted")
    assert backend.events[-1] == "close"


def test_empty_batch_does_not_call_teacher():
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        assert trainer._score_teacher_tokens([]) == []
    assert "score" not in backend.events


def test_reward_fn_not_allowed():
    backend = FakeBackend()
    cfg = OPDTrainerConfig(
        algo="opd", ckpt="student", dataset_path="prompts",
        teacher_ckpt="teacher",
    )
    with pytest.raises(ValueError, match="reward"):
        OPDTrainer(cfg, instance=backend, dataset=[],
                   reward_fn=lambda _: 1, loss_fn=lambda *a: None)


def test_training_still_fails_explicitly():
    trainer = make_trainer(FakeBackend())
    with pytest.raises(NotImplementedError, match="Stage 3"):
        trainer.fit()
