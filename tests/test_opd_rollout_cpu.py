"""Stage 3: prompt rollout -> frozen teacher -> aligned TrainSequence (CPU).

These tests use AReno's real data models and PolicyOnlyTrainer path. No CUDA,
model downloads, or actual gradient updates are required.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from areno.api.data import PromptBatch, PromptItem
from areno.api.models import RolloutResult, RolloutSequence, TrainSequence
from areno.api.trainer_config import OPDTrainerConfig
from areno.api.trainers.opd import OPDTrainer


class FakeBackend:
    def __init__(self, *, scores=None, fail_score=False, rollout_results=None):
        self.events = []
        self.scores = scores
        self.fail_score = fail_score
        self.rollout_results = rollout_results
        self.scored = []
        self.training_batches = []

    def init(self):
        self.events.append("student_init")

    def ensure_roles(self, roles):
        self.events.append("teacher_init")
        assert list(roles) == ["ref"]
        assert roles["ref"].trainable is False
        assert roles["ref"].path == "teacher"

    def score_logprobs(self, role, rows, *, features, microbatch_size):
        self.events.append("teacher_score")
        self.scored.append((role, [list(r) for r in rows], list(features), microbatch_size))
        if self.fail_score:
            raise RuntimeError("teacher backend failure")
        return self.scores if self.scores is not None else [
            [-0.25] * len(r) for r in rows
        ]

    def close(self):
        self.events.append("close")

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=999)

    def get_processor(self):
        return None

    def load_prompt_batches(self, dataset, *, batch_size, max_prompt_tokens):
        self.events.append("dataset")
        yield dataset

    @asynccontextmanager
    async def rollout_session(self, *, sampling_params, max_running_prompts, proxy):
        self.events.append("rollout_begin")
        try:
            yield
        finally:
            self.events.append("rollout_end")

    async def rollout_token_batch_async(self, prompts, n_samples, sampling_params, *, prompt_features):
        self.events.append("student_rollout")
        return self.rollout_results

    def train(self, batch, loss_fn, *, mini_bs, gradient_accumulation_steps):
        self.events.append("train")
        self.training_batches.append(batch)
        return {"mock_train": 1.0}


def make_batch(*items):
    return PromptBatch(
        items=[PromptItem(prompt=f"prompt{i}", solutions=None,
                          input_tokens=list(tokens), record={"features": feat})
               for i, (tokens, feat) in enumerate(items)],
        scanned=len(items), skipped_long=0, total_skipped_long=0,
    )


def rollouts(*groups):
    return [RolloutResult(sequences=[RolloutSequence(resp_tokens=list(t), resp_logprobs=list(lp))
                                     for t, lp in group]) for group in groups]


def make_trainer(backend, **overrides):
    cfg = dict(algo="opd", ckpt="student", teacher_ckpt="teacher",
               dataset_path="prompts.jsonl", score_micro_bs=3,
               metrics_log_dir=None, max_steps=1, epochs=1, n_samples=2,
               batch_size=2, save_path=None)
    cfg.update(overrides)
    return OPDTrainer(
        OPDTrainerConfig(**cfg), instance=backend, dataset=[],
        reward_fn=None, loss_fn=lambda *_args, **_kwargs: (0, {}),
    )


def materialize(trainer, prompts, responses):
    return trainer._materialize_train_batch(
        SimpleNamespace(eos_token_id=999), prompts, responses
    )


def test_multiprompt_multisample_exact_alignment_and_metrics():
    backend = FakeBackend(scores=[
        [-9, -8, -0.3, -0.4], [-9, -8, -0.5], [-7, -0.7, -0.8, -0.9],
    ])
    trainer = make_trainer(backend)
    prompts = make_batch(([11, 12], {"tag": "a"}), ([21], None))
    result = rollouts(
        [([31, 32], [-1.0, -1.1]), ([33], [-1.2])],
        [([41, 42, 43], [-1.3, -1.4, -1.5])],
    )
    with trainer.student_teacher_session():
        rows, rewards, old_scores = materialize(trainer, prompts, result)

    assert len(rows) == 3
    assert all(isinstance(x, TrainSequence) for x in rows)
    assert [x.tokens for x in rows] == [[11, 12, 31, 32], [11, 12, 33], [21, 41, 42, 43]]
    assert [x.logprobs for x in rows] == [
        [0.0, 0.0, -1.0, -1.1], [0.0, 0.0, -1.2], [0.0, -1.3, -1.4, -1.5],
    ]
    assert [x.prompt_mask for x in rows] == [
        [True, True, False, False], [True, True, False],
        [True, False, False, False],
    ]
    assert [x.ref_logprobs for x in rows] == backend.scores
    assert [x.features for x in rows] == [{"tag": "a"}, {"tag": "a"}, None]
    assert [x.eos_token_id for x in rows] == [999] * 3
    assert all(not x.advantages and not x.values and not x.returns for x in rows)
    assert rewards == []
    assert old_scores == [-1.0, -1.1, -1.2, -1.3, -1.4, -1.5]
    assert backend.scored == [("ref", [[11, 12, 31, 32], [11, 12, 33], [21, 41, 42, 43]],
                               [{"tag": "a"}, {"tag": "a"}, None], 3)]
    stats = trainer._last_opd_stats
    assert stats["opd_response_token_count"] == 6
    assert stats["opd_empty_responses_skipped"] == 0
    assert stats["opd_teacher_score_time_s"] >= 0
    expected_gap = ((-0.3 + 1.0) + (-0.4 + 1.1) + (-0.5 + 1.2)
                    + (-0.7 + 1.3) + (-0.8 + 1.4) + (-0.9 + 1.5)) / 6
    assert stats["opd_teacher_minus_rollout_logprob_mean"] == pytest.approx(expected_gap)
    assert backend.events == ["student_init", "teacher_init", "teacher_score", "close"]


def test_unchanged_input_lists_and_no_reward_scoring():
    prompt = [8, 9]
    tokens = [33, 34]
    lps = [-0.1, -0.2]
    result = rollouts([ (tokens, lps) ])
    prompts = make_batch((prompt, None))
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        materialize(trainer, prompts, result)
    assert prompt == [8, 9]
    assert tokens == [33, 34]
    assert lps == [-0.1, -0.2]
    assert "reward_score" not in backend.events


def test_skip_empty_completion_but_keep_nonempty():
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        rows, rewards, old = materialize(
            trainer, make_batch(([1], None)),
            rollouts([([], []), ([3], [-0.4]), ([], [])]),
        )
    assert len(rows) == 1
    assert rows[0].tokens == [1, 3]
    assert old == [-0.4] and rewards == []
    assert trainer._last_opd_stats["opd_empty_responses_skipped"] == 2


def test_all_empty_completions_skip_scoring():
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        rows, rewards, old = materialize(
            trainer, make_batch(([7], None)), rollouts([([], []), ([], [])]),
        )
    assert (rows, rewards, old) == ([], [], [])
    assert "teacher_score" not in backend.events
    assert trainer._last_opd_stats["opd_response_token_count"] == 0
    assert trainer._last_opd_stats["opd_teacher_score_time_s"] == 0


def test_empty_batch_no_teacher_call():
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        assert materialize(trainer, make_batch(), []) == ([], [], [])
    assert "teacher_score" not in backend.events


@pytest.mark.parametrize("results", [[], rollouts([([4], [-0.5])], [([5], [-0.6])])])
def test_reject_prompt_count_mismatch_before_scoring(results):
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        with pytest.raises(ValueError, match="prompt count mismatch"):
            materialize(trainer, make_batch(([1], None)), results)
    assert "teacher_score" not in backend.events


def test_empty_prompt_prefix_rejected():
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        with pytest.raises(ValueError, match="nonempty prompt"):
            materialize(trainer, make_batch(([], None)), rollouts([([1], [-1.0])]))
    assert "teacher_score" not in backend.events


@pytest.mark.parametrize("tokens,logprobs", [([7, 8], [-0.1]), ([7], [-0.1, -0.2])])
def test_mismatched_response_and_rollout_scores_rejected(tokens, logprobs):
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        with pytest.raises(ValueError, match="different lengths"):
            materialize(trainer, make_batch(([1], None)), rollouts([(tokens, logprobs)]))
    assert "teacher_score" not in backend.events


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -float("inf")])
def test_nonfinite_rollout_scores_rejected(bad):
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        with pytest.raises(ValueError, match="non-finite"):
            materialize(trainer, make_batch(([1], None)), rollouts([([3], [bad])]))
    assert "teacher_score" not in backend.events


@pytest.mark.parametrize("teacher,match", [
    ([], "returned 0 rows"),
    ([[-0.1]], "row 0"),
    ([[-0.1, float("nan")]], "non-finite"),
    ([[-0.1, float("inf")]], "non-finite"),
])
def test_teacher_alignment_rejected_and_cleanup(teacher, match):
    backend = FakeBackend(scores=teacher)
    trainer = make_trainer(backend)
    with pytest.raises(ValueError, match=match):
        with trainer.student_teacher_session():
            materialize(trainer, make_batch(([1], None)), rollouts([([3], [-0.5])]))
    assert backend.events[-1] == "close"


def test_score_backend_exception_still_closes():
    backend = FakeBackend(fail_score=True)
    trainer = make_trainer(backend)
    with pytest.raises(RuntimeError, match="teacher backend failure"):
        with trainer.student_teacher_session():
            materialize(trainer, make_batch(([1], None)), rollouts([([3], [-0.5])]))
    assert backend.events[-1] == "close"


def test_materialization_before_role_initialization_is_rejected():
    trainer = make_trainer(FakeBackend())
    with pytest.raises(RuntimeError, match="initialized"):
        materialize(trainer, make_batch(([1], None)), rollouts([([3], [-0.5])]))


def test_augment_train_stats_and_non_dict_stats():
    backend = FakeBackend()
    trainer = make_trainer(backend)
    with trainer.student_teacher_session():
        materialize(trainer, make_batch(([1], None)), rollouts([([3], [-0.5])]))
    merged = trainer._augment_train_stats({"mock_train": 1.0})
    assert merged["mock_train"] == 1.0
    assert merged["opd_response_token_count"] == 1
    assert trainer._augment_train_stats("unchanged") == "unchanged"


def test_existing_rollout_loop_orders_student_then_teacher_then_train(monkeypatch):
    # Exercise the actual inherited PolicyOnlyTrainer loop, not a reimplemented
    # imitation. The fake `train` accepts the batch without invoking the
    # intentionally unimplemented Stage 4 loss function.
    monkeypatch.setenv("ARENO_LOG_COMPLETIONS", "0")
    monkeypatch.setattr(
        "areno.api.trainers.policy_only.record_dashboard_state",
        lambda *args, **kwargs: None,
    )
    backend = FakeBackend(rollout_results=rollouts([([3], [-0.5])]))
    prompts = make_batch(([1, 2], None))
    trainer = make_trainer(backend, n_samples=1, batch_size=1)
    trainer.dataset = prompts
    trainer.fit()
    assert backend.events == [
        "student_init", "teacher_init", "dataset", "rollout_begin",
        "student_rollout", "rollout_end", "teacher_score", "train", "close",
    ]
    assert len(backend.training_batches) == 1
    assert backend.training_batches[0][0].tokens == [1, 2, 3]
    assert backend.training_batches[0][0].ref_logprobs == [-0.25, -0.25, -0.25]


def test_automatic_cleanup_when_inherited_training_loop_fails(monkeypatch):
    monkeypatch.setenv("ARENO_LOG_COMPLETIONS", "0")
    monkeypatch.setattr(
        "areno.api.trainers.policy_only.record_dashboard_state",
        lambda *args, **kwargs: None,
    )
    backend = FakeBackend(fail_score=True, rollout_results=rollouts([([3], [-0.5])]))
    trainer = make_trainer(backend, n_samples=1, batch_size=1)
    trainer.dataset = make_batch(([1, 2], None))
    with pytest.raises(RuntimeError, match="teacher backend failure"):
        trainer.fit()
    assert "train" not in backend.events
    assert backend.events[-1] == "close"
