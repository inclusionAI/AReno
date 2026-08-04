from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from areno.api.roles import ModelRole
from areno.api.trainers.dpo import DPOTrainer
from areno.api.trainers.ppo import PPOTrainer
from areno.api.trainers.role_aware import (
    RoleAwarePolicyTrainer,
    RoleAwareTrainerMixin,
)


class _FakeAreno:
    def __init__(
        self,
        *,
        score_rows=None,
        ensure_error: Exception | None = None,
    ):
        self.score_rows = [] if score_rows is None else score_rows
        self.ensure_error = ensure_error
        self.events = []
        self.score_calls = []

    def init(self):
        self.events.append("init")

    def ensure_roles(self, roles):
        self.events.append(
            ("ensure_roles", tuple(roles))
        )

        if self.ensure_error is not None:
            raise self.ensure_error

    def close(self):
        self.events.append("close")

    def score_logprobs(
        self,
        role,
        token_rows,
        *,
        microbatch_size,
    ):
        self.score_calls.append(
            (role, token_rows, microbatch_size)
        )
        return self.score_rows


class _DummyRoleTrainer(RoleAwareTrainerMixin):
    def __init__(self, areno):
        self.areno = areno
        self.config = SimpleNamespace(
            score_micro_bs=3
        )
        self.logger = logging.getLogger(
            "test.role_aware"
        )
        self.roles = {
            "teacher": ModelRole(
                "teacher",
                "teacher-ckpt",
                trainable=False,
            )
        }
        self.initialized_fit_called = False

    def _fit_initialized(self):
        self.initialized_fit_called = True
        self.areno.events.append(
            "fit_initialized"
        )


def test_existing_role_trainers_use_shared_role_boundary():
    assert issubclass(
        DPOTrainer,
        RoleAwareTrainerMixin,
    )
    assert issubclass(
        PPOTrainer,
        RoleAwarePolicyTrainer,
    )


def test_role_aware_fit_initializes_roles_before_training_and_closes():
    areno = _FakeAreno()
    trainer = _DummyRoleTrainer(areno)

    trainer.fit()

    assert trainer.initialized_fit_called
    assert areno.events == [
        "init",
        ("ensure_roles", ("teacher",)),
        "fit_initialized",
        "close",
    ]


def test_role_aware_fit_closes_when_role_initialization_fails():
    areno = _FakeAreno(
        ensure_error=RuntimeError("load failed")
    )
    trainer = _DummyRoleTrainer(areno)

    with pytest.raises(
        RuntimeError,
        match="load failed",
    ):
        trainer.fit()

    assert areno.events == [
        "init",
        ("ensure_roles", ("teacher",)),
        "close",
    ]


def test_role_logprob_scoring_forwards_microbatch_and_validates_alignment():
    token_rows = [
        [1, 2],
        [3],
    ]
    areno = _FakeAreno(
        score_rows=[
            [-0.1, -0.2],
            [-0.3],
        ]
    )
    trainer = _DummyRoleTrainer(areno)

    rows = trainer._score_logprobs(
        "teacher",
        token_rows,
    )

    assert rows == [
        [-0.1, -0.2],
        [-0.3],
    ]
    assert areno.score_calls == [
        ("teacher", token_rows, 3)
    ]


def test_role_logprob_scoring_rejects_missing_rows():
    trainer = _DummyRoleTrainer(
        _FakeAreno(score_rows=[])
    )

    with pytest.raises(
        ValueError,
        match=(
            "returned 0 logprob rows "
            "for 1 token rows"
        ),
    ):
        trainer._score_logprobs(
            "teacher",
            [[1]],
        )


def test_role_logprob_scoring_rejects_misaligned_token_scores():
    trainer = _DummyRoleTrainer(
        _FakeAreno(
            score_rows=[[-0.1]]
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "returned 1 logprobs for token row 0 "
            "with 2 tokens"
        ),
    ):
        trainer._score_logprobs(
            "teacher",
            [[1, 2]],
        )