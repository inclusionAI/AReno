"""On-policy distillation trainer: Stage 2 student/teacher role support.

Stage 2 deliberately does not perform rollout, build train batches, or optimize.
Those operations are added in Stage 3 and Stage 4. The backend owns model
loading, trainability, devices, and scoring; never load the teacher here with
Hugging Face directly.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from areno.api.roles import ModelRole
from areno.api.trainers.policy_only import PolicyOnlyTrainer


def align_teacher_scores(
    token_rows: Sequence[Sequence[int]],
    score_rows: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Validate one finite teacher log-probability per input token.

    AReno's scoring contract returns a row aligned to each token row. This
    includes prompt positions; response-only masking is added in Stage 3.
    """
    if len(token_rows) != len(score_rows):
        raise ValueError(
            f"OPD teacher returned {len(score_rows)} rows for {len(token_rows)} input rows"
        )

    aligned: list[list[float]] = []
    for index, (tokens, scores) in enumerate(zip(token_rows, score_rows, strict=True)):
        if len(tokens) != len(scores):
            raise ValueError(
                f"OPD teacher row {index} returned {len(scores)} log-probabilities "
                f"for {len(tokens)} tokens"
            )
        values = [float(value) for value in scores]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"OPD teacher row {index} contains non-finite scores")
        aligned.append(values)
    return aligned


class OPDTrainer(PolicyOnlyTrainer):
    """Trainable student plus an auxiliary frozen teacher scoring role.

    The already-configured actor/student is loaded through self.areno.init().
    AReno's supported auxiliary scoring role is named ``ref``; this role
    represents the OPD teacher, not a PPO KL reference policy.
    """

    _TEACHER_ROLE = "ref"

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn) -> None:
        if reward_fn is not None:
            raise ValueError("OPD does not accept a reward function")
        if not getattr(config, "teacher_ckpt", None):
            raise ValueError("OPD requires a teacher checkpoint")
        if getattr(config, "agent_fn", None) is not None:
            raise ValueError("Stage 2 OPD supports prompt-based training only")

        super().__init__(
            config, instance=instance, dataset=dataset,
            reward_fn=None, loss_fn=loss_fn,
        )
        self.roles = {
            self._TEACHER_ROLE: ModelRole(
                self._TEACHER_ROLE,
                config.teacher_ckpt,
                trainable=False,
            )
        }
        self._roles_ready = False

    def _policy_role_name(self) -> str:
        return "student"

    @contextmanager
    def student_teacher_session(self) -> Iterator[None]:
        """Start student/teacher roles and always close the backend.

        Stage 3 will call this around the existing rollout/train loop.
        """
        if self._roles_ready:
            raise RuntimeError("OPD model roles are already initialized")
        try:
            # The primary student comes from Trainer(model_path=config.ckpt).
            self.areno.init()
            # The backend loads the teacher in a non-trainable auxiliary role.
            self.areno.ensure_roles(self.roles)
            self._roles_ready = True
            yield
        finally:
            self._roles_ready = False
            self.areno.close()

    def _score_teacher_tokens(
        self,
        token_rows: list[list[int]],
        *,
        features: list[Any] | None = None,
    ) -> list[list[float]]:
        """Score exactly the supplied token rows with the frozen teacher."""
        if not self._roles_ready:
            raise RuntimeError("OPD teacher role must be initialized before scoring")
        if features is None:
            features = [None] * len(token_rows)
        if len(features) != len(token_rows):
            raise ValueError("OPD token rows and feature rows have different lengths")
        if not token_rows:
            return []
        if any(not row for row in token_rows):
            raise ValueError("OPD teacher cannot score an empty token row")

        score_rows = self.areno.score_logprobs(
            self._TEACHER_ROLE,
            token_rows,
            features=features,
            microbatch_size=self.config.score_micro_bs,
        )
        return align_teacher_scores(token_rows, score_rows)

    def fit(self) -> None:
        raise NotImplementedError(
            "OPD Stage 2 supports student/teacher initialization and scoring, "
            "but does not run training until Stage 3 (rollout) and Stage 4 (loss)."
        )
