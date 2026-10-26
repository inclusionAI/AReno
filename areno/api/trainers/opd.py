"""On-policy distillation trainer: student rollout and frozen teacher scoring.

Stage 3 materializes student-generated responses as teacher-scored training
sequences; Stage 4 supplies the differentiable OPD loss. The backend owns
model loading, trainability, devices, and scoring.
"""

from __future__ import annotations

import math
import time
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
            raise ValueError("Basic OPD supports prompt-based training only")

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
        self._last_opd_stats: dict[str, float] = {}

    def _policy_role_name(self) -> str:
        return "student"

    @contextmanager
    def student_teacher_session(self) -> Iterator[None]:
        """Start student/teacher roles and always close the backend.

        This wraps the existing prompt-rollout/training loop.
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

    def _materialize_train_batch(self, tokenizer, prompt_batch, rollout_results):
        """Turn on-policy student completions into teacher-scored training rows.

        Reuses PolicyOnlyTrainer's prompt rollout; unlike GRPO/GSPO this path
        NEVER calls a reward function or computes group-relative advantages.

        Returns (train_sequences, [], response_rollout_logprobs), matching the
        parent trainer loop's hook contract. `ref_logprobs` carries *teacher*
        scores, not the usual PPO reference-model KL baseline.
        """
        from areno.api.models import TrainSequence

        self._last_opd_stats = {}
        if len(prompt_batch.items) != len(rollout_results):
            raise ValueError(
                "OPD rollout prompt count mismatch: "
                f"{len(rollout_results)} results for {len(prompt_batch.items)} prompts"
            )

        token_rows: list[list[int]] = []
        features: list[dict | None] = []
        metadata: list[tuple[int, list[float]]] = []
        response_rollout_logprobs: list[float] = []
        skipped_empty = 0

        # Flatten prompt/sample pairs in the same order as rollout results.
        for item, result in zip(prompt_batch.items, rollout_results, strict=True):
            prompt = list(item.input_tokens)
            if not prompt:
                raise ValueError("OPD requires a nonempty prompt token prefix")
            for sequence in result.sequences:
                response = list(sequence.resp_tokens)
                old_scores = [float(lp) for lp in sequence.resp_logprobs]
                if len(response) != len(old_scores):
                    raise ValueError(
                        "OPD response tokens and rollout logprobs have different lengths"
                    )
                if not all(math.isfinite(lp) for lp in old_scores):
                    raise ValueError("OPD rollout contains non-finite logprobs")
                if not response:
                    skipped_empty += 1
                    continue

                token_rows.append(prompt + response)
                metadata.append((len(prompt), old_scores))
                features.append(item.record.get("features"))
                response_rollout_logprobs.extend(old_scores)

        n_tokens = len(response_rollout_logprobs)
        if not token_rows:
            # No response-only tokens: no teacher call and no optimizer step.
            self._last_opd_stats = {
                "opd_response_token_count": 0.0,
                "opd_empty_responses_skipped": float(skipped_empty),
                "opd_teacher_score_time_s": 0.0,
            }
            return [], [], []

        self.logger.info("role=ref stage=logprob_score_start rows=%d", len(token_rows))
        start = time.perf_counter()
        teacher_scores = self._score_teacher_tokens(token_rows, features=features)
        elapsed = time.perf_counter() - start
        self.logger.info(
            "role=ref stage=logprob_score_end rows=%d time_s=%.6f",
            len(token_rows), elapsed,
        )

        train_batch = []
        teacher_minus_rollout_sum = 0.0
        for tokens, scores, (prompt_len, old_scores), feature in zip(
            token_rows, teacher_scores, metadata, features, strict=True
        ):
            response_len = len(old_scores)
            # Only response scores enter the teacher-student gap metric.
            for teacher_lp, rollout_lp in zip(
                scores[prompt_len:], old_scores, strict=True
            ):
                teacher_minus_rollout_sum += teacher_lp - rollout_lp
            train_batch.append(
                TrainSequence(
                    tokens=tokens,
                    prompt_mask=[True] * prompt_len + [False] * response_len,
                    # Prompt positions are placeholders masked from the loss.
                    logprobs=[0.0] * prompt_len + old_scores,
                    # Reuse the backend's reference-score slot for teacher LPs.
                    ref_logprobs=scores,
                    features=feature,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )

        self._last_opd_stats = {
            "opd_response_token_count": float(n_tokens),
            "opd_empty_responses_skipped": float(skipped_empty),
            "opd_teacher_score_time_s": elapsed,
            "opd_teacher_minus_rollout_logprob_mean": (
                teacher_minus_rollout_sum / n_tokens
            ),
        }
        return train_batch, [], response_rollout_logprobs

    def _augment_train_stats(self, result):
        """Attach teacher-scoring diagnostics to the parent loop's metrics."""
        if isinstance(result, dict):
            result.update(self._last_opd_stats)
        return result

    def fit(self) -> None:
        """Reuse the existing prompt rollout/training loop with teacher roles.

        Full optimization requires the Stage 4 OPD loss; the Stage 1 loss stub
        raises NotImplementedError rather than silently running a wrong loss.
        """
        with self.student_teacher_session():
            self._fit_initialized()
