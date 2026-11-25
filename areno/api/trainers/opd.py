"""On-policy distillation trainer.

Per step the student samples responses, a frozen teacher scores those exact
tokens, and a token-level distillation loss is applied on the response window.

Teacher scores travel in ``TrainSequence.ref_logprobs``, the framework's
generic frozen-reference slot. CUDA converts them to ``packed_ref_logprobs``
on the response-token action view; MLX reads ``ref_logprobs`` from its padded
batch. OPD adds one frozen scoring role and no reward or critic.
"""

from __future__ import annotations

import math
import time
from typing import Any

import areno.api
from areno.api.roles import ModelRole
from areno.api.tokenizer import assert_compatible_tokenizers, load_tokenizer
from areno.api.trainers.export import export_gguf_if_configured
from areno.api.trainers.policy_only import PolicyOnlyTrainer


class OPDTrainer(PolicyOnlyTrainer):
    """Student rollout -> frozen teacher scoring -> token-level distillation."""

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        super().__init__(config, instance=instance, dataset=dataset, reward_fn=reward_fn, loss_fn=loss_fn)
        self._last_opd_stats: dict[str, float] = {}
        # CUDA offloads frozen roles between scoring passes; MLX keeps them
        # resident. Either way the backend owns role lifetime.
        self.roles = {
            "actor": ModelRole("actor", config.ckpt, trainable=True),
            "teacher": ModelRole("teacher", config.teacher_ckpt, trainable=False),
        }

    def _policy_role_name(self) -> str:
        return "actor"

    def fit(self) -> None:
        self.areno.init()
        self._check_tokenizer_compatibility()
        self.areno.ensure_roles(self.roles)
        try:
            self._fit_initialized()
            export_gguf_if_configured(self)
        finally:
            self.areno.close()

    def _check_tokenizer_compatibility(self) -> None:
        """Fail at start-up if teacher and student disagree on token ids."""

        from areno.cli.model_refs import resolve_model_ref

        student = self.areno.get_tokenizer()
        teacher_path = resolve_model_ref(self.config.teacher_ckpt, model_hub=self.config.model_hub)
        try:
            teacher = load_tokenizer(teacher_path)
        except Exception as exc:
            raise RuntimeError(
                f"could not load the teacher tokenizer from {teacher_path!r}; "
                "OPD needs a teacher tokenizer compatible with the student"
            ) from exc
        assert_compatible_tokenizers(student, teacher)

    def _materialize_train_batch(self, tokenizer, prompt_batch, rollout_results):
        """Score this step's rollout with the teacher, then assemble rows.

        The teacher scores the tokens the student just generated; scores are
        never reused across steps.
        """

        self._last_opd_stats = {}

        token_rows: list[list[int]] = []
        token_features: list[dict | None] = []
        row_meta: list[tuple[Any, Any, int, int]] = []
        for item, result in zip(prompt_batch.items, rollout_results, strict=True):
            prefix_len = len(item.input_tokens)
            for seq in result.sequences:
                if not seq.resp_tokens:
                    continue
                token_rows.append(item.input_tokens + seq.resp_tokens)
                token_features.append(item.record.get("features"))
                row_meta.append((item, seq, prefix_len, len(seq.resp_tokens)))
        if not token_rows:
            return [], [], []

        started = time.perf_counter()
        teacher_rows = self.areno.score_logprobs(
            "teacher",
            token_rows,
            features=token_features if any(feature is not None for feature in token_features) else None,
            microbatch_size=self.config.opd_teacher_micro_bs,
        )
        self._last_opd_stats["teacher_score_time_s"] = time.perf_counter() - started
        if len(teacher_rows) != len(token_rows):
            raise ValueError("teacher returned a different number of logprob rows than rollout sequences")

        train_batch: list[Any] = []
        rollout_logprobs: list[float] = []
        for (item, seq, prefix_len, resp_len), teacher_logprobs in zip(row_meta, teacher_rows, strict=True):
            tokens = item.input_tokens + seq.resp_tokens
            if len(teacher_logprobs) != len(tokens):
                raise ValueError(
                    f"teacher logprobs length {len(teacher_logprobs)} does not match token length {len(tokens)}"
                )
            if len(seq.resp_logprobs) != resp_len:
                raise ValueError("rollout logprobs must align with response tokens")
            # Index 0 is the scorer's alignment placeholder; later prompt
            # tokens still enter the masked loss, where a NaN would spread.
            if not all(math.isfinite(float(value)) for value in teacher_logprobs[1:]):
                raise ValueError("teacher logprobs contain non-finite values")
            rollout_logprobs.extend(float(value) for value in seq.resp_logprobs)
            train_batch.append(
                areno.api.TrainSequence(
                    tokens=tokens,
                    logprobs=[0.0] * prefix_len + [float(value) for value in seq.resp_logprobs],
                    prompt_len=prefix_len,
                    scalar_advantage=0.0,
                    ref_logprobs=[float(value) for value in teacher_logprobs],
                    features=item.record.get("features"),
                    eos_token_id=tokenizer.eos_token_id,
                    routed_experts=seq.routed_experts,
                )
            )
        self._last_opd_stats["teacher_rows"] = float(len(teacher_rows))
        return train_batch, [], rollout_logprobs

    def _augment_train_stats(self, result):
        result = super()._augment_train_stats(result)
        if isinstance(result, dict) and self._last_opd_stats:
            result.update(self._last_opd_stats)
        return result
