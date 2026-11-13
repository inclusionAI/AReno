"""On-policy distillation trainer.

Per step the student samples responses, a frozen teacher scores those exact
tokens, and a token-level distillation loss is applied on the response window.

Teacher scores travel in ``TrainSequence.ref_logprobs`` -- the framework's
generic frozen-reference slot -- and the existing CUDA pack builder converts
them to ``packed_ref_logprobs`` on the response-token action view. No engine
data structures are changed: OPD has no reward function and no critic, it only
adds one frozen scoring role.
"""

from __future__ import annotations

import math
import time
from typing import Any

import areno.api
from areno.api.dashboard import record_dashboard_state
from areno.api.eval_utils import benchmark_due, summarize_benchmark
from areno.api.roles import ModelRole
from areno.api.tokenizer import assert_compatible_tokenizers
from areno.api.trainers.export import export_gguf_if_configured
from areno.api.trainers.policy_only import PolicyOnlyTrainer


class OPDTrainer(PolicyOnlyTrainer):
    """Student rollout -> frozen teacher scoring -> token-level distillation."""

    def __init__(self, config, *, instance, dataset, reward_fn, loss_fn):
        super().__init__(config, instance=instance, dataset=dataset, reward_fn=reward_fn, loss_fn=loss_fn)
        self.validation_dataset: Any = None
        self._last_opd_stats: dict[str, float] = {}
        # The framework's RoleManager loads every non-actor role lazily and
        # offloads it after scoring, so no manual memory movement is needed.
        self.roles = {
            "actor": ModelRole("actor", config.ckpt, trainable=True),
            "teacher": ModelRole("teacher", config.teacher_ckpt, trainable=False),
        }

    def _policy_role_name(self) -> str:
        return "actor"

    def fit(self) -> None:
        self.areno.init()
        self.areno.ensure_roles(self.roles)
        self._check_tokenizer_compatibility()
        try:
            self._fit_initialized()
            export_gguf_if_configured(self)
        finally:
            self.areno.close()

    def _check_tokenizer_compatibility(self) -> None:
        """Fail at start-up if teacher and student disagree on token ids."""

        from transformers import AutoTokenizer

        from areno.cli.model_refs import resolve_model_ref

        student = self.areno.get_tokenizer()
        teacher_path = resolve_model_ref(self.config.teacher_ckpt, model_hub=self.config.model_hub)
        try:
            teacher = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                f"could not load the teacher tokenizer from {teacher_path!r}; "
                "OPD needs a teacher tokenizer compatible with the student"
            ) from exc
        assert_compatible_tokenizers(student, teacher)

    def _materialize_train_batch(self, tokenizer, prompt_batch, rollout_results):
        """Score this step's rollout with the teacher, then assemble rows.

        The teacher sees the tokens the student just generated. Nothing is
        reused across steps, which is what keeps the objective on-policy.
        """

        self._last_opd_stats = {}
        self._maybe_benchmark(phase="before")

        token_rows: list[list[int]] = []
        token_features: list[dict | None] = []
        row_meta: list[tuple[Any, Any, int, int]] = []
        for item, result in zip(prompt_batch.items, rollout_results, strict=True):
            prefix_len = len(item.input_tokens)
            for seq in result.sequences:
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
            if resp_len < 1:
                continue
            tokens = item.input_tokens + seq.resp_tokens
            if len(teacher_logprobs) != len(tokens):
                raise ValueError(
                    f"teacher logprobs length {len(teacher_logprobs)} does not match "
                    f"token length {len(tokens)}"
                )
            if len(seq.resp_logprobs) != resp_len:
                raise ValueError("rollout logprobs must align with response tokens")
            # Index 0 is the alignment placeholder added by every backend's
            # scorer. Every later token enters the masked loss, so a non-finite
            # value in the prompt section would still poison the microbatch.
            # Checking here keeps the validation outside backend graph transforms.
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
        self._maybe_benchmark(phase="after")
        return result

    # -- held-out benchmark -------------------------------------------------

    def _benchmark_token_rows(self, tokenizer):
        """Turn held-out rows into fixed (prompt + reference answer) sequences."""

        rows: list[list[int]] = []
        prompt_lengths: list[int] = []
        for record in self.validation_dataset:
            prompt = record.get("prompt") or record.get("question") or ""
            answer = record.get("solutions") or record.get("answer") or ""
            if isinstance(answer, (list, tuple)):
                answer = answer[0] if answer else ""
            prompt_ids = list(tokenizer.encode(str(prompt), add_special_tokens=False))
            answer_ids = list(tokenizer.encode(str(answer), add_special_tokens=False))
            if not prompt_ids or not answer_ids:
                continue
            rows.append(prompt_ids + answer_ids)
            prompt_lengths.append(len(prompt_ids))
        return rows, prompt_lengths

    def _run_benchmark(self) -> dict[str, float]:
        tokenizer = self.areno.get_tokenizer()
        rows, prompt_lengths = self._benchmark_token_rows(tokenizer)
        if self.config.eval_batches is not None:
            limit = max(1, int(self.config.eval_batches) * int(self.config.batch_size))
            rows, prompt_lengths = rows[:limit], prompt_lengths[:limit]
        if not rows:
            raise ValueError("OPD validation dataset produced no benchmark rows")
        student_rows = self.areno.score_logprobs(
            "actor", rows, microbatch_size=self.config.score_micro_bs
        )
        teacher_rows = self.areno.score_logprobs(
            "teacher", rows, microbatch_size=self.config.opd_teacher_micro_bs
        )
        return summarize_benchmark(teacher_rows, student_rows, prompt_lengths)

    # -- generation benchmark (task accuracy) ------------------------------

    def _generation_prompt_batches(self):
        """Iterate held-out prompts with the framework's tokenizing loader."""

        from itertools import islice

        batches = self.areno.load_prompt_batches(
            self.validation_dataset,
            batch_size=self.config.batch_size,
            max_prompt_tokens=self.config.max_prompt_tokens,
        )
        if self.config.eval_batches is not None:
            batches = islice(batches, max(1, int(self.config.eval_batches)))
        return batches

    def _score_generation_batch(self, tokenizer, prompt_batch, rollout_results) -> list[float]:
        """Score completions through the parent trainer's reward record path."""

        from areno.api.rewards import make_reward_record
        from areno.api.trainers.policy_only import _batch_decode

        token_lists = [seq.resp_tokens for result in rollout_results for seq in result.sequences]
        completions = _batch_decode(tokenizer, token_lists)
        if len(completions) != len(token_lists):
            raise ValueError("generation decode returned a different number of completions")
        completion_iter = iter(completions)
        records = []
        for item, result in zip(prompt_batch.items, rollout_results, strict=True):
            prefix_len = len(item.input_tokens)
            for seq in result.sequences:
                records.append(
                    make_reward_record(
                        prompt=item.prompt,
                        completion=next(completion_iter),
                        source_record=item.record,
                        answer=item.solutions,
                        tokens=item.input_tokens + seq.resp_tokens,
                        logprobs=[0.0] * prefix_len + list(seq.resp_logprobs),
                        loss_mask=[False] * prefix_len + [True] * len(seq.resp_tokens),
                    )
                )
        return self._score_reward_records(records)

    def _run_generation_benchmark(self) -> dict[str, float]:
        """Score current-policy generations on held-out tasks.

        Training ignores ``reward_fn`` because OPD has no reward term, but the
        same callable is used here through the parent trainer's rollout and
        reward plumbing. No second evaluation stack is introduced.
        """

        if self.reward_fn is None or self.validation_dataset is None:
            return {}
        import asyncio

        import areno.api
        from areno.api.eval_utils import summarize_generation_rewards

        tokenizer = self.areno.get_tokenizer()
        sampling_params = areno.api.SamplingParams(
            greedy=True,
            temperature=1.0,
            max_new_tokens=self.config.max_new_tokens,
            max_context_len=getattr(self.config, "max_context_len", None),
            max_prompt_len=self.config.max_prompt_tokens,
            top_k=-1,
            top_p=1.0,
        )
        rewards: list[float] = []
        batches = 0
        for prompt_batch in self._generation_prompt_batches():
            # Greedy decoding makes repeated samples identical, so one sample
            # per prompt is enough and avoids n_samples-times wasted decoding.
            rollout_results = asyncio.run(
                self._run_prompt_rollout(sampling_params, prompt_batch, n_samples=1)
            )
            rewards.extend(self._score_generation_batch(tokenizer, prompt_batch, rollout_results))
            batches += 1
        if not rewards:
            raise ValueError("OPD generation benchmark produced no scored samples")
        stats = summarize_generation_rewards(rewards)
        stats["batches"] = float(batches)
        return {f"generation_{key}": value for key, value in stats.items()}

    def _maybe_benchmark(self, *, phase: str) -> None:
        if self.validation_dataset is None:
            return
        step = int(getattr(self, "_dashboard_step", 0) or 0)
        if not benchmark_due(
            step,
            eval_interval=int(self.config.eval_interval),
            max_steps=self.config.max_steps,
            eval_before_step=bool(self.config.eval_before_step),
            phase=phase,
        ):
            return
        # The logprob gap is a cheap training-progress monitor. Generation
        # accuracy is the actual task-level outcome and runs on the same
        # schedule whenever a --reward-fn-path was supplied for evaluation.
        stats = self._run_benchmark()
        stats.update(self._run_generation_benchmark())
        self.areno.record_validation_stats(
            {f"{phase}_{key}": value for key, value in stats.items()}, step=step
        )
        self.logger.info("step=%d phase=%s benchmark_stats=%s", step, phase, stats)
        record_dashboard_state(
            self.areno,
            stage="validation_end",
            epoch=getattr(self, "_dashboard_epoch", None),
            step=step,
            role=self._policy_role_name(),
            extra=stats,
        )
