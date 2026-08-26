"""High-level entrypoint that algorithm scripts interact with.

`Trainer` ties together tokenizer loading, backend creation, the rollout/train
cycle, and (optionally) TensorBoard recording. A typical RL script constructs
one `Trainer`, calls ``init()`` once, and then loops:
``rollout_batch() -> train()``. PPO additionally calls `ensure_roles` so that
ref/reward/critic models become available behind the backend boundary.
"""

import time
from collections.abc import Callable, Iterable
from typing import Any

from areno.api.agentic import LossMaskPolicy, RolloutSession
from areno.api.backend.base import Backend, get_backend_cls
from areno.api.config import BackendConfig, coerce_backend_config, resolve_backend_type
from areno.api.context import Context
from areno.api.data import PromptBatch, PromptItem
from areno.api.metrics import MetricsRecorder
from areno.api.models import BackendType, RolloutResult, SamplingParams, TrainSequence
from areno.api.multimodal import (
    encode_multimodal_prompt,
    expand_image_tokens,
    image_token_counts_from_features,
    mrope_position_ids_from_image_grid,
    record_has_image,
)
from areno.api.roles import ModelRole
from areno.api.tokenizer import (
    encode_generation_prompt,
    eos_token_ids,
    load_processor,
    load_tokenizer,
    normalize_token_ids,
)


class Trainer:
    """High-level API used by algorithm code.

    `Trainer` owns tokenizer loading, backend construction, rollout, training,
    checkpointing, and optional metric recording. A typical RL loop calls
    `init()`, repeatedly runs `rollout_batch() -> train()`, and finally
    `close()`.
    """

    def __init__(
        self,
        world_size: int,
        model_path: str,
        backend_type: BackendType | None = None,
        custom_config: BackendConfig | None = None,
        metrics_log_dir: str | None = None,
        score_micro_bs: int = 8,
    ) -> None:
        """Create a trainer without starting backend workers.

        Call `init()` before rollout or training. `world_size` is the total
        number of devices/workers visible to the selected backend.
        """

        self._tokenizer = None
        self._processor = None
        self._backend: Backend | None = None
        # Resolve backend type from the explicit value or default to Areno.
        self._backend_type = resolve_backend_type(backend_type, custom_config)
        self._model_path = model_path
        self._ctx: Context | None = None
        self._world_size = world_size
        self._initialized = False
        self._custom_config = coerce_backend_config(self._backend_type, custom_config)
        self._metrics = MetricsRecorder(metrics_log_dir) if metrics_log_dir else None
        self._score_micro_bs = int(score_micro_bs)
        # Per-step wall-time bag accumulated by the rollout/train helpers
        # so `record_train_step` can flush a complete timing snapshot.
        self._metric_timings: dict[str, float] = {}
        self._step_active = False
        self._step_wall_start: float | None = None
        self._rollout_session_depth = 0
        self._rollout_wall_start: float | None = None

    def init(self) -> None:
        """Load tokenizer, create backend context, and initialize workers."""

        real_path = self._model_path
        self._tokenizer = load_tokenizer(real_path)
        self._processor = load_processor(real_path)
        self._ctx = Context(
            self._world_size, real_path, self._tokenizer, self._custom_config, eos_token_ids(real_path, self._tokenizer)
        )
        backend_cls = get_backend_cls(self._backend_type)
        if backend_cls is None:
            raise ValueError(f"unsupported backend type: {self._backend_type}")
        self._backend = backend_cls()
        self._backend.initialize(self._ctx)
        components = self._backend.runtime_components()
        if components.tokenizer is not None:
            self._tokenizer = components.tokenizer
            self._ctx.tokenizer = components.tokenizer
            self._ctx.eos_token_ids = eos_token_ids(real_path, components.tokenizer)
        if components.processor is not None:
            self._processor = components.processor
        self._initialized = True

    def get_tokenizer(self) -> Any:
        """Return the initialized tokenizer for prompt and completion handling."""

        return self._tokenizer

    def get_processor(self) -> Any:
        """Return the initialized multimodal processor when the checkpoint provides one."""

        return self._processor

    def _begin_step(self) -> None:
        """Open a trainer-owned step if rollout/train has not already done so."""

        if self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._step_active:
            return
        self._ctx.step()
        self._metric_timings = {}
        self._step_active = True
        self._step_wall_start = time.perf_counter()

    def finish_step(self) -> None:
        """Close the current trainer-owned step without running actor train."""

        self._step_active = False
        self._step_wall_start = None

    def begin_rollout_session(self) -> None:
        """Prepare backend rollout state for one or more rollout calls."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth == 0:
            self._begin_step()
            self._rollout_wall_start = time.perf_counter()
            self._backend.begin_rollout_session(self._ctx)
        self._rollout_session_depth += 1

    async def begin_rollout_session_async(self) -> None:
        """Async variant of :meth:`begin_rollout_session`."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth == 0:
            self._begin_step()
            self._rollout_wall_start = time.perf_counter()
            await self._backend.begin_rollout_session_async(self._ctx)
        self._rollout_session_depth += 1

    async def sync_rollout_session_async(self) -> None:
        """Synchronize backend rollout workers before request-driven rollout."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth <= 0:
            raise RuntimeError(
                "sync_rollout_session_async must be called inside `async with trainer.rollout_session(...)`"
            )
        await self._backend.sync_rollout_session_async(self._ctx)

    def dp_size(self) -> int:
        """Return the initialized backend's effective data-parallel size."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        try:
            return int(self._backend.dp_size(self._ctx))
        except AttributeError:
            config = self._ctx.custom_config
            if config is None:
                return int(self._ctx.world_size)
            return max(int(self._ctx.world_size) // int(config.tp_size), 1)

    def model_context_len(self) -> int | None:
        """Return the loaded model's context length when the backend exposes it."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        return self._backend.model_context_len(self._ctx)

    def probe_rollout_cache(self, *, max_new_tokens: int, max_running_prompts: int, max_prompt_len: int) -> float:
        """Allocate rollout KV cache/decode graphs without running rollout decode."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        return self._backend.probe_rollout_cache(
            self._ctx,
            max_new_tokens=max_new_tokens,
            max_running_prompts=max_running_prompts,
            max_prompt_len=max_prompt_len,
        )

    def end_rollout_session(self) -> None:
        """Finalize backend rollout state when a rollout group completes."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth <= 0:
            return
        self._rollout_session_depth -= 1
        if self._rollout_session_depth == 0:
            try:
                self._backend.end_rollout_session(self._ctx)
            finally:
                self._finish_rollout_timing()

    async def end_rollout_session_async(self) -> None:
        """Async variant of :meth:`end_rollout_session`."""

        if self._backend is None or self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._rollout_session_depth <= 0:
            return
        self._rollout_session_depth -= 1
        if self._rollout_session_depth == 0:
            try:
                await self._backend.end_rollout_session_async(self._ctx)
            finally:
                self._finish_rollout_timing()

    def _finish_rollout_timing(self) -> None:
        """Record one rollout-session wall time for the current policy step."""

        if self._rollout_wall_start is None:
            return
        self._metric_timings["rollout"] = (
            self._metric_timings.get("rollout", 0.0) + time.perf_counter() - self._rollout_wall_start
        )
        self._rollout_wall_start = None

    def load_prompt_batches(
        self,
        dataset,
        *,
        batch_size: int,
        max_prompt_tokens: int,
        prompt_key: str = "prompt",
        solutions_key: str = "solutions",
    ) -> Iterable[PromptBatch]:
        """Yield tokenized prompt batches from a dataset-like object.

        Records whose prompt exceeds `max_prompt_tokens` are skipped. The full
        original record is preserved on each `PromptItem` so reward functions
        can read task-specific fields. The cursor advances even when records
        are skipped, so the iterator eventually walks the entire dataset.
        """

        cursor = 0
        total_skipped_long = 0
        shortest_skipped = None
        longest_skipped = None
        while cursor < len(dataset):
            items = []
            scanned = 0
            skipped_long = 0
            # Keep scanning until we accumulate `batch_size` accepted rows or
            # exhaust the dataset; over-long prompts increment the skip counter
            # but do not fill the batch.
            while len(items) < batch_size and cursor < len(dataset):
                record = dict(dataset[cursor])
                cursor += 1
                scanned += 1
                if record_has_image(record):
                    prompt = str(record.get(prompt_key, ""))
                    input_tokens, features = encode_multimodal_prompt(
                        self._tokenizer,
                        self._processor,
                        record,
                        prompt_key=prompt_key,
                    )
                    if features is not None:
                        record["features"] = features
                    record["tokens"] = input_tokens
                elif "tokens" in record:
                    prompt = str(record.get(prompt_key, "<encoded prompt>"))
                    input_tokens = [int(token) for token in record["tokens"]]
                    features = record.get("features")
                    image_counts = image_token_counts_from_features(features if isinstance(features, dict) else None)
                    if image_counts:
                        image_token_id = features.get("image_token_id") if isinstance(features, dict) else None
                        if image_token_id is None:
                            raise ValueError("multimodal prompt rows require features.image_token_id")
                        input_tokens, _ = expand_image_tokens(
                            input_tokens,
                            image_token_id=int(image_token_id),
                            image_token_counts=image_counts,
                        )
                        mrope_position_ids = mrope_position_ids_from_image_grid(
                            input_tokens,
                            image_token_id=int(image_token_id),
                            features=features,
                        )
                        if mrope_position_ids is not None:
                            features = dict(features)
                            features["mrope_position_ids"] = mrope_position_ids
                            record["features"] = features
                    record["tokens"] = input_tokens
                elif prompt_key in record:
                    prompt = record[prompt_key]
                    input_tokens = encode_generation_prompt(self._tokenizer, prompt)
                else:
                    raise ValueError(
                        f"dataset row must contain `{prompt_key}`; use --dataset-loader-fn to normalize raw rows"
                    )
                if len(input_tokens) > max_prompt_tokens:
                    skipped_long += 1
                    total_skipped_long += 1
                    shortest_skipped = (
                        len(input_tokens) if shortest_skipped is None else min(shortest_skipped, len(input_tokens))
                    )
                    longest_skipped = (
                        len(input_tokens) if longest_skipped is None else max(longest_skipped, len(input_tokens))
                    )
                    continue
                items.append(
                    PromptItem(
                        prompt=prompt,
                        solutions=record[solutions_key] if solutions_key in record else None,
                        input_tokens=input_tokens,
                        record=record,
                    )
                )
            if not items:
                if total_skipped_long == len(dataset) and total_skipped_long > 0:
                    raise ValueError(
                        f"all {total_skipped_long} dataset prompts exceed "
                        f"--max-prompt-tokens={max_prompt_tokens} "
                        f"(shortest={shortest_skipped}, longest={longest_skipped}); "
                        "increase --max-prompt-tokens or shorten the dataset prompts"
                    )
                break
            yield PromptBatch(
                items=items,
                scanned=scanned,
                skipped_long=skipped_long,
                total_skipped_long=total_skipped_long,
            )

    def rollout_batch(self, prompts: list[str], n_samples: int, sampling_params: SamplingParams) -> list[RolloutResult]:
        """Generate `n_samples` completions for each prompt in order."""

        prompt_tokens = [encode_generation_prompt(self._tokenizer, prompt) for prompt in prompts]
        return self.rollout_token_batch(prompt_tokens, n_samples, sampling_params)

    def rollout_token_batch(
        self,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
        prompt_features: list[dict | None] | None = None,
    ) -> list[RolloutResult]:
        """Generate completions for prompts that were already tokenized."""

        # Rollout is the natural boundary of a new policy step. Consecutive
        # rollouts before train stay on the same step instead of bumping twice.
        if self._rollout_session_depth <= 0:
            raise RuntimeError("rollout_token_batch must be called inside `async with trainer.rollout_session(...)`")
        self._begin_step()
        return self._backend.rollout_batch(
            self._ctx,
            _normalize_prompt_token_batch(prompt_tokens),
            n_samples,
            sampling_params,
            prompt_features=prompt_features,
        )

    async def rollout_token_batch_async(
        self,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
        prompt_features: list[dict | None] | None = None,
    ) -> list[RolloutResult]:
        """Async rollout variant for request-concurrent callers."""

        if self._rollout_session_depth <= 0:
            raise RuntimeError(
                "rollout_token_batch_async must be called inside `async with trainer.rollout_session(...)`"
            )
        self._begin_step()
        rollout_async = getattr(self._backend, "rollout_batch_async")
        return await rollout_async(
            self._ctx,
            _normalize_prompt_token_batch(prompt_tokens),
            n_samples,
            sampling_params,
            prompt_features=prompt_features,
        )

    def rollout_session(
        self,
        *,
        sampling_params: SamplingParams,
        loss_mask_policy: LossMaskPolicy | None = None,
        max_running_prompts: int | None = None,
        timeout_s: float = 300.0,
        proxy: bool = True,
    ) -> RolloutSession:
        """Create an async rollout session, optionally with an OpenAI-compatible proxy."""

        return RolloutSession(
            self,
            sampling_params=sampling_params,
            loss_mask_policy=loss_mask_policy,
            max_running_prompts=max_running_prompts,
            timeout_s=timeout_s,
            proxy=proxy,
        )

    def train(
        self,
        batch_data: list[TrainSequence],
        loss_fn: Callable,
        mini_bs: int = 8,
        gradient_accumulation_steps: int | None = None,
    ) -> dict[str, float]:
        """Run one backend training step with a caller-provided loss function.

        Returns whatever scalar metric dict the backend produces; when a
        `MetricsRecorder` is attached the dict and the accumulated step timings
        are also dispatched to TensorBoard.
        """

        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        self._begin_step()
        start = time.perf_counter()
        result = self._backend.train(self._ctx, batch_data, loss_fn, mini_bs, gradient_accumulation_steps)
        self._metric_timings["train"] = time.perf_counter() - start
        if isinstance(result, dict):
            if "rollout" in self._metric_timings:
                result["step_rollout_time_s"] = self._metric_timings["rollout"]
            result["step_train_time_s"] = self._metric_timings["train"]
            if self._step_wall_start is not None:
                result["step_e2e_time_s"] = time.perf_counter() - self._step_wall_start
        if self._metrics is not None:
            self._metrics.record_train_step(
                step=self._ctx.global_step,
                train_result=result,
                train_batch=batch_data,
                timings=self._metric_timings,
            )
        self.finish_step()
        return result

    def record_rollout_sample(self, sample: dict[str, Any]) -> None:
        """Persist a representative rollout sample when metrics recording is enabled."""

        if self._metrics is not None:
            self._metrics.record_rollout_sample(sample)

    def record_dashboard_state(
        self,
        *,
        stage: str,
        step: int | None = None,
        epoch: int | None = None,
        role: str | None = None,
        status: str = "running",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Persist dashboard state independently from TensorBoard scalar events."""

        if self._metrics is not None:
            self._metrics.record_dashboard_state(
                stage=stage, step=step, epoch=epoch, role=role, status=status, extra=extra
            )

    def ensure_roles(self, roles: dict[str, ModelRole]) -> None:
        """Prepare backend-owned auxiliary model roles for algorithms like PPO."""

        self._backend.ensure_roles(self._ctx, roles)

    def score_logprobs(
        self,
        role: str,
        token_rows: list[list[int]],
        *,
        features: list[dict | None] | None = None,
        routed_experts: list[object] | None = None,
        microbatch_size: int | None = None,
    ) -> list[list[float]]:
        """Score fixed token sequences with a backend-owned model role."""

        kwargs = {
            "features": features,
            "microbatch_size": self._score_micro_bs if microbatch_size is None else int(microbatch_size),
        }
        if routed_experts is not None:
            kwargs["routed_experts"] = routed_experts
        return self._backend.score_logprobs(self._ctx, role, token_rows, **kwargs)

    def score_values(
        self, role: str, token_rows: list[list[int]], *, features: list[dict | None] | None = None
    ) -> list[list[float]]:
        """Score per-token critic values with a backend-owned model role."""

        return self._backend.score_values(self._ctx, role, token_rows, features=features)

    def score_rewards(
        self, role: str, token_rows: list[list[int]], *, features: list[dict | None] | None = None
    ) -> list[float]:
        """Score sequence rewards with a backend-owned reward model role."""

        return self._backend.score_rewards(self._ctx, role, token_rows, features=features)

    def train_values(
        self,
        role: str,
        batch_data: list[TrainSequence],
        mini_bs: int,
        gradient_accumulation_steps: int | None = None,
        *,
        cliprange_value: float = 0.5,
        value_loss_coef: float = 0.5,
    ) -> dict[str, float]:
        """Train a backend-owned critic/value role.

        `cliprange_value` is the value-function clipping range from the PPO
        paper; `value_loss_coef` scales the MSE loss before it is added to the
        critic's objective.
        """

        return self._backend.train_values(
            self._ctx,
            role,
            batch_data,
            mini_bs,
            gradient_accumulation_steps,
            cliprange_value=cliprange_value,
            value_loss_coef=value_loss_coef,
        )

    def save_checkpoint(self, path: str) -> str:
        """Save a native backend checkpoint, or a PEFT artifact for native LoRA."""

        return self._backend.save_checkpoint(self._ctx, path)

    def export_adapter(self, path: str) -> str:
        """Export the live native LoRA weights as a standard PEFT adapter."""

        return self._backend.export_adapter(self._ctx, path)

    def close(self) -> None:
        """Release backend workers and local resources such as metric writers."""

        try:
            if self._backend is not None:
                self._backend.close()
        finally:
            self._backend = None
            self._initialized = False
            if self._metrics is not None:
                self._metrics.close()


def _normalize_prompt_token_batch(prompt_tokens: list[list[int]]) -> list[list[int]]:
    return [normalize_token_ids(row) for row in prompt_tokens]
