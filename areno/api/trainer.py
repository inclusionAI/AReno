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

from areno.api.config import BackendConfig, coerce_backend_config, resolve_backend_type
from areno.api.context import Context
from areno.api.data import PromptBatch, PromptItem
from areno.api.metrics import MetricsRecorder
from areno.api.tokenizer import encode_generation_prompt, eos_token_ids, load_tokenizer
from areno.api.backend.base import Backend, get_backend_cls
from areno.api.models import SamplingParams, RolloutResult, TrainSequence, BackendType
from areno.api.roles import ModelRole


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
    ) -> None:
        """Create a trainer without starting backend workers.

        Call `init()` before rollout or training. `world_size` is the total
        number of devices/workers visible to the selected backend.
        """

        self._tokenizer = None
        self._backend: Backend | None = None
        # Resolve backend type from the explicit value or default to Areno.
        self._backend_type = resolve_backend_type(backend_type, custom_config)
        self._model_path = model_path
        self._ctx: Context | None = None
        self._world_size = world_size
        self._initialized = False
        self._custom_config = coerce_backend_config(self._backend_type, custom_config)
        self._metrics = MetricsRecorder(metrics_log_dir) if metrics_log_dir else None
        # Per-step wall-time bag accumulated by the rollout/train helpers
        # so `record_train_step` can flush a complete timing snapshot.
        self._metric_timings: dict[str, float] = {}
        self._step_active = False

    def init(self) -> None:
        """Load tokenizer, create backend context, and initialize workers."""

        real_path = self._model_path
        self._tokenizer = load_tokenizer(real_path)
        self._ctx = Context(self._world_size, real_path, self._tokenizer, self._custom_config, eos_token_ids(real_path, self._tokenizer))
        backend_cls = get_backend_cls(self._backend_type)
        if backend_cls is None:
            raise ValueError(f"unsupported backend type: {self._backend_type}")
        self._backend = backend_cls()
        self._backend.initialize(self._ctx)
        self._initialized = True

    def get_tokenizer(self) -> Any:
        """Return the initialized tokenizer for prompt and completion handling."""

        return self._tokenizer

    def _begin_step(self) -> None:
        """Open a trainer-owned step if rollout/train has not already done so."""

        if self._ctx is None:
            raise RuntimeError("Trainer is not initialized")
        if self._step_active:
            return
        self._ctx.step()
        self._metric_timings = {}
        self._step_active = True

    def finish_step(self) -> None:
        """Close the current trainer-owned step without running actor train."""

        self._step_active = False

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
        while cursor < len(dataset):
            items = []
            scanned = 0
            skipped_long = 0
            # Keep scanning until we accumulate `batch_size` accepted rows or
            # exhaust the dataset; over-long prompts increment the skip counter
            # but do not fill the batch.
            while len(items) < batch_size and cursor < len(dataset):
                record = dataset[cursor]
                cursor += 1
                scanned += 1
                if prompt_key not in record:
                    raise ValueError(f"dataset row must contain `{prompt_key}`; use --dataset-loader-fn to normalize raw rows")
                prompt = record[prompt_key]
                input_tokens = encode_generation_prompt(self._tokenizer, prompt)
                if len(input_tokens) > max_prompt_tokens:
                    skipped_long += 1
                    total_skipped_long += 1
                    continue
                items.append(
                    PromptItem(
                        prompt=prompt,
                        solutions=record[solutions_key] if solutions_key in record else None,
                        input_tokens=input_tokens,
                        record=dict(record),
                    )
                )
            if not items:
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
    ) -> list[RolloutResult]:
        """Generate completions for prompts that were already tokenized."""

        # Rollout is the natural boundary of a new policy step. Consecutive
        # rollouts before train stay on the same step instead of bumping twice.
        self._begin_step()
        start = time.perf_counter()
        result = self._backend.rollout_batch(self._ctx, prompt_tokens, n_samples, sampling_params)
        self._metric_timings["rollout"] = self._metric_timings.get("rollout", 0.0) + time.perf_counter() - start
        return result

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
        if self._metrics is not None:
            self._metrics.record_train_step(
                step=self._ctx.global_step,
                train_result=result,
                train_batch=batch_data,
                timings=self._metric_timings,
            )
        self.finish_step()
        return result

    def ensure_roles(self, roles: dict[str, ModelRole]) -> None:
        """Prepare backend-owned auxiliary model roles for algorithms like PPO."""

        self._backend.ensure_roles(self._ctx, roles)

    def score_logprobs(self, role: str, token_rows: list[list[int]]) -> list[list[float]]:
        """Score fixed token sequences with a backend-owned model role."""

        return self._backend.score_logprobs(self._ctx, role, token_rows)

    def score_values(self, role: str, token_rows: list[list[int]]) -> list[list[float]]:
        """Score per-token critic values with a backend-owned model role."""

        return self._backend.score_values(self._ctx, role, token_rows)

    def score_rewards(self, role: str, token_rows: list[list[int]]) -> list[float]:
        """Score sequence rewards with a backend-owned reward model role."""

        return self._backend.score_rewards(self._ctx, role, token_rows)

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
        """Save a HuggingFace-compatible checkpoint when supported by backend."""

        return self._backend.save_checkpoint(self._ctx, path)

    def close(self) -> None:
        """Release local resources such as metric writers."""

        if self._metrics is not None:
            self._metrics.close()
