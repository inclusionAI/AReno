"""In-process MLX backend using one MLX-LM model for rollout and training."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from typing import Any

from areno.api.algorithms import describe_loss_fn
from areno.api.backend.base import Backend, BackendCapabilities, BackendRuntimeComponents, register_backend
from areno.api.backend.common import (
    accumulation_group_size,
    accumulation_steps,
    reduce_microbatch_metrics,
)
from areno.api.backend.mlx.checkpoint import save_checkpoint
from areno.api.backend.mlx.generation import ContinuousBatchScheduler, GenerationConfig
from areno.api.backend.mlx.losses import mlx_loss
from areno.api.backend.mlx.numerics import selected_token_logprobs
from areno.api.backend.mlx.optimizer import apply_optimizer_update, build_optimizer, set_group_learning_rates
from areno.api.backend.mlx.provider import MlxModelProvider, load_provider
from areno.api.backend.mlx.roles import MlxRole, load_language_role, load_value_role, role_output
from areno.api.backend.mlx.training import clip_grad_norm, make_train_batch, sft_target_token_count
from areno.api.config import MlxConfig
from areno.api.context import Context
from areno.api.models import BackendType, RolloutResult, SamplingParams, TrainSequence
from areno.api.roles import ModelRole


@register_backend(BackendType.MLX)
class MlxBackend(Backend):
    """Train and decode one policy object without cross-runtime weight copies."""

    def __init__(self) -> None:
        self.model = None
        self.provider: MlxModelProvider | None = None
        self.tokenizer = None
        self.processor = None
        self.optimizer = None
        self._optimizer_groups: list[tuple[str, Any, dict[str, Any]]] = []
        self.config: MlxConfig | None = None
        self.model_config: dict[str, Any] = {}
        self._rollout_active = False
        self._policy_version = 0
        self._rollout_version = -1
        self._rollout_scheduler: ContinuousBatchScheduler | None = None
        self._roles: dict[str, MlxRole] = {}
        self._compiled_losses: dict[tuple[str, tuple[tuple[str, object], ...]], Callable] = {}

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            algorithms=frozenset({"sft", "dpo", "gspo", "grpo", "ppo"}),
            model_roles=frozenset({"actor", "ref", "reward", "critic"}),
            multimodal=True,
        )

    def initialize(self, ctx: Context):
        """Load an MLX-LM model and create its optimizer in this process."""

        if ctx.world_size != 1:
            raise ValueError("MLX backend currently requires world_size=1")
        config = ctx.custom_config
        if config is None:
            config = MlxConfig()
        if not isinstance(config, MlxConfig):
            raise TypeError(f"MlxBackend requires MlxConfig, got {type(config)!r}")
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise RuntimeError("MLX backend requires an Apple Silicon AReno installation") from exc

        self.config = config
        model_path = self.config.model_path or ctx.model_path
        self.provider = load_provider(model_path, adapter_path=self.config.adapter_path)
        self.model = self.provider.model
        self.tokenizer = self.provider.tokenizer
        self.processor = self.provider.processor
        self.model_config = self.provider.config
        if mx.metal.is_available():
            max_working_set_size = mx.device_info().get("max_recommended_working_set_size")
            if max_working_set_size is not None:
                mx.set_wired_limit(int(max_working_set_size))
            # A unified rollout/training process cannot let Metal retain all
            # temporary generation buffers for reuse across the phase change.
            mx.set_cache_limit(0)
        if self.model_config.get("quantization") is not None:
            raise ValueError(
                "MLX training requires a non-quantized checkpoint so saved weights remain loadable by Transformers"
            )
        self._validate_tokenizer(ctx.tokenizer)
        optimizer_config = self.config.optimizer
        self.provider.configure_trainability(optimizer_config)
        self.optimizer, self._optimizer_groups = build_optimizer(optimizer_config)
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        self.model.train()
        mx.eval(self.model.parameters(), self.optimizer.state)

    def runtime_components(self) -> BackendRuntimeComponents:
        """Expose the provider tokenizer and multimodal processor through the backend contract."""

        return BackendRuntimeComponents(tokenizer=self.tokenizer, processor=self.processor)

    def close(self) -> None:
        try:
            import mlx.core as mx
        except ImportError:
            return
        self._compiled_losses.clear()
        if self._rollout_scheduler is not None:
            self._rollout_scheduler.close()
            self._rollout_scheduler = None
        self._roles.clear()
        self.provider = None
        self.model = None
        self.processor = None
        self.optimizer = None
        self._optimizer_groups = []
        mx.clear_cache()

    def begin_rollout_session(self, ctx: Context) -> None:
        del ctx
        if self._rollout_active:
            raise RuntimeError("MLX rollout session is already active")
        self._require_runtime()
        self.model.eval()
        if self._rollout_scheduler is None:
            self._rollout_scheduler = ContinuousBatchScheduler(
                self.provider,
                GenerationConfig(
                    max_running_prompts=self.config.max_running_prompts,
                    completion_batch_size=self.config.completion_batch_size,
                    prefill_batch_size=self.config.prefill_batch_size,
                    prefill_step_size=self.config.prefill_step_size,
                    max_kv_size=self.config.max_kv_size,
                    decode_progress_interval_s=self.config.decode_progress_interval_s,
                ),
            )
        self._rollout_version = self._policy_version
        self._rollout_active = True

    async def begin_rollout_session_async(self, ctx: Context) -> None:
        self.begin_rollout_session(ctx)

    async def sync_rollout_session_async(self, ctx: Context) -> None:
        del ctx
        if not self._rollout_active or self._rollout_version != self._policy_version:
            raise RuntimeError("MLX rollout session does not match the current policy version")

    def end_rollout_session(self, ctx: Context) -> None:
        del ctx
        self._rollout_active = False
        if not self.config.keep_rollout_state and self._rollout_scheduler is not None:
            self._rollout_scheduler.drop_state()
        if self.model is not None:
            self.model.train()

    async def end_rollout_session_async(self, ctx: Context) -> None:
        self.end_rollout_session(ctx)

    def rollout_batch(
        self,
        ctx: Context,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
        prompt_features: list[dict | None] | None = None,
    ) -> list[RolloutResult]:
        """Generate with MLX-LM continuous batching on the current policy."""

        self._require_runtime()
        if not self._rollout_active:
            raise RuntimeError("rollout_batch must run inside a rollout session")
        if self._rollout_scheduler is None:
            raise RuntimeError("MLX rollout scheduler is not initialized")
        return self._rollout_scheduler.submit(prompt_tokens, n_samples, sampling_params, prompt_features).result()

    async def rollout_batch_async(
        self,
        ctx: Context,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
        prompt_features: list[dict | None] | None = None,
    ) -> list[RolloutResult]:
        """Submit agent requests to the shared continuous-batch scheduler."""

        self._require_runtime()
        if not self._rollout_active or self._rollout_scheduler is None:
            raise RuntimeError("rollout_batch_async must run inside a rollout session")
        return await self._rollout_scheduler.submit_async(prompt_tokens, n_samples, sampling_params, prompt_features)

    def train(
        self,
        ctx: Context,
        batch_data: list[TrainSequence],
        loss_fn: Callable,
        mini_bs: int,
        gradient_accumulation_steps: int | None = None,
    ) -> dict[str, float]:
        """Apply native MLX gradients to the same model used for rollout."""

        self._require_runtime()
        if self._rollout_active:
            raise RuntimeError("cannot update MLX policy during an active rollout session")
        if not batch_data:
            raise ValueError("train batch is empty")
        if mini_bs < 1:
            raise ValueError("mini_bs must be positive")

        import mlx.core as mx
        from mlx.utils import tree_map

        spec = describe_loss_fn(loss_fn)
        microbatches = [batch_data[start : start + mini_bs] for start in range(0, len(batch_data), mini_bs)]
        accumulation = accumulation_steps(len(microbatches), gradient_accumulation_steps)

        self.model.train()
        started = time.perf_counter()
        grad_accum = None
        group_count = 0
        losses: list[float] = []
        metric_rows: list[dict[str, float]] = []
        grad_norms: list[float] = []
        learning_rates = set_group_learning_rates(self._optimizer_groups, ctx.global_step)
        learning_rate = learning_rates["model"]
        value_and_grad = self._value_and_grad(spec.name, spec.kwargs)

        for index, rows in enumerate(microbatches):
            batch = self.provider.prepare_train_batch(make_train_batch(rows), rows)
            if spec.name == "sft":
                group_size = accumulation_group_size(index, len(microbatches), accumulation)
                group_start = (index // accumulation) * accumulation
                group_end = group_start + group_size
                batch["_sft_total_target_tokens"] = mx.array(
                    sum(sft_target_token_count(group) for group in microbatches[group_start:group_end]),
                    dtype=mx.float32,
                )
                batch["_sft_grad_scale"] = mx.array(group_size, dtype=mx.float32)
            (loss, stats), grads = value_and_grad(batch)
            grad_accum = grads if grad_accum is None else tree_map(lambda left, right: left + right, grad_accum, grads)
            group_count += 1
            should_step = group_count == accumulation or index + 1 == len(microbatches)
            if should_step:
                mx.eval(loss, grad_accum)
                mx.clear_cache()
                grads = tree_map(lambda value: value / group_count, grad_accum)
                grads, grad_norm = clip_grad_norm(grads, self.config.optimizer.get("grad_clip_norm"))
                mx.eval(grad_norm)
                apply_optimizer_update(self.model, self.optimizer, grads)
                grad_norms.append(float(grad_norm.item()))
                grad_accum = None
                group_count = 0
            else:
                mx.eval(loss, grads)

            losses.append(float(loss.item()))
            metric_rows.append({str(key): float(value.item()) for key, value in stats.items()})

        self._policy_version += 1
        self._rollout_version = -1
        mx.clear_cache()
        result = reduce_microbatch_metrics(metric_rows)
        result.update(
            {
                "loss": sum(losses) / len(losses),
                "lr": learning_rate,
                "grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
                "policy_train_wall_time_s": time.perf_counter() - started,
            }
        )
        for group, rate in learning_rates.items():
            if group != "model":
                result[f"{group}_lr"] = rate
        return result

    def save_checkpoint(self, ctx: Context, path: str) -> str:
        """Save MLX weights plus tokenizer/config files in MLX-LM format."""

        self._require_runtime()
        return save_checkpoint(
            self.provider,
            self.optimizer,
            source_path=self.config.model_path or ctx.model_path,
            destination_path=path,
            policy_version=self._policy_version,
            global_step=ctx.global_step,
        )

    def ensure_roles(self, ctx: Context, roles: dict[str, ModelRole]) -> None:
        """Load PPO/DPO auxiliary roles with MLX-native modules."""

        del ctx
        self._require_runtime()
        unsupported = set(roles) - self.capabilities().model_roles
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(f"MLX backend does not support model roles: {names}")

        for name, role in roles.items():
            if name == "actor" or name in self._roles:
                continue
            if name == "ref":
                loaded = load_language_role(role.path)
            elif name == "critic":
                loaded = load_value_role(
                    role.path,
                    trainable=True,
                    learning_rate=role.optimizer_lr,
                    reward=False,
                )
            elif name == "reward":
                loaded = load_value_role(
                    role.path,
                    trainable=False,
                    learning_rate=None,
                    reward=True,
                )
            else:
                raise ValueError(f"unknown MLX model role: {name}")
            self._validate_loaded_tokenizer(loaded.tokenizer)
            self._roles[name] = loaded

    def score_logprobs(
        self,
        ctx: Context,
        role: str,
        token_rows: list[list[int]],
        *,
        features: list[dict | None] | None = None,
        microbatch_size: int = 8,
    ) -> list[list[float]]:
        """Score fixed token rows with the actor or a frozen reference model."""

        del ctx
        self._require_runtime()
        if microbatch_size < 1:
            raise ValueError("microbatch_size must be positive")
        if role == "actor":
            provider = self.provider
        else:
            loaded = self._roles.get(role)
            if loaded is None:
                raise RuntimeError(f"MLX model role {role!r} is not initialized")
            provider = loaded.provider

        results: list[list[float]] = []
        for start in range(0, len(token_rows), microbatch_size):
            row_features = features[start : start + microbatch_size] if features is not None else None
            results.extend(
                _score_token_rows(
                    provider,
                    token_rows[start : start + microbatch_size],
                    self.tokenizer,
                    row_features,
                )
            )
        return results

    def score_values(
        self,
        ctx: Context,
        role: str,
        token_rows: list[list[int]],
        *,
        features: list[dict | None] | None = None,
    ) -> list[list[float]]:
        """Score one critic value for every input-token position."""

        del ctx
        loaded = self._require_value_role(role, features)
        return _score_value_rows(loaded, token_rows, self.tokenizer, features=features)

    def score_rewards(
        self,
        ctx: Context,
        role: str,
        token_rows: list[list[int]],
        *,
        features: list[dict | None] | None = None,
    ) -> list[float]:
        """Return the reward head's final-token score for every row."""

        del ctx
        loaded = self._require_value_role(role, features)
        rows = _score_value_rows(loaded, token_rows, self.tokenizer, features=features, all_outputs=True)
        return [float(row[-1][-1] if isinstance(row[-1], list) else row[-1]) for row in rows]

    def train_values(
        self,
        ctx: Context,
        role: str,
        batch_data: list[TrainSequence],
        mini_bs: int,
        gradient_accumulation_steps: int | None = None,
        *,
        cliprange_value: float = 0.5,
        value_loss_coef: float = 0.5,
    ) -> dict[str, float]:
        """Train the MLX critic with the same clipped value objective as CUDA."""

        del ctx
        loaded = self._roles.get(role)
        if loaded is None or loaded.module is None or loaded.optimizer is None:
            raise RuntimeError(f"MLX trainable value role {role!r} is not initialized")
        if not batch_data:
            raise ValueError("critic train batch is empty")
        if mini_bs < 1:
            raise ValueError("mini_bs must be positive")

        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_map

        microbatches = [batch_data[start : start + mini_bs] for start in range(0, len(batch_data), mini_bs)]
        accumulation = accumulation_steps(len(microbatches), gradient_accumulation_steps)
        value_and_grad = nn.value_and_grad(
            loaded.module,
            partial(
                _critic_loss,
                cliprange_value=float(cliprange_value),
                value_loss_coef=float(value_loss_coef),
            ),
        )
        loaded.module.train()
        grad_accum = None
        group_count = 0
        losses: list[float] = []
        clipfracs: list[float] = []
        grad_norms: list[float] = []
        for index, rows in enumerate(microbatches):
            batch = loaded.provider.prepare_train_batch(make_train_batch(rows), rows)
            (loss, stats), grads = value_and_grad(loaded.module, batch)
            grad_accum = grads if grad_accum is None else tree_map(lambda left, right: left + right, grad_accum, grads)
            group_count += 1
            if group_count == accumulation or index + 1 == len(microbatches):
                grads = tree_map(lambda value: value / group_count, grad_accum)
                grads, grad_norm = clip_grad_norm(grads, self.config.optimizer.get("grad_clip_norm"))
                loaded.optimizer.update(loaded.module, grads)
                mx.eval(loaded.module.parameters(), loaded.optimizer.state, loss, grad_norm)
                grad_norms.append(float(grad_norm.item()))
                grad_accum = None
                group_count = 0
            else:
                mx.eval(loss, grads)
            losses.append(float(loss.item()))
            clipfracs.append(float(stats["critic_value_clipfrac"].item()))
        loaded.module.eval()
        return {
            "critic_value_loss": sum(losses) / len(losses),
            "critic_value_clipfrac": sum(clipfracs) / len(clipfracs),
            "critic_grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
        }

    def dp_size(self, ctx: Context) -> int:
        del ctx
        return 1

    def model_context_len(self, ctx: Context) -> int | None:
        del ctx
        for key in ("max_position_embeddings", "max_sequence_length", "model_max_length"):
            value = self.model_config.get(key)
            if value is not None:
                return int(value)
        return None

    def probe_rollout_cache(
        self,
        ctx: Context,
        *,
        max_new_tokens: int,
        max_running_prompts: int,
        max_prompt_len: int,
    ) -> float:
        del ctx, max_new_tokens, max_running_prompts, max_prompt_len
        import mlx.core as mx

        return float(mx.get_peak_memory() / 1e9)

    def _value_and_grad(self, name: str, kwargs: dict[str, object]):
        import mlx.core as mx
        import mlx.nn as nn

        key = (name, tuple(sorted(kwargs.items())))
        cached = self._compiled_losses.get(key)
        if cached is not None:
            return cached

        def loss(model, batch):
            targets = batch["input_ids"][:, 1:]
            selected = self.provider.selected_token_logprobs(
                batch,
                targets,
                chunk_size=self.config.logits_chunk_size,
            )
            return mlx_loss(name, batch, selected, **kwargs)

        value_and_grad = nn.value_and_grad(self.model, loss)
        if self.config.compile_train_step and not self.provider.is_multimodal:
            state = [self.model.state, mx.random.state]

            @partial(mx.compile, inputs=state, outputs=state)
            def compiled(batch):
                return value_and_grad(self.model, batch)

            value_and_grad_fn = compiled
        else:

            def value_and_grad_fn(batch):
                return value_and_grad(self.model, batch)

        self._compiled_losses[key] = value_and_grad_fn
        return value_and_grad_fn

    def _validate_tokenizer(self, trainer_tokenizer: Any) -> None:
        probe = "AReno MLX tokenizer probe"
        expected = list(trainer_tokenizer.encode(probe, add_special_tokens=False))
        actual = list(self.tokenizer.encode(probe, add_special_tokens=False))
        if expected != actual:
            raise ValueError("AReno and MLX-LM tokenizers produce different token ids")

    def _validate_loaded_tokenizer(self, tokenizer: Any) -> None:
        probe = "AReno MLX tokenizer probe"
        expected = list(self.tokenizer.encode(probe, add_special_tokens=False))
        actual = list(tokenizer.encode(probe, add_special_tokens=False))
        if expected != actual:
            raise ValueError("MLX actor and reference tokenizers produce different token ids")

    def _require_value_role(self, role: str, features: list[dict | None] | None) -> MlxRole:
        self._require_runtime()
        loaded = self._roles.get(role)
        if loaded is None or loaded.module is None:
            raise RuntimeError(f"MLX value role {role!r} is not initialized")
        loaded.module.eval()
        return loaded

    def _enable_gradient_checkpointing(self) -> None:
        try:
            from mlx_lm.tuner.trainer import grad_checkpoint

            model = self.provider.generation_model
            layers = getattr(model, "layers", None)
            if layers is None and hasattr(model, "model"):
                layers = getattr(model.model, "layers", None)
            if layers:
                grad_checkpoint(layers[0])
        except (AttributeError, IndexError, TypeError):
            return

    def _require_runtime(self) -> None:
        if (
            self.provider is None
            or self.model is None
            or self.tokenizer is None
            or self.optimizer is None
            or self.config is None
        ):
            raise RuntimeError("MLX backend is not initialized")


def _score_token_rows(
    provider: MlxModelProvider,
    token_rows: list[list[int]],
    tokenizer: Any,
    features: list[dict | None] | None,
) -> list[list[float]]:
    import mlx.core as mx
    import numpy as np

    if not token_rows:
        return []
    if any(len(row) < 1 for row in token_rows):
        raise ValueError("token rows must be non-empty")
    lengths = [len(row) for row in token_rows]
    width = max(lengths)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    tokens = np.full((len(token_rows), width), int(pad_id or 0), dtype=np.int32)
    for index, row in enumerate(token_rows):
        tokens[index, : len(row)] = row
    input_ids = mx.array(tokens)
    batch = provider.prepare_score_batch(input_ids, lengths, features)
    logits = provider.forward_logits(batch)
    selected = selected_token_logprobs(logits, input_ids[:, 1:])
    mx.eval(selected)
    values = np.asarray(selected.astype(mx.float32))
    return [[0.0, *values[index, : length - 1].astype(float).tolist()] for index, length in enumerate(lengths)]


def _score_value_rows(
    role: MlxRole,
    token_rows: list[list[int]],
    tokenizer: Any,
    *,
    features: list[dict | None] | None = None,
    all_outputs: bool = False,
) -> list[list[Any]]:
    import mlx.core as mx
    import numpy as np

    if not token_rows:
        return []
    if any(not row for row in token_rows):
        raise ValueError("token rows must be non-empty")
    lengths = [len(row) for row in token_rows]
    width = max(lengths)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    tokens = np.full((len(token_rows), width), int(pad_id or 0), dtype=np.int32)
    for index, row in enumerate(token_rows):
        tokens[index, : len(row)] = row
    input_ids = mx.array(tokens)
    batch = role.provider.prepare_score_batch(input_ids, lengths, features)
    output = role_output(role, batch)
    mx.eval(output)
    values = np.asarray(output.astype(mx.float32))
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim == 2:
        return [values[index, :length].astype(float).tolist() for index, length in enumerate(lengths)]
    if values.ndim == 3 and all_outputs:
        return [values[index, :length].astype(float).tolist() for index, length in enumerate(lengths)]
    raise ValueError(f"value role returned unsupported shape {values.shape}")


def _critic_loss(module: Any, batch: dict[str, Any], *, cliprange_value: float, value_loss_coef: float):
    import mlx.core as mx

    predicted = module(batch)
    if predicted.shape[-1] != 1:
        raise ValueError("critic head must return exactly one value per token")
    predicted = predicted[:, :-1, 0]
    baseline = batch["values"]
    returns = batch["returns"]
    mask = batch["response_mask"]
    clipped = baseline + mx.clip(predicted - baseline, -cliprange_value, cliprange_value)
    losses = mx.maximum((predicted - returns) ** 2, (clipped - returns) ** 2)
    count = mx.maximum(mask.sum(), mx.array(1.0))
    loss = 0.5 * value_loss_coef * (losses * mask).sum() / count
    clipfrac = (((predicted - baseline) > cliprange_value) | ((predicted - baseline) < -cliprange_value)).astype(
        mx.float32
    )
    return loss, {"critic_value_loss": loss, "critic_value_clipfrac": (clipfrac * mask).sum() / count}


__all__ = ["MlxBackend"]
