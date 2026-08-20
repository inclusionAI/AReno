"""CUDA adapter from the public `Trainer` API onto `ArenoEngine`.

areno runs a co-located train + rollout engine in the same process group.
This file is the thin glue that:

- starts the engine with the dataclass-validated `CudaConfig`,
- forwards rollout requests through `generate_rollout` while translating
  SDK `SamplingParams` into the engine's own type,
- packs `TrainSequence` objects with the CUDA training adapter, and
- routes the caller's loss function into the engine's training step via the
  small `_external_loss_dispatcher` hook so the engine itself stays
  algorithm-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from areno.api.backend.base import Backend, BackendCapabilities, register_backend
from areno.api.backend.common import (
    MetricReduction,
    expand_prompt_features,
    expand_prompts,
    group_rollout_sequences,
    metric_reduction,
)
from areno.api.backend.cuda.checkpoint import save_checkpoint
from areno.api.backend.cuda.generation import rollout_options
from areno.api.backend.cuda.losses import dispatch_loss
from areno.api.backend.cuda.training import (
    annotate_sft_token_mean_packs,
    is_sft_loss_fn,
    make_train_pack,
    pad_token_id,
    sft_target_token_count,
)
from areno.api.config import CudaConfig
from areno.api.context import Context
from areno.api.models import BackendType, RolloutResult, RolloutSequence, SamplingParams, TrainSequence
from areno.api.roles import ModelRole

logger = logging.getLogger(__name__)
_SYS_PATH_LOCK = Lock()
_SYS_PATH_PREFERRED = False


def _prefer_repo_areno() -> None:
    """Prefer this repository's engine packages over installed wheels.

    Promote the repository root so local code wins over stale installed
    packages for `areno`, `areno.models`, and `areno.accel`.
    """

    global _SYS_PATH_PREFERRED
    with _SYS_PATH_LOCK:
        if _SYS_PATH_PREFERRED:
            return
        repo_root = Path(__file__).resolve().parents[4]
        if not (repo_root / "areno").is_dir():
            _SYS_PATH_PREFERRED = True
            return
        repo_root_str = str(repo_root)
        try:
            sys.path.remove(repo_root_str)
        except ValueError:
            pass
        sys.path.insert(0, repo_root_str)
        _SYS_PATH_PREFERRED = True


@register_backend(BackendType.CUDA)
class CudaBackend(Backend):
    """Backend adapter that maps `Trainer` calls onto `areno.ArenoEngine`."""

    def __init__(self):
        """Create an adapter; workers are started by `initialize`."""

        super().__init__()
        self._train_engine = None
        self._rollout_engine = None
        self._separate_rollout = False
        self._train_policy_version = 0
        self._rollout_policy_version = 0
        self._policy_sync_lock = Lock()
        self._rollout_session_active = False
        self._policy_sync_bucket_bytes = 64 * 1024 * 1024
        self._pending_policy_sync_metrics: dict[str, float] = {}
        # Per-step wall-time accumulators used to print the
        # rollout/train/end-to-end breakdown after `train` completes.
        self._step_e2e_start: float | None = None
        self._step_rollout_time_s = 0.0

    @classmethod
    def capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            algorithms=frozenset({"sft", "dpo", "gspo", "grpo", "ppo"}),
            model_roles=frozenset({"actor", "ref", "reward", "critic"}),
            multimodal=True,
            distributed=True,
            custom_loss=True,
        )

    def _require_train_engine(self):
        """Return the initialized training engine."""

        if self._train_engine is None:
            raise RuntimeError("CudaBackend is not initialized")
        return self._train_engine

    def _require_rollout_engine(self):
        """Return the independent rollout engine or the colocated engine."""

        return self._rollout_engine or self._require_train_engine()

    def close(self) -> None:
        """Stop backend worker processes and release engine resources."""

        engines = tuple(engine for engine in (self._rollout_engine, self._train_engine) if engine is not None)
        self._train_engine = None
        self._rollout_engine = None
        self._separate_rollout = False
        self._rollout_session_active = False
        self._pending_policy_sync_metrics = {}
        for engine in engines:
            engine.close()

    def initialize(self, ctx: Context):
        _prefer_repo_areno()
        from areno import ArenoEngine, OptimizerConfig, RuntimeConfig

        cfg = ctx.custom_config
        if cfg is None:
            cfg = CudaConfig()
        if not isinstance(cfg, CudaConfig):
            raise TypeError(f"CudaBackend requires CudaConfig, got {type(cfg)!r}")

        # Derive the DP/TP layout: world = dp * tp must hold exactly. When the
        # caller omits `dp_size` we infer it from `world_size / tp_size`.
        world_size = int(ctx.world_size)
        tp_size = int(cfg.tp_size)
        if world_size % tp_size != 0:
            raise ValueError(f"world_size={world_size} must be divisible by tp_size={tp_size}")
        dp_size = cfg.dp_size
        dp_size = world_size // tp_size if dp_size is None else int(dp_size)
        if dp_size * tp_size != world_size:
            raise ValueError(f"dp_size * tp_size must equal world_size, got {dp_size} * {tp_size} != {world_size}")
        devices = cfg.devices
        if devices is None and ctx.world_size:
            devices = list(range(world_size))
        if devices is None or len(devices) != world_size:
            raise ValueError(f"training device count must equal world_size={world_size}")
        if cfg.rollout_tp_size is not None and cfg.rollout_devices is None:
            raise ValueError("rollout_tp_size requires rollout_devices")

        if not cfg.uses_separate_rollout_engine():
            self._train_engine = ArenoEngine.from_pretrained(
                cfg.model_path or ctx.model_path,
                tp_size=tp_size,
                dp_size=dp_size,
                devices=devices,
                dummy_load=cfg.dummy_load,
                optimizer_config=OptimizerConfig(**cfg.optimizer),
                runtime_config=RuntimeConfig(**cfg.runtime),
                loss_fn=dispatch_loss,
                policy_sync_bucket_mb=cfg.policy_sync_bucket_mb,
            )
            return
        self._policy_sync_bucket_bytes = cfg.policy_sync_bucket_mb * 1024 * 1024

        from areno.engine.protocol import (
            ClusterPartition,
            DistributedWorldSpec,
            find_free_port,
            start_partitioned_clusters,
        )

        rollout_devices = list(cfg.rollout_devices or ())
        rollout_tp_size = cfg.resolved_rollout_tp_size()
        if not rollout_devices:
            raise ValueError("rollout_devices must be non-empty")
        if len(rollout_devices) % rollout_tp_size != 0:
            raise ValueError("len(rollout_devices) must be divisible by rollout_tp_size")
        train_partition = ClusterPartition(
            role="train",
            global_rank_offset=0,
            local_world_size=world_size,
            tp_size=tp_size,
            devices=tuple(devices or ()),
        )
        rollout_partition = ClusterPartition(
            role="rollout",
            global_rank_offset=world_size,
            local_world_size=len(rollout_devices),
            tp_size=rollout_tp_size,
            devices=tuple(rollout_devices),
        )
        world_spec = DistributedWorldSpec(
            master_addr="127.0.0.1",
            master_port=find_free_port(),
            global_world_size=world_size + len(rollout_devices),
            train=train_partition,
            rollout=rollout_partition,
        )
        common = {
            "dummy_load": cfg.dummy_load,
            "runtime_config": RuntimeConfig(**cfg.runtime),
            "start": False,
            "policy_sync_bucket_mb": cfg.policy_sync_bucket_mb,
        }
        self._train_engine = ArenoEngine.from_pretrained(
            cfg.model_path or ctx.model_path,
            tp_size=tp_size,
            dp_size=dp_size,
            devices=devices,
            optimizer_config=OptimizerConfig(**cfg.optimizer),
            loss_fn=dispatch_loss,
            role="train",
            cluster_kwargs={"world_spec": world_spec, "partition": train_partition},
            **common,
        )
        rollout_runtime = RuntimeConfig(**cfg.runtime)
        self._rollout_engine = ArenoEngine.from_pretrained(
            cfg.model_path or ctx.model_path,
            tp_size=rollout_tp_size,
            dp_size=len(rollout_devices) // rollout_tp_size,
            devices=rollout_devices,
            dummy_load=cfg.dummy_load,
            runtime_config=rollout_runtime,
            loss_fn=None,
            role="rollout",
            policy_sync_bucket_mb=cfg.policy_sync_bucket_mb,
            start=False,
            cluster_kwargs={"world_spec": world_spec, "partition": rollout_partition},
        )
        try:
            start_partitioned_clusters(
                self._train_engine.cluster,
                self._rollout_engine.cluster,
                world_spec,
            )
        except BaseException:
            self.close()
            raise
        self._separate_rollout = True

    def _validate_policy_plans(self) -> None:
        """Refresh live tensor views and reject any cross-rank layout mismatch."""

        # Model onload/offload replaces Parameter storage. Rebuild the plan for
        # every version so tasks always reference the currently-live tensors;
        # the metadata comparison is cheap and also guards adapter drift.
        from areno.engine.protocol import Op

        train_results = self._require_train_engine().cluster.call(Op.POLICY_SYNC_PLAN)
        rollout_results = self._require_rollout_engine().cluster.call(Op.POLICY_SYNC_PLAN)
        train_plan = train_results[0]
        rollout_plan = rollout_results[0]
        if any(result != train_plan for result in train_results):
            raise RuntimeError("training ranks produced inconsistent policy synchronization plans")
        if any(result != rollout_plan for result in rollout_results):
            raise RuntimeError("rollout ranks produced inconsistent policy synchronization plans")
        if train_plan != rollout_plan:
            raise RuntimeError("train and rollout policy synchronization layouts do not match")

    def _sync_policy_if_needed(self) -> None:
        """Synchronize one new optimizer version before rollout starts."""

        if not self._separate_rollout or self._train_policy_version == self._rollout_policy_version:
            return
        with self._policy_sync_lock:
            if self._train_policy_version == self._rollout_policy_version:
                return
            sync_started = time.perf_counter()
            self._validate_policy_plans()
            from areno.engine.protocol import Op, PolicySyncPayload

            version = self._train_policy_version
            payload = PolicySyncPayload(version=version, bucket_bytes=self._policy_sync_bucket_bytes)
            train_call = self._require_train_engine().cluster.submit(Op.POLICY_SYNC_PUBLISH, payload)
            rollout_call = self._require_rollout_engine().cluster.submit(Op.POLICY_SYNC_RECEIVE, payload)
            train_call.result()
            rollout_results = rollout_call.result()
            self._rollout_policy_version = version
            summary = next((result for result in rollout_results if isinstance(result, dict)), None)
            if summary is not None:
                total_s = time.perf_counter() - sync_started
                transfer_s = max(float(result["elapsed_s"]) for result in rollout_results if isinstance(result, dict))
                transferred_bytes = int(summary["bytes"])
                tensors = int(summary["tensors"])
                throughput_gbps = transferred_bytes * 8 / max(transfer_s, 1e-12) / 1e9
                self._pending_policy_sync_metrics = {
                    "policy_sync_time_s": total_s,
                    "policy_sync_transfer_time_s": transfer_s,
                    "policy_sync_bytes": float(transferred_bytes),
                    "policy_sync_tensors": float(tensors),
                    "policy_sync_throughput_gbps": throughput_gbps,
                }
                logger.info(
                    "policy_sync version=%d total_s=%.6f transfer_s=%.6f bytes=%d tensors=%d throughput_gbps=%.3f",
                    version,
                    total_s,
                    transfer_s,
                    transferred_bytes,
                    tensors,
                    throughput_gbps,
                )

    def rollout_batch(
        self,
        ctx: Context,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
        prompt_features: list[dict | None] | None = None,
    ) -> list[RolloutResult]:
        engine = self._require_rollout_engine()
        if not prompt_tokens:
            return []
        if prompt_features is not None and len(prompt_features) != len(prompt_tokens):
            raise ValueError("prompt_features must have the same length as prompt_tokens")
        if not self._rollout_session_active:
            self._sync_policy_if_needed()
        # Replicate each already-tokenized prompt `n_samples` times so the
        # engine treats each completion as independent while preserving the
        # `[prompt0_sample0, prompt0_sample1, ..., promptN_sampleK]` layout.
        flat_prompts = expand_prompts(prompt_tokens, n_samples)
        flat_features = expand_prompt_features(prompt_features, len(prompt_tokens), n_samples)
        options = rollout_options(ctx, sampling_params)

        if self._step_e2e_start is None:
            self._step_e2e_start = time.perf_counter()
            self._step_rollout_time_s = 0.0
        # Translate the public SamplingParams into the engine's native type.
        # Greedy decoding is implemented by forcing temperature to zero.
        rollout = engine.generate_rollout(
            flat_prompts,
            prompt_features=flat_features,
            max_new_tokens=sampling_params.max_new_tokens,
            max_running_prompts=options["max_running_prompts"],
            max_prompt_len=options["max_prompt_len"],
            eos_token_id=options["eos_token_id"],
            decode_progress_interval_s=options["decode_progress_interval_s"],
            sampling_params=options["sampling_params"],
        )
        # Repack the flat result into per-prompt groups of `n_samples`
        # completions so downstream code can iterate `for item, result`.
        sequences = [
            RolloutSequence(resp_tokens=tokens, resp_logprobs=rollout.logprobs[index, : len(tokens)].tolist())
            for index, tokens in enumerate(rollout.response_ids)
        ]
        return group_rollout_sequences(sequences, len(prompt_tokens), n_samples)

    def begin_rollout_session(self, ctx: Context) -> None:
        """Prepare colocated actor state before rollout requests are issued."""

        del ctx
        self._sync_policy_if_needed()
        self._require_rollout_engine().begin_rollout_session()
        self._rollout_session_active = True

    async def begin_rollout_session_async(self, ctx: Context) -> None:
        """Async rollout-session begin hook for agentic callers."""

        del ctx
        await asyncio.to_thread(self._sync_policy_if_needed)
        await self._require_rollout_engine().begin_rollout_session_async()
        self._rollout_session_active = True

    async def sync_rollout_session_async(self, ctx: Context) -> None:
        """Synchronize worker TP groups before agentic request rollout."""

        del ctx
        await self._require_rollout_engine().sync_rollout_session_async()

    def dp_size(self, ctx: Context) -> int:
        """Return the engine's effective DP size after backend initialization."""

        del ctx
        return int(self._require_rollout_engine().config.dp_size)

    def model_context_len(self, ctx: Context) -> int | None:
        """Return the checkpoint's max position embeddings from the loaded engine config."""

        del ctx
        return int(self._require_rollout_engine().config.model.max_position_embeddings)

    def probe_rollout_cache(
        self,
        ctx: Context,
        *,
        max_new_tokens: int,
        max_running_prompts: int,
        max_prompt_len: int,
    ) -> float:
        """Allocate rollout cache and capture decode graphs without generating."""

        del ctx
        return self._require_rollout_engine().probe_rollout_cache(
            max_new_tokens=max_new_tokens,
            max_running_prompts=max_running_prompts,
            max_prompt_len=max_prompt_len,
        )

    def end_rollout_session(self, ctx: Context) -> None:
        """Finalize rollout-only state before scoring or training."""

        del ctx
        self._require_rollout_engine().end_rollout_session()
        self._rollout_session_active = False

    async def end_rollout_session_async(self, ctx: Context) -> None:
        """Async rollout-session end hook for agentic callers."""

        del ctx
        await self._require_rollout_engine().end_rollout_session_async()
        self._rollout_session_active = False

    async def rollout_batch_async(
        self,
        ctx: Context,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
        prompt_features: list[dict | None] | None = None,
    ) -> list[RolloutResult]:
        """Async rollout entry for serving/agentic callers."""

        engine = self._require_rollout_engine()
        if not prompt_tokens:
            return []
        if prompt_features is not None and len(prompt_features) != len(prompt_tokens):
            raise ValueError("prompt_features must have the same length as prompt_tokens")
        if not self._rollout_session_active:
            await asyncio.to_thread(self._sync_policy_if_needed)
        prompts = expand_prompts(prompt_tokens, n_samples)
        flat_features = expand_prompt_features(prompt_features, len(prompt_tokens), n_samples)
        options = rollout_options(ctx, sampling_params)
        if self._step_e2e_start is None:
            self._step_e2e_start = time.perf_counter()
            self._step_rollout_time_s = 0.0
        rollout = await engine.generate_rollout_async(
            prompts,
            prompt_features=flat_features,
            max_new_tokens=sampling_params.max_new_tokens,
            max_running_prompts=options["max_running_prompts"],
            max_prompt_len=options["max_prompt_len"],
            eos_token_id=options["eos_token_id"],
            decode_progress_interval_s=options["decode_progress_interval_s"],
            sampling_params=options["sampling_params"],
        )
        sequences = [
            RolloutSequence(resp_tokens=tokens, resp_logprobs=rollout.logprobs[index, : len(tokens)].tolist())
            for index, tokens in enumerate(rollout.response_ids)
        ]
        return group_rollout_sequences(sequences, len(prompt_tokens), n_samples)

    def train(
        self,
        ctx: Context,
        batch_data: list[TrainSequence],
        loss_fn: Callable,
        mini_bs: int,
        gradient_accumulation_steps: int | None = None,
    ) -> dict[str, float]:
        engine = self._require_train_engine()
        if not callable(loss_fn):
            raise ValueError("CudaBackend requires a callable loss_fn")
        if self._separate_rollout and self._rollout_session_active:
            raise RuntimeError("cannot update policy weights during an active rollout session")

        train_start = time.perf_counter()
        if self._step_e2e_start is None:
            self._step_e2e_start = train_start
            self._step_rollout_time_s = 0.0
        losses = []
        metrics: dict[str, float] = {}
        # Slice the batch into `mini_bs` chunks; each chunk becomes one tensor
        # pack and the loss function is stamped onto each pack so the engine's
        # forward worker can call it without having to know about the loss API.
        packs = []
        is_sft = is_sft_loss_fn(loss_fn)
        sft_target_counts = [] if is_sft else None
        for start in range(0, len(batch_data), mini_bs):
            seqs = batch_data[start : start + mini_bs]
            pack = make_train_pack(seqs)
            pack["_loss_fn"] = loss_fn
            packs.append(pack)
            if is_sft:
                sft_target_counts.append(sft_target_token_count(seqs))
        if is_sft:
            annotate_sft_token_mean_packs(
                packs,
                sft_target_counts,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        stats_list = engine.step(packs, gradient_accumulation_steps=gradient_accumulation_steps)
        if self._separate_rollout and any(bool(stats.stepped) for stats in stats_list):
            self._train_policy_version += 1
        train_time_s = time.perf_counter() - train_start
        # `first_policy_metrics` keeps the per-step rollout/policy diagnostics
        # untouched (we want the value seen on the first microbatch, not the
        # average over microbatches), while everything else gets mean-averaged.
        first_policy_metrics: dict[str, float] = {}
        averaged_metric_counts: dict[str, int] = {}
        for stats in stats_list:
            losses.append(stats.loss)
            if stats.metrics:
                for key, value in stats.metrics.items():
                    key = str(key)
                    value_float = float(value)
                    if metric_reduction(key) is MetricReduction.FIRST:
                        first_policy_metrics.setdefault(key, value_float)
                    else:
                        metrics[key] = metrics.get(key, 0.0) + value_float
                        averaged_metric_counts[key] = averaged_metric_counts.get(key, 0) + 1
        if metrics:
            metrics = {key: value / averaged_metric_counts[key] for key, value in metrics.items()}
        metrics.update(first_policy_metrics)
        result = {"loss": sum(losses) / max(len(losses), 1)}
        result.update(metrics)
        result.update(self._pending_policy_sync_metrics)
        self._pending_policy_sync_metrics = {}
        if self._step_e2e_start is not None:
            step_e2e_time_s = time.perf_counter() - self._step_e2e_start
            self._step_e2e_start = None
            logger.info(
                "time rollout=%.6f train=%.6f total=%.6f",
                self._step_rollout_time_s,
                train_time_s,
                step_e2e_time_s,
            )
            result["step_rollout_time_s"] = self._step_rollout_time_s
            result["step_train_time_s"] = train_time_s
            result["step_e2e_time_s"] = step_e2e_time_s
        return result

    def save_checkpoint(self, ctx: Context, path: str) -> str:
        engine = self._require_train_engine()
        return save_checkpoint(engine, path)

    def ensure_roles(self, ctx: Context, roles: dict[str, ModelRole]) -> None:
        engine = self._require_train_engine()
        engine.ensure_roles(roles)

    def score_logprobs(
        self,
        ctx: Context,
        role: str,
        token_rows: list[list[int]],
        *,
        features: list[dict | None] | None = None,
        microbatch_size: int = 8,
    ) -> list[list[float]]:
        engine = self._require_train_engine()
        if features is not None and len(features) != len(token_rows):
            raise ValueError("features must have the same length as token_rows")
        return engine.score_logprobs(
            role,
            token_rows,
            pad_token_id=pad_token_id(ctx),
            features=features,
            microbatch_size=microbatch_size,
        )

    def score_values(
        self, ctx: Context, role: str, token_rows: list[list[int]], *, features: list[dict | None] | None = None
    ) -> list[list[float]]:
        engine = self._require_train_engine()
        if features is not None and len(features) != len(token_rows):
            raise ValueError("features must have the same length as token_rows")
        return engine.score_values(role, token_rows, pad_token_id=pad_token_id(ctx), features=features)

    def score_rewards(
        self, ctx: Context, role: str, token_rows: list[list[int]], *, features: list[dict | None] | None = None
    ) -> list[float]:
        engine = self._require_train_engine()
        if features is not None and len(features) != len(token_rows):
            raise ValueError("features must have the same length as token_rows")
        return engine.score_rewards(role, token_rows, pad_token_id=pad_token_id(ctx), features=features)

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
        engine = self._require_train_engine()
        # The critic shares the pack layout with the actor; we reuse the same
        # packer but drop the loss-function pointer (the engine has a dedicated
        # value loss path that takes (cliprange_value, value_loss_coef)).
        packs = []
        for start in range(0, len(batch_data), mini_bs):
            packs.append(make_train_pack(batch_data[start : start + mini_bs]))
        return engine.train_values(
            role,
            packs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            cliprange_value=cliprange_value,
            value_loss_coef=value_loss_coef,
        )
