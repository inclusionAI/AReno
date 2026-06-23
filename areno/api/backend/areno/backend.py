"""Adapter from the public `Trainer` API onto the areno `ArenoEngine`.

areno runs a co-located train + rollout engine in the same process group.
This file is the thin glue that:

- starts the engine with the dataclass-validated `ArenoConfig`,
- forwards rollout requests through `generate_rollout` while translating
  SDK `SamplingParams` into the engine's own type,
- packs a list of `TrainSequence` objects into the tensor batch
  (`_make_train_pack`) the engine consumes, and
- routes the caller's loss function into the engine's training step via the
  small `_external_loss_dispatcher` hook so the engine itself stays
  algorithm-agnostic.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Lock

import torch

from areno.api.backend.base import Backend, register_backend
from areno.api.config import ArenoConfig
from areno.api.context import Context
from areno.api.loss_fns.sft import sft_loss_fn
from areno.api.models import BackendType, RolloutResult, RolloutSequence, SamplingParams, TrainSequence
from areno.api.roles import ModelRole

logger = logging.getLogger(__name__)
_SYS_PATH_LOCK = Lock()
_SYS_PATH_PREFERRED = False


def _rollout_options(ctx: Context, sampling_params: SamplingParams):
    """Translate public rollout params to areno engine-native options."""

    max_prompt_len = sampling_params.max_prompt_len
    eos_token_ids = () if sampling_params.ignore_eos else ctx.eos_token_ids
    stop_token_ids = tuple(sampling_params.stop_token_ids or ())
    suppress_candidates = set(_explicit_suppress_token_ids(ctx.tokenizer))
    if not sampling_params.ignore_eos:
        suppress_candidates.update(int(token_id) for token_id in getattr(ctx.tokenizer, "all_special_ids", ()) or ())
    suppress_token_ids = tuple(
        sorted(token_id for token_id in suppress_candidates if token_id not in {*eos_token_ids, *stop_token_ids})
    )
    cfg = ctx.custom_config
    if cfg is None:
        cfg = ArenoConfig()
    if not isinstance(cfg, ArenoConfig):
        raise TypeError(f"ArenoBackend requires ArenoConfig, got {type(cfg)!r}")

    from areno import SamplingParams as ArenoSamplingParams

    return {
        "max_prompt_len": max_prompt_len,
        "eos_token_id": eos_token_ids,
        "max_running_prompts": cfg.max_running_prompts,
        "decode_progress_interval_s": cfg.decode_progress_interval_s,
        "sampling_params": ArenoSamplingParams(
            temperature=0.0 if sampling_params.greedy else sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=max(0, sampling_params.top_k),
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            suppress_special_tokens=not sampling_params.ignore_eos,
        ),
    }


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


def _external_loss_dispatcher(pack: dict, logprobs: torch.Tensor):
    """Call the loss function attached to an areno train pack.

    areno workers execute in a backend-owned process/thread. Carrying the
    callable inside each pack keeps the engine API stable while allowing the
    algorithm script to provide GSPO, GRPO, or custom losses.
    """

    loss_fn = pack.get("_loss_fn")
    if not callable(loss_fn):
        raise ValueError("Areno train data pack is missing callable _loss_fn")
    return loss_fn(pack, logprobs)


@register_backend(BackendType.Areno)
class ArenoBackend(Backend):
    """Backend adapter that maps `Trainer` calls onto `areno.ArenoEngine`."""

    def __init__(self):
        """Create an adapter; workers are started by `initialize`."""

        super().__init__()
        self._engine = None
        # Per-step wall-time accumulators used to print the
        # rollout/train/end-to-end breakdown after `train` completes.
        self._step_e2e_start: float | None = None
        self._step_rollout_time_s = 0.0

    def _require_engine(self):
        """Return the initialized engine or raise a consistent error."""

        if self._engine is None:
            raise RuntimeError("ArenoBackend is not initialized")
        return self._engine

    def initialize(self, ctx: Context):
        _prefer_repo_areno()
        from areno import ArenoEngine, OptimizerConfig, RuntimeConfig

        cfg = ctx.custom_config
        if cfg is None:
            cfg = ArenoConfig()
        if not isinstance(cfg, ArenoConfig):
            raise TypeError(f"ArenoBackend requires ArenoConfig, got {type(cfg)!r}")

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

        # ArenoEngine construction is synchronous and CUDA-heavy. The SDK is
        # intentionally synchronous because engine IPC is blocking as well.
        self._engine = ArenoEngine.from_pretrained(
            cfg.model_path or ctx.model_path,
            tp_size=tp_size,
            dp_size=dp_size,
            devices=devices,
            dummy_load=cfg.dummy_load,
            optimizer_config=OptimizerConfig(**cfg.optimizer),
            runtime_config=RuntimeConfig(**cfg.runtime),
            loss_fn=_external_loss_dispatcher,
        )

    def rollout_batch(
        self,
        ctx: Context,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
    ) -> list[RolloutResult]:
        engine = self._require_engine()
        if not prompt_tokens:
            return []
        # Replicate each already-tokenized prompt `n_samples` times so the
        # engine treats each completion as independent while preserving the
        # `[prompt0_sample0, prompt0_sample1, ..., promptN_sampleK]` layout.
        flat_prompts = [ids for ids in prompt_tokens for _ in range(n_samples)]
        options = _rollout_options(ctx, sampling_params)

        if self._step_e2e_start is None:
            self._step_e2e_start = time.perf_counter()
            self._step_rollout_time_s = 0.0
        # Translate the public SamplingParams into the engine's native type.
        # Greedy decoding is implemented by forcing temperature to zero.
        rollout = engine.generate_rollout(
            flat_prompts,
            max_new_tokens=sampling_params.max_new_tokens,
            max_running_prompts=options["max_running_prompts"],
            max_prompt_len=options["max_prompt_len"],
            eos_token_id=options["eos_token_id"],
            decode_progress_interval_s=options["decode_progress_interval_s"],
            sampling_params=options["sampling_params"],
        )
        # Repack the flat result into per-prompt groups of `n_samples`
        # completions so downstream code can iterate `for item, result`.
        results = []
        for prompt_idx in range(len(prompt_tokens)):
            start = prompt_idx * n_samples
            end = start + n_samples
            results.append(
                RolloutResult(
                    sequences=[
                        RolloutSequence(
                            resp_tokens=tokens,
                            resp_logprobs=rollout.logprobs[i, : len(tokens)].tolist(),
                        )
                        for i, tokens in enumerate(rollout.response_ids[start:end], start=start)
                    ]
                )
            )
        return results

    def begin_rollout_session(self, ctx: Context) -> None:
        """Prepare colocated actor state before rollout requests are issued."""

        del ctx
        self._require_engine().begin_rollout_session()

    async def begin_rollout_session_async(self, ctx: Context) -> None:
        """Async rollout-session begin hook for agentic callers."""

        del ctx
        await self._require_engine().begin_rollout_session_async()

    async def sync_rollout_session_async(self, ctx: Context) -> None:
        """Synchronize worker TP groups before agentic request rollout."""

        del ctx
        await self._require_engine().sync_rollout_session_async()

    def dp_size(self, ctx: Context) -> int:
        """Return the engine's effective DP size after backend initialization."""

        del ctx
        return int(self._require_engine().config.dp_size)

    def model_context_len(self, ctx: Context) -> int | None:
        """Return the checkpoint's max position embeddings from the loaded engine config."""

        del ctx
        return int(self._require_engine().config.model.max_position_embeddings)

    def end_rollout_session(self, ctx: Context) -> None:
        """Finalize rollout-only state before scoring or training."""

        del ctx
        self._require_engine().end_rollout_session()

    async def end_rollout_session_async(self, ctx: Context) -> None:
        """Async rollout-session end hook for agentic callers."""

        del ctx
        await self._require_engine().end_rollout_session_async()

    async def rollout_batch_async(
        self,
        ctx: Context,
        prompt_tokens: list[list[int]],
        n_samples: int,
        sampling_params: SamplingParams,
    ) -> list[RolloutResult]:
        """Async rollout entry for serving/agentic callers."""

        engine = self._require_engine()
        if not prompt_tokens:
            return []
        prompts = [tokens for tokens in prompt_tokens for _ in range(n_samples)]
        options = _rollout_options(ctx, sampling_params)
        if self._step_e2e_start is None:
            self._step_e2e_start = time.perf_counter()
            self._step_rollout_time_s = 0.0
        rollout = await engine.generate_rollout_async(
            prompts,
            max_new_tokens=sampling_params.max_new_tokens,
            max_running_prompts=options["max_running_prompts"],
            max_prompt_len=options["max_prompt_len"],
            eos_token_id=options["eos_token_id"],
            decode_progress_interval_s=options["decode_progress_interval_s"],
            sampling_params=options["sampling_params"],
        )
        results = []
        for prompt_idx in range(len(prompt_tokens)):
            start = prompt_idx * n_samples
            end = start + n_samples
            results.append(
                RolloutResult(
                    sequences=[
                        RolloutSequence(
                            resp_tokens=tokens,
                            resp_logprobs=rollout.logprobs[i, : len(tokens)].tolist(),
                        )
                        for i, tokens in enumerate(rollout.response_ids[start:end], start=start)
                    ]
                )
            )
        return results

    def train(
        self,
        ctx: Context,
        batch_data: list[TrainSequence],
        loss_fn: Callable,
        mini_bs: int,
        gradient_accumulation_steps: int | None = None,
    ) -> dict[str, float]:
        engine = self._require_engine()
        if not callable(loss_fn):
            raise ValueError("ArenoBackend requires a callable loss_fn")

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
        sft_target_counts = []
        for start in range(0, len(batch_data), mini_bs):
            seqs = batch_data[start : start + mini_bs]
            pack = _make_train_pack(seqs)
            pack["_loss_fn"] = loss_fn
            packs.append(pack)
            sft_target_counts.append(_sft_target_token_count(seqs))
        if _is_sft_loss_fn(loss_fn):
            _annotate_sft_token_mean_packs(
                packs,
                sft_target_counts,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
        stats_list = engine.step(packs, gradient_accumulation_steps=gradient_accumulation_steps)
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
                    value_float = float(value)
                    if _is_rollout_policy_metric(key):
                        first_policy_metrics.setdefault(key, value_float)
                    else:
                        metrics[key] = metrics.get(key, 0.0) + value_float
                        averaged_metric_counts[key] = averaged_metric_counts.get(key, 0) + 1
        if metrics:
            metrics = {key: value / averaged_metric_counts[key] for key, value in metrics.items()}
        metrics.update(first_policy_metrics)
        result = {"loss": sum(losses) / max(len(losses), 1)}
        result.update(metrics)
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
        engine = self._require_engine()
        return engine.save_checkpoint(path)

    def ensure_roles(self, ctx: Context, roles: dict[str, ModelRole]) -> None:
        engine = self._require_engine()
        engine.ensure_roles(roles)

    def score_logprobs(self, ctx: Context, role: str, token_rows: list[list[int]]) -> list[list[float]]:
        engine = self._require_engine()
        return engine.score_logprobs(role, token_rows, pad_token_id=_pad_token_id(ctx))

    def score_values(self, ctx: Context, role: str, token_rows: list[list[int]]) -> list[list[float]]:
        engine = self._require_engine()
        return engine.score_values(role, token_rows, pad_token_id=_pad_token_id(ctx))

    def score_rewards(self, ctx: Context, role: str, token_rows: list[list[int]]) -> list[float]:
        engine = self._require_engine()
        return engine.score_rewards(role, token_rows, pad_token_id=_pad_token_id(ctx))

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
        engine = self._require_engine()
        # The critic shares the pack layout with the actor; we reuse the same
        # packer but drop the loss-function pointer (the engine has a dedicated
        # value loss path that takes (cliprange_value, value_loss_coef)).
        packs = []
        for start in range(0, len(batch_data), mini_bs):
            packs.append(_make_train_pack(batch_data[start : start + mini_bs]))
        return engine.train_values(
            role,
            packs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            cliprange_value=cliprange_value,
            value_loss_coef=value_loss_coef,
        )


def _make_train_pack(seqs: list[TrainSequence]) -> dict[str, torch.Tensor]:
    """Pack a list of `TrainSequence` rows into right-padded 2D tensors.

    Output layout (all shapes are (B, max_len) unless noted):
        input_ids       int64,  prompt+response token ids, padded with EOS
        labels          int64,  copy of input_ids reused as next-token targets
        lengths         int32,  (B,), real sequence length before padding
        prompt_mask     bool,   True at prompt positions (and on padded tail)
        logprobs        float,  rollout-policy logprobs aligned with input_ids
        advantages      float,  per-token advantage (zero on prompt prefix)
        returns/values/ref_logprobs (optional, only populated when at least one
            sequence carries the field; needed for PPO critic + reference KL).
    `prompt_mask` defaults to True for padding tokens so loss functions
    automatically ignore them when they restrict computation to response
    positions.
    """

    if not seqs:
        raise ValueError("train batch is empty")
    from areno.engine.runtime.common import pad_rows

    eos_token_id = seqs[0].eos_token_id
    batch = len(seqs)
    lengths = torch.tensor([len(seq.tokens) for seq in seqs], dtype=torch.int32)
    max_len = int(lengths.max().item()) if batch else 0
    input_ids = pad_rows([seq.tokens for seq in seqs], dtype=torch.long, fill_value=eos_token_id, width=max_len)
    prompt_mask = pad_rows([seq.prompt_mask for seq in seqs], dtype=torch.bool, fill_value=True, width=max_len)
    has_loss_mask = any(bool(seq.loss_mask) for seq in seqs)
    loss_mask_rows = [seq.loss_mask if seq.loss_mask else [not item for item in seq.prompt_mask] for seq in seqs]
    loss_mask = pad_rows(loss_mask_rows, dtype=torch.bool, fill_value=False, width=max_len) if has_loss_mask else None
    logprobs = pad_rows([seq.logprobs for seq in seqs], dtype=torch.float32, width=max_len)
    advantages = pad_rows([seq.advantages for seq in seqs], dtype=torch.float32, width=max_len)
    # Allocate optional fields only when at least one sequence carries them so
    # the engine can branch on key presence and avoid unused tensors.
    has_returns = any(bool(seq.returns) for seq in seqs)
    has_values = any(bool(seq.values) for seq in seqs)
    has_ref_logprobs = any(bool(seq.ref_logprobs) for seq in seqs)
    returns = pad_rows([seq.returns for seq in seqs], dtype=torch.float32, width=max_len) if has_returns else None
    values = pad_rows([seq.values for seq in seqs], dtype=torch.float32, width=max_len) if has_values else None
    ref_logprobs = (
        pad_rows([seq.ref_logprobs for seq in seqs], dtype=torch.float32, width=max_len) if has_ref_logprobs else None
    )

    pack = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "lengths": lengths,
        "prompt_mask": prompt_mask,
        "logprobs": logprobs,
        "advantages": advantages,
    }
    if loss_mask is not None:
        pack["loss_mask"] = loss_mask
    if returns is not None:
        pack["returns"] = returns
    if values is not None:
        pack["values"] = values
    if ref_logprobs is not None:
        pack["ref_logprobs"] = ref_logprobs
    return pack


def _is_sft_loss_fn(loss_fn: Callable) -> bool:
    """Return true for the built-in SFT loss, including simple partial wrappers."""

    return loss_fn is sft_loss_fn or getattr(loss_fn, "func", None) is sft_loss_fn


def _annotate_sft_token_mean_packs(
    packs: list[dict],
    target_counts: list[int],
    *,
    gradient_accumulation_steps: int | None,
) -> None:
    """Attach per-accumulation-group token normalizers for global token-mean SFT."""

    if not packs:
        return
    accumulation_steps = len(packs) if gradient_accumulation_steps is None else max(int(gradient_accumulation_steps), 1)
    for group_start in range(0, len(packs), accumulation_steps):
        group_end = min(group_start + accumulation_steps, len(packs))
        group_total = max(sum(target_counts[group_start:group_end]), 1)
        group_size = group_end - group_start
        for pack in packs[group_start:group_end]:
            pack["_sft_total_target_tokens"] = group_total
            pack["_sft_grad_scale"] = group_size


def _sft_target_token_count(seqs: list[TrainSequence]) -> int:
    """Count target tokens using the same next-token masks as packed training."""

    count = 0
    for seq in seqs:
        length = min(len(seq.tokens), len(seq.prompt_mask))
        for idx in range(1, length):
            is_target = not bool(seq.prompt_mask[idx])
            if seq.loss_mask:
                is_target = is_target and idx < len(seq.loss_mask) and bool(seq.loss_mask[idx])
            if is_target:
                count += 1
    return count


def _pad_token_id(ctx: Context) -> int:
    # Score helpers right-pad their batched inputs with this id; fall back to
    # the EOS token when the tokenizer does not define a dedicated pad token.
    token_id = ctx.tokenizer.pad_token_id
    if token_id is None:
        token_id = ctx.tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return int(token_id)


def _explicit_suppress_token_ids(tokenizer) -> tuple[int, ...]:
    """Return marker ids that should never be sampled as normal text."""

    out: list[int] = []
    for attr in ("pad_token_id", "bos_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            out.append(value)
    return tuple(dict.fromkeys(out))


def _is_rollout_policy_metric(key: str) -> bool:
    # These metrics describe the rollout-vs-train policy gap; the value on the
    # first microbatch is most representative because subsequent microbatches
    # already see updated weights once the engine fires gradient steps.
    return key in {"ratio_mean", "ratio_std", "rollout_logprobs_mean", "train_logprobs_mean"} or key.startswith(
        "logp_diff"
    )
