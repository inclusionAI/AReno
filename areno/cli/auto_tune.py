"""Automatic train/rollout parameter selection for ``areno train``.

The tuner intentionally probes the real AReno backend with dummy-loaded
weights and synthetic token rows. That keeps the estimate close to the
selected model architecture and TP/DP layout without requiring a real dataset
or writing checkpoints during the search.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TypeVar

from areno.api.trainer_config import RolloutTrainerConfig, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AutoTuneCandidate:
    tp_size: int
    batch_size: int
    n_samples: int
    mini_bs: int
    max_running_prompts: int
    adam_8bit: bool
    keep_rollout_state: bool

    @property
    def train_rows(self) -> int:
        return self.batch_size * self.n_samples


@dataclass(frozen=True, slots=True)
class AutoTuneMeasurement:
    candidate: AutoTuneCandidate
    peak_mem_frac: float
    ok: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AutoTuneResult:
    config: TrainerConfig
    measurement: AutoTuneMeasurement | None
    measurements: tuple[AutoTuneMeasurement, ...]


ProbeFn = Callable[[TrainerConfig, AutoTuneCandidate, str], AutoTuneMeasurement]
T = TypeVar("T")


def auto_tune_config(
    config: TrainerConfig,
    *,
    mem_frac: float = 0.9,
    auto_max_samples: int = 256,
    probe_fn: ProbeFn | None = None,
) -> AutoTuneResult:
    """Return a copy of ``config`` with safe train/rollout knobs filled in.

    The search probes large candidates first and falls back to smaller points
    until it finds one whose observed peak memory fraction is below
    ``mem_frac``. Rollout tunes concurrency. Train derives batch size from the
    selected rollout concurrency and only tunes mini-batch size.
    """

    if not 0 < float(mem_frac) <= 1:
        raise ValueError("--mem-frac must be in (0, 1]")
    if int(auto_max_samples) < 1:
        raise ValueError("--tune-max-samples must be >= 1")
    if not isinstance(config, RolloutTrainerConfig):
        raise ValueError("--tune-params currently tunes rollout-based trainers only")
    probe = probe_fn or probe_candidate_with_dummy_run
    rollout_candidates = _rollout_candidates(config, auto_max_samples=auto_max_samples)
    logger.info(
        "auto tune start mem_frac=%.3f auto_max_samples=%d world_size=%d rollout_candidates=%d",
        mem_frac,
        auto_max_samples,
        config.world_size,
        len(rollout_candidates),
    )
    rollout_result = _first_under_target_desc(
        config,
        rollout_candidates,
        mem_frac=mem_frac,
        probe=probe,
        stage="rollout",
    )
    measurements = list(rollout_result.measurements)
    logger.info(
        "auto tune rollout selected %s",
        _format_measurement(rollout_result.measurement),
    )
    train_result = _best_train_mini_bs(
        config,
        rollout_result.measurement.candidate,
        auto_max_samples=auto_max_samples,
        mem_frac=mem_frac,
        probe=probe,
    )
    measurements.extend(train_result.measurements)
    train_best = train_result.measurement.candidate
    tuned = replace(
        config,
        tp_size=train_best.tp_size,
        batch_size=train_best.batch_size,
        n_samples=train_best.n_samples,
        mini_bs=train_best.mini_bs,
        max_running_prompts=train_best.max_running_prompts,
        keep_rollout_state=train_best.keep_rollout_state,
    )
    logger.info("auto tune selected %s", _format_measurement(train_result.measurement))
    return AutoTuneResult(config=tuned, measurement=train_result.measurement, measurements=tuple(measurements))


def enumerate_candidates(config: RolloutTrainerConfig, *, auto_max_samples: int = 256) -> list[AutoTuneCandidate]:
    """Generate a compact, monotonic-ish search space around common RL knobs."""

    rollout_candidates = _rollout_candidates(config, auto_max_samples=auto_max_samples)
    train_candidates = [
        candidate
        for template in rollout_candidates
        for candidate in _train_candidates(config, template, auto_max_samples=auto_max_samples)
    ]
    return sorted(
        {candidate: None for candidate in [*rollout_candidates, *train_candidates]}.keys(), key=_candidate_key
    )


def enumerate_candidate_groups(
    config: RolloutTrainerConfig, *, auto_max_samples: int = 256
) -> list[list[AutoTuneCandidate]]:
    """Group rollout candidates for auto-tune inspection."""

    return [_rollout_candidates(config, auto_max_samples=auto_max_samples)]


def _rollout_candidates(
    config: RolloutTrainerConfig,
    template: AutoTuneCandidate | None = None,
    *,
    auto_max_samples: int = 256,
) -> list[AutoTuneCandidate]:
    """Return up to 16 sparse candidates over rollout concurrency."""

    tp_size = int(config.tp_size)
    if template is None:
        template = _default_template_candidate(config)
    batch_size = template.batch_size
    n_samples = template.n_samples
    rows = max(int(batch_size) * int(n_samples), 1)
    max_running_prompts = max(int(auto_max_samples), 1)
    running_prompt_values = _powers_of_two_up_to(max_running_prompts)
    candidates = []
    for max_running_prompts in running_prompt_values:
        candidates.append(
            AutoTuneCandidate(
                tp_size=tp_size,
                batch_size=batch_size,
                n_samples=n_samples,
                mini_bs=min(template.mini_bs, rows),
                max_running_prompts=max_running_prompts,
                adam_8bit=bool(config.adam_8bit),
                keep_rollout_state=template.keep_rollout_state,
            )
        )
    selected = _sparse_select(sorted(set(candidates), key=_rollout_candidate_key), 16)
    return list(reversed(selected))


def _train_candidates(
    config: RolloutTrainerConfig,
    template: AutoTuneCandidate,
    *,
    auto_max_samples: int = 256,
) -> list[AutoTuneCandidate]:
    """Return sparse train-memory candidates over batch and mini-batch size."""

    n_samples = max(int(config.n_samples), 1)
    max_batch = max(1, min(int(config.batch_size), max(int(auto_max_samples), 1) // n_samples))
    max_mini_bs = max(int(config.mini_bs), 1)
    candidates = []
    for batch_size in _powers_of_two_up_to(max_batch):
        rows = batch_size * n_samples
        for mini_bs in _mini_bs_values(max_mini_bs, rows):
            candidates.append(
                AutoTuneCandidate(
                    tp_size=template.tp_size,
                    batch_size=batch_size,
                    n_samples=n_samples,
                    mini_bs=mini_bs,
                    max_running_prompts=template.max_running_prompts,
                    adam_8bit=bool(config.adam_8bit),
                    keep_rollout_state=False,
                )
            )
    return sorted(_sparse_select(sorted(set(candidates), key=_train_candidate_key), 32), key=_train_candidate_key)


def _train_template_from_rollout(
    config: RolloutTrainerConfig,
    template: AutoTuneCandidate,
    *,
    auto_max_samples: int = 256,
) -> AutoTuneCandidate:
    """Derive a fixed train batch size from the selected rollout concurrency."""

    n_samples = max(int(config.n_samples), 1)
    max_rows = max(1, min(int(auto_max_samples), int(template.max_running_prompts)))
    batch_size = max(1, _floor_power_of_two(max_rows // n_samples))
    rows = batch_size * n_samples
    return AutoTuneCandidate(
        tp_size=template.tp_size,
        batch_size=batch_size,
        n_samples=n_samples,
        mini_bs=min(max(int(config.mini_bs), 1), rows),
        max_running_prompts=template.max_running_prompts,
        adam_8bit=bool(config.adam_8bit),
        keep_rollout_state=False,
    )


def _default_template_candidate(config: RolloutTrainerConfig) -> AutoTuneCandidate:
    return AutoTuneCandidate(
        tp_size=int(config.tp_size),
        batch_size=_floor_power_of_two(max(int(config.batch_size), 1)),
        n_samples=max(int(config.n_samples), 1),
        mini_bs=1,
        max_running_prompts=1,
        adam_8bit=bool(config.adam_8bit),
        keep_rollout_state=False,
    )


def _first_under_target_desc(
    config: TrainerConfig,
    candidates: list[AutoTuneCandidate],
    *,
    mem_frac: float,
    probe: ProbeFn,
    stage: str,
) -> AutoTuneResult:
    measurements: list[AutoTuneMeasurement] = []
    logger.info("auto tune %s stage start candidates=%d direction=desc", stage, len(candidates))
    for index, candidate in enumerate(candidates, start=1):
        logger.info(
            "auto tune %s probe start index=%d/%d %s",
            stage,
            index,
            len(candidates),
            _format_candidate(candidate),
        )
        measurement = probe(config, candidate, stage)
        measurements.append(measurement)
        _log_measurement(measurement)
        if measurement.ok and measurement.peak_mem_frac <= mem_frac:
            logger.info("auto tune %s stage first fit %s", stage, _format_measurement(measurement))
            return AutoTuneResult(config=config, measurement=measurement, measurements=tuple(measurements))
        logger.info("auto tune %s probe rejected target %.3f with %s", stage, mem_frac, _format_candidate(candidate))
    raise RuntimeError(
        f"auto tune {stage} stage could not find a candidate under mem_frac={mem_frac}; "
        f"first error: {measurements[0].error if measurements else 'no candidates'}"
    )


def _best_train_mini_bs(
    config: RolloutTrainerConfig,
    rollout: AutoTuneCandidate,
    *,
    auto_max_samples: int,
    mem_frac: float,
    probe: ProbeFn,
) -> AutoTuneResult:
    measurements: list[AutoTuneMeasurement] = []
    template = _train_template_from_rollout(config, rollout, auto_max_samples=auto_max_samples)
    mini_values = list(reversed(_mini_bs_values(max(int(config.mini_bs), 1), template.train_rows)))
    logger.info(
        "auto tune train stage start fixed_batch_size=%d train_rows=%d mini_bs_candidates=%d direction=desc",
        template.batch_size,
        template.train_rows,
        len(mini_values),
    )
    for mini_index, mini_bs in enumerate(mini_values, start=1):
        candidate = replace(template, mini_bs=mini_bs)
        logger.info(
            "auto tune train probe start mini_bs index=%d/%d %s",
            mini_index,
            len(mini_values),
            _format_candidate(candidate),
        )
        measurement = probe(config, candidate, "train")
        measurements.append(measurement)
        _log_measurement(measurement)
        if measurement.ok and measurement.peak_mem_frac <= mem_frac:
            logger.info(
                "auto tune train stage first fit %.3f with %s",
                mem_frac,
                _format_candidate(candidate),
            )
            return AutoTuneResult(config=config, measurement=measurement, measurements=tuple(measurements))
        logger.info("auto tune train probe rejected target %.3f with %s", mem_frac, _format_candidate(candidate))
    raise RuntimeError(
        f"auto tune train stage could not find a candidate under mem_frac={mem_frac}; "
        f"first error: {measurements[0].error if measurements else 'no candidates'}"
    )


def _log_measurement(measurement: AutoTuneMeasurement) -> None:
    logger.info("auto tune probe result %s", _format_measurement(measurement))


def _format_measurement(measurement: AutoTuneMeasurement) -> str:
    suffix = f" error={measurement.error}" if measurement.error else ""
    return f"{_format_candidate(measurement.candidate)} ok={measurement.ok} peak_mem_frac={measurement.peak_mem_frac:.4f}{suffix}"


def _format_candidate(candidate: AutoTuneCandidate) -> str:
    return (
        f"tp_size={candidate.tp_size} batch_size={candidate.batch_size} n_samples={candidate.n_samples} "
        f"mini_bs={candidate.mini_bs} max_running_prompts={candidate.max_running_prompts} "
        f"adam_8bit={candidate.adam_8bit} keep_rollout_state={candidate.keep_rollout_state}"
    )


def _candidate_key(item: AutoTuneCandidate) -> tuple:
    return (
        item.max_running_prompts,
        item.train_rows,
        item.mini_bs,
        item.batch_size,
        item.n_samples,
        -item.tp_size,
        not item.adam_8bit,
        item.keep_rollout_state,
    )


def _rollout_candidate_key(item: AutoTuneCandidate) -> tuple:
    return (
        -item.tp_size,
        item.max_running_prompts,
        item.train_rows,
        item.batch_size,
        item.n_samples,
        item.mini_bs,
        item.keep_rollout_state,
    )


def _train_candidate_key(item: AutoTuneCandidate) -> tuple:
    return (
        item.tp_size,
        item.mini_bs,
        not item.adam_8bit,
        item.keep_rollout_state,
        item.max_running_prompts,
        item.train_rows,
    )


def probe_candidate_with_dummy_run(
    config: TrainerConfig, candidate: AutoTuneCandidate, stage: str
) -> AutoTuneMeasurement:
    """Measure one candidate with dummy weights/data on the configured GPUs."""

    try:
        peak = _run_dummy_probe(config, candidate, stage=stage)
        return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=peak, ok=True)
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
        if _is_oom_error(message) or _is_tp_compat_error(message):
            return AutoTuneMeasurement(candidate=candidate, peak_mem_frac=1.0, ok=False, error=message)
        raise


def _run_dummy_probe(config: TrainerConfig, candidate: AutoTuneCandidate, *, stage: str) -> float:
    import torch

    import areno.api
    from areno.api.models import BackendType

    probe_config = copy.deepcopy(config)
    probe_config.batch_size = candidate.batch_size
    probe_config.n_samples = candidate.n_samples
    probe_config.mini_bs = candidate.mini_bs
    probe_config.max_running_prompts = candidate.max_running_prompts
    probe_config.tp_size = candidate.tp_size
    probe_config.adam_8bit = candidate.adam_8bit
    probe_config.keep_rollout_state = candidate.keep_rollout_state
    custom_config = probe_config.areno_config()
    custom_config.dummy_load = True
    custom_config.max_running_prompts = candidate.max_running_prompts
    devices = _probe_devices(probe_config.world_size)
    custom_config.devices = devices

    trainer = areno.api.Trainer(
        probe_config.world_size,
        probe_config.ckpt,
        backend_type=BackendType.Areno,
        custom_config=custom_config,
        metrics_log_dir=None,
    )
    _reset_cuda_peak_stats(devices)
    try:
        trainer.init()
        tokenizer = trainer.get_tokenizer()
        prompt_len, response_len = _dummy_token_budgets(
            max_prompt_tokens=probe_config.max_prompt_tokens,
            max_new_tokens=probe_config.max_new_tokens,
            max_context_len=probe_config.max_context_len,
        )
        prompt_tokens = _dummy_prompt_tokens(tokenizer, prompt_len)
        response_tokens = _dummy_response_tokens(
            tokenizer,
            response_len=response_len,
        )
        if stage == "rollout":
            peak = trainer.probe_rollout_cache(
                max_new_tokens=len(response_tokens),
                max_running_prompts=candidate.max_running_prompts,
                max_prompt_len=len(prompt_tokens),
            )
            for device in devices:
                torch.cuda.synchronize(device)
            return float(peak)
        trainer.begin_rollout_session()
        try:
            trainer.probe_rollout_cache(
                max_new_tokens=len(response_tokens),
                max_running_prompts=candidate.max_running_prompts,
                max_prompt_len=len(prompt_tokens),
            )
        finally:
            with suppress(Exception):
                trainer.end_rollout_session()
        train_rows = _dummy_train_rows(
            prompt_tokens=prompt_tokens,
            eos_token_id=_eos_token_id(tokenizer),
            target_rows=candidate.train_rows,
            response_tokens=response_tokens,
        )
        train_stats = trainer.train(train_rows, _dummy_policy_loss, mini_bs=candidate.mini_bs)
        for device in devices:
            torch.cuda.synchronize(device)
        worker_peak = float(train_stats.get("auto_tune_worker_peak_mem_frac", 0.0))
        return worker_peak if worker_peak > 0.0 else _peak_cuda_memory_fraction(devices)
    finally:
        with suppress(Exception):
            trainer.close()
        torch.cuda.empty_cache()


def _dummy_token_budgets(
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    max_context_len: int | None,
) -> tuple[int, int]:
    if max_context_len is None:
        return max(1, int(max_prompt_tokens)), max(1, int(max_new_tokens))

    total_len = max(2, int(max_context_len))
    prompt_len = min(max(1, int(max_prompt_tokens)), total_len - 1)
    return prompt_len, total_len - prompt_len


def _dummy_prompt_tokens(tokenizer, max_prompt_tokens: int) -> list[int]:
    encoded = tokenizer.encode("AReno auto tune prompt.", add_special_tokens=False)
    if not encoded:
        encoded = [_eos_token_id(tokenizer)]
    width = max(1, int(max_prompt_tokens))
    return _repeat_to_width([int(token) for token in encoded], width)


def _dummy_response_tokens(
    tokenizer,
    *,
    response_len: int,
) -> list[int]:
    encoded = tokenizer.encode("AReno auto tune response.", add_special_tokens=False)
    if not encoded:
        encoded = [_eos_token_id(tokenizer)]
    return _repeat_to_width([int(token) for token in encoded], max(1, int(response_len)))


def _repeat_to_width(tokens: list[int], width: int) -> list[int]:
    if width <= len(tokens):
        return tokens[:width]
    repeats = (width + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:width]


def _dummy_policy_loss(pack: dict, logprobs):
    from areno.api.loss_fns.layout import response_layout

    layout = response_layout(pack, logprobs, need_advantages=True)
    advantages = layout.advantages
    if advantages is None:
        raise ValueError("auto tune dummy loss requires advantages")
    response_mask = layout.response_mask.to(dtype=logprobs.dtype)
    target = (advantages.abs() * response_mask).sum().clamp_min(1.0)
    loss = -(logprobs * advantages * response_mask).sum() / target
    return loss, {"auto_tune_dummy_loss": loss.detach()}


def _dummy_train_rows(
    *,
    prompt_tokens: list[int],
    response_tokens: list[int],
    eos_token_id: int,
    target_rows: int,
):
    from areno.api.models import TrainSequence

    rows = []
    tokens = [*prompt_tokens, *response_tokens]
    prompt_mask = [True] * len(prompt_tokens) + [False] * len(response_tokens)
    logprobs = [0.0] * len(tokens)
    advantages = [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens)
    while len(rows) < target_rows:
        rows.append(
            TrainSequence(
                tokens=list(tokens),
                prompt_mask=list(prompt_mask),
                logprobs=list(logprobs),
                advantages=list(advantages),
                reward=1.0,
                eos_token_id=eos_token_id,
            )
        )
    return rows


def _eos_token_id(tokenizer) -> int:
    token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(token_id, list | tuple):
        return int(token_id[0]) if token_id else 0
    return int(token_id) if token_id is not None else 0


def _probe_devices(world_size: int) -> list[int]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--tune-params requires CUDA so it can measure GPU memory")
    visible = int(torch.cuda.device_count())
    required = int(world_size)
    if required > visible:
        raise RuntimeError(
            f"--tune-params requires --world-size <= visible CUDA devices; got world_size={required}, visible_cuda_devices={visible}"
        )
    return list(range(required))


def _reset_cuda_peak_stats(devices: list[int]) -> None:
    import torch

    for device in devices:
        device_idx = int(device)
        torch.cuda.set_device(device_idx)
        torch.empty((), device=f"cuda:{device_idx}")
        torch.cuda.synchronize(device_idx)
        torch.cuda.reset_peak_memory_stats(device_idx)


def _peak_cuda_memory_fraction(devices: list[int]) -> float:
    import torch

    fractions = []
    for device in devices:
        device_idx = int(device)
        total = torch.cuda.get_device_properties(device_idx).total_memory
        peak = torch.cuda.max_memory_allocated(device_idx)
        fractions.append(float(peak) / float(total))
    return max(fractions) if fractions else 0.0


def _powers_of_two_up_to(value: int) -> list[int]:
    out = []
    current = 1
    while current <= value:
        out.append(current)
        current *= 2
    return out or [1]


def _floor_power_of_two(value: int) -> int:
    return _powers_of_two_up_to(max(int(value), 1))[-1]


def _sparse_select(items: list[T], limit: int) -> list[T]:
    if len(items) <= limit:
        return items
    if limit <= 1:
        return [items[-1]]
    last = len(items) - 1
    indices = sorted({round(index * last / (limit - 1)) for index in range(limit)})
    return [items[index] for index in indices]


def _mini_bs_values(max_mini_bs: int, rows: int) -> list[int]:
    limit = min(max_mini_bs, rows)
    return [value for value in (1, 2, 4, 8, 16) if value <= limit] or [1]


def _is_oom_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "out of memory" in lowered
        or "cuda error: out of memory" in lowered
        or (
            "worker exited without reporting result" in lowered
            and ("exitcode -9" in lowered or "during op.train" in lowered or "during op.probe_rollout_cache" in lowered)
        )
        or (
            "failed during op.probe_rollout_cache" in lowered
            and (
                "nccl" in lowered
                or "device error" in lowered
                or "external library call failed" in lowered
                or "system call" in lowered
            )
        )
    )


def _is_tp_compat_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "tp_size" in lowered
        or "divisible by tp" in lowered
        or "divisible by tp_size" in lowered
        or ("head" in lowered and "divisible" in lowered)
        or ("kv" in lowered and "divisible" in lowered)
    )
