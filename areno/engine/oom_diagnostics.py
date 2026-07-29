"""Stage-specific CUDA OOM diagnostics (issue #244).

When a CUDA out-of-memory error occurs, this module produces stage-specific
suggestions. The stage is determined by **where** the error was caught in
the training pipeline (model loading, rollout, training), NOT by guessing
from traceback text.

Design principles:

* Stage comes from explicit call-site boundaries, not string matching.
* Augment, never replace, the original OOM error.
* Suggestions must use real AReno CLI option names and current resolved values.
* Omit advice irrelevant to the failing stage.
* Do not mutate configuration or retry automatically.
* Unknown stage (no boundary marker) produces no guidance (backward compatible).

Public API:

* :class:`OOMStage` — enum for the three stages + UNKNOWN.
* :class:`OOMGuidance` — structured result with ``to_dict()`` and ``to_json()``.
* :func:`build_oom_guidance` — build suggestions from stage + config snapshot.
* :func:`format_oom_guidance` — human-readable multi-line string.
* :func:`is_oom_error` — detect OOM from exception types and messages.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("areno.engine.oom_diagnostics")

_TROUBLESHOOTING_URL = "https://github.com/inclusionAI/AReno/blob/main/docs/troubleshooting/oom-timeout.rst"


class OOMStage(str, Enum):
    """The three stages where CUDA OOM can occur in AReno."""

    MODEL_LOADING = "model_loading"
    ROLLOUT = "rollout"
    TRAINING = "training"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OOMSuggestion:
    """A single actionable suggestion for resolving an OOM error.

    Attributes:
        option: The AReno CLI option name to adjust, or ``None`` for a
            diagnostic action that does not change configuration.
        current_value: The resolved value currently in effect.
        recommended_action: Human-readable description of what to do.
        priority: Lower numbers appear first in the output.
    """

    option: str | None
    current_value: Any
    recommended_action: str
    priority: int = 0


@dataclass(frozen=True)
class OOMGuidance:
    """Structured OOM diagnostic output.

    Attributes:
        stage: The detected stage where OOM occurred.
        suggestions: Ordered list of :class:`OOMSuggestion` objects.
        config_snapshot: The config values used to generate suggestions.
        troubleshooting_url: Link to focused documentation.
    """

    stage: OOMStage
    suggestions: list[OOMSuggestion] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    troubleshooting_url: str = _TROUBLESHOOTING_URL

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""

        return {
            "stage": self.stage.value,
            "suggestions": [
                {
                    "option": s.option,
                    "current_value": s.current_value,
                    "recommended_action": s.recommended_action,
                    "priority": s.priority,
                }
                for s in self.suggestions
            ],
            "config_snapshot": self.config_snapshot,
            "troubleshooting_url": self.troubleshooting_url,
        }

    def to_json(self) -> str:
        """Return a JSON string."""

        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# OOM detection
# ---------------------------------------------------------------------------


def is_oom_error(error: BaseException) -> bool:
    """Return ``True`` if *error* is a CUDA out-of-memory error."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if any("OutOfMemory" in et.__name__ for et in type(current).__mro__):
            return True
        message = str(current).lower()
        if "out of memory" in message and any(marker in message for marker in ("cuda", "gpu", "cublas")):
            return True
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        nested = getattr(current, "exceptions", ())
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)
        pending.extend(item for item in nested if isinstance(item, BaseException))
    return False


@contextmanager
def oom_stage(stage: OOMStage):
    """Annotate a CUDA OOM at an explicit runtime boundary and re-raise it."""

    try:
        yield
    except Exception as exc:
        if is_oom_error(exc) and not hasattr(exc, "_oom_stage"):
            exc._oom_stage = stage.value
        raise


# ---------------------------------------------------------------------------
# Config snapshot (only from real resolved TrainerConfig fields)
# ---------------------------------------------------------------------------

# CLI options that exist in `areno train --help`.
# Only these can be suggested to the user.
_VALID_CLI_OPTIONS: frozenset[str] = frozenset(
    {
        "--tp-size",
        "--world-size",
        "--batch-size",
        "--n-samples",
        "--mini-bs",
        "--max-running-prompts",
        "--max-new-tokens",
        "--max-prompt-tokens",
        "--attn-backend",
        "--activation-checkpointing",
        "--drop-rollout-state",
        "--eager-decode",
        "--adam-8bit",
        "--gradient-accumulation-steps",
    }
)


def build_oom_config_snapshot(trainer_config: Any) -> dict[str, Any]:
    """Build a config snapshot from a real TrainerConfig.

    Only reads fields that exist on TrainerConfig. Does NOT hardcode
    values like compile_model.
    """

    snapshot: dict[str, Any] = {
        "tp_size": getattr(trainer_config, "tp_size", None),
        "world_size": getattr(trainer_config, "world_size", None),
        "batch_size": getattr(trainer_config, "batch_size", None),
        "mini_bs": getattr(trainer_config, "mini_bs", None),
        "attn_backend": getattr(trainer_config, "attn_backend", None),
        "activation_checkpointing": getattr(trainer_config, "activation_checkpointing", None),
        "keep_rollout_state": getattr(trainer_config, "keep_rollout_state", None),
        "eager_decode": getattr(trainer_config, "eager_decode", None),
        "adam_8bit": getattr(trainer_config, "adam_8bit", None),
        "gradient_accumulation_steps": getattr(trainer_config, "gradient_accumulation_steps", None),
    }
    dp_size = None
    ts = snapshot.get("tp_size")
    ws = snapshot.get("world_size")
    if ts and ws and isinstance(ts, int) and isinstance(ws, int) and ts > 0:
        dp_size = ws // ts
    snapshot["dp_size"] = dp_size
    snapshot["drop_rollout_state"] = (
        not snapshot.get("keep_rollout_state", True) if snapshot.get("keep_rollout_state") is not None else None
    )
    # rollout-specific
    if hasattr(trainer_config, "n_samples"):
        snapshot["n_samples"] = trainer_config.n_samples
    if hasattr(trainer_config, "max_running_prompts"):
        snapshot["max_running_prompts"] = trainer_config.resolved_max_running_prompts()
    return snapshot


# ---------------------------------------------------------------------------
# Suggestion builders (per stage, only real CLI options)
# ---------------------------------------------------------------------------


def _build_model_loading_suggestions(cfg: dict[str, Any]) -> list[OOMSuggestion]:
    """Suggestions for OOM during model loading / weight initialisation.

    Note: optimizer-related options (--adam-8bit) are NOT included here
    because optimizer state is not allocated during model loading.
    """

    suggestions: list[OOMSuggestion] = [
        OOMSuggestion(
            option=None,
            current_value=None,
            recommended_action=(
                "Inspect competing GPU processes with nvidia-smi and stop only stale jobs you own before retrying."
            ),
            priority=0,
        )
    ]
    tp_size = cfg.get("tp_size")
    world_size = cfg.get("world_size")
    next_tp_size = _next_tp_size(tp_size, world_size)
    if next_tp_size is not None:
        suggestions.append(
            OOMSuggestion(
                option="--tp-size",
                current_value=tp_size,
                recommended_action=(
                    f"Increase --tp-size from {tp_size} to {next_tp_size} to shard the model "
                    "across more GPUs, reducing per-GPU memory for weights."
                ),
                priority=1,
            )
        )

    attn_backend = cfg.get("attn_backend")
    if attn_backend == "flash":
        suggestions.append(
            OOMSuggestion(
                option="--attn-backend",
                current_value=attn_backend,
                recommended_action=(
                    "Try --attn-backend native if flash-attn workspace allocations "
                    "contribute to the OOM (slower but lower peak workspace memory)."
                ),
                priority=2,
            )
        )

    return suggestions


def _build_rollout_suggestions(cfg: dict[str, Any]) -> list[OOMSuggestion]:
    """Suggestions for OOM during rollout / inference / KV-cache allocation."""

    suggestions: list[OOMSuggestion] = []
    max_running = cfg.get("max_running_prompts")
    if isinstance(max_running, int) and max_running > 0:
        suggestions.append(
            OOMSuggestion(
                option="--max-running-prompts",
                current_value=max_running,
                recommended_action=(
                    f"Reduce --max-running-prompts (currently {max_running}) to lower "
                    "concurrent decode memory and KV-cache footprint."
                ),
                priority=0,
            )
        )

    batch_size = cfg.get("batch_size")
    n_samples = cfg.get("n_samples")
    if isinstance(batch_size, int) and isinstance(n_samples, int):
        total = batch_size * n_samples
        if total > 0:
            suggestions.append(
                OOMSuggestion(
                    option="--batch-size",
                    current_value=batch_size,
                    recommended_action=(
                        f"Reduce --batch-size (currently {batch_size}) or --n-samples "
                        f"(currently {n_samples}) to lower concurrent rollout sequences "
                        f"(total={total})."
                    ),
                    priority=1,
                )
            )

    eager_decode = cfg.get("eager_decode")
    if eager_decode is False:
        suggestions.append(
            OOMSuggestion(
                option="--eager-decode",
                current_value=eager_decode,
                recommended_action=(
                    "Add --eager-decode to disable decode CUDA graph capture, which "
                    "pre-allocates workspace memory per decode bucket."
                ),
                priority=2,
            )
        )

    tp_size = cfg.get("tp_size")
    next_tp_size = _next_tp_size(tp_size, cfg.get("world_size"))
    if next_tp_size is not None:
        suggestions.append(
            OOMSuggestion(
                option="--tp-size",
                current_value=tp_size,
                recommended_action=(
                    f"Increase --tp-size from {tp_size} to {next_tp_size} to shard KV-cache across more GPUs."
                ),
                priority=4,
            )
        )

    return suggestions


def _build_training_suggestions(cfg: dict[str, Any]) -> list[OOMSuggestion]:
    """Suggestions for OOM during training (forward/backward/optimizer)."""

    suggestions: list[OOMSuggestion] = []
    mini_bs = cfg.get("mini_bs")
    if isinstance(mini_bs, int) and mini_bs > 0:
        suggestions.append(
            OOMSuggestion(
                option="--mini-bs",
                current_value=mini_bs,
                recommended_action=(
                    f"Reduce --mini-bs (currently {mini_bs}) to shrink the training "
                    "microbatch and lower activation memory."
                ),
                priority=0,
            )
        )

    activation_checkpointing = cfg.get("activation_checkpointing")
    if activation_checkpointing is False:
        suggestions.append(
            OOMSuggestion(
                option="--activation-checkpointing",
                current_value=activation_checkpointing,
                recommended_action=(
                    "Enable --activation-checkpointing to trade compute for memory "
                    "by recomputing activations during backward."
                ),
                priority=1,
            )
        )

    drop_rollout = cfg.get("drop_rollout_state")
    if drop_rollout is False:
        suggestions.append(
            OOMSuggestion(
                option="--drop-rollout-state",
                current_value=drop_rollout,
                recommended_action=(
                    "Add --drop-rollout-state to release rollout state before training, "
                    "freeing GPU memory for the backward pass."
                ),
                priority=2,
            )
        )

    adam_8bit = cfg.get("adam_8bit")
    if adam_8bit is False:
        suggestions.append(
            OOMSuggestion(
                option="--adam-8bit",
                current_value=adam_8bit,
                recommended_action=(
                    "Enable --adam-8bit to use 8-bit Adam moment states, which roughly halves optimizer memory."
                ),
                priority=3,
            )
        )

    gradient_accumulation = cfg.get("gradient_accumulation_steps")
    if isinstance(gradient_accumulation, int) and gradient_accumulation > 1:
        suggestions.append(
            OOMSuggestion(
                option="--gradient-accumulation-steps",
                current_value=gradient_accumulation,
                recommended_action=(
                    f"Increase --gradient-accumulation-steps (currently {gradient_accumulation}) "
                    "and further reduce --mini-bs to keep the same effective batch size."
                ),
                priority=4,
            )
        )

    tp_size = cfg.get("tp_size")
    next_tp_size = _next_tp_size(tp_size, cfg.get("world_size"))
    if next_tp_size is not None:
        suggestions.append(
            OOMSuggestion(
                option="--tp-size",
                current_value=tp_size,
                recommended_action=(
                    f"Increase --tp-size from {tp_size} to {next_tp_size} to shard model weights, "
                    "gradients, and optimizer states across more GPUs."
                ),
                priority=5,
            )
        )

    return suggestions


def _next_tp_size(tp_size: Any, world_size: Any) -> int | None:
    """Return the next valid tensor-parallel divisor, if one exists."""

    if not isinstance(tp_size, int) or not isinstance(world_size, int) or tp_size < 1 or world_size < 1:
        return None
    return next((candidate for candidate in range(tp_size + 1, world_size + 1) if world_size % candidate == 0), None)


_STAGE_BUILDERS: dict[OOMStage, Any] = {
    OOMStage.MODEL_LOADING: _build_model_loading_suggestions,
    OOMStage.ROLLOUT: _build_rollout_suggestions,
    OOMStage.TRAINING: _build_training_suggestions,
    OOMStage.UNKNOWN: lambda cfg: [],
}

# Keys relevant to each stage (for snapshot filtering).
_STAGE_RELEVANT_KEYS: dict[OOMStage, list[str]] = {
    OOMStage.MODEL_LOADING: ["tp_size", "dp_size", "world_size", "attn_backend"],
    OOMStage.ROLLOUT: [
        "max_running_prompts",
        "batch_size",
        "n_samples",
        "eager_decode",
        "drop_rollout_state",
        "tp_size",
        "world_size",
    ],
    OOMStage.TRAINING: [
        "mini_bs",
        "activation_checkpointing",
        "drop_rollout_state",
        "adam_8bit",
        "gradient_accumulation_steps",
        "tp_size",
        "world_size",
    ],
    OOMStage.UNKNOWN: [],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_oom_guidance(stage: OOMStage, config_snapshot: dict[str, Any]) -> OOMGuidance:
    """Build structured OOM guidance from a stage and config snapshot.

    Args:
        stage: The stage where OOM occurred (from explicit call-site boundary).
        config_snapshot: Dict of resolved option values from TrainerConfig.

    Returns:
        An :class:`OOMGuidance` with ordered suggestions.
    """

    builder = _STAGE_BUILDERS.get(stage, lambda cfg: [])
    relevant_keys = _STAGE_RELEVANT_KEYS.get(stage, [])
    filtered_cfg = {k: config_snapshot[k] for k in relevant_keys if k in config_snapshot}
    suggestions = builder(filtered_cfg)
    suggestions.sort(key=lambda s: s.priority)
    return OOMGuidance(
        stage=stage,
        suggestions=suggestions,
        config_snapshot=filtered_cfg,
    )


def format_oom_guidance(stage: OOMStage, config_snapshot: dict[str, Any]) -> str:
    """Return a human-readable multi-line OOM guidance string.

    Returns empty string for UNKNOWN stage (backward compatible).
    """

    if stage is OOMStage.UNKNOWN:
        return ""

    guidance = build_oom_guidance(stage, config_snapshot)
    if not guidance.suggestions:
        return ""

    lines: list[str] = []
    stage_label = {
        OOMStage.MODEL_LOADING: "model loading",
        OOMStage.ROLLOUT: "rollout generation",
        OOMStage.TRAINING: "training",
    }.get(stage, str(stage))

    lines.append(f"CUDA OOM during {stage_label}. Suggestions (in priority order):")
    lines.append("")
    for i, s in enumerate(guidance.suggestions, 1):
        lines.append(f"  {i}. {s.recommended_action}")
        if s.option is not None:
            lines.append(f"     Option: {s.option}  (current value: {s.current_value})")
    lines.append("")
    lines.append(f"See {guidance.troubleshooting_url} for detailed OOM troubleshooting.")
    return "\n".join(lines)


def validate_suggestions_use_real_cli_options(guidance: OOMGuidance) -> bool:
    """Check that all suggestion options exist in CLI --help.

    Returns True if all options are valid, False otherwise.
    """

    for s in guidance.suggestions:
        if s.option is not None and s.option not in _VALID_CLI_OPTIONS:
            return False
    return True
