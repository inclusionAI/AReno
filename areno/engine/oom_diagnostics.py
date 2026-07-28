"""Stage-specific CUDA OOM diagnostics.

When a CUDA out-of-memory error occurs, this module augments — but never
replaces — the original error with ordered, actionable suggestions based on
which stage the error happened in: model loading, rollout generation, or
training.  Suggestions reference real AReno option names and the resolved
values currently in effect so the user knows exactly what to change.

The module exposes two public entry points:

* ``format_oom_guidance(stage, config_snapshot)`` – returns a human-readable
  multi-line string to log or print after the original traceback.
* ``OOMGuidance`` dataclass – a structured representation with ``stage``,
  ``suggestions`` list, and ``config_snapshot`` for machine consumers.

Design principles (from issue #244):

* Augment, never replace, the original OOM error.
* Suggestions must use real AReno option names and current resolved values.
* Omit advice irrelevant to the failing stage.
* Do not mutate configuration or retry automatically.
* Link to focused troubleshooting documentation.
* Default behavior (no stage detected) preserves backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("areno.engine.oom_diagnostics")

# Documentation link for OOM troubleshooting.
_TROUBLESHOOTING_URL = "https://github.com/inclusionAI/AReno/blob/main/docs/cli/training.rst#troubleshooting-oom"


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
        option: The AReno CLI option name to adjust (e.g. ``--tp-size``).
        current_value: The resolved value currently in effect.
        recommended_action: Human-readable description of what to do.
        priority: Lower numbers appear first in the output.
    """

    option: str
    current_value: Any
    recommended_action: str
    priority: int = 0


@dataclass(frozen=True)
class OOMGuidance:
    """Structured OOM diagnostic output for machine consumers.

    Attributes:
        stage: The detected stage where OOM occurred.
        suggestions: Ordered list of ``OOMSuggestion`` objects.
        config_snapshot: The config values used to generate suggestions.
        troubleshooting_url: Link to focused documentation.
    """

    stage: OOMStage
    suggestions: list[OOMSuggestion]
    config_snapshot: dict[str, Any]
    troubleshooting_url: str = _TROUBLESHOOTING_URL

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for structured output."""

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


# ---------------------------------------------------------------------------
# Stage detection
# ---------------------------------------------------------------------------

# Keywords mapped to stages.  Checked in order; first match wins.
_STAGE_KEYWORDS: list[tuple[OOMStage, list[str]]] = [
    (
        OOMStage.MODEL_LOADING,
        [
            "build_model_on_device",
            "load_model_weights",
            "onload_train_weights",
            "model loading",
            "model_weights",
            "init_weights",
            "skipping decode cuda graph capture after oom",
        ],
    ),
    (
        OOMStage.ROLLOUT,
        [
            "infer_rollout",
            "rollout",
            "decode_graph",
            "prefill",
            "kv_cache",
            "kv cache",
            "decode cuda graph",
            "max_running_prompts",
            "sampling",
            "generate",
        ],
    ),
    (
        OOMStage.TRAINING,
        [
            "train",
            "backward",
            "loss",
            "optimizer",
            "gradient",
            "mini_bs",
            "mini-bs",
            "activation_checkpointing",
            "_train_step",
        ],
    ),
]


def detect_stage(error_text: str) -> OOMStage:
    """Detect which stage an OOM occurred in from the error traceback.

    Args:
        error_text: The full traceback string from the OOM exception.

    Returns:
        The detected :class:`OOMStage`, or :class:`OOMStage.UNKNOWN` if
        no stage can be determined.
    """

    lowered = error_text.lower()
    for stage, keywords in _STAGE_KEYWORDS:
        for kw in keywords:
            if kw in lowered:
                return stage
    return OOMStage.UNKNOWN


# ---------------------------------------------------------------------------
# Suggestion builders
# ---------------------------------------------------------------------------

# Config keys that are relevant for each stage.  Only these keys are read
# from the snapshot to keep suggestions focused and avoid irrelevant advice.
_STAGE_RELEVANT_KEYS: dict[OOMStage, list[str]] = {
    OOMStage.MODEL_LOADING: [
        "tp_size",
        "dp_size",
        "world_size",
        "model_path",
        "attn_backend",
        "compile_model",
        "activation_checkpointing",
        "adam_8bit",
        "dummy_load",
    ],
    OOMStage.ROLLOUT: [
        "max_running_prompts",
        "batch_size",
        "n_samples",
        "max_new_tokens",
        "max_prompt_tokens",
        "tp_size",
        "eager_decode",
        "keep_rollout_state",
        "drop_rollout_state",
        "attn_backend",
    ],
    OOMStage.TRAINING: [
        "mini_bs",
        "batch_size",
        "n_samples",
        "tp_size",
        "activation_checkpointing",
        "drop_rollout_state",
        "keep_rollout_state",
        "adam_8bit",
        "gradient_accumulation_steps",
        "max_new_tokens",
        "max_prompt_tokens",
    ],
    OOMStage.UNKNOWN: [],
}


def _build_model_loading_suggestions(cfg: dict[str, Any]) -> list[OOMSuggestion]:
    """Suggestions for OOM during model loading / weight initialisation."""

    suggestions: list[OOMSuggestion] = []
    tp_size = cfg.get("tp_size")
    if tp_size is not None and isinstance(tp_size, int) and tp_size > 0:
        suggestions.append(
            OOMSuggestion(
                option="--tp-size",
                current_value=tp_size,
                recommended_action=(
                    f"Increase --tp-size (currently {tp_size}) to shard the model "
                    "across more GPUs, reducing per-GPU memory for weights."
                ),
                priority=0,
            )
        )

    dp_size = cfg.get("dp_size")
    world_size = cfg.get("world_size")
    if dp_size is not None and world_size is not None and isinstance(dp_size, int) and isinstance(world_size, int):
        if dp_size > 1:
            suggestions.append(
                OOMSuggestion(
                    option="--world-size / --dp-size",
                    current_value=f"world_size={world_size}, dp_size={dp_size}",
                    recommended_action=(
                        "Reduce data-parallel size (or increase --tp-size within the same "
                        "world_size) so each rank holds a smaller model shard."
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

    compile_model = cfg.get("compile_model")
    if compile_model:
        suggestions.append(
            OOMSuggestion(
                option="--no-compile-model (runtime.compile_model)",
                current_value=compile_model,
                recommended_action=(
                    "Disable torch.compile via runtime config to avoid extra "
                    "compile-time memory overhead during model construction."
                ),
                priority=3,
            )
        )

    adam_8bit = cfg.get("adam_8bit")
    if adam_8bit is not None and not adam_8bit:
        suggestions.append(
            OOMSuggestion(
                option="--adam-8bit",
                current_value=adam_8bit,
                recommended_action=(
                    "Enable --adam-8bit to use 8-bit Adam moment states, which roughly halves optimizer memory."
                ),
                priority=4,
            )
        )

    return suggestions


def _build_rollout_suggestions(cfg: dict[str, Any]) -> list[OOMSuggestion]:
    """Suggestions for OOM during rollout / inference / KV-cache allocation."""

    suggestions: list[OOMSuggestion] = []
    max_running = cfg.get("max_running_prompts")
    if max_running is not None and isinstance(max_running, int) and max_running > 0:
        suggestions.append(
            OOMSuggestion(
                option="--max-running-prompts",
                current_value=max_running,
                recommended_action=(
                    f"Reduce --max-running-prompts (currently {max_running}) to lower "
                    "concurrent decode memory and KV-cache footprint.  Try halving it."
                ),
                priority=0,
            )
        )

    batch_size = cfg.get("batch_size")
    n_samples = cfg.get("n_samples")
    if batch_size is not None and n_samples is not None:
        total = batch_size * n_samples if isinstance(batch_size, int) and isinstance(n_samples, int) else None
        if total is not None and total > 0:
            suggestions.append(
                OOMSuggestion(
                    option="--batch-size / --n-samples",
                    current_value=f"batch_size={batch_size}, n_samples={n_samples} (total={total})",
                    recommended_action=(
                        "Reduce --batch-size or --n-samples to lower the total number of concurrent rollout sequences."
                    ),
                    priority=1,
                )
            )

    max_new_tokens = cfg.get("max_new_tokens")
    if max_new_tokens is not None and isinstance(max_new_tokens, int) and max_new_tokens > 0:
        suggestions.append(
            OOMSuggestion(
                option="--max-new-tokens",
                current_value=max_new_tokens,
                recommended_action=(
                    f"Reduce --max-new-tokens (currently {max_new_tokens}) to shorten "
                    "the maximum generation length and reduce per-sequence KV-cache memory."
                ),
                priority=2,
            )
        )

    eager_decode = cfg.get("eager_decode")
    if eager_decode is not None and not eager_decode:
        suggestions.append(
            OOMSuggestion(
                option="--eager-decode",
                current_value=eager_decode,
                recommended_action=(
                    "Add --eager-decode to disable decode CUDA graph capture, which "
                    "pre-allocates workspace memory per decode bucket."
                ),
                priority=3,
            )
        )

    drop_rollout = cfg.get("drop_rollout_state")
    keep_rollout = cfg.get("keep_rollout_state")
    if drop_rollout is not None and not drop_rollout and keep_rollout is not None and keep_rollout:
        suggestions.append(
            OOMSuggestion(
                option="--drop-rollout-state",
                current_value="keep_rollout_state=True",
                recommended_action=(
                    "Add --drop-rollout-state to release rollout KV-cache and decode "
                    "state after rollout, freeing memory before training."
                ),
                priority=4,
            )
        )

    tp_size = cfg.get("tp_size")
    if tp_size is not None and isinstance(tp_size, int) and tp_size > 0:
        suggestions.append(
            OOMSuggestion(
                option="--tp-size",
                current_value=tp_size,
                recommended_action=(f"Increase --tp-size (currently {tp_size}) to shard KV-cache across more GPUs."),
                priority=5,
            )
        )

    return suggestions


def _build_training_suggestions(cfg: dict[str, Any]) -> list[OOMSuggestion]:
    """Suggestions for OOM during training (forward/backward/optimizer)."""

    suggestions: list[OOMSuggestion] = []
    mini_bs = cfg.get("mini_bs")
    if mini_bs is not None and isinstance(mini_bs, int) and mini_bs > 0:
        suggestions.append(
            OOMSuggestion(
                option="--mini-bs",
                current_value=mini_bs,
                recommended_action=(
                    f"Reduce --mini-bs (currently {mini_bs}) to shrink the training "
                    "microbatch and lower activation memory.  Try halving it."
                ),
                priority=0,
            )
        )

    activation_checkpointing = cfg.get("activation_checkpointing")
    if activation_checkpointing is not None and not activation_checkpointing:
        suggestions.append(
            OOMSuggestion(
                option="--activation-checkpointing",
                current_value=activation_checkpointing,
                recommended_action=(
                    "Enable --activation-checkpointing to trade compute for memory "
                    "by recomputing activations during backward instead of storing them."
                ),
                priority=1,
            )
        )

    drop_rollout = cfg.get("drop_rollout_state")
    if drop_rollout is not None and not drop_rollout:
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
    if adam_8bit is not None and not adam_8bit:
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

    max_new_tokens = cfg.get("max_new_tokens")
    if max_new_tokens is not None and isinstance(max_new_tokens, int) and max_new_tokens > 0:
        suggestions.append(
            OOMSuggestion(
                option="--max-new-tokens",
                current_value=max_new_tokens,
                recommended_action=(
                    f"Reduce --max-new-tokens (currently {max_new_tokens}) to shorten "
                    "training sequences and lower peak activation memory."
                ),
                priority=4,
            )
        )

    gradient_accumulation = cfg.get("gradient_accumulation_steps")
    if gradient_accumulation is not None and isinstance(gradient_accumulation, int) and gradient_accumulation > 1:
        suggestions.append(
            OOMSuggestion(
                option="--gradient-accumulation-steps",
                current_value=gradient_accumulation,
                recommended_action=(
                    f"Increase --gradient-accumulation-steps (currently {gradient_accumulation}) "
                    "and further reduce --mini-bs to keep the same effective batch size "
                    "with smaller microbatches."
                ),
                priority=5,
            )
        )

    tp_size = cfg.get("tp_size")
    if tp_size is not None and isinstance(tp_size, int) and tp_size > 0:
        suggestions.append(
            OOMSuggestion(
                option="--tp-size",
                current_value=tp_size,
                recommended_action=(
                    f"Increase --tp-size (currently {tp_size}) to shard model weights, "
                    "gradients, and optimizer states across more GPUs."
                ),
                priority=6,
            )
        )

    return suggestions


_STAGE_BUILDERS: dict[OOMStage, Any] = {
    OOMStage.MODEL_LOADING: _build_model_loading_suggestions,
    OOMStage.ROLLOUT: _build_rollout_suggestions,
    OOMStage.TRAINING: _build_training_suggestions,
    OOMStage.UNKNOWN: lambda cfg: [],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_oom_guidance(
    stage: OOMStage,
    config_snapshot: dict[str, Any],
) -> OOMGuidance:
    """Build structured OOM guidance for the given stage and config.

    Args:
        stage: The detected OOM stage.
        config_snapshot: A dict of resolved AReno option values.  Keys that
            are not relevant to the stage are ignored.

    Returns:
        An :class:`OOMGuidance` with ordered suggestions.
    """

    builder = _STAGE_BUILDERS.get(stage, lambda cfg: [])
    # Only pass stage-relevant keys to keep suggestions focused.
    relevant_keys = _STAGE_RELEVANT_KEYS.get(stage, [])
    filtered_cfg = {k: config_snapshot[k] for k in relevant_keys if k in config_snapshot}
    suggestions = builder(filtered_cfg)
    suggestions.sort(key=lambda s: s.priority)
    return OOMGuidance(
        stage=stage,
        suggestions=suggestions,
        config_snapshot=filtered_cfg,
    )


def format_oom_guidance(
    stage: OOMStage,
    config_snapshot: dict[str, Any],
) -> str:
    """Return a human-readable multi-line OOM guidance string.

    This is intended to be logged or printed *after* the original OOM
    traceback.  It never replaces the original error.

    Args:
        stage: The detected OOM stage (use :func:`detect_stage` from the
            error traceback, or pass :class:`OOMStage.UNKNOWN` if unknown).
        config_snapshot: A dict of resolved AReno option values.

    Returns:
        A formatted string with a header, ordered suggestions, and a
        documentation link.  Returns an empty string for
        :class:`OOMStage.UNKNOWN` to preserve backward compatibility.
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
        lines.append(f"     Option: {s.option}  (current value: {s.current_value})")
    lines.append("")
    lines.append(f"See {_TROUBLESHOOTING_URL} for detailed OOM troubleshooting.")
    return "\n".join(lines)


def is_oom_error(error: BaseException) -> bool:
    """Return ``True`` if *error* is a CUDA out-of-memory error."""

    # PyTorch raises ``torch.cuda.OutOfMemoryError`` (subclass of
    # ``torch.OutOfMemoryError`` since PyTorch 2.x).  We also check the
    # error message for robustness across PyTorch versions.
    error_types = type(error).__mro__
    for et in error_types:
        if "OutOfMemory" in et.__name__:
            return True
    msg = str(error).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def diagnose_oom_from_exception(
    error: BaseException,
    config_snapshot: dict[str, Any],
) -> str:
    """Convenience wrapper: detect stage from an exception and format guidance.

    Args:
        error: The OOM exception (or any exception whose traceback
            text contains stage hints).
        config_snapshot: Resolved AReno option values.

    Returns:
        Human-readable guidance string, or ``""`` if the stage cannot be
        determined (backward-compatible default).
    """

    error_text = ""
    # Prefer the traceback if attached, otherwise use str(error).
    tb = getattr(error, "__traceback__", None)
    if tb is not None:
        import traceback as tb_mod

        error_text = tb_mod.format_exception(type(error), error, tb)
        error_text = "".join(error_text)
    if not error_text:
        error_text = str(error)

    stage = detect_stage(error_text)
    return format_oom_guidance(stage, config_snapshot)
