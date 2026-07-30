#!/usr/bin/env python3
"""Generate a complete, editable training recipe and launch command.

Given a training mode (SFT/DPO/GSPO/GRPO/PPO), GPU count, context length, and
target batch size, this script derives a full training configuration from
AReno's ``TrainerConfig`` dataclass hierarchy defaults, validates all inputs,
and emits both structured JSON and a human-readable summary.  The output
includes per-value provenance explaining why each configuration value was
chosen.
"""

from __future__ import annotations

import argparse
import json
import shlex
from typing import Any

# ---------------------------------------------------------------------------
# Constants — mirrors of AReno's public contracts (no runtime import required)
# ---------------------------------------------------------------------------

MODES = ("sft", "dpo", "gspo", "grpo", "ppo")

ROLLOUT_MODES = {"gspo", "grpo", "ppo"}

# Dataclass field name -> CLI flag name.  Most fields simply replace ``_`` with
# ``-`` and prefix ``--``; the entries below capture the exceptions that exist
# in ``areno/cli/train.py``.
_SPECIAL_CLI_FLAGS: dict[str, str] = {
    "optimizer_lr": "--lr",
    "optimizer_min_lr": "--min-lr",
    "optimizer_beta1": "--adam-beta1",
    "optimizer_beta2": "--adam-beta2",
}

# Boolean flags whose CLI representation is an ``is_flag`` (emit only when
# the value is *True*).  Each entry maps a recipe key to the CLI flag string.
_INVERTED_BOOL_FLAGS: dict[str, str] = {
    "eager_decode": "--eager-decode",
    "greedy": "--greedy",
    "adam_8bit": "--adam-8bit",
    "train_tool_results": "--train-tool-results",
    "disable_thinking": "--disable-thinking",
}

# ``--drop-rollout-state`` is an ``is_flag`` that means
# ``keep_rollout_state=False``.  Emit it only when keep_rollout_state is False.
_DROP_ROLLOUT_FLAG = "--drop-rollout-state"

# ``--no-activation-checkpointing`` is the negated form; emit it only when
# activation_checkpointing is False.
_NO_ACTIVATION_FLAG = "--no-activation-checkpointing"


def _cli_flag(field_name: str) -> str:
    """Return the CLI flag string for a dataclass field name."""

    if field_name in _SPECIAL_CLI_FLAGS:
        return _SPECIAL_CLI_FLAGS[field_name]
    return "--" + field_name.replace("_", "-")


# ---------------------------------------------------------------------------
# Per-mode field sets (ordered for stable output)
# ---------------------------------------------------------------------------

# Common fields shared by *all* modes, in display order.
_BASE_FIELDS: tuple[str, ...] = (
    "algo",
    "ckpt",
    "dataset_path",
    "model_hub",
    "dataset_loader_fn",
    "tp_size",
    "world_size",
    "batch_size",
    "mini_bs",
    "score_micro_bs",
    "gradient_accumulation_steps",
    "max_prompt_tokens",
    "max_new_tokens",
    "max_context_len",
    "optimizer_lr",
    "optimizer_min_lr",
    "lr_decay_steps",
    "lr_decay_style",
    "optimizer_beta1",
    "optimizer_beta2",
    "weight_decay",
    "grad_clip_norm",
    "adam_8bit",
    "activation_checkpointing",
    "keep_rollout_state",
    "eager_decode",
    "attn_backend",
    "epochs",
    "max_steps",
    "save_path",
    "save_interval",
    "metrics_log_dir",
)

# Additional fields for rollout-capable modes (gspo/grpo/ppo).
_ROLLOUT_FIELDS: tuple[str, ...] = (
    "n_samples",
    "temperature",
    "top_k",
    "top_p",
    "greedy",
    "max_running_prompts",
)

# Additional fields for policy-gradient modes (gspo/grpo).
_POLICY_FIELDS: tuple[str, ...] = (
    "reward_fn_path",
    "agent_fn",
    "agent_timeout_s",
    "train_tool_results",
    "chat_template_enable_thinking",
)

# GSPO-specific clip epsilon.
_GSPO_FIELDS: tuple[str, ...] = ("gspo_clip_eps",)

# GRPO-specific clip epsilon.
_GRPO_FIELDS: tuple[str, ...] = ("grpo_clip_eps",)

# DPO-specific fields.
_DPO_FIELDS: tuple[str, ...] = ("ref_ckpt", "dpo_beta")

# PPO-specific fields (extends policy config).
_PPO_FIELDS: tuple[str, ...] = (
    "ref_ckpt",
    "reward_ckpt",
    "critic_ckpt",
    "critic_lr",
    "critic_warmup_steps",
    "use_kl_loss",
    "kl_loss_coef",
    "kl_loss_type",
    "clip_eps",
    "clip_ratio_c",
    "value_clip_eps",
    "value_loss_coef",
    "gamma",
    "lam",
)


def _fields_for_mode(mode: str) -> tuple[str, ...]:
    """Return the ordered field tuple for a given training mode."""

    fields = list(_BASE_FIELDS)
    if mode in ROLLOUT_MODES:
        fields.extend(_ROLLOUT_FIELDS)
    if mode in {"gspo", "grpo", "ppo"}:
        fields.extend(_POLICY_FIELDS)
    if mode == "gspo":
        fields.extend(_GSPO_FIELDS)
    if mode == "grpo":
        fields.extend(_GRPO_FIELDS)
    if mode == "dpo":
        fields.extend(_DPO_FIELDS)
    if mode == "ppo":
        fields.extend(_PPO_FIELDS)
    return tuple(fields)


# ---------------------------------------------------------------------------
# Defaults — copied from ``areno/api/trainer_config.py`` dataclass defaults
# ---------------------------------------------------------------------------

_BASE_DEFAULTS: dict[str, Any] = {
    "model_hub": "modelscope",
    "dataset_loader_fn": None,
    "save_path": None,
    "save_interval": 100,
    "epochs": 10,
    "max_steps": None,
    "tp_size": 4,
    "world_size": 8,
    "batch_size": 32,
    "mini_bs": 16,
    "score_micro_bs": 8,
    "gradient_accumulation_steps": None,
    "max_prompt_tokens": 1024,
    "max_new_tokens": 3071,
    "max_context_len": None,
    "optimizer_lr": 1.0e-6,
    "optimizer_min_lr": 1.0e-7,
    "lr_decay_steps": 1000,
    "lr_decay_style": "cosine",
    "optimizer_beta1": 0.9,
    "optimizer_beta2": 0.999,
    "weight_decay": 1.0e-2,
    "grad_clip_norm": 1.0,
    "adam_8bit": False,
    "activation_checkpointing": True,
    "keep_rollout_state": True,
    "eager_decode": False,
    "attn_backend": "flash",
    "metrics_log_dir": "/tmp/areno/tfevent",
    "agent_fn": None,
    "agent_timeout_s": 300.0,
    "train_tool_results": False,
    "chat_template_enable_thinking": None,
}

_ROLLOUT_DEFAULTS: dict[str, Any] = {
    "n_samples": 8,
    "greedy": False,
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "max_running_prompts": None,
}

_POLICY_DEFAULTS: dict[str, Any] = {
    "reward_fn_path": None,
    "gspo_clip_eps": 3.0e-4,
    "grpo_clip_eps": 0.2,
}

_DPO_DEFAULTS: dict[str, Any] = {
    "ref_ckpt": None,
    "dpo_beta": 0.1,
}

_PPO_DEFAULTS: dict[str, Any] = {
    "ref_ckpt": None,
    "reward_ckpt": None,
    "critic_ckpt": None,
    "role_device": None,
    "critic_lr": 1.0e-5,
    "kl_coef": 0.02,
    "use_kl_loss": True,
    "kl_loss_coef": 0.001,
    "kl_loss_type": "low_var_kl",
    "clip_eps": 0.2,
    "clip_ratio_c": 3.0,
    "value_clip_eps": 0.5,
    "value_loss_coef": 0.5,
    "gamma": 1.0,
    "lam": 0.95,
    "critic_warmup_steps": 20,
}


def _defaults_for_mode(mode: str) -> dict[str, Any]:
    """Return the full default-value dict for a given mode."""

    defaults = dict(_BASE_DEFAULTS)
    if mode in ROLLOUT_MODES:
        defaults.update(_ROLLOUT_DEFAULTS)
    if mode in {"gspo", "grpo", "ppo"}:
        defaults.update({k: v for k, v in _POLICY_DEFAULTS.items() if k != "gspo_clip_eps" or mode == "gspo"})
        defaults.update({k: v for k, v in _POLICY_DEFAULTS.items() if k != "grpo_clip_eps" or mode == "grpo"})
    if mode == "ppo":
        defaults.update(_PPO_DEFAULTS)
    if mode == "dpo":
        defaults.update(_DPO_DEFAULTS)
    return defaults


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_inputs(
    mode: str,
    gpu_count: int,
    context_length: int,
    target_batch: int,
    overrides: dict[str, Any],
) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""

    errors: list[str] = []
    if mode not in MODES:
        errors.append(f"mode must be one of {MODES}, got '{mode}'")
    if gpu_count <= 0:
        errors.append("gpu_count must be positive")
    if context_length <= 0:
        errors.append("context_length must be positive")
    if target_batch <= 0:
        errors.append("target_batch must be positive")
    tp_size = overrides.get("tp_size")
    if tp_size is not None:
        if tp_size <= 0:
            errors.append("tp_size override must be positive")
        elif gpu_count % tp_size != 0:
            errors.append(f"gpu_count ({gpu_count}) must be divisible by tp_size ({tp_size})")
    n_samples = overrides.get("n_samples")
    if n_samples is not None and mode in ROLLOUT_MODES and n_samples <= 0:
        errors.append("n_samples override must be positive for rollout modes")
    mini_bs = overrides.get("mini_bs")
    if mini_bs is not None and mini_bs <= 0:
        errors.append("mini_bs override must be positive")
    max_new = overrides.get("max_new_tokens")
    if max_new is not None and max_new <= 0:
        errors.append("max_new_tokens override must be positive")
    return errors


# ---------------------------------------------------------------------------
# Recipe derivation
# ---------------------------------------------------------------------------


def derive_recipe(
    mode: str,
    gpu_count: int,
    context_length: int,
    target_batch: int,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Derive the full recipe, provenance map, and warnings list.

    Returns ``(recipe, provenance, warnings)``.
    """

    fields = _fields_for_mode(mode)
    defaults = _defaults_for_mode(mode)
    recipe: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    warnings: list[str] = []

    small_gpu = gpu_count <= 2

    # ---- topology ----
    tp_size: int
    if "tp_size" in overrides:
        tp_size = int(overrides["tp_size"])
        provenance["tp_size"] = "user override"
    elif small_gpu:
        tp_size = 1
        provenance["tp_size"] = f"set to 1 because gpu_count ({gpu_count}) <= 2 (small-GPU friendly)"
    else:
        tp_size = min(4, gpu_count)
        provenance["tp_size"] = f"set to min(4, gpu_count) = {tp_size} (TrainerConfig default capped to GPU count)"

    world_size = gpu_count
    dp_size = world_size // tp_size if tp_size else 0

    # ---- batch sizing ----
    if "mini_bs" in overrides:
        mini_bs = int(overrides["mini_bs"])
        provenance["mini_bs"] = "user override"
    elif small_gpu:
        mini_bs = min(target_batch, 4)
        provenance["mini_bs"] = f"capped to min(target_batch, 4) = {mini_bs} for small GPU count (<=2) to avoid OOM"
    else:
        mini_bs = min(target_batch, 16)
        provenance["mini_bs"] = f"set to min(target_batch, 16) = {mini_bs} (TrainerConfig default capped to batch)"

    if "score_micro_bs" in overrides:
        score_micro_bs = int(overrides["score_micro_bs"])
        provenance["score_micro_bs"] = "user override"
    elif small_gpu:
        score_micro_bs = 4
        provenance["score_micro_bs"] = "reduced to 4 for small GPU count (<=2)"
    else:
        score_micro_bs = 8
        provenance["score_micro_bs"] = "TrainerConfig default (8)"

    # ---- context-length split ----
    max_prompt = min(context_length // 2, 1024)
    max_new = min(context_length - max_prompt, 3071)

    if "max_prompt_tokens" in overrides:
        max_prompt = int(overrides["max_prompt_tokens"])
        provenance["max_prompt_tokens"] = "user override"
    else:
        provenance["max_prompt_tokens"] = (
            f"set to min(context_length//2, 1024) = {max_prompt} (half context for prompt, capped at TrainerConfig default)"
        )

    if "max_new_tokens" in overrides:
        max_new = int(overrides["max_new_tokens"])
        provenance["max_new_tokens"] = "user override"
    else:
        provenance["max_new_tokens"] = (
            f"set to min(context_length - max_prompt_tokens, 3071) = {max_new} (remaining context for generation, capped at TrainerConfig default)"
        )

    # ---- n_samples for rollout modes ----
    n_samples = 8
    if "n_samples" in overrides:
        n_samples = int(overrides["n_samples"])
        provenance["n_samples"] = "user override"
    elif mode in ROLLOUT_MODES:
        provenance["n_samples"] = "RolloutTrainerConfig default (8)"

    # ---- build the recipe dict ----
    for field_name in fields:
        if field_name == "algo":
            recipe[field_name] = mode
            provenance[field_name] = "user-specified mode"
        elif field_name == "ckpt":
            recipe[field_name] = "<ckpt>"
            provenance[field_name] = "placeholder; user must provide model checkpoint path"
            warnings.append("ckpt is a placeholder; replace <ckpt> with your model checkpoint path or repo ID")
        elif field_name == "dataset_path":
            recipe[field_name] = "<dataset-path>"
            provenance[field_name] = "placeholder; user must provide dataset path"
            warnings.append("dataset_path is a placeholder; replace <dataset-path> with your dataset path or ref")
        elif field_name == "tp_size":
            recipe[field_name] = tp_size
        elif field_name == "world_size":
            recipe[field_name] = world_size
            provenance[field_name] = "set to gpu_count (user request)"
        elif field_name == "batch_size":
            recipe[field_name] = target_batch
            provenance[field_name] = "set to target_batch (user request)"
        elif field_name == "mini_bs":
            recipe[field_name] = mini_bs
        elif field_name == "score_micro_bs":
            recipe[field_name] = score_micro_bs
        elif field_name == "max_prompt_tokens":
            recipe[field_name] = max_prompt
        elif field_name == "max_new_tokens":
            recipe[field_name] = max_new
        elif field_name == "max_context_len":
            val = overrides.get("max_context_len", context_length)
            recipe[field_name] = val
            if "max_context_len" in overrides:
                provenance[field_name] = "user override"
            else:
                provenance[field_name] = f"set to context_length ({context_length}) (user request)"
        elif field_name == "n_samples":
            if mode in ROLLOUT_MODES:
                recipe[field_name] = n_samples
            # For non-rollout modes, n_samples is not in the field set.
        elif field_name == "reward_fn_path":
            val = overrides.get("reward_fn_path")
            recipe[field_name] = val
            if val:
                provenance[field_name] = "user override"
            else:
                provenance[field_name] = "not set; required for rollout RL — provide --reward-fn-path before training"
                warnings.append("reward_fn_path is not set; provide --reward-fn-path before training")
        elif field_name == "ref_ckpt":
            val = overrides.get("ref_ckpt")
            recipe[field_name] = val
            if val:
                provenance[field_name] = "user override"
            else:
                provenance[field_name] = "not set; DPO/PPO require a reference checkpoint — provide --ref-ckpt"
                if mode in {"dpo", "ppo"}:
                    warnings.append("ref_ckpt is not set; provide --ref-ckpt before training")
        elif field_name == "critic_ckpt":
            val = overrides.get("critic_ckpt")
            recipe[field_name] = val
            if val:
                provenance[field_name] = "user override"
            else:
                provenance[field_name] = "not set; PPO requires a critic checkpoint — provide --critic-ckpt"
                warnings.append("critic_ckpt is not set; provide --critic-ckpt before training")
        elif field_name == "reward_ckpt":
            val = overrides.get("reward_ckpt")
            recipe[field_name] = val
            if val:
                provenance[field_name] = "user override"
            else:
                provenance[field_name] = (
                    "not set; PPO may use a reward model checkpoint — provide --reward-ckpt if needed"
                )
                warnings.append("reward_ckpt is not set; provide --reward-ckpt if using a reward model")
        elif field_name == "dataset_loader_fn":
            val = overrides.get("dataset_loader_fn")
            if mode == "sft":
                val = val or "<loader-path>"
                provenance[field_name] = "placeholder; SFT requires a dataset loader — provide --dataset-loader-fn"
                warnings.append(
                    "dataset_loader_fn is a placeholder; SFT requires --dataset-loader-fn (e.g. examples/sft/alpaca/dataset_loader.py)"
                )
            else:
                provenance[field_name] = (
                    "optional; set if your dataset needs normalization (see inspect_dataset.py)"
                    if not val
                    else "user override"
                )
            recipe[field_name] = val
        elif field_name in overrides:
            recipe[field_name] = overrides[field_name]
            provenance[field_name] = "user override"
        elif field_name in defaults:
            recipe[field_name] = defaults[field_name]
            provenance[field_name] = f"TrainerConfig default ({defaults[field_name]!r})"
        else:
            # Fallback: should not reach here, but keep output complete.
            recipe[field_name] = None
            provenance[field_name] = "no default available"

    # Add DP size to provenance as supplementary info (not a recipe field).
    provenance["_dp_size"] = f"world_size // tp_size = {world_size} // {tp_size} = {dp_size}"

    return recipe, provenance, warnings


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def build_command(mode: str, recipe: dict[str, Any]) -> str:
    """Build a directly runnable ``areno train`` command string from the recipe."""

    parts: list[str] = ["areno", "train"]

    # Always include algo, ckpt, dataset-path first.
    parts.extend(["--algo", str(recipe["algo"])])
    parts.extend(["--ckpt", str(recipe["ckpt"])])
    parts.extend(["--dataset-path", str(recipe["dataset_path"])])

    # SFT always needs a dataset loader.
    if mode == "sft" and recipe.get("dataset_loader_fn") is not None:
        parts.extend(["--dataset-loader-fn", str(recipe["dataset_loader_fn"])])

    # Rollout modes need reward_fn_path (emit placeholder if not set).
    if mode in ROLLOUT_MODES and "reward_fn_path" in recipe:
        rwd = recipe["reward_fn_path"]
        parts.extend(["--reward-fn-path", str(rwd) if rwd else "<reward-fn-path>"])

    # DPO and PPO need ref_ckpt (emit placeholder if not set).
    if mode in {"dpo", "ppo"} and "ref_ckpt" in recipe:
        ref = recipe["ref_ckpt"]
        parts.extend(["--ref-ckpt", str(ref) if ref else "<ref-ckpt>"])

    # PPO needs reward_ckpt and critic_ckpt.
    if mode == "ppo":
        if "reward_ckpt" in recipe and recipe["reward_ckpt"] is not None:
            parts.extend(["--reward-ckpt", str(recipe["reward_ckpt"])])
        if "critic_ckpt" in recipe and recipe["critic_ckpt"] is not None:
            parts.extend(["--critic-ckpt", str(recipe["critic_ckpt"])])

    # Emit derived/configured values in field order, skipping already-emitted
    # and None-valued fields.
    skip = {
        "algo",
        "ckpt",
        "dataset_path",
        "dataset_loader_fn",
        "reward_fn_path",
        "ref_ckpt",
        "reward_ckpt",
        "critic_ckpt",
        "save_path",
        "metrics_log_dir",
        "model_hub",
        "agent_fn",
        "agent_timeout_s",
        "chat_template_enable_thinking",
        "role_device",
    }

    for field_name, value in recipe.items():
        if field_name in skip:
            continue
        if value is None:
            continue

        # Handle inverted boolean flags.
        if field_name == "activation_checkpointing":
            if value is False:
                parts.append(_NO_ACTIVATION_FLAG)
            continue
        if field_name == "keep_rollout_state":
            if value is False:
                parts.append(_DROP_ROLLOUT_FLAG)
            continue
        if field_name in _INVERTED_BOOL_FLAGS:
            if value is True:
                parts.append(_INVERTED_BOOL_FLAGS[field_name])
            continue

        # Skip booleans that are already False (no flag to emit).
        if isinstance(value, bool) and not value:
            continue

        flag = _cli_flag(field_name)
        parts.extend([flag, _format_value(value)])

    return shlex.join(parts)


def _format_value(value: Any) -> str:
    """Format a recipe value for CLI output."""

    if isinstance(value, float):
        # Preserve scientific notation for very small numbers.
        if abs(value) < 1e-3 and value != 0:
            return repr(value)
        return str(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def build_human_readable(
    mode: str,
    gpu_count: int,
    context_length: int,
    target_batch: int,
    recipe: dict[str, Any],
    provenance: dict[str, str],
    warnings: list[str],
) -> str:
    """Build a human-readable summary of the recipe."""

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("AReno Training Recipe")
    lines.append("=" * 60)
    lines.append(f"Mode:           {mode.upper()}")
    lines.append(f"GPU count:      {gpu_count}")
    lines.append(f"Context length: {context_length} tokens")
    lines.append(f"Target batch:   {target_batch}")
    lines.append(
        f"Topology:       world_size={recipe.get('world_size')}, tp_size={recipe.get('tp_size')}, "
        f"dp={recipe.get('world_size', 0) // max(recipe.get('tp_size', 1), 1)}"
    )
    lines.append("")

    lines.append("-" * 40)
    lines.append("Configuration")
    lines.append("-" * 40)
    for field_name, value in recipe.items():
        prov = provenance.get(field_name, "")
        # Truncate provenance for display.
        if len(prov) > 70:
            prov = prov[:67] + "..."
        lines.append(f"  {field_name:<30} {str(value):<20} # {prov}")
    lines.append("")

    if warnings:
        lines.append("-" * 40)
        lines.append("Warnings")
        lines.append("-" * 40)
        for warning in warnings:
            lines.append(f"  ! {warning}")
        lines.append("")

    lines.append("-" * 40)
    lines.append("Launch Command")
    lines.append("-" * 40)
    lines.append(f"  {build_command(mode, recipe)}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Override parsing
# ---------------------------------------------------------------------------


def parse_override(raw: str) -> tuple[str, Any]:
    """Parse a ``key=value`` override string into ``(field_name, typed_value)``."""

    if "=" not in raw:
        raise ValueError(f"override '{raw}' must be key=value format")
    key, _, value_str = raw.partition("=")
    key = key.strip()
    value_str = value_str.strip()

    # Try bool.
    if value_str.lower() in ("true", "false"):
        return key, value_str.lower() == "true"

    # Try int.
    try:
        return key, int(value_str)
    except ValueError:
        pass

    # Try float.
    try:
        return key, float(value_str)
    except ValueError:
        pass

    # String fallback (strip surrounding quotes if present).
    if value_str.startswith('"') and value_str.endswith('"'):
        value_str = value_str[1:-1]
    elif value_str.startswith("'") and value_str.endswith("'"):
        value_str = value_str[1:-1]
    return key, value_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a complete, editable training recipe and launch command.",
    )
    parser.add_argument("--mode", choices=MODES, required=True, help="Training algorithm mode.")
    parser.add_argument("--gpu-count", type=int, required=True, help="Number of GPUs available.")
    parser.add_argument("--context-length", type=int, required=True, help="Total context length in tokens.")
    parser.add_argument("--target-batch", type=int, required=True, help="Target prompt/pair batch size.")
    parser.add_argument("--tp-size", type=int, default=None, help="Override tensor-parallel size.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override rollout samples per prompt.")
    parser.add_argument("--mini-bs", type=int, default=None, help="Override training microbatch size.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max generation tokens.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Field-level override in key=value format (can be repeated).",
    )
    parser.add_argument("--output", type=str, default=None, help="Write JSON output to this file instead of stdout.")
    args = parser.parse_args()

    # Collect named overrides.
    overrides: dict[str, Any] = {}
    if args.tp_size is not None:
        overrides["tp_size"] = args.tp_size
    if args.n_samples is not None:
        overrides["n_samples"] = args.n_samples
    if args.mini_bs is not None:
        overrides["mini_bs"] = args.mini_bs
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = args.max_new_tokens

    # Parse --override key=value entries.
    for raw in args.override:
        try:
            key, value = parse_override(raw)
            overrides[key] = value
        except ValueError as exc:
            result = {"ok": False, "errors": [str(exc)]}
            print(json.dumps(result, indent=2))
            return 1

    # Validate.
    errors = validate_inputs(args.mode, args.gpu_count, args.context_length, args.target_batch, overrides)
    if errors:
        result: dict[str, Any] = {"ok": False, "errors": errors}
        print(json.dumps(result, indent=2))
        return 1

    # Derive recipe.
    recipe, provenance, warnings = derive_recipe(
        args.mode,
        args.gpu_count,
        args.context_length,
        args.target_batch,
        overrides,
    )

    # Build command.
    command = build_command(args.mode, recipe)

    # Build human-readable summary.
    human = build_human_readable(
        args.mode,
        args.gpu_count,
        args.context_length,
        args.target_batch,
        recipe,
        provenance,
        warnings,
    )

    # Assemble structured output.
    output: dict[str, Any] = {
        "ok": True,
        "mode": args.mode,
        "recipe": recipe,
        "provenance": provenance,
        "command": command,
        "warnings": warnings,
        "human_readable": human,
    }

    json_str = json.dumps(output, indent=2, sort_keys=False)

    if args.output:
        from pathlib import Path

        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Recipe written to {args.output}")
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
