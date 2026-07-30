#!/usr/bin/env python3
"""Generate a complete, editable training recipe and launch command.

Given a training mode (SFT/DPO/GSPO/GRPO/PPO), GPU count, context length, and
target batch size, this script derives a full training configuration from
AReno's ``TrainerConfig`` dataclass hierarchy defaults, validates all inputs,
and emits both structured JSON and a human-readable summary.  Each config
value is annotated with provenance explaining its derivation.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants — mirrors of AReno's public contracts (no runtime import required)
# ---------------------------------------------------------------------------

MODES = ("sft", "dpo", "gspo", "grpo", "ppo")

# Modes that require rollout (rollout-capable trainers).
ROLLOUT_MODES = {"gspo", "grpo", "ppo"}

# Fields whose CLI flag name differs from the ``_`` → ``-`` convention.
# These map dataclass field names to the actual ``--flag`` used by
# ``areno/cli/train.py``.
_SPECIAL_CLI_FLAGS: dict[str, str] = {
    "optimizer_lr": "--lr",
    "optimizer_min_lr": "--min-lr",
    "optimizer_beta1": "--adam-beta1",
    "optimizer_beta2": "--adam-beta2",
}

# ``is_flag`` booleans: emit the flag only when the value is True.
_IS_FLAG_BOOLS: dict[str, str] = {
    "eager_decode": "--eager-decode",
    "greedy": "--greedy",
    "adam_8bit": "--adam-8bit",
    "train_tool_results": "--train-tool-results",
    "disable_thinking": "--disable-thinking",
}

# Negated booleans: emit the flag only when the value is False.
_NEGATED_FLAG_BOOLS: dict[str, str] = {
    "activation_checkpointing": "--no-activation-checkpointing",
    "keep_rollout_state": "--drop-rollout-state",
}


def _cli_flag(field_name: str) -> str:
    """Return the CLI flag string for a dataclass field name."""

    if field_name in _SPECIAL_CLI_FLAGS:
        return _SPECIAL_CLI_FLAGS[field_name]
    return "--" + field_name.replace("_", "-")


# ---------------------------------------------------------------------------
# Per-mode field sets (ordered for stable output)
# ---------------------------------------------------------------------------

# Common fields for all modes, in display order.
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

# Rollout-specific fields (gspo/grpo/ppo).
_ROLLOUT_FIELDS: tuple[str, ...] = (
    "n_samples",
    "temperature",
    "top_k",
    "top_p",
    "greedy",
    "max_running_prompts",
)

# Policy-gradient fields (gspo/grpo/ppo).
_POLICY_FIELDS: tuple[str, ...] = (
    "reward_fn_path",
    "agent_fn",
    "agent_timeout_s",
    "train_tool_results",
    "chat_template_enable_thinking",
)

_GSPO_FIELDS: tuple[str, ...] = ("gspo_clip_eps",)
_GRPO_FIELDS: tuple[str, ...] = ("grpo_clip_eps",)
_DPO_FIELDS: tuple[str, ...] = ("ref_ckpt", "dpo_beta")
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

# Fields excluded from the generated command (handled specially or not CLI-exposed).
_COMMAND_SKIP_FIELDS: set[str] = {
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
    "agent_fn": None,
    "agent_timeout_s": 300.0,
    "train_tool_results": False,
    "chat_template_enable_thinking": None,
}

_DPO_DEFAULTS: dict[str, Any] = {
    "ref_ckpt": None,
    "dpo_beta": 0.1,
}

_GSPO_DEFAULTS: dict[str, Any] = {"gspo_clip_eps": 3.0e-4}
_GRPO_DEFAULTS: dict[str, Any] = {"grpo_clip_eps": 0.2}

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

# Mode → list of default dicts to merge (later dicts override earlier).
_MODE_DEFAULT_CHAIN: dict[str, tuple[dict[str, Any], ...]] = {
    "sft": (_BASE_DEFAULTS,),
    "dpo": (_BASE_DEFAULTS, _DPO_DEFAULTS),
    "gspo": (_BASE_DEFAULTS, _ROLLOUT_DEFAULTS, _POLICY_DEFAULTS, _GSPO_DEFAULTS),
    "grpo": (_BASE_DEFAULTS, _ROLLOUT_DEFAULTS, _POLICY_DEFAULTS, _GRPO_DEFAULTS),
    "ppo": (_BASE_DEFAULTS, _ROLLOUT_DEFAULTS, _POLICY_DEFAULTS, _PPO_DEFAULTS),
}


def _defaults_for_mode(mode: str) -> dict[str, Any]:
    """Return the merged default-value dict for a given mode."""

    merged: dict[str, Any] = {}
    for defaults in _MODE_DEFAULT_CHAIN[mode]:
        merged.update(defaults)
    return merged


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

# Fields that require special handling in ``derive_recipe`` beyond a simple
# default lookup.  Each handler returns (value, provenance, optional warning).
# Handlers that set ``warning`` will have it appended to the warnings list.


def _derive_topo(gpu_count: int, overrides: dict[str, Any]) -> tuple[int, str, str | None]:
    """Derive tp_size from GPU count.  Returns (tp_size, provenance, warning)."""

    if "tp_size" in overrides:
        return int(overrides["tp_size"]), "user override", None
    if gpu_count <= 2:
        return 1, f"set to 1 because gpu_count ({gpu_count}) <= 2 (small-GPU friendly)", None
    tp = min(4, gpu_count)
    return tp, f"set to min(4, gpu_count) = {tp} (TrainerConfig default capped to GPU count)", None


def _derive_mini_bs(target_batch: int, gpu_count: int, overrides: dict[str, Any]) -> tuple[int, str, str | None]:
    """Derive mini_bs from target batch and GPU count."""

    if "mini_bs" in overrides:
        return int(overrides["mini_bs"]), "user override", None
    if gpu_count <= 2:
        cap = min(target_batch, 4)
        return cap, f"capped to min(target_batch, 4) = {cap} for small GPU count (<=2) to avoid OOM", None
    cap = min(target_batch, 16)
    return cap, f"set to min(target_batch, 16) = {cap} (TrainerConfig default capped to batch)", None


def _derive_score_micro_bs(gpu_count: int, overrides: dict[str, Any]) -> tuple[int, str, str | None]:
    """Derive score_micro_bs from GPU count."""

    if "score_micro_bs" in overrides:
        return int(overrides["score_micro_bs"]), "user override", None
    if gpu_count <= 2:
        return 4, "reduced to 4 for small GPU count (<=2)", None
    return 8, "TrainerConfig default (8)", None


# Fields that emit a warning when unset (placeholder required).
_REQUIRED_PLACEHOLDERS: dict[str, tuple[str, str]] = {
    # field_name: (provenance_text, warning_text)
    "reward_fn_path": (
        "not set; required for rollout RL — provide --reward-fn-path before training",
        "reward_fn_path is not set; provide --reward-fn-path before training",
    ),
    "ref_ckpt": (
        "not set; DPO/PPO require a reference checkpoint — provide --ref-ckpt",
        "ref_ckpt is not set; provide --ref-ckpt before training",
    ),
    "critic_ckpt": (
        "not set; PPO requires a critic checkpoint — provide --critic-ckpt",
        "critic_ckpt is not set; provide --critic-ckpt before training",
    ),
    "reward_ckpt": (
        "not set; PPO may use a reward model checkpoint — provide --reward-ckpt if needed",
        "reward_ckpt is not set; provide --reward-ckpt if using a reward model",
    ),
}


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

    # Pre-derive capacity-sensitive values.
    tp_size, _, _ = _derive_topo(gpu_count, overrides)
    mini_bs, _, _ = _derive_mini_bs(target_batch, gpu_count, overrides)
    score_micro_bs, _, _ = _derive_score_micro_bs(gpu_count, overrides)
    world_size = gpu_count

    # Pre-derive context-length split.
    max_prompt = min(context_length // 2, 1024)
    max_new = min(context_length - max_prompt, 3071)

    for field_name in fields:
        # --- Fixed values (mode, placeholders) ---
        if field_name == "algo":
            recipe[field_name] = mode
            provenance[field_name] = "user-specified mode"
            continue

        if field_name == "ckpt":
            recipe[field_name] = "<ckpt>"
            provenance[field_name] = "placeholder; user must provide model checkpoint path"
            warnings.append("ckpt is a placeholder; replace <ckpt> with your model checkpoint path or repo ID")
            continue

        if field_name == "dataset_path":
            recipe[field_name] = "<dataset-path>"
            provenance[field_name] = "placeholder; user must provide dataset path"
            warnings.append("dataset_path is a placeholder; replace <dataset-path> with your dataset path or ref")
            continue

        # --- Capacity-derived values (pre-computed above) ---
        if field_name == "tp_size":
            recipe[field_name] = tp_size
            provenance[field_name] = _derive_topo(gpu_count, overrides)[1]
            continue

        if field_name == "world_size":
            recipe[field_name] = world_size
            provenance[field_name] = "set to gpu_count (user request)"
            continue

        if field_name == "batch_size":
            recipe[field_name] = target_batch
            provenance[field_name] = "set to target_batch (user request)"
            continue

        if field_name == "mini_bs":
            recipe[field_name] = mini_bs
            provenance[field_name] = _derive_mini_bs(target_batch, gpu_count, overrides)[1]
            continue

        if field_name == "score_micro_bs":
            recipe[field_name] = score_micro_bs
            provenance[field_name] = _derive_score_micro_bs(gpu_count, overrides)[1]
            continue

        # --- Context-length fields ---
        if field_name == "max_prompt_tokens":
            if "max_prompt_tokens" in overrides:
                max_prompt = int(overrides["max_prompt_tokens"])
                recipe[field_name] = max_prompt
                provenance[field_name] = "user override"
            else:
                recipe[field_name] = max_prompt
                provenance[field_name] = (
                    f"set to min(context_length//2, 1024) = {max_prompt} "
                    "(half context for prompt, capped at TrainerConfig default)"
                )
            continue

        if field_name == "max_new_tokens":
            if "max_new_tokens" in overrides:
                max_new = int(overrides["max_new_tokens"])
                recipe[field_name] = max_new
                provenance[field_name] = "user override"
            else:
                recipe[field_name] = max_new
                provenance[field_name] = (
                    f"set to min(context_length - max_prompt_tokens, 3071) = {max_new} "
                    "(remaining context for generation, capped at TrainerConfig default)"
                )
            continue

        if field_name == "max_context_len":
            val = overrides.get("max_context_len", context_length)
            recipe[field_name] = val
            provenance[field_name] = (
                "user override"
                if "max_context_len" in overrides
                else (f"set to context_length ({context_length}) (user request)")
            )
            continue

        # --- Rollout-specific: n_samples ---
        if field_name == "n_samples":
            if "n_samples" in overrides:
                recipe[field_name] = int(overrides["n_samples"])
                provenance[field_name] = "user override"
            else:
                recipe[field_name] = 8
                provenance[field_name] = "RolloutTrainerConfig default (8)"
            continue

        # --- Dataset loader (SFT requires it) ---
        if field_name == "dataset_loader_fn":
            val = overrides.get("dataset_loader_fn")
            if mode == "sft":
                val = val or "<loader-path>"
                provenance[field_name] = "placeholder; SFT requires a dataset loader — provide --dataset-loader-fn"
                warnings.append(
                    "dataset_loader_fn is a placeholder; SFT requires --dataset-loader-fn "
                    "(e.g. examples/sft/alpaca/dataset_loader.py)"
                )
            else:
                provenance[field_name] = (
                    "user override"
                    if val
                    else "optional; set if your dataset needs normalization (see inspect_dataset.py)"
                )
            recipe[field_name] = val
            continue

        # --- Required-but-optional placeholders (reward_fn_path, ref_ckpt, etc.) ---
        if field_name in _REQUIRED_PLACEHOLDERS:
            val = overrides.get(field_name)
            recipe[field_name] = val
            if val:
                provenance[field_name] = "user override"
            else:
                prov_text, warn_text = _REQUIRED_PLACEHOLDERS[field_name]
                provenance[field_name] = prov_text
                warnings.append(warn_text)
            continue

        # --- Generic fallback: user override or dataclass default ---
        if field_name in overrides:
            recipe[field_name] = overrides[field_name]
            provenance[field_name] = "user override"
        elif field_name in defaults:
            recipe[field_name] = defaults[field_name]
            provenance[field_name] = f"TrainerConfig default ({defaults[field_name]!r})"
        else:
            recipe[field_name] = None
            provenance[field_name] = "no default available"

    # Supplementary provenance (not a recipe field).
    dp_size = world_size // tp_size if tp_size else 0
    provenance["_dp_size"] = f"world_size // tp_size = {world_size} // {tp_size} = {dp_size}"

    return recipe, provenance, warnings


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Format a recipe value for CLI output."""

    if isinstance(value, float):
        # Preserve scientific notation for small numbers (e.g. 1e-06).
        if abs(value) < 1e-3 and value != 0:
            return repr(value)
        return str(value)
    return str(value)


def build_command(mode: str, recipe: dict[str, Any]) -> str:
    """Build a directly runnable ``areno train`` command from the recipe."""

    parts: list[str] = ["areno", "train"]

    # --- Mandatory positional-equivalent flags ---
    parts.extend(["--algo", str(recipe["algo"])])
    parts.extend(["--ckpt", str(recipe["ckpt"])])
    parts.extend(["--dataset-path", str(recipe["dataset_path"])])

    # SFT requires a dataset loader.
    if mode == "sft" and recipe.get("dataset_loader_fn") is not None:
        parts.extend(["--dataset-loader-fn", str(recipe["dataset_loader_fn"])])

    # Rollout modes need reward_fn_path (emit placeholder if unset).
    if mode in ROLLOUT_MODES and "reward_fn_path" in recipe:
        rwd = recipe["reward_fn_path"]
        parts.extend(["--reward-fn-path", str(rwd) if rwd else "<reward-fn-path>"])

    # DPO and PPO need ref_ckpt.
    if mode in {"dpo", "ppo"} and "ref_ckpt" in recipe:
        ref = recipe["ref_ckpt"]
        parts.extend(["--ref-ckpt", str(ref) if ref else "<ref-ckpt>"])

    # PPO: reward_ckpt and critic_ckpt (only when explicitly set).
    if mode == "ppo":
        if recipe.get("reward_ckpt") is not None:
            parts.extend(["--reward-ckpt", str(recipe["reward_ckpt"])])
        if recipe.get("critic_ckpt") is not None:
            parts.extend(["--critic-ckpt", str(recipe["critic_ckpt"])])

    # --- Derived/configured values (skip already-emitted and None) ---
    for field_name, value in recipe.items():
        if field_name in _COMMAND_SKIP_FIELDS:
            continue
        if value is None:
            continue

        # Negated booleans: emit flag when value is False.
        if field_name in _NEGATED_FLAG_BOOLS:
            if value is False:
                parts.append(_NEGATED_FLAG_BOOLS[field_name])
            continue

        # is_flag booleans: emit flag when value is True.
        if field_name in _IS_FLAG_BOOLS:
            if value is True:
                parts.append(_IS_FLAG_BOOLS[field_name])
            continue

        # Skip remaining False booleans (no flag to emit).
        if isinstance(value, bool) and not value:
            continue

        parts.extend([_cli_flag(field_name), _format_value(value)])

    return shlex.join(parts)


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
    command: str,
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
    ws = recipe.get("world_size", 0)
    tp = recipe.get("tp_size", 1)
    lines.append(f"Topology:       world_size={ws}, tp_size={tp}, dp={ws // max(tp, 1)}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("Configuration")
    lines.append("-" * 40)
    for field_name, value in recipe.items():
        prov = provenance.get(field_name, "")
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
    lines.append(f"  {command}")
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

    # Type coercion: bool → int → float → string.
    if value_str.lower() in ("true", "false"):
        return key, value_str.lower() == "true"
    try:
        return key, int(value_str)
    except ValueError:
        pass
    try:
        return key, float(value_str)
    except ValueError:
        pass
    # Strip surrounding quotes from string values.
    if len(value_str) >= 2 and value_str[0] == value_str[-1] and value_str[0] in "\"'":
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

    # Collect named overrides from dedicated flags.
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
            print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2))
            return 1

    # Validate all inputs before derivation.
    errors = validate_inputs(args.mode, args.gpu_count, args.context_length, args.target_batch, overrides)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    # Derive recipe, build command, and assemble output.
    recipe, provenance, warnings = derive_recipe(
        args.mode,
        args.gpu_count,
        args.context_length,
        args.target_batch,
        overrides,
    )
    command = build_command(args.mode, recipe)
    human = build_human_readable(
        args.mode,
        args.gpu_count,
        args.context_length,
        args.target_batch,
        recipe,
        provenance,
        warnings,
        command,
    )

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
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Recipe written to {args.output}")
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
