#!/usr/bin/env python3
"""Generate a complete, editable training recipe and launch command.

Given a training mode (SFT/DPO/GSPO/GRPO/PPO), GPU count, context length, and
target batch size, this script derives a full training configuration from
AReno's ``TrainerConfig`` dataclass hierarchy defaults, validates all inputs,
and emits both structured JSON and a human-readable summary.  Each config
value is annotated with provenance explaining its derivation.

When ``--ckpt`` points to a model whose architecture can be inferred (e.g.
``Qwen/Qwen3-0.6B``), the script also estimates per-GPU memory usage for
weights, optimizer, KV-cache, and activations, and warns if the estimate
exceeds typical GPU VRAM.
"""

from __future__ import annotations

import argparse
import json
import re
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

_ROLLOUT_FIELDS: tuple[str, ...] = (
    "n_samples",
    "temperature",
    "top_k",
    "top_p",
    "greedy",
    "max_running_prompts",
)

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

# Fields always emitted in the command (required for every mode).
_COMMAND_REQUIRED_BASE: tuple[str, ...] = (
    "tp_size",
    "world_size",
    "batch_size",
    "mini_bs",
    "max_prompt_tokens",
    "max_new_tokens",
)

# Context-length split ratios per algorithm: (prompt_fraction, response_fraction).
# SFT: all context for prompt, no generation needed.
# DPO: equal split between prompt and response.
# RL: 25% for prompt (capped at 1024), 75% for generation.
_CONTEXT_SPLIT: dict[str, tuple[float, float]] = {
    "sft": (1.0, 0.0),
    "dpo": (0.5, 0.5),
    "gspo": (0.25, 0.75),
    "grpo": (0.25, 0.75),
    "ppo": (0.25, 0.75),
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
# Model architecture inference (for memory estimation)
# ---------------------------------------------------------------------------

# Known model families with approximate parameter counts (in billions).
# Used to infer model size from checkpoint name when areno is not importable.
_MODEL_SIZE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"0\.6[bB]", re.IGNORECASE), 0.6),
    (re.compile(r"1\.5[bB]", re.IGNORECASE), 1.5),
    (re.compile(r"1\.7[bB]", re.IGNORECASE), 1.7),
    (re.compile(r"4[bB]", re.IGNORECASE), 4.0),
    (re.compile(r"7[bB]", re.IGNORECASE), 7.0),
    (re.compile(r"8[bB]", re.IGNORECASE), 8.0),
    (re.compile(r"14[bB]", re.IGNORECASE), 14.0),
    (re.compile(r"32[bB]", re.IGNORECASE), 32.0),
    (re.compile(r"72[bB]", re.IGNORECASE), 72.0),
]

# Rough architecture heuristics: param_count → (num_layers, hidden_size, num_heads, num_kv_heads).
# Based on typical Qwen/Llama-style architectures.  These are coarse estimates
# sufficient for memory budgeting, not exact specs.
_ARCH_HEURISTICS: list[tuple[float, tuple[int, int, int, int]]] = [
    # (min_params_b, (num_layers, hidden, num_heads, num_kv_heads))
    (0.0, (28, 1024, 16, 8)),  # ~0.6B
    (1.0, (28, 2048, 16, 8)),  # ~1.5B
    (3.0, (36, 2560, 20, 8)),  # ~4B
    (6.0, (32, 3584, 28, 4)),  # ~7B
    (12.0, (40, 5120, 40, 8)),  # ~14B
    (24.0, (64, 5120, 40, 8)),  # ~32B
    (60.0, (80, 8192, 64, 8)),  # ~72B
]

# Typical VRAM per GPU type (in bytes).
_GPU_VRAM: dict[str, int] = {
    "T4": 16 * 1024**3,
    "V100": 16 * 1024**3,
    "A10": 24 * 1024**3,
    "A100": 80 * 1024**3,
    "H100": 80 * 1024**3,
    "H200": 141 * 1024**3,
}


def _infer_param_count(ckpt: str) -> float | None:
    """Infer approximate parameter count (in billions) from a checkpoint name."""

    for pattern, size in _MODEL_SIZE_PATTERNS:
        if pattern.search(ckpt):
            return size
    return None


def _infer_architecture(param_count_b: float) -> tuple[int, int, int, int]:
    """Return (num_layers, hidden_size, num_heads, num_kv_heads) for a param count."""

    for min_params, arch in _ARCH_HEURISTICS:
        if param_count_b <= min_params + 0.5:
            return arch
    return _ARCH_HEURISTICS[-1][1]


def estimate_memory(
    param_count_b: float,
    tp_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    batch_size: int,
    n_samples: int,
    mini_bs: int,
    max_new_tokens: int,
    context_length: int,
    activation_checkpointing: bool,
    adam_8bit: bool,
    mode: str,
    gpu_type: str | None = None,
) -> dict[str, Any]:
    """Estimate per-GPU memory usage for the four major components.

    Returns a dict with bytes for weights, optimizer, KV-cache, activations,
    total, and (if gpu_type is known) headroom and an OOM warning flag.
    """

    dtype_bytes = 2  # bf16/fp16
    param_count = int(param_count_b * 1e9)

    # --- Weights: TP-sharded ---
    weights_bytes = param_count * dtype_bytes // tp_size

    # --- Optimizer: fp32 Adam states (2 moments + grad) ---
    # 8-bit Adam uses 6 bytes/param, standard Adam uses 12 bytes/param.
    opt_bytes_per_param = 6 if adam_8bit else 12
    optimizer_bytes = param_count * (dtype_bytes + opt_bytes_per_param) // tp_size

    # --- KV-cache: per-GPU ---
    local_kv_heads = max(num_kv_heads // tp_size, 1)
    head_dim = hidden_size // num_heads
    # For RL: running sequences = batch_size * n_samples; for offline: batch_size.
    max_running_seqs = batch_size * n_samples if mode in ROLLOUT_MODES else batch_size
    # Each sequence may use up to max_new_tokens tokens of KV cache (coarse).
    cache_tokens = max(max_new_tokens, context_length)
    kv_cache_bytes = num_layers * 2 * max_running_seqs * cache_tokens * local_kv_heads * head_dim * dtype_bytes

    # --- Activations: per-GPU, coarse estimate ---
    # ~34 * hidden * seq * mini_bs * dtype_bytes, reduced 70% with checkpointing.
    act_bytes = 34 * hidden_size * context_length * mini_bs * dtype_bytes
    if activation_checkpointing:
        act_bytes = int(act_bytes * 0.3)

    total = weights_bytes + optimizer_bytes + kv_cache_bytes + act_bytes

    result: dict[str, Any] = {
        "weights_bytes": weights_bytes,
        "optimizer_bytes": optimizer_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "activations_bytes": act_bytes,
        "total_estimated_bytes": total,
        "param_count": param_count,
        "tp_size": tp_size,
    }

    # Add headroom if GPU type is known.
    if gpu_type and gpu_type in _GPU_VRAM:
        vram = _GPU_VRAM[gpu_type]
        result["gpu_type"] = gpu_type
        result["per_gpu_vram_bytes"] = vram
        result["headroom_bytes"] = vram - total
        result["headroom_ok"] = vram > total
        if vram <= total:
            result["oom_warning"] = (
                f"Estimated memory ({total / 1024**3:.1f} GB) exceeds {gpu_type} VRAM "
                f"({vram / 1024**3:.0f} GB). Consider reducing batch_size, mini_bs, "
                f"or context_length, or using --adam-8bit."
            )

    return result


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


def _derive_topo(gpu_count: int, overrides: dict[str, Any]) -> tuple[int, str]:
    """Derive tp_size from GPU count.  Returns (tp_size, provenance)."""

    if "tp_size" in overrides:
        return int(overrides["tp_size"]), "user override"
    if gpu_count <= 2:
        return 1, f"set to 1 because gpu_count ({gpu_count}) <= 2 (small-GPU friendly)"
    tp = min(4, gpu_count)
    return tp, f"set to min(4, gpu_count) = {tp} (TrainerConfig default capped to GPU count)"


def _derive_mini_bs(target_batch: int, gpu_count: int, overrides: dict[str, Any]) -> tuple[int, str]:
    """Derive mini_bs from target batch and GPU count."""

    if "mini_bs" in overrides:
        return int(overrides["mini_bs"]), "user override"
    if gpu_count <= 2:
        cap = min(target_batch, 4)
        return cap, f"capped to min(target_batch, 4) = {cap} for small GPU count (<=2) to avoid OOM"
    cap = min(target_batch, 16)
    return cap, f"set to min(target_batch, 16) = {cap} (TrainerConfig default capped to batch)"


def _derive_score_micro_bs(gpu_count: int, overrides: dict[str, Any]) -> tuple[int, str]:
    """Derive score_micro_bs from GPU count."""

    if "score_micro_bs" in overrides:
        return int(overrides["score_micro_bs"]), "user override"
    if gpu_count <= 2:
        return 4, "reduced to 4 for small GPU count (<=2)"
    return 8, "TrainerConfig default (8)"


def _derive_context_split(mode: str, context_length: int) -> tuple[int, int]:
    """Split context_length into (max_prompt_tokens, max_new_tokens) per algorithm.

    SFT: all context for prompt, no generation.
    DPO: equal split.
    RL (gspo/grpo/ppo): 25% for prompt (capped at 1024), 75% for generation.
    """

    prompt_frac, response_frac = _CONTEXT_SPLIT.get(mode, (0.5, 0.5))
    if mode == "sft":
        # SFT: full context for prompt, no generation tokens.
        return context_length, 0
    max_prompt = min(int(context_length * prompt_frac), 1024)
    max_new = min(int(context_length * response_frac), 3071)
    if max_new <= 0:
        max_new = context_length - max_prompt
    return max_prompt, max_new


# Fields that emit a warning when unset (placeholder required).
_REQUIRED_PLACEHOLDERS: dict[str, tuple[str, str]] = {
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
    tp_size, tp_prov = _derive_topo(gpu_count, overrides)
    mini_bs, mini_prov = _derive_mini_bs(target_batch, gpu_count, overrides)
    score_micro_bs, smb_prov = _derive_score_micro_bs(gpu_count, overrides)
    world_size = gpu_count

    # Pre-derive context-length split (algorithm-aware).
    max_prompt, max_new = _derive_context_split(mode, context_length)

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
            provenance[field_name] = tp_prov
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
            provenance[field_name] = mini_prov
            continue

        if field_name == "score_micro_bs":
            recipe[field_name] = score_micro_bs
            provenance[field_name] = smb_prov
            continue

        # --- Context-length fields (algorithm-aware split) ---
        if field_name == "max_prompt_tokens":
            if "max_prompt_tokens" in overrides:
                max_prompt = int(overrides["max_prompt_tokens"])
                recipe[field_name] = max_prompt
                provenance[field_name] = "user override"
            else:
                recipe[field_name] = max_prompt
                prompt_frac = _CONTEXT_SPLIT[mode][0]
                provenance[field_name] = (
                    f"set to min(context_length*{prompt_frac:.0%}, 1024) = {max_prompt} "
                    f"(algorithm-aware split for {mode})"
                )
            continue

        if field_name == "max_new_tokens":
            if "max_new_tokens" in overrides:
                max_new = int(overrides["max_new_tokens"])
                recipe[field_name] = max_new
                provenance[field_name] = "user override"
            else:
                recipe[field_name] = max_new
                response_frac = _CONTEXT_SPLIT[mode][1]
                if mode == "sft":
                    provenance[field_name] = "set to 0 (SFT does not generate tokens)"
                else:
                    provenance[field_name] = (
                        f"set to min(context_length*{response_frac:.0%}, 3071) = {max_new} "
                        f"(algorithm-aware split for {mode})"
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
# Command builder (concise: only required fields + explicit overrides)
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Format a recipe value for CLI output."""

    if isinstance(value, float):
        if abs(value) < 1e-3 and value != 0:
            return repr(value)
        return str(value)
    return str(value)


def build_command(
    mode: str,
    recipe: dict[str, Any],
    provenance: dict[str, str],
    overrides: dict[str, Any],
) -> str:
    """Build a concise ``areno train`` command.

    Only emits:
    1. Mandatory fields (algo, ckpt, dataset-path, tp_size, world_size, etc.)
    2. Algorithm-specific required fields (reward_fn_path for RL, ref_ckpt for DPO/PPO, etc.)
    3. Fields explicitly overridden by the user (provenance == 'user override')

    Non-required defaults (epochs, lr, weight_decay, etc.) are omitted to keep
    the command short.  They remain in the full recipe JSON for reference.
    """

    parts: list[str] = ["areno", "train"]

    # --- Always-required flags ---
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

    # --- Required recipe fields (topology, batch, context) ---
    for field_name in _COMMAND_REQUIRED_BASE:
        value = recipe.get(field_name)
        if value is None:
            continue
        parts.extend([_cli_flag(field_name), _format_value(value)])

    # RL-specific required fields.
    if mode in ROLLOUT_MODES:
        n_samples = recipe.get("n_samples")
        if n_samples is not None:
            parts.extend(["--n-samples", str(n_samples)])

    # Algorithm-specific clip/loss fields (always emitted, they define the algorithm behavior).
    if mode == "gspo" and "gspo_clip_eps" in recipe and recipe["gspo_clip_eps"] is not None:
        parts.extend(["--gspo-clip-eps", _format_value(recipe["gspo_clip_eps"])])
    if mode == "grpo" and "grpo_clip_eps" in recipe and recipe["grpo_clip_eps"] is not None:
        parts.extend(["--grpo-clip-eps", _format_value(recipe["grpo_clip_eps"])])
    if mode == "dpo" and "dpo_beta" in recipe and recipe["dpo_beta"] is not None:
        parts.extend(["--dpo-beta", _format_value(recipe["dpo_beta"])])
    if mode == "ppo":
        for field in ("clip_eps", "critic_lr", "critic_warmup_steps", "kl_loss_coef", "gamma", "lam"):
            if field in recipe and recipe[field] is not None:
                parts.extend([_cli_flag(field), _format_value(recipe[field])])

    # --- User overrides (fields explicitly set by --override or named flags) ---
    already_emitted = set(_COMMAND_REQUIRED_BASE)
    if mode in ROLLOUT_MODES:
        already_emitted.add("n_samples")
    if mode == "gspo":
        already_emitted.add("gspo_clip_eps")
    if mode == "grpo":
        already_emitted.add("grpo_clip_eps")
    if mode == "dpo":
        already_emitted.add("dpo_beta")
    if mode == "ppo":
        already_emitted.update({"clip_eps", "critic_lr", "critic_warmup_steps", "kl_loss_coef", "gamma", "lam"})

    for field_name, value in recipe.items():
        if field_name in already_emitted:
            continue  # Already emitted above.

        # Only emit if this was a user override.
        prov = provenance.get(field_name, "")
        if prov != "user override":
            continue
        if value is None:
            continue

        # Handle boolean flags.
        if field_name in _NEGATED_FLAG_BOOLS:
            if value is False:
                parts.append(_NEGATED_FLAG_BOOLS[field_name])
            continue
        if field_name in _IS_FLAG_BOOLS:
            if value is True:
                parts.append(_IS_FLAG_BOOLS[field_name])
            continue
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
    memory: dict[str, Any] | None,
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

    # Memory estimate section.
    if memory:
        lines.append("-" * 40)
        lines.append("Memory Estimate (per GPU)")
        lines.append("-" * 40)
        lines.append(f"  weights:      {memory['weights_bytes'] / 1024**3:.2f} GB")
        lines.append(f"  optimizer:    {memory['optimizer_bytes'] / 1024**3:.2f} GB")
        lines.append(f"  kv_cache:     {memory['kv_cache_bytes'] / 1024**3:.2f} GB")
        lines.append(f"  activations:  {memory['activations_bytes'] / 1024**3:.2f} GB")
        lines.append(f"  total:        {memory['total_estimated_bytes'] / 1024**3:.2f} GB")
        if "per_gpu_vram_bytes" in memory:
            vram = memory["per_gpu_vram_bytes"]
            headroom = memory.get("headroom_bytes", 0)
            lines.append(f"  GPU VRAM:     {vram / 1024**3:.0f} GB ({memory.get('gpu_type', '?')})")
            lines.append(
                f"  headroom:     {headroom / 1024**3:.2f} GB ({'OK' if headroom > 0 else 'WARNING: OOM risk'})"
            )
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
    parser.add_argument("--ckpt", type=str, default=None, help="Model checkpoint path (enables memory estimation).")
    parser.add_argument("--gpu-type", type=str, default=None, help="GPU type for VRAM check (T4, A100, H100, etc.).")
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

    # If --ckpt is provided, override the placeholder and estimate memory.
    ckpt = args.ckpt or recipe.get("ckpt", "<ckpt>")
    if args.ckpt:
        recipe["ckpt"] = args.ckpt
        provenance["ckpt"] = "user-specified checkpoint"
        # Remove the placeholder warning if present.
        warnings = [w for w in warnings if not w.startswith("ckpt is a placeholder")]

    command = build_command(args.mode, recipe, provenance, overrides)

    # Estimate memory if model size can be inferred from ckpt.
    memory: dict[str, Any] | None = None
    param_count = _infer_param_count(ckpt) if ckpt != "<ckpt>" else None
    if param_count is not None:
        num_layers, hidden_size, num_heads, num_kv_heads = _infer_architecture(param_count)
        tp_size = recipe.get("tp_size", 1)
        batch_size = recipe.get("batch_size", 1)
        n_samples = recipe.get("n_samples", 8) if args.mode in ROLLOUT_MODES else 1
        mini_bs = recipe.get("mini_bs", 4)
        max_new = recipe.get("max_new_tokens", 3071)
        act_ckpt = recipe.get("activation_checkpointing", True)
        adam_8bit = recipe.get("adam_8bit", False)
        memory = estimate_memory(
            param_count,
            tp_size,
            num_layers,
            hidden_size,
            num_heads,
            num_kv_heads,
            batch_size,
            n_samples,
            mini_bs,
            max_new,
            args.context_length,
            act_ckpt,
            adam_8bit,
            args.mode,
            args.gpu_type,
        )
        # Add OOM warning to warnings list.
        if memory.get("oom_warning"):
            warnings.append(memory["oom_warning"])

    human = build_human_readable(
        args.mode,
        args.gpu_count,
        args.context_length,
        args.target_batch,
        recipe,
        provenance,
        warnings,
        command,
        memory,
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

    if memory:
        output["memory"] = memory

    json_str = json.dumps(output, indent=2, sort_keys=False)

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Recipe written to {args.output}")
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
