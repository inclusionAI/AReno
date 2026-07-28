#!/usr/bin/env python3
"""Generate capacity-tuning recommendations without starting a training run.

Reads peak memory, throughput, and resolved training parameters, then emits
conservative, balanced, and throughput-oriented override sets with explanations.
Each recommendation is validated against AReno's ``RolloutTrainerConfig``
contract so the output can be applied directly.

Usage::

    # With measured profile data
    python .agents/skills/areno-tune-capacity/scripts/recommend_capacity.py \
      --tp-size 4 --world-size 8 --batch-size 32 --n-samples 8 --mini-bs 16 \
      --peak-mem-frac 0.82 --throughput-tps 1200.0 --json

    # Without profile data (fallback estimation)
    python .agents/skills/areno-tune-capacity/scripts/recommend_capacity.py \
      --tp-size 4 --world-size 8 --batch-size 32 --n-samples 8 --mini-bs 16 \
      --gpu-memory-gb 80 --model-params-billions 7.0 --output-dir /tmp/areno-overrides
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers (mirror auto_tune.py power-of-two logic without importing it)
# ---------------------------------------------------------------------------


def _floor_power_of_two(value: int) -> int:
    """Largest power of two <= ``value`` (minimum 1)."""
    v = max(int(value), 1)
    power = 1
    while power * 2 <= v:
        power *= 2
    return power


def _ceil_power_of_two(value: int) -> int:
    """Smallest power of two >= ``value`` (minimum 1)."""
    v = max(int(value), 1)
    power = 1
    while power < v:
        power *= 2
    return power


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecommenderInput:
    """Resolved input parameters for the recommender."""

    tp_size: int
    world_size: int
    batch_size: int
    n_samples: int
    mini_bs: int
    max_running_prompts: int  # always resolved (never None)
    adam_8bit: bool
    activation_checkpointing: bool
    keep_rollout_state: bool
    max_new_tokens: int  # read-only semantic parameter
    max_context_len: int | None  # read-only semantic parameter
    max_prompt_tokens: int  # read-only semantic parameter


@dataclass(frozen=True)
class ProfileData:
    """Measured or estimated memory/throughput profile."""

    peak_mem_frac: float
    throughput_tps: float | None
    source: str  # "measured", "estimated", or "default"

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_mem_frac": self.peak_mem_frac,
            "throughput_tps": self.throughput_tps,
            "source": self.source,
        }


@dataclass
class Recommendation:
    """A single capacity recommendation."""

    mode: str
    overrides: dict[str, Any]
    explanation: str
    estimated_mem_frac: float
    validation: dict[str, Any] = field(default_factory=lambda: {"ok": True, "errors": []})


# ---------------------------------------------------------------------------
# Recommendation algorithms
# ---------------------------------------------------------------------------


def _conervative_overrides(inp: RecommenderInput, profile: ProfileData) -> Recommendation:
    """Conservative: memory safety first. Halve concurrency and microbatch."""
    new_prompts = _floor_power_of_two(inp.max_running_prompts // 2)
    new_mini_bs = _floor_power_of_two(inp.mini_bs // 2)
    target_mem = min(0.7, profile.peak_mem_frac * 0.75) if profile.peak_mem_frac > 0 else 0.6

    overrides = {
        "max_running_prompts": new_prompts,
        "mini_bs": new_mini_bs,
        "activation_checkpointing": True,
        "keep_rollout_state": False,
        "adam_8bit": True,
    }

    explanation = (
        f"Reduces rollout concurrency by 50% ({inp.max_running_prompts} -> {new_prompts}) "
        f"and training microbatch by 50% ({inp.mini_bs} -> {new_mini_bs}) per ops_knowledge "
        f"OOM triage rules. Drops rollout state to free GPU memory between phases and "
        f"enables 8-bit Adam to halve optimizer state memory. "
        f"Target peak memory fraction: {target_mem:.2f}."
    )

    return Recommendation(
        mode="conservative",
        overrides=overrides,
        explanation=explanation,
        estimated_mem_frac=round(target_mem, 2),
    )


def _balanced_overrides(inp: RecommenderInput, profile: ProfileData) -> Recommendation:
    """Balanced: trade off memory headroom against throughput."""
    new_prompts = _floor_power_of_two(int(inp.max_running_prompts * 0.75)) if inp.max_running_prompts > 1 else 1
    new_mini_bs = inp.mini_bs
    if profile.peak_mem_frac > 0.7:
        new_mini_bs = _floor_power_of_two(int(inp.mini_bs * 0.75)) if inp.mini_bs > 1 else 1
    target_mem = min(0.85, profile.peak_mem_frac * 0.9) if profile.peak_mem_frac > 0 else 0.75

    overrides = {
        "max_running_prompts": new_prompts,
        "mini_bs": new_mini_bs,
        "activation_checkpointing": True,
        "keep_rollout_state": False,
        "adam_8bit": inp.adam_8bit,
    }

    explanation = (
        f"Reduces rollout concurrency by 25% ({inp.max_running_prompts} -> {new_prompts}) "
        f"and keeps microbatch at {new_mini_bs}. Drops rollout state to reclaim "
        f"inter-phase memory. Activates checkpointing to bound peak memory. "
        f"Target peak memory fraction: {target_mem:.2f}."
    )

    return Recommendation(
        mode="balanced",
        overrides=overrides,
        explanation=explanation,
        estimated_mem_frac=round(target_mem, 2),
    )


def _throughput_overrides(inp: RecommenderInput, profile: ProfileData) -> Recommendation:
    """Throughput: maximize utilization when memory headroom exists."""
    full_demand = inp.batch_size * inp.n_samples
    new_prompts = min(full_demand, max(inp.max_running_prompts, _ceil_power_of_two(inp.max_running_prompts)))
    new_mini_bs = min(max(full_demand, 1), max(inp.mini_bs, _ceil_power_of_two(inp.mini_bs)))

    headroom = 1.0 - (profile.peak_mem_frac if profile.peak_mem_frac > 0 else 0.5)
    disable_checkpointing = headroom > 0.3 and profile.peak_mem_frac < 0.7

    target_mem = profile.peak_mem_frac if profile.peak_mem_frac > 0 else 0.9

    overrides = {
        "max_running_prompts": new_prompts,
        "mini_bs": new_mini_bs,
        "activation_checkpointing": not disable_checkpointing,
        "keep_rollout_state": True,
        "adam_8bit": False,
    }

    checkpoint_note = "disabled to reduce recomputation overhead" if disable_checkpointing else "kept enabled"
    explanation = (
        f"Increases rollout concurrency to {new_prompts} and microbatch to {new_mini_bs} "
        f"to utilize available GPU headroom ({headroom:.0%} free). Activation checkpointing "
        f"is {checkpoint_note}. Rollout state is kept resident to reduce step latency. "
        f"Target peak memory fraction: {target_mem:.2f}."
    )

    return Recommendation(
        mode="throughput",
        overrides=overrides,
        explanation=explanation,
        estimated_mem_frac=round(target_mem, 2),
    )


# ---------------------------------------------------------------------------
# Profile estimation (fallback when no measured data)
# ---------------------------------------------------------------------------

# Bytes-per-parameter for model weights (FP16/BF16)
_WEIGHT_BYTES = 2.0
# Bytes-per-parameter for Adam optimizer states (FP32 momentum + variance)
_ADAM_STATE_BYTES_32BIT = 8.0
# Bytes-per-parameter for 8-bit Adam optimizer states
_ADAM_STATE_BYTES_8BIT = 2.0


def _estimate_profile(
    inp: RecommenderInput,
    gpu_memory_gb: float | None,
    model_params_billions: float | None,
) -> ProfileData:
    """Estimate peak memory fraction from GPU capacity and model size."""

    if gpu_memory_gb is not None and model_params_billions is not None:
        params = float(model_params_billions) * 1e9
        weights_bytes = params * _WEIGHT_BYTES / inp.tp_size
        optimizer_bytes = params * (_ADAM_STATE_BYTES_8BIT if inp.adam_8bit else _ADAM_STATE_BYTES_32BIT) / inp.tp_size
        base_bytes = weights_bytes + optimizer_bytes
        base_gb = base_bytes / 1e9
        estimated_frac = min(base_gb / float(gpu_memory_gb), 0.95)
        return ProfileData(
            peak_mem_frac=round(estimated_frac, 4),
            throughput_tps=None,
            source="estimated",
        )

    # No GPU info either: use safe defaults
    return ProfileData(
        peak_mem_frac=0.6,
        throughput_tps=None,
        source="default",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_recommendation(rec: Recommendation, inp: RecommenderInput) -> Recommendation:
    """Validate a recommendation against AReno config constraints.

    Performs:
    1. Static parameter checks (positive values, world_size % tp_size == 0).
    2. RolloutTrainerConfig construction (lazy import, CPU-safe).
    """

    errors: list[str] = []

    # --- static checks ---
    ov = rec.overrides
    for key in ("max_running_prompts", "mini_bs"):
        val = ov.get(key, 0)
        if not isinstance(val, int) or val <= 0:
            errors.append(f"{key} must be a positive integer, got {val!r}")

    if inp.world_size % inp.tp_size:
        errors.append(f"world_size ({inp.world_size}) must be divisible by tp_size ({inp.tp_size})")

    rollout_demand = inp.batch_size * inp.n_samples
    if rollout_demand <= 0:
        errors.append(f"rollout demand (batch_size * n_samples) must be positive, got {rollout_demand}")

    # --- config construction check (skipped if areno is not installed) ---
    try:
        from areno.api.trainer_config import RolloutTrainerConfig

        RolloutTrainerConfig(
            algo="gspo",
            ckpt="/dummy",
            dataset_path="/dummy",
            tp_size=inp.tp_size,
            world_size=inp.world_size,
            batch_size=inp.batch_size,
            mini_bs=ov.get("mini_bs", inp.mini_bs),
            n_samples=inp.n_samples,
            max_running_prompts=ov.get("max_running_prompts", inp.max_running_prompts),
            adam_8bit=ov.get("adam_8bit", inp.adam_8bit),
            activation_checkpointing=ov.get("activation_checkpointing", inp.activation_checkpointing),
            keep_rollout_state=ov.get("keep_rollout_state", inp.keep_rollout_state),
        )
    except ModuleNotFoundError:
        rec.validation = {"ok": not errors, "errors": errors, "config_check": "skipped (areno not installed)"}
        return rec
    except ValueError as exc:
        errors.append(f"config validation failed: {exc}")
    except Exception as exc:  # noqa: BLE001 - construction failure
        errors.append(f"config construction error: {type(exc).__name__}: {exc}")

    rec.validation = {"ok": not errors, "errors": errors}
    return rec


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def generate_recommendations(
    inp: RecommenderInput,
    profile: ProfileData,
) -> dict[str, Recommendation]:
    """Generate all three recommendation modes and validate each."""

    raw = {
        "conservative": _conervative_overrides(inp, profile),
        "balanced": _balanced_overrides(inp, profile),
        "throughput": _throughput_overrides(inp, profile),
    }
    validated: dict[str, Recommendation] = {}
    for mode, rec in raw.items():
        validated[mode] = _validate_recommendation(rec, inp)
    return validated


def _build_input_from_args(args: argparse.Namespace) -> RecommenderInput:
    """Resolve argparse namespace into a ``RecommenderInput``."""

    max_running = args.max_running_prompts
    if max_running is None:
        max_running = max(args.batch_size * args.n_samples, 1)

    return RecommenderInput(
        tp_size=args.tp_size,
        world_size=args.world_size,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        mini_bs=args.mini_bs,
        max_running_prompts=max_running,
        adam_8bit=args.adam_8bit,
        activation_checkpointing=not args.no_activation_checkpointing,
        keep_rollout_state=not args.drop_rollout_state,
        max_new_tokens=args.max_new_tokens,
        max_context_len=args.max_context_len,
        max_prompt_tokens=args.max_prompt_tokens,
    )


def _validate_args(args: argparse.Namespace) -> list[str]:
    """Pre-flight argument validation."""

    errors: list[str] = []
    for name in ("tp_size", "world_size", "batch_size", "n_samples", "mini_bs", "max_new_tokens", "max_prompt_tokens"):
        val = getattr(args, name, 0)
        if val <= 0:
            errors.append(f"--{name.replace('_', '-')} must be positive, got {val}")
    if args.world_size % args.tp_size:
        errors.append(f"--world-size ({args.world_size}) must be divisible by --tp-size ({args.tp_size})")
    if not 0 < args.mem_frac <= 0.9:
        errors.append(f"--mem-frac must be in (0, 0.9], got {args.mem_frac}")
    if args.max_running_prompts is not None and args.max_running_prompts <= 0:
        errors.append(f"--max-running-prompts must be positive, got {args.max_running_prompts}")
    if args.gpu_memory_gb is not None and args.gpu_memory_gb <= 0:
        errors.append(f"--gpu-memory-gb must be positive, got {args.gpu_memory_gb}")
    if args.model_params_billions is not None and args.model_params_billions <= 0:
        errors.append(f"--model-params-billions must be positive, got {args.model_params_billions}")
    return errors


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _format_human_readable(
    inp: RecommenderInput,
    profile: ProfileData,
    recs: dict[str, Recommendation],
) -> str:
    """Produce a human-readable text report."""

    lines: list[str] = [
        "AReno Capacity Recommendations",
        "=" * 50,
        "",
        "Input:",
        f"  tp_size={inp.tp_size}, world_size={inp.world_size}, batch_size={inp.batch_size},"
        f" n_samples={inp.n_samples}, mini_bs={inp.mini_bs}",
        f"  max_running_prompts={inp.max_running_prompts}",
        f"  adam_8bit={inp.adam_8bit}, activation_checkpointing={inp.activation_checkpointing},"
        f" keep_rollout_state={inp.keep_rollout_state}",
        "",
        f"Profile: peak_mem_frac={profile.peak_mem_frac:.4f},"
        f" throughput_tps={profile.throughput_tps}, source={profile.source}",
        "",
    ]

    for mode in ("conservative", "balanced", "throughput"):
        rec = recs[mode]
        title = mode.capitalize()
        lines.append(f"--- {title} ---")
        for key, val in rec.overrides.items():
            lines.append(f"  {key}: {val}")
        val_status = "OK" if rec.validation["ok"] else f"FAILED: {rec.validation['errors']}"
        lines.append(f"  Validation: {val_status}")
        lines.append(f"  Estimated mem_frac: {rec.estimated_mem_frac}")
        lines.append(f"  Rationale: {rec.explanation}")
        lines.append("")

    lines.append("Each recommendation has been validated against AReno config constraints.")
    lines.append("Override files are written only when --output-dir is provided.")
    return "\n".join(lines)


def _format_json(
    inp: RecommenderInput,
    profile: ProfileData,
    recs: dict[str, Recommendation],
) -> str:
    """Produce a structured JSON report."""

    result: dict[str, Any] = {
        "ok": all(r.validation["ok"] for r in recs.values()),
        "input": {
            "tp_size": inp.tp_size,
            "world_size": inp.world_size,
            "batch_size": inp.batch_size,
            "n_samples": inp.n_samples,
            "mini_bs": inp.mini_bs,
            "max_running_prompts": inp.max_running_prompts,
            "adam_8bit": inp.adam_8bit,
            "activation_checkpointing": inp.activation_checkpointing,
            "keep_rollout_state": inp.keep_rollout_state,
            "max_new_tokens": inp.max_new_tokens,
            "max_context_len": inp.max_context_len,
            "max_prompt_tokens": inp.max_prompt_tokens,
        },
        "profile": profile.as_dict(),
        "recommendations": {
            mode: {
                "overrides": rec.overrides,
                "explanation": rec.explanation,
                "estimated_mem_frac": rec.estimated_mem_frac,
                "validation": rec.validation,
            }
            for mode, rec in recs.items()
        },
    }
    return json.dumps(result, indent=2, sort_keys=True)


def _write_override_files(
    recs: dict[str, Recommendation],
    output_dir: Path,
) -> list[str]:
    """Write one JSON override file per recommendation mode."""

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for mode, rec in recs.items():
        path = output_dir / f"capacity_override_{mode}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec.overrides, f, indent=2, sort_keys=True)
        written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (separated for testability)."""

    parser = argparse.ArgumentParser(
        description="Generate capacity-tuning recommendations without starting a training run.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Current configuration
    g_cfg = parser.add_argument_group("current configuration")
    g_cfg.add_argument("--tp-size", type=int, default=4, help="Tensor-parallel size (default: 4)")
    g_cfg.add_argument("--world-size", type=int, default=8, help="Total GPU count (default: 8)")
    g_cfg.add_argument("--batch-size", type=int, default=32, help="Prompts per step (default: 32)")
    g_cfg.add_argument("--n-samples", type=int, default=8, help="Rollout samples per prompt (default: 8)")
    g_cfg.add_argument("--mini-bs", type=int, default=16, help="Training microbatch size (default: 16)")
    g_cfg.add_argument("--max-running-prompts", type=int, default=None, help="Rollout concurrency (default: batch_size*n_samples)")
    g_cfg.add_argument("--adam-8bit", action="store_true", help="Enable 8-bit Adam optimizer states")
    g_cfg.add_argument("--no-activation-checkpointing", action="store_true", help="Disable activation checkpointing")
    g_cfg.add_argument("--drop-rollout-state", action="store_true", help="Drop rollout state between phases to save memory")

    # Read-only semantic parameters (never modified by recommendations)
    g_sem = parser.add_argument_group("semantic parameters (read-only, never adjusted)")
    g_sem.add_argument("--max-new-tokens", type=int, default=3071, help="Max generation tokens (read-only)")
    g_sem.add_argument("--max-context-len", type=int, default=None, help="Max context length (read-only)")
    g_sem.add_argument("--max-prompt-tokens", type=int, default=1024, help="Max prompt tokens (read-only)")

    # Profile data (optional, measured)
    g_prof = parser.add_argument_group("profile data (measured)")
    g_prof.add_argument("--peak-mem-frac", type=float, default=None, help="Measured peak GPU memory fraction (0-1)")
    g_prof.add_argument("--throughput-tps", type=float, default=None, help="Measured throughput in tokens/s")

    # Fallback inputs (when no profile available)
    g_fb = parser.add_argument_group("fallback inputs (when no profile data)")
    g_fb.add_argument("--gpu-memory-gb", type=float, default=None, help="Per-GPU memory in GB (e.g. 80 for H100)")
    g_fb.add_argument("--model-params-billions", type=float, default=None, help="Model parameter count in billions (e.g. 7.0)")

    # Output control
    g_out = parser.add_argument_group("output control")
    g_out.add_argument("--json", action="store_true", help="Output structured JSON instead of human-readable text")
    g_out.add_argument("--output-dir", type=Path, default=None, help="Write override JSON files to this directory")
    g_out.add_argument("--mem-frac", type=float, default=0.9, help="Safety target fraction upper bound (default: 0.9)")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point — returns exit code (0 success, 1 validation error)."""

    parser = build_parser()
    args = parser.parse_args(argv)

    # Pre-flight validation
    arg_errors = _validate_args(args)
    if arg_errors:
        for err in arg_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    # Build input
    inp = _build_input_from_args(args)

    # Resolve profile
    if args.peak_mem_frac is not None:
        profile = ProfileData(
            peak_mem_frac=args.peak_mem_frac,
            throughput_tps=args.throughput_tps,
            source="measured",
        )
    else:
        profile = _estimate_profile(inp, args.gpu_memory_gb, args.model_params_billions)

    # Generate recommendations
    recs = generate_recommendations(inp, profile)

    # Write override files if requested
    override_paths: list[str] = []
    if args.output_dir is not None:
        override_paths = _write_override_files(recs, args.output_dir)

    # Emit output
    if args.json:
        output = _format_json(inp, profile, recs)
        print(output)
    else:
        output = _format_human_readable(inp, profile, recs)
        print(output)
        if override_paths:
            print(f"\nOverride files written to:")
            for p in override_paths:
                print(f"  {p}")

    # Exit code: 0 if all recommendations valid, 1 if any failed validation
    return 0 if all(r.validation["ok"] for r in recs.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())