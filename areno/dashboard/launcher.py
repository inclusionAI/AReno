"""Validation and preview helpers for dashboard launcher submissions."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Callable

CommandBuilder = Callable[[dict[str, Any]], list[str]]


def preview_launcher(
    kind: str,
    config: dict[str, Any],
    command_builder: CommandBuilder,
    *,
    acknowledge_warnings: bool = False,
) -> dict[str, Any]:
    """Resolve a launcher config without starting a process or loading weights."""

    normalized = dict(config)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if kind not in {"train", "serve"}:
        errors.append({"field": "kind", "message": "must be train or serve"})
    if kind == "train":
        _validate_train(normalized, errors, warnings)
    elif kind == "serve":
        _validate_serve(normalized, errors, warnings)
    try:
        command = command_builder(normalized)
    except (TypeError, ValueError) as exc:
        errors.append(
            {"field": "extra_args", "message": f"invalid shell arguments: {exc}"}
        )
        command = []
    shell_command = shlex.join(command) if not errors else ""
    return {
        "ok": not errors and (acknowledge_warnings or not warnings),
        "kind": kind,
        "resolved_args": normalized,
        "command": command,
        "shell_command": shell_command,
        "errors": errors,
        "warnings": warnings,
        "requires_acknowledgement": bool(warnings) and not acknowledge_warnings,
    }


def _validate_train(
    config: dict[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]
) -> None:
    _required(config, "ckpt", errors)
    _required(config, "dataset_path", errors)
    _validate_local_reference(config.get("ckpt"), "ckpt", errors, warnings)
    _validate_local_reference(
        config.get("dataset_path"), "dataset_path", errors, warnings
    )
    _validate_positive_fields(
        config,
        (
            "epochs",
            "max_steps",
            "world_size",
            "tp_size",
            "batch_size",
            "mini_bs",
            "score_micro_bs",
            "gradient_accumulation_steps",
            "max_prompt_tokens",
            "max_new_tokens",
            "save_interval",
            "tune_max_samples",
        ),
        errors,
    )
    _validate_positive_fields(config, ("n_samples", "max_running_prompts"), errors)
    _validate_positive_fields(
        config, ("lr", "temperature", "top_p", "agent_timeout_s"), errors
    )
    _validate_gpu_topology(config, errors)
    _validate_parallelism(config, errors)
    if config.get("model_hub", "modelscope") not in {"hf", "modelscope"}:
        errors.append({"field": "model_hub", "message": "must be hf or modelscope"})
    if config.get("algo", "sft").lower() == "sft" and not config.get(
        "dataset_loader_fn"
    ):
        errors.append({"field": "dataset_loader_fn", "message": "is required for sft"})
    if config.get("algo", "sft").lower() in {"gspo", "grpo", "ppo"} and not (
        config.get("reward_fn_path") or config.get("reward_ckpt")
    ):
        errors.append(
            {
                "field": "reward_fn_path",
                "message": "a reward function or reward checkpoint is required",
            }
        )
    if config.get("smoke_infer") and config.get("smoke_train"):
        errors.append(
            {
                "field": "smoke_train",
                "message": "smoke_infer and smoke_train are mutually exclusive",
            }
        )
    _validate_range(
        config, "top_p", errors, minimum=0, maximum=1, minimum_inclusive=False
    )
    _validate_range(
        config, "mem_frac", errors, minimum=0, maximum=1, minimum_inclusive=False
    )
    if config.get("greedy") and config.get("temperature") not in (None, "", 0, 1, 1.0):
        warnings.append(
            {"field": "temperature", "message": "greedy decoding ignores temperature"}
        )
    if not config.get("save_path") and not config.get("save_dir"):
        warnings.append(
            {"field": "save_path", "message": "no checkpoint output path is configured"}
        )


def _validate_serve(
    config: dict[str, Any], errors: list[dict[str, str]], warnings: list[dict[str, str]]
) -> None:
    _required(config, "model_path", errors)
    _validate_local_reference(config.get("model_path"), "model_path", errors, warnings)
    _validate_positive_fields(
        config,
        ("port", "world_size", "tp_size", "max_running_prompts", "default_max_tokens"),
        errors,
    )
    _validate_gpu_topology(config, errors)
    _validate_parallelism(config, errors)
    _validate_range(config, "port", errors, minimum=1, maximum=65535)
    if config.get("model_hub", "modelscope") not in {"hf", "modelscope"}:
        errors.append({"field": "model_hub", "message": "must be hf or modelscope"})
    if config.get("host") in {"", None}:
        errors.append({"field": "host", "message": "must not be empty"})


def _required(config: dict[str, Any], field: str, errors: list[dict[str, str]]) -> None:
    if config.get(field) in (None, ""):
        errors.append({"field": field, "message": "is required"})


def _validate_positive_fields(
    config: dict[str, Any], fields: tuple[str, ...], errors: list[dict[str, str]]
) -> None:
    for field in fields:
        value = config.get(field)
        if value in (None, ""):
            continue
        try:
            if float(value) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append({"field": field, "message": "must be positive"})


def _validate_gpu_topology(
    config: dict[str, Any], errors: list[dict[str, str]]
) -> None:
    devices = config.get("train_devices")
    if devices in (None, ""):
        return
    if isinstance(devices, str):
        try:
            devices = [part.strip() for part in devices.split(",") if part.strip()]
            devices = [int(item) for item in devices]
        except ValueError:
            errors.append(
                {
                    "field": "train_devices",
                    "message": "must contain integer GPU indices",
                }
            )
            return
    if not isinstance(devices, list) or any(
        not isinstance(item, int) or item < 0 for item in devices
    ):
        errors.append(
            {
                "field": "train_devices",
                "message": "must contain non-negative GPU indices",
            }
        )
        return
    if len(devices) != len(set(devices)):
        errors.append(
            {
                "field": "train_devices",
                "message": "must not contain duplicate GPU indices",
            }
        )
    try:
        world_size = int(config.get("world_size", 1))
    except (TypeError, ValueError):
        return
    if len(devices) != world_size:
        errors.append(
            {"field": "world_size", "message": "must equal the number of train_devices"}
        )


def _validate_parallelism(config: dict[str, Any], errors: list[dict[str, str]]) -> None:
    raw_world_size = config.get("world_size", 1)
    raw_tp_size = config.get("tp_size", 1)
    try:
        world_size = int(raw_world_size)
        tp_size = int(raw_tp_size)
        world_size_number = float(raw_world_size)
        tp_size_number = float(raw_tp_size)
    except (TypeError, ValueError):
        return
    if world_size != world_size_number:
        errors.append({"field": "world_size", "message": "must be an integer"})
        return
    if tp_size != tp_size_number:
        errors.append({"field": "tp_size", "message": "must be an integer"})
        return
    if world_size > 0 and tp_size > 0 and world_size % tp_size != 0:
        errors.append({"field": "tp_size", "message": "must divide world_size"})


def _validate_range(
    config: dict[str, Any],
    field: str,
    errors: list[dict[str, str]],
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> None:
    value = config.get(field)
    if value in (None, ""):
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    minimum_ok = number >= minimum if minimum_inclusive else number > minimum
    if not minimum_ok or number > maximum:
        left = "[" if minimum_inclusive else "("
        errors.append(
            {"field": field, "message": f"must be in {left}{minimum}, {maximum}]"}
        )


def _validate_local_reference(
    value: Any, field: str, errors: list[dict[str, str]], warnings: list[dict[str, str]]
) -> None:
    if value in (None, ""):
        return
    text = str(value)
    path = Path(text).expanduser()
    if path.is_absolute() or text.startswith(".") or path.exists():
        if not path.exists():
            errors.append(
                {"field": field, "message": f"local path does not exist: {text}"}
            )
    else:
        warnings.append(
            {
                "field": field,
                "message": "remote reference is not resolved during preview",
            }
        )
