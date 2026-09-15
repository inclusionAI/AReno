"""Validated, serializable job plans independent of Modal and the HTTP layer."""

from __future__ import annotations

import math
import shlex

from arenoflow.catalog import arguments

GPU_TYPES = ("T4", "L4", "A10G", "L40S", "A100-40GB", "A100-80GB", "H100", "H200", "B200")


def bounded(value, name, lower, upper, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(value) or not lower <= value <= upper or (integer and not value.is_integer()):
        raise ValueError(f"{name} must be between {lower} and {upper}" + (" (integer)" if integer else ""))
    return int(value) if integer else value


def timeout_seconds(raw):
    if "timeout_seconds" in raw:
        return bounded(raw["timeout_seconds"], "Timeout seconds", 1, 86400, True)
    # Older exported workflows and stored runs used whole hours.
    return bounded(raw.get("timeout_hours", 4), "Timeout hours", 1, 24, True) * 3600


def resources(raw):
    gpu = raw.get("gpu", "H100")
    if gpu not in GPU_TYPES:
        raise ValueError("Unsupported GPU reservation")
    result = dict(
        gpu=gpu,
        count=bounded(raw.get("count", 1), "GPU count", 1, 8, True),
        cpu=bounded(raw.get("cpu", 4), "CPU cores", 1, 64),
        memory_gib=bounded(raw.get("memory_gib", 32), "Memory GiB", 4, 512, True),
        timeout_seconds=timeout_seconds(raw),
    )
    return result


def plan(request, catalog, job_id="preview"):
    kind = request.get("kind", "training")
    if kind in ("image_build", "model_download"):
        return preparation_plan(request, catalog)
    if kind not in ("training", "deployment"):
        raise ValueError("Job kind must be training or deployment")
    resource = resources(request.get("resources", {}))
    model = request.get("model", {})
    if model.get("adapter") not in {m["id"] for m in catalog["models"]}:
        raise ValueError("Select a model adapter from the repository catalog")
    if not str(model.get("checkpoint", "")).strip():
        raise ValueError("A model checkpoint or repository ID is required")
    result = dict(
        kind=kind,
        revision=catalog["revision"],
        schema_version=catalog["schema_version"],
        image=request.get("image") or catalog["image"],
        model=model,
        stages=[],
        input_assets=request.get("input_assets", []),
    )
    commands = []
    if kind == "training":
        stages = request.get("stages", [])
        if not isinstance(stages, list) or not 1 <= len(stages) <= 8:
            raise ValueError("A workflow needs 1–8 training stages")
        for index, stage in enumerate(stages):
            algo = stage.get("algo")
            if algo not in {a["id"] for a in catalog["algorithms"]}:
                raise ValueError("Select a registered algorithm")
            params = {**catalog["presets"].get(algo, {}), **stage.get("params", {}), "algo": algo}
            incompatible = [
                p["name"] for p in catalog["train"] if algo not in p["algorithms"] and params.get(p["name"]) is not None
            ]
            if incompatible:
                raise ValueError(f"{algo.upper()} does not use: {', '.join(incompatible)}")
            params.setdefault("ckpt", model["checkpoint"] if index == 0 else "__previous__")
            params.setdefault("save_path", f"/artifacts/runs/{job_id}/stage-{index}")
            params.setdefault("metrics_log_dir", f"/artifacts/runs/{job_id}/metrics-{index}")
            if index and params["ckpt"] != "__previous__":
                raise ValueError("Later stages must use __previous__ to consume the preceding checkpoint")
            if not params.get("dataset_path"):
                raise ValueError(f"Stage {index + 1}: dataset is required")
            if algo in ("grpo", "gspo", "ppo") and not (params.get("reward_fn_path") or params.get("reward_ckpt")):
                raise ValueError(f"Stage {index + 1}: configure a reward function or reward checkpoint")
            if params.get("model_hub") not in (None, "hf"):
                raise ValueError("Only Hugging Face is supported")
            params["model_hub"] = "hf"
            validate_devices(params, resource)
            for field in ("save_path", "metrics_log_dir"):
                if not str(params.get(field, "")).startswith("/artifacts/") or ".." in params[field].split("/"):
                    raise ValueError(f"{field} must be inside /artifacts/ so outputs persist")
            # Full-model checkpoints are the explicit artifact contract of this workflow runner.
            if params.get("lora_rank") and len(stages) > 1:
                raise ValueError(
                    "Multi-stage LoRA chaining is not supported; use a single-stage run and deploy its adapter"
                )
            argv = arguments("train", params, catalog["train"])
            result["stages"].append(dict(algo=algo, args=argv, save_path=params["save_path"], params=params))
            commands.append(shlex.join(["areno", "train", *argv]))
    else:
        params = dict(request.get("serve", {}))
        params.setdefault("model_path", model["checkpoint"])
        if params.get("model_hub") not in (None, "hf"):
            raise ValueError("Only Hugging Face is supported")
        params["model_hub"] = "hf"
        params.setdefault("tp_size", resource["count"])
        params.setdefault("world_size", resource["count"])
        params.update(host="127.0.0.1", port=8000)
        validate_devices(params, resource)
        result["serve_args"] = arguments("serve", params, catalog["serve"])
        commands.append(shlex.join(["areno", "serve", *result["serve_args"]]))
    return dict(manifest=result, resources=resource, commands=commands)


def validate_devices(params, resource):
    world = bounded(params.get("world_size", 1), "world_size", 1, resource["count"], True)
    tp = bounded(params.get("tp_size", 1), "tp_size", 1, world, True)
    if world % tp:
        raise ValueError("world_size must be divisible by tp_size")


def preparation_plan(request, catalog):
    """Preparation jobs share the training image and Volume, without GPU reservations."""
    kind = request["kind"]
    model = request.get("model", {}) if kind == "model_download" else {"checkpoint": ""}
    if kind == "model_download":
        if model.get("adapter") not in {m["id"] for m in catalog["models"]}:
            raise ValueError("Select a model adapter from the repository catalog")
        checkpoint = model.get("checkpoint", "")
        if not isinstance(checkpoint, str) or not checkpoint.strip() or checkpoint.startswith(("/", ".")):
            raise ValueError("Select a model repository ID to download")
    hub = request.get("model_hub", "hf")
    if hub != "hf":
        raise ValueError("Only Hugging Face is supported")
    return dict(
        manifest=dict(
            kind=kind,
            revision=catalog["revision"],
            schema_version=catalog["schema_version"],
            image=request.get("image") or catalog["image"],
            model=model,
            model_hub=hub,
            stages=[],
        ),
        resources=dict(
            gpu=None, count=0, cpu=2, memory_gib=8, timeout_seconds=timeout_seconds(request.get("resources", {}))
        ),
        commands=[],
    )
