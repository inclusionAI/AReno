"""Online MLX quantization and validation of prequantized QLoRA checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from areno.adapters.config import LoraConfig, read_adapter_quantization

METADATA_FILE = "areno_mlx_quantization.json"


def validate_quantization(bits: int, group_size: int) -> None:
    if bits not in (1, 2, 3, 4, 5, 6, 8):
        raise ValueError("MLX affine quantization bits must be 1,2,3,4,5,6 or 8")
    if group_size not in (32, 64, 128):
        raise ValueError("MLX affine quantization group size must be 32,64 or 128")


def adapter_quantization(path: str | Path) -> dict[str, Any] | None:
    metadata = Path(path) / METADATA_FILE
    if not metadata.is_file():
        return None
    config = json.loads(metadata.read_text())
    if config.get("format") != "mlx-affine-v1":
        raise ValueError("unsupported MLX online quantization metadata")
    validate_quantization(config["bits"], config["group_size"])
    return config


def _module_at(model: Any, path: str) -> Any:
    current = model
    for key in path.split("."):
        current = current[int(key)] if isinstance(current, list | tuple) else current[key]
    return current


def quantize_online(model: Any, config: dict[str, Any], *, bits: int, group_size: int) -> dict[str, Any]:
    """Quantize supported leaf modules one at a time, releasing source arrays.

    Uses the model's existing quantization predicate and each module's public
    to_quantized capability. No model source, checkpoint file or source tensor
    is mutated. Floating head/adapter modules are added only AFTER this step.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    validate_quantization(bits, group_size)
    if config.get("quantization") is not None:
        raise ValueError("online quantization requires an unquantized source checkpoint")
    mx.set_cache_limit(0)
    predicate = getattr(model, "quant_predicate", None)
    # Keep paths only; retaining original module objects would retain all BF16
    # source weights even after replacement and defeat streaming quantization.
    paths = [path for path, module in model.named_modules() if path and callable(getattr(module, "to_quantized", None))]
    overrides, quantized = {}, 0
    for path in paths:
        module = _module_at(model, path)
        weight = getattr(module, "weight", None)
        if weight is None or weight.shape[-1] % group_size:
            continue
        decision = predicate(path, module) if predicate is not None else True
        if not decision:
            continue
        parameters = {"bits": bits, "group_size": group_size, "mode": "affine"}
        if isinstance(decision, dict):
            parameters.update(decision)
            overrides[path] = parameters
        replacement = module.to_quantized(**parameters)
        mx.eval(replacement.parameters())
        model.update_modules(tree_unflatten([(path, replacement)]))
        del module, weight, replacement
        mx.clear_cache()
        quantized += 1
    if not quantized:
        raise ValueError("checkpoint contains no quantizable modules for the requested group size")
    quantization = {"bits": bits, "group_size": group_size, "mode": "affine", **overrides}
    return {
        **config,
        "quantization": quantization,
        "areno_online_quantization": {
            "format": "mlx-affine-v1",
            "bits": bits,
            "group_size": group_size,
            "modules": quantized,
        },
    }


def logical_weight_shape(module: Any) -> tuple[int, ...]:
    shape = tuple(getattr(getattr(module, "weight", None), "shape", ()))
    if shape and hasattr(module, "scales") and hasattr(module, "bits"):
        return (*shape[:-1], shape[-1] * 32 // module.bits)
    return shape


def validate_quantized_base(
    model: Any, config: dict[str, Any], lora: LoraConfig | None, path: str
) -> dict[str, Any] | None:
    """Inspect before adapter injection; never quantize a floating base implicitly."""

    quantization = config.get("quantization")
    if lora is None or not lora.qlora:
        if quantization is not None or config.get("quantization_config") is not None:
            raise ValueError("MLX quantized training requires LoRA with qlora=True (--qlora)")
        return None
    if config.get("model_type") not in {"qwen3", "bailing_hybrid"}:
        raise ValueError("MLX QLoRA supports Qwen3 dense and Ling Bailing V3 only")
    if not isinstance(quantization, dict):
        raise ValueError("MLX QLoRA requires a preconverted MLX 4-bit affine checkpoint")
    if config.get("quantization_config", quantization) != quantization:
        raise ValueError("MLX QLoRA requires native MLX quantization metadata")
    _validate_parameters(quantization)
    for name, value in quantization.items():
        if name in {"bits", "group_size", "mode"}:
            continue
        if value is not False:
            if not isinstance(value, dict):
                raise ValueError(f"unsupported MLX quantization entry: {name}")
            _validate_parameters(value)

    layers = {}
    predicate = getattr(model, "quant_predicate", None)
    for name, module in model.named_modules():
        if not hasattr(module, "scales") or not hasattr(module, "bits"):
            continue
        parameters = {"bits": module.bits, "group_size": module.group_size, "mode": module.mode}
        _validate_parameters(parameters)
        if predicate is not None and predicate(name, module) is False:
            raise ValueError(f"MLX QLoRA requires floating weights for model-protected layer {name}")
        layers[name] = {**parameters, "shape": list(logical_weight_shape(module))}
    if not layers:
        raise ValueError("MLX QLoRA checkpoint contains no quantized modules")

    root = Path(path)
    weights = sorted(root.glob("*.safetensors"))
    if not weights or not (root / "config.json").is_file():
        raise ValueError("MLX QLoRA requires a local checkpoint with config.json and safetensors weights")
    descriptor = {
        "format": "mlx-affine-v1",
        "model_type": config["model_type"],
        "quantization": quantization,
        "layers": layers,
        "base_config_sha256": _file_digest(root / "config.json"),
        "base_weights_sha256": {file.name: _file_digest(file) for file in weights},
    }
    validate_adapter_base(lora.adapter_path, descriptor)
    return descriptor


def validate_adapter_base(path: str | Path | None, descriptor: dict[str, Any] | None) -> None:
    if path is not None and read_adapter_quantization(path) != descriptor:
        raise ValueError("adapter quantization/base fingerprint does not match the loaded MLX base checkpoint")


def _validate_parameters(parameters: dict[str, Any]) -> None:
    if (parameters.get("bits"), parameters.get("group_size"), parameters.get("mode", "affine")) != (4, 64, "affine"):
        raise ValueError("MLX QLoRA currently requires 4-bit affine quantization with group_size=64")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
