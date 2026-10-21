"""Model-independent, bounded-memory online quantization for MLX providers."""

import json
from pathlib import Path
from typing import Any

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
