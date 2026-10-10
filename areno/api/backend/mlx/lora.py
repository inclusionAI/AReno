"""PEFT-compatible LoRA injection for supported MLX-LM projections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from areno.adapters.config import LoraConfig

_PEFT_PREFIX = "base_model.model.model."
_BAILING_KDA_TARGETS = ("q_proj", "k_proj", "v_proj", "f_proj", "o_proj")
_BAILING_MLA_TARGETS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "dense")


@dataclass(slots=True)
class MlxLoraState:
    """Index the MLX-LM LoRA modules that own trainable adapter arrays."""

    slots: dict[str, Any]
    config: LoraConfig


@dataclass(frozen=True, slots=True)
class _MlxLoraApi:
    mx: Any
    linear_type: type
    quantized_linear_type: type
    lora_linear_type: type
    tree_flatten: Any
    tree_unflatten: Any


def _qwen3_targets(model: Any, requested: set[str], model_config: dict[str, Any]) -> list[tuple[str, Any]]:
    del model_config
    targets = []
    for path, module in model.named_modules():
        if path.rsplit(".", 1)[-1] not in requested:
            continue
        if not path.startswith("model.layers."):
            raise ValueError(f"MLX LoRA target {path!r} is outside the dense transformer layers")
        targets.append((path, module))
    return targets


def _bailing_v3_targets(model: Any, requested: set[str], model_config: dict[str, Any]) -> list[tuple[str, Any]]:
    architectures = model_config.get("architectures") or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    if "BailingMoeV3ForCausalLM" not in architectures:
        raise ValueError("MLX Ling LoRA requires architecture BailingMoeV3ForCausalLM")
    allowed = set(_BAILING_KDA_TARGETS + _BAILING_MLA_TARGETS)
    if not requested <= allowed:
        raise ValueError(
            f"unsupported MLX Ling LoRA targets: {', '.join(sorted(requested - allowed))}; "
            f"use --lora-target-modules {','.join(_BAILING_KDA_TARGETS + _BAILING_MLA_TARGETS)}"
        )
    if model_config.get("quantization") is not None:
        raise ValueError("MLX Ling LoRA requires a non-quantized checkpoint; QLoRA is not supported yet")
    args = getattr(model, "args", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if args is None or not layers or len(layers) != args.num_hidden_layers:
        raise ValueError("MLX Ling LoRA requires the MLX-LM Bailing V3 model and complete decoder layers")
    if not args.no_kda_lora or not args.kda_safe_gate or not args.rope_interleave or args.q_lora_rank is None:
        raise ValueError(
            "MLX Ling LoRA requires no_kda_lora=true, kda_safe_gate=true, rope_interleave=true and q_lora_rank"
        )

    # Validate all layers before freezing or replacing any module. Dimensions
    # follow the loaded model args, including reduced models used in tests.
    hidden, heads = args.hidden_size, args.num_attention_heads
    kda_width = heads * args.head_dim
    kda_shapes = {name: (kda_width, hidden) for name in _BAILING_KDA_TARGETS if name != "o_proj"}
    kda_shapes["o_proj"] = (hidden, kda_width)
    mla_shapes = {
        "q_a_proj": (args.q_lora_rank, hidden),
        "q_b_proj": (heads * (args.qk_nope_head_dim + args.qk_rope_head_dim), args.q_lora_rank),
        "kv_a_proj_with_mqa": (args.kv_lora_rank + args.qk_rope_head_dim, hidden),
        "dense": (hidden, heads * args.v_head_dim),
    }
    modules = dict(model.named_modules())
    targets = []
    for index, layer in enumerate(layers):
        if not isinstance(getattr(layer, "is_linear", None), bool):
            raise ValueError(f"MLX Ling LoRA layer {index} has no KDA/MLA discriminator")
        shapes = kda_shapes if layer.is_linear else mla_shapes
        for name in sorted(requested & shapes.keys()):
            path = f"model.layers.{index}.attention.{name}"
            module = modules.get(path)
            if module is None:
                raise ValueError(f"MLX Ling LoRA target {path!r} is missing")
            shape = tuple(getattr(getattr(module, "weight", None), "shape", ()))
            if shape != shapes[name]:
                raise ValueError(f"MLX Ling LoRA target {path!r} has shape {shape}, expected {shapes[name]}")
            targets.append((path, module))
    expected_paths = {path for path, _ in targets}
    unexpected = {path for path in modules if path.rsplit(".", 1)[-1] in requested and path not in expected_paths}
    if unexpected:
        raise ValueError(f"MLX Ling LoRA targets outside supported attention paths: {sorted(unexpected)}")
    return targets


_TARGET_RESOLVERS = {"qwen3": _qwen3_targets, "bailing_hybrid": _bailing_v3_targets}


def initialize_lora(
    model: Any,
    config: LoraConfig,
    *,
    model_type: str,
    model_config: dict[str, Any] | None = None,
) -> MlxLoraState:
    """Validate a model family's projections, freeze the base and inject LoRA."""

    if config.full_parameter_targets:
        raise ValueError("full_parameter_targets are currently supported by the CUDA backend only")
    resolver = _TARGET_RESOLVERS.get(model_type)
    if resolver is None:
        supported = ", ".join(sorted(_TARGET_RESOLVERS))
        raise ValueError(f"MLX LoRA supports model types: {supported}")

    requested = set(config.target_modules)
    resolved = resolver(model, requested, model_config or {})
    api = _mlx_lora_api()
    matched: set[str] = set()
    targets: list[tuple[str, str, Any]] = []
    for module_path, module in resolved:
        target_name = module_path.rsplit(".", 1)[-1]
        matched.add(target_name)
        if isinstance(module, api.quantized_linear_type):
            raise ValueError(f"MLX LoRA target {module_path!r} is quantized; QLoRA is not supported yet")
        if not isinstance(module, api.linear_type):
            raise TypeError(
                f"MLX LoRA target {module_path!r} has unsupported type {type(module).__name__}; "
                "only dense Linear layers are supported"
            )
        logical_name = module_path.removeprefix("model.")
        targets.append((module_path, logical_name, module))

    missing = requested - matched
    if missing:
        raise ValueError(f"target_modules are not present in {model_type}: {', '.join(sorted(missing))}")

    model.freeze()
    replacements = []
    slots: dict[str, Any] = {}
    module_paths: list[str] = []
    for module_path, logical_name, module in sorted(targets):
        lora_module = api.lora_linear_type.from_base(
            module,
            r=config.rank,
            dropout=config.dropout,
            scale=config.scale,
        )
        replacements.append((module_path, lora_module))
        slots[logical_name] = lora_module
        module_paths.append(module_path)
    model.update_modules(api.tree_unflatten(replacements))
    _validate_trainable_parameters(model, module_paths, api.tree_flatten)

    state = MlxLoraState(slots=slots, config=config)
    if config.adapter_path is not None:
        _load_peft_adapter(state, config.adapter_path, api)
    return state


def load_peft_adapter(state: MlxLoraState, path: str | Path) -> None:
    """Load one standard dense PEFT adapter into already-injected MLX slots."""

    _load_peft_adapter(state, path, _mlx_lora_api())


def export_peft_adapter(
    state: MlxLoraState,
    path: str | Path,
    *,
    base_model_name_or_path: str | None,
) -> str:
    """Write the live MLX LoRA arrays as one standard PEFT artifact."""

    return _export_peft_adapter(
        state,
        path,
        base_model_name_or_path=base_model_name_or_path,
        api=_mlx_lora_api(),
    )


def _load_peft_adapter(state: MlxLoraState, path: str | Path, api: _MlxLoraApi) -> None:
    adapter_file = Path(path) / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"PEFT adapter weights do not exist: {adapter_file}")
    tensors = api.mx.load(str(adapter_file))
    expected_shapes = _expected_peft_shapes(state)
    actual_keys = set(tensors)
    expected_keys = set(expected_shapes)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "PEFT adapter tensor keys do not match the MLX LoRA registry: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(tensors[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"PEFT adapter tensor {key!r} has shape {actual_shape}, expected {expected_shape}")

    converted: dict[str, tuple[Any, Any]] = {}
    for logical_name, slot in state.slots.items():
        lora_a = tensors[_peft_key(logical_name, "A")].T.astype(slot.lora_a.dtype)
        lora_b = tensors[_peft_key(logical_name, "B")].T.astype(slot.lora_b.dtype)
        converted[logical_name] = (lora_a, lora_b)
    api.mx.eval(*(array for pair in converted.values() for array in pair))
    for logical_name, (lora_a, lora_b) in converted.items():
        slot = state.slots[logical_name]
        slot.lora_a = lora_a
        slot.lora_b = lora_b


def _export_peft_adapter(
    state: MlxLoraState,
    path: str | Path,
    *,
    base_model_name_or_path: str | None,
    api: _MlxLoraApi,
) -> str:
    import numpy as np
    from safetensors.numpy import save_file

    arrays: dict[str, Any] = {}
    for logical_name in sorted(state.slots):
        slot = state.slots[logical_name]
        arrays[_peft_key(logical_name, "A")] = slot.lora_a.T.astype(api.mx.float32)
        arrays[_peft_key(logical_name, "B")] = slot.lora_b.T.astype(api.mx.float32)
    api.mx.eval(*arrays.values())
    tensors = {name: np.ascontiguousarray(np.array(array, copy=True)) for name, array in arrays.items()}

    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": state.config.alpha,
        "lora_dropout": state.config.dropout,
        "peft_type": "LORA",
        "r": state.config.rank,
        "target_modules": list(state.config.target_modules),
        "task_type": "CAUSAL_LM",
    }
    with TemporaryDirectory(dir=output_path, prefix=".areno-peft-") as staging_directory:
        staging_path = Path(staging_directory)
        staging_config = staging_path / "adapter_config.json"
        staging_weights = staging_path / "adapter_model.safetensors"
        staging_config.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        save_file(tensors, staging_weights)
        staging_weights.replace(output_path / staging_weights.name)
        staging_config.replace(output_path / staging_config.name)
    return str(output_path)


def _expected_peft_shapes(state: MlxLoraState) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for logical_name, slot in state.slots.items():
        shapes[_peft_key(logical_name, "A")] = (int(slot.lora_a.shape[1]), int(slot.lora_a.shape[0]))
        shapes[_peft_key(logical_name, "B")] = (int(slot.lora_b.shape[1]), int(slot.lora_b.shape[0]))
    return shapes


def _validate_trainable_parameters(model: Any, module_paths: list[str], tree_flatten: Any) -> None:
    expected = {f"{path}.{name}" for path in module_paths for name in ("lora_a", "lora_b")}
    actual = {name for name, _ in tree_flatten(model.trainable_parameters())}
    if actual != expected:
        unexpected = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise RuntimeError(
            f"MLX LoRA must leave only adapter parameters trainable: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )


def _peft_key(logical_name: str, component: str) -> str:
    return f"{_PEFT_PREFIX}{logical_name}.lora_{component}.weight"


def _mlx_lora_api() -> _MlxLoraApi:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear

    return _MlxLoraApi(
        mx=mx,
        linear_type=nn.Linear,
        quantized_linear_type=nn.QuantizedLinear,
        lora_linear_type=LoRALinear,
        tree_flatten=tree_flatten,
        tree_unflatten=tree_unflatten,
    )
