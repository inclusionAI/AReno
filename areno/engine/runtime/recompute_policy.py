"""Selective recompute configuration, resolution and effective execution plans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MODULES = ("softmax_attention", "kda_attention", "dense_mlp", "routed_experts", "shared_experts")


@dataclass(frozen=True, slots=True)
class RecomputePolicy:
    modules: tuple[str, ...] = MODULES
    layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if set(self.modules) - set(MODULES):
            raise ValueError(f"Unknown recompute module; expected one of {MODULES}")
        if self.layers is not None and any(type(index) is not int or index < 0 for index in self.layers):
            raise ValueError("Recompute layer indices must be nonnegative integers")

    def selects(self, module: str, layer_idx: int) -> bool:
        return module in self.modules and (self.layers is None or layer_idx in self.layers)


def parse_layer_selection(selection: str, num_layers: int | None) -> tuple[int, ...] | None:
    """Parse zero-based inclusive indices/ranges, e.g. '0,4..7'."""
    if selection == "all":
        return None
    if selection == "none":
        return ()
    indices = set()
    for part in selection.split(","):
        endpoints = part.strip().split("..")
        if len(endpoints) not in (1, 2) or not all(value.isdecimal() for value in endpoints):
            raise ValueError("Expected layer indices or inclusive ranges, such as '0,4..7'")
        start, end = int(endpoints[0]), int(endpoints[-1])
        if start > end or (num_layers is not None and end >= num_layers):
            raise ValueError(f"Recompute layers must be in [0, {num_layers}) with ascending ranges")
        indices.update(range(start, end + 1))
    return tuple(sorted(indices))


def normalize_recompute_selection(modules, layers):
    """Normalize SDK sequences and CLI comma-separated selectors without Torch."""
    if isinstance(modules, str):
        modules = modules.strip()
        modules = MODULES if modules == "all" else () if modules == "none" else tuple(modules.split(","))
        modules = tuple(value.strip() for value in modules)
    if modules is not None:
        if not isinstance(modules, (tuple, list)) or any(value not in MODULES for value in modules):
            raise ValueError(f"recompute_modules must contain only {MODULES}, or 'all'/'none'")
        modules = tuple(module for module in MODULES if module in modules)
    if isinstance(layers, str):
        layers = parse_layer_selection(layers.strip(), None)
    if layers is not None:
        if not isinstance(layers, (tuple, list)) or any(type(index) is not int or index < 0 for index in layers):
            raise ValueError("recompute_layers must contain nonnegative integer indices or inclusive ranges")
        layers = tuple(sorted(set(layers)))
    return modules, layers


def resolve_runtime_recompute_policy(model, runtime) -> RecomputePolicy | None:
    """Validate selectors against the actual adapter and decoder layer count."""
    modules, layers = normalize_recompute_selection(runtime.recompute_modules, runtime.recompute_layers)
    explicit = modules is not None or layers is not None
    if explicit and (model.model_type != "bailing_moe_v3" or runtime.device_type != "cuda"):
        raise ValueError("module/layer recompute selection currently requires CUDA Ling v3 (bailing_moe_v3)")
    if layers is not None and any(index >= model.num_hidden_layers for index in layers):
        raise ValueError(f"recompute_layers must be in [0, {model.num_hidden_layers})")
    # An explicit decoder-off request must also bypass legacy tokenwise MST.
    if not explicit and not (model.model_type == "bailing_moe_v3" and not runtime.activation_checkpointing):
        return None
    return RecomputePolicy(MODULES if modules is None else modules, layers)


def effective_recompute_plan(
    *,
    activation_checkpointing: bool,
    policy: RecomputePolicy,
    layer_modules: Sequence[Sequence[str]],
    output_recompute: bool,
    output_chunk_size: int,
    mst_chunk_size: int = 0,
) -> dict:
    """Resolve the explicit policy against actual adapter module availability."""
    if output_chunk_size <= 0 or mst_chunk_size < 0:
        raise ValueError("Output chunk size must be positive; MST chunk size must be nonnegative")
    if policy.layers is not None and any(index >= len(layer_modules) for index in policy.layers):
        raise ValueError("Recompute layer index exceeds the decoder layer count")
    layers = []
    for index, available in enumerate(layer_modules):
        strategies = {}
        for module in available:
            if module not in MODULES:
                raise ValueError(f"Unknown decoder module {module!r}")
            selected = activation_checkpointing and policy.selects(module, index)
            strategies[module] = (
                "mst"
                if selected and mst_chunk_size and module not in MODULES[:2]
                else "checkpoint"
                if selected
                else "direct"
            )
        layers.append({"index": index, "modules": strategies})
    return {
        "decoder": {"activation_checkpointing": activation_checkpointing, "layers": layers},
        "sft_output": {"recompute": output_recompute, "chunk_size": output_chunk_size},
        "mst_chunk_size": mst_chunk_size,
    }
