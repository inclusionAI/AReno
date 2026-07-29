#!/usr/bin/env python3
"""Interactive model-adaptation scaffold generator (issue #274).

Reads a local HuggingFace model config (``config.json``) and generates an
adapter directory structure with registration code, checkpoint-mapping
placeholders, and a minimal load example.

Two scaffold types are supported:

* **dense**: Standard decoder-only transformer (attention + MLP per layer).
* **moe**: Mixture-of-experts transformer (attention + MoE per layer).

The generator never downloads assets or writes outside the requested
destination.  Reruns preserve user-edited files and report conflicts.

Usage::

    python generate_adapter_scaffold.py \\
        --hf-config /path/to/model/config.json \\
        --adapter-name mymodel \\
        --dest-dir areno/models/mymodel

    # Non-interactive (accept all inferred choices):
    python generate_adapter_scaffold.py \\
        --hf-config /path/to/config.json \\
        --adapter-name mymodel \\
        --dest-dir areno/models/mymodel \\
        --yes
"""

from __future__ import annotations

import argparse
import json
import keyword
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AdapterConfig:
    """Inferred adapter configuration from a HuggingFace config.

    Attributes:
        adapter_name: Lowercase identifier for the adapter (e.g. ``mymodel``).
        model_type: The ``model_type`` field from the HF config.
        is_moe: Whether the model uses mixture-of-experts.
        class_name: PascalCase class name for the adapter (e.g. ``MymodelAdapter``).
        hf_config_path: Path to the source ``config.json``.
        dest_dir: Destination directory for generated files.
        hidden_size: Model hidden dimension.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads (GQA).
        intermediate_size: MLP intermediate dimension.
        vocab_size: Vocabulary size.
        num_experts: Number of experts (MoE only, None for dense).
        num_experts_per_tok: Experts per token (MoE only, None for dense).
    """

    adapter_name: str
    model_type: str
    is_moe: bool
    class_name: str
    hf_config_path: str
    dest_dir: str
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    num_experts: int | None = None
    num_experts_per_tok: int | None = None


@dataclass
class GenerationResult:
    """Result of scaffold generation.

    Attributes:
        created_files: List of files created (new).
        preserved_files: List of files that already existed and were preserved.
        conflicted_files: List of files where user edits differ from template.
        dest_dir: The destination directory.
    """

    created_files: list[str] = field(default_factory=list)
    preserved_files: list[str] = field(default_factory=list)
    conflicted_files: list[str] = field(default_factory=list)
    dest_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for structured output."""

        return {
            "created_files": self.created_files,
            "preserved_files": self.preserved_files,
            "conflicted_files": self.conflicted_files,
            "dest_dir": self.dest_dir,
        }


# ---------------------------------------------------------------------------
# Config inference
# ---------------------------------------------------------------------------


def infer_config(
    hf_config_path: str,
    adapter_name: str,
    dest_dir: str,
) -> AdapterConfig:
    """Infer adapter configuration from a HuggingFace ``config.json``.

    Args:
        hf_config_path: Path to the HF ``config.json`` file.
        adapter_name: Lowercase adapter identifier.
        dest_dir: Destination directory for generated files.

    Returns:
        An :class:`AdapterConfig` with inferred values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is missing required fields or adapter_name is invalid.
    """

    module_name = adapter_name.replace("-", "_")
    if not module_name.isidentifier() or keyword.iskeyword(module_name):
        raise ValueError(
            f"adapter_name must form a valid Python module name after replacing hyphens with underscores; got {adapter_name!r}"
        )
    if not dest_dir.strip():
        raise ValueError("dest_dir must be non-empty")

    path = Path(hf_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {hf_config_path}")

    with path.open("r", encoding="utf-8") as f:
        hf_config = json.load(f)

    if not isinstance(hf_config, dict):
        raise ValueError("Config file must contain a JSON object")

    model_type = hf_config.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("Config file is missing 'model_type' field")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", model_type) is None:
        raise ValueError(f"model_type contains unsupported characters: {model_type!r}")

    # Detect MoE: check common fields across model families.
    is_moe = _detect_moe(hf_config)

    # Extract architecture parameters.
    hidden_size = _positive_int(hf_config, "hidden_size")
    num_hidden_layers = _positive_int(hf_config, "num_hidden_layers")
    num_attention_heads = _positive_int(hf_config, "num_attention_heads")
    num_key_value_heads = _positive_int(
        hf_config,
        "num_key_value_heads",
        fallback_key="num_kv_heads",
        default=num_attention_heads,
    )
    intermediate_size = _positive_int(hf_config, "intermediate_size")
    vocab_size = _positive_int(hf_config, "vocab_size")
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    num_experts = None
    num_experts_per_tok = None
    if is_moe:
        num_experts = _positive_int(hf_config, "num_experts")
        num_experts_per_tok = _positive_int(hf_config, "num_experts_per_tok")
        if num_experts_per_tok > num_experts:
            raise ValueError("num_experts_per_tok must not exceed num_experts")

    class_name = _to_pascal_case(adapter_name) + "Adapter"

    return AdapterConfig(
        adapter_name=module_name,
        model_type=model_type,
        is_moe=is_moe,
        class_name=class_name,
        hf_config_path=hf_config_path,
        dest_dir=dest_dir,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )


def _positive_int(
    config: dict[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
    default: int | None = None,
) -> int:
    """Read a required positive integer without accepting booleans or floats."""

    value = config.get(key)
    if value is None and fallback_key is not None:
        value = config.get(fallback_key)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config field {key!r} must be a positive integer; got {value!r}")
    return value


def _detect_moe(hf_config: dict[str, Any]) -> bool:
    """Detect whether a HF config describes a MoE model."""

    if hf_config.get("num_experts"):
        return True
    if hf_config.get("moe_intermediate_size"):
        return True
    archs = hf_config.get("architectures", [])
    if isinstance(archs, list):
        for arch in archs:
            if "moe" in arch.lower() or "MoE" in arch:
                return True
    return False


def _to_pascal_case(name: str) -> str:
    """Convert a lowercase name to PascalCase (e.g. ``my_model`` -> ``MyModel``)."""

    parts = name.replace("-", "_").split("_")
    return "".join(part.capitalize() for part in parts if part)


# ---------------------------------------------------------------------------
# File templates
# ---------------------------------------------------------------------------


def _template_init_py(config: AdapterConfig) -> str:
    """Generate ``__init__.py`` for the adapter package."""

    model_module = "model_moe" if config.is_moe else "model"
    return f'''"""{config.adapter_name} plugin.

Re-exports the adapter and the causal LM module so the registry can pick them
up.  The implementation is intentionally thin and reuses the generic
``CausalSelfAttention`` / ``GatedMLP`` building blocks.
"""

from __future__ import annotations

from areno.models.{config.adapter_name}.{model_module} import {config.class_name}


def register() -> None:
    """Register this adapter with AReno's model registry."""

    from areno.models.registry import register_adapter

    register_adapter({config.class_name}())

__all__ = ["{config.class_name}", "register"]
'''


def _template_model_py(config: AdapterConfig) -> str:
    """Generate ``model.py`` for a dense adapter."""

    return f'''"""{config.adapter_name} causal-LM adapter.

Targets the {config.model_type} family of HF checkpoints
(``model_type == "{config.model_type}"``).
The architecture is a standard GQA decoder:
    * Pre-norm RMSNorm + CausalSelfAttention + GatedMLP (SwiGLU).
    * GQA head ratio comes from ``num_attention_heads`` / ``num_key_value_heads``
      in the HF config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from areno.engine.checkpoints.common import load_checkpoint_weights, save_checkpoint_weights
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention import CausalSelfAttention
from areno.engine.layers.mlp import GatedMLP
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.parallel.collectives import all_reduce
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.recompute import checkpoint_layer
from areno.models.base import CausalLMOutput, ModelAdapter
from areno.models.{config.adapter_name}.checkpoint import CHECKPOINT_SPEC


class {config.class_name.replace("Adapter", "DecoderLayer")}(nn.Module):
    """One transformer block: pre-norm attention + pre-norm SwiGLU MLP."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = GatedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, position_ids, train_meta, infer_meta)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class {config.class_name.replace("Adapter", "ForCausalLM")}(nn.Module):
    """{config.model_type} causal LM."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([
            {config.class_name.replace("Adapter", "DecoderLayer")}(config, i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> CausalLMOutput:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = checkpoint_layer(
                layer, hidden_states, position_ids, train_meta, infer_meta
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits_shard=logits, hidden_states=hidden_states)


class {config.class_name}(ModelAdapter):
    """Adapter for {config.model_type} HF checkpoints."""

    name = "{config.model_type}"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        return hf_config.get("model_type") == "{config.model_type}"

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        return ModelConfig(
            model_type="{config.model_type}",
            vocab_size=int(hf_config.get("vocab_size", {config.vocab_size})),
            hidden_size=int(hf_config.get("hidden_size", {config.hidden_size})),
            intermediate_size=int(hf_config.get("intermediate_size", {config.intermediate_size})),
            num_hidden_layers=int(hf_config.get("num_hidden_layers", {config.num_hidden_layers})),
            num_attention_heads=int(hf_config.get("num_attention_heads", {config.num_attention_heads})),
            num_key_value_heads=int(hf_config.get("num_key_value_heads", {config.num_key_value_heads})),
            head_dim=int(hf_config.get("head_dim", {config.hidden_size} // {config.num_attention_heads})) if {config.num_attention_heads} else 128,
            rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(hf_config.get("rope_theta", 1e6)),
            dtype=_parse_dtype(hf_config.get("torch_dtype", "bfloat16")),
        )

    def build(self, config: ModelConfig) -> nn.Module:
        return {config.class_name.replace("Adapter", "ForCausalLM")}(config)

    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        load_checkpoint_weights(model, model_path, CHECKPOINT_SPEC)

    def save_weights(
        self, model: nn.Module, output_path: str | Path, source_path: str | Path | None
    ) -> str | None:
        return save_checkpoint_weights(model, output_path, source_path, CHECKPOINT_SPEC)
'''


def _template_model_moe_py(config: AdapterConfig) -> str:
    """Generate ``model_moe.py`` for a MoE adapter."""

    return f'''"""{config.adapter_name} MoE causal-LM adapter.

Targets the {config.model_type} family of HF checkpoints with Mixture-of-Experts
(``model_type == "{config.model_type}"``, MoE variant).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from areno.engine.checkpoints.common import load_checkpoint_weights, save_checkpoint_weights
from areno.engine.config import ModelConfig, _parse_dtype
from areno.engine.layers.attention import CausalSelfAttention
from areno.engine.layers.norm import RMSNorm
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.parallel.context import get_tp_context
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.recompute import checkpoint_layer
from areno.models.base import CausalLMOutput, ModelAdapter
from areno.models.{config.adapter_name}.checkpoint import CHECKPOINT_SPEC


class {config.class_name.replace("Adapter", "DecoderLayer")}(nn.Module):
    """One MoE transformer block: pre-norm attention + pre-norm MoE MLP."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # TODO: Replace with actual MoE MLP implementation.
        # See areno/models/qwen3/model.py:Qwen3MoeExperts for reference.
        self.mlp = nn.Identity()  # Placeholder

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, position_ids, train_meta, infer_meta)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class {config.class_name.replace("Adapter", "ForCausalLM")}(nn.Module):
    """{config.model_type} MoE causal LM."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([
            {config.class_name.replace("Adapter", "DecoderLayer")}(config, i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = VocabParallelLMHead(config.hidden_size, config.vocab_size, dtype=config.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        train_meta: TrainMeta | None = None,
        infer_meta: InferMeta | None = None,
    ) -> CausalLMOutput:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = checkpoint_layer(
                layer, hidden_states, position_ids, train_meta, infer_meta
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits_shard=logits, hidden_states=hidden_states)


class {config.class_name}(ModelAdapter):
    """Adapter for {config.model_type} MoE HF checkpoints."""

    name = "{config.model_type}"

    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        return hf_config.get("model_type") == "{config.model_type}"

    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        return ModelConfig(
            model_type="{config.model_type}",
            vocab_size=int(hf_config.get("vocab_size", {config.vocab_size})),
            hidden_size=int(hf_config.get("hidden_size", {config.hidden_size})),
            intermediate_size=int(hf_config.get("intermediate_size", {config.intermediate_size})),
            num_hidden_layers=int(hf_config.get("num_hidden_layers", {config.num_hidden_layers})),
            num_attention_heads=int(hf_config.get("num_attention_heads", {config.num_attention_heads})),
            num_key_value_heads=int(hf_config.get("num_key_value_heads", {config.num_key_value_heads})),
            head_dim=int(hf_config.get("head_dim", {config.hidden_size} // {config.num_attention_heads})) if {config.num_attention_heads} else 128,
            rms_norm_eps=float(hf_config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(hf_config.get("rope_theta", 1e6)),
            dtype=_parse_dtype(hf_config.get("torch_dtype", "bfloat16")),
            enable_moe_block=True,
            num_experts=int(hf_config.get("num_experts", {config.num_experts or 0})),
            num_experts_per_tok=int(hf_config.get("num_experts_per_tok", {config.num_experts_per_tok or 1})),
        )

    def build(self, config: ModelConfig) -> nn.Module:
        return {config.class_name.replace("Adapter", "ForCausalLM")}(config)

    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        load_checkpoint_weights(model, model_path, CHECKPOINT_SPEC)

    def save_weights(
        self, model: nn.Module, output_path: str | Path, source_path: str | Path | None
    ) -> str | None:
        return save_checkpoint_weights(model, output_path, source_path, CHECKPOINT_SPEC)
'''


def _template_checkpoint_py(config: AdapterConfig) -> str:
    """Generate ``checkpoint.py`` with checkpoint-mapping placeholders."""

    moe_note = ""
    if config.is_moe:
        moe_note = """
# MoE checkpoint specs:
# TODO: Define MoeSpec for expert weights. See qwen3/checkpoint.py:MOE_SPEC.
# TODO: Define DenseOrMoeSpec for MoE MLP layers.
"""
    return f'''"""{config.adapter_name} HF safetensors load/save specs.

Maps HF parameter names to areno's fused parallel layers.  This is a
placeholder — fill in the actual key mappings for {config.model_type}.
"""

from __future__ import annotations

from areno.engine.checkpoints.common import (
    CheckpointSpec,
    LayerSpec,
    MergedColumnSpec,
    ParallelTensorSpec,
    ReplicatedTensorSpec,
    SplitColumnSpec,
    TopLevelSpec,
)

# TODO: Adjust embedding key and attribute for {config.model_type}.
TOP_LEVEL_SPEC = TopLevelSpec(
    embedding_key="model.embed_tokens.weight",
    embedding_attr="embed_tokens",
)

# Per-layer RMSNorm weights are replicated across TP ranks.
LAYER_NORM_SPECS = (
    ReplicatedTensorSpec("{{prefix}}.input_layernorm.weight", "input_layernorm.weight"),
    ReplicatedTensorSpec("{{prefix}}.post_attention_layernorm.weight", "post_attention_layernorm.weight"),
)

# TODO: Adjust QKV key names for {config.model_type}.
QKV_WEIGHT_SPEC = MergedColumnSpec(
    dst_attr="self_attn.qkv_proj.weight",
    keys=(
        "{{prefix}}.self_attn.q_proj.weight",
        "{{prefix}}.self_attn.k_proj.weight",
        "{{prefix}}.self_attn.v_proj.weight",
    ),
)

# TODO: Adjust output projection key for {config.model_type}.
ATTN_ROW_SPEC = ParallelTensorSpec(
    "{{prefix}}.self_attn.o_proj.weight", "self_attn.o_proj.weight", 1
)

# TODO: Adjust gate/up key names for {config.model_type}.
GATE_UP_WEIGHT_SPEC = MergedColumnSpec(
    dst_attr="mlp.gate_up_proj.weight",
    keys=(
        "{{prefix}}.mlp.gate_proj.weight",
        "{{prefix}}.mlp.up_proj.weight",
    ),
)
GATE_UP_SAVE_SPEC = SplitColumnSpec(
    src_attr="mlp.gate_up_proj.weight",
    size_attr="mlp.gate_up_proj.local_out_features",
    keys=GATE_UP_WEIGHT_SPEC.keys,
)

# TODO: Adjust down projection key for {config.model_type}.
MLP_ROW_SPEC = ParallelTensorSpec(
    "{{prefix}}.mlp.down_proj.weight", "mlp.down_proj.weight", 1
)
{moe_note}
LAYER_SPEC = LayerSpec(
    prefix="model.layers.{{layer}}",
    replicated=LAYER_NORM_SPECS,
    load_ops=(
        QKV_WEIGHT_SPEC,
        ATTN_ROW_SPEC,
        GATE_UP_WEIGHT_SPEC,
        MLP_ROW_SPEC,
    ),
    save_ops=(
        QKV_WEIGHT_SPEC,  # TODO: Add RangedSplitColumnSpec for save
        ATTN_ROW_SPEC,
        GATE_UP_SAVE_SPEC,
        MLP_ROW_SPEC,
    ),
)

CHECKPOINT_SPEC = CheckpointSpec(top_level=TOP_LEVEL_SPEC, layer=LAYER_SPEC)
'''


def _template_example_py(config: AdapterConfig) -> str:
    """Generate a minimal load example script."""

    return f'''"""Minimal load example for the {config.adapter_name} adapter.

Run after filling in the checkpoint specs::

    python {config.adapter_name}/example.py --model-path /path/to/{config.model_type}-checkpoint
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="Load {config.model_type} with the {config.adapter_name} adapter")
    parser.add_argument("--model-path", required=True, help="Path to the HF checkpoint directory")
    args = parser.parse_args()

    # This example requires a CUDA-capable GPU and torch.
    # Uncomment the following lines to test after filling in checkpoint specs:

    # from areno.models.{config.adapter_name} import {config.class_name}
    # from areno.models.registry import adapter_from_hf, config_from_hf, build_model, load_model_weights
    #
    # adapter = adapter_from_hf(args.model_path)
    # config = config_from_hf(args.model_path)
    # model = build_model(config)
    # load_model_weights(model, config, args.model_path)
    # print(f"Loaded {{adapter.name}} with {{config.num_hidden_layers}} layers")

    print(f"Adapter: {config.class_name}")
    print(f"Model type: {config.model_type}")
    print(f"MoE: {config.is_moe}")
    print(f"Model path: {{args.model_path}}")
    print("Fill in checkpoint specs before running a real load.")


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# File generation with rerun safety
# ---------------------------------------------------------------------------


# Files that the generator writes, mapped to their template functions.
def _file_templates(config: AdapterConfig) -> dict[str, str]:
    """Return a dict of relative path -> file content for the adapter."""

    if config.is_moe:
        return {
            "__init__.py": _template_init_py(config),
            "model_moe.py": _template_model_moe_py(config),
            "checkpoint.py": _template_checkpoint_py(config),
            "example.py": _template_example_py(config),
        }
    return {
        "__init__.py": _template_init_py(config),
        "model.py": _template_model_py(config),
        "checkpoint.py": _template_checkpoint_py(config),
        "example.py": _template_example_py(config),
    }


# Sentinel placed at the top of generated files so reruns can detect them.
_GENERATED_SENTINEL = "# GENERATED BY areno-model-adaptation scaffold generator"


def _add_sentinel(content: str) -> str:
    """Add the generated-file sentinel as a comment after the docstring."""

    if _GENERATED_SENTINEL in content:
        return content
    # Insert after the module docstring (first closing triple quote).
    idx = content.find('"""', 3)
    if idx != -1:
        insert_pos = idx + 3
        return content[:insert_pos] + "\n" + _GENERATED_SENTINEL + content[insert_pos:]
    return _GENERATED_SENTINEL + "\n" + content


def _is_generated(path: Path) -> bool:
    """Check if a file was generated by this tool (contains the sentinel)."""

    try:
        text = path.read_text(encoding="utf-8")
        return _GENERATED_SENTINEL in text
    except (OSError, UnicodeDecodeError):
        return False


def generate_scaffold(
    config: AdapterConfig,
    *,
    overwrite: bool = False,
) -> GenerationResult:
    """Generate adapter scaffold files.

    Args:
        config: The inferred :class:`AdapterConfig`.
        overwrite: If ``True``, overwrite existing generated files.
            If ``False`` (default), preserve user-edited files and report
            conflicts.

    Returns:
        A :class:`GenerationResult` with created/preserved/conflicted file lists.

    Raises:
        ValueError: If the destination is not a valid directory path.
    """

    dest = Path(config.dest_dir)
    templates = _file_templates(config)
    result = GenerationResult(dest_dir=str(dest))

    if dest.exists() and not dest.is_dir():
        raise ValueError(f"Destination exists and is not a directory: {dest}")
    if dest.is_symlink():
        raise ValueError(f"Destination must not be a symbolic link: {dest}")

    # Create destination directory.
    dest.mkdir(parents=True, exist_ok=True)

    for rel_path, content in templates.items():
        file_path = dest / rel_path
        tagged_content = _add_sentinel(content)

        if file_path.is_symlink():
            result.conflicted_files.append(rel_path)
        elif file_path.exists():
            if _is_generated(file_path):
                # This is a previously generated file.
                if overwrite:
                    file_path.write_text(tagged_content, encoding="utf-8")
                    result.created_files.append(rel_path)
                else:
                    # Check if user has modified it from the template.
                    current = file_path.read_text(encoding="utf-8")
                    # Strip sentinel for comparison.
                    current_clean = current.replace(_GENERATED_SENTINEL + "\n", "").replace(_GENERATED_SENTINEL, "")
                    if current_clean.strip() == content.strip():
                        result.preserved_files.append(rel_path)
                    else:
                        result.conflicted_files.append(rel_path)
            else:
                # User-created file with the same name — never overwrite.
                result.preserved_files.append(rel_path)
        else:
            file_path.write_text(tagged_content, encoding="utf-8")
            result.created_files.append(rel_path)

    return result


def format_result(result: GenerationResult) -> str:
    """Return a human-readable summary of the generation result."""

    lines: list[str] = []
    lines.append(f"Adapter scaffold generated in: {result.dest_dir}")
    lines.append("")
    if result.created_files:
        lines.append("Created files:")
        for f in result.created_files:
            lines.append(f"  + {f}")
    if result.preserved_files:
        lines.append("Preserved files (already exist):")
        for f in result.preserved_files:
            lines.append(f"  = {f}")
    if result.conflicted_files:
        lines.append("Conflicted files (user-edited, not overwritten):")
        for f in result.conflicted_files:
            lines.append(f"  ! {f}")
        lines.append("")
        lines.append("  Re-run with --overwrite to replace conflicted files.")
    lines.append("")
    lines.append("Next steps:")
    lines.append("  1. Edit checkpoint.py to fill in HF key mappings.")
    model_file = "model_moe.py" if "model_moe.py" in result.created_files + result.preserved_files + result.conflicted_files else "model.py"
    lines.append(f"  2. Edit {model_file} to adjust architecture details.")
    lines.append("  3. Call the generated register() function from areno/models/__init__.py.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a model adapter scaffold from a HuggingFace config.",
    )
    parser.add_argument("--hf-config", required=True, help="Path to the HuggingFace config.json file.")
    parser.add_argument("--adapter-name", required=True, help="Lowercase adapter name (e.g. mymodel).")
    parser.add_argument("--dest-dir", required=True, help="Destination directory for generated files.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Accept all inferred choices without prompting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previously generated files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    # Infer config.
    try:
        config = infer_config(args.hf_config, args.adapter_name, args.dest_dir)
    except (OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.json:
        print(f"Adapter name:     {config.adapter_name}")
        print(f"Model type:       {config.model_type}")
        print(f"Scaffold type:    {'MoE' if config.is_moe else 'Dense'}")
        print(f"Class name:       {config.class_name}")
        print(f"Hidden size:      {config.hidden_size}")
        print(f"Num layers:       {config.num_hidden_layers}")
        print(f"Attention heads:  {config.num_attention_heads}")
        print(f"KV heads:         {config.num_key_value_heads}")
        if config.is_moe:
            print(f"Num experts:      {config.num_experts}")
            print(f"Experts per tok:  {config.num_experts_per_tok}")
        print(f"Destination:      {config.dest_dir}")
        print()

    # Confirm unless --yes.
    if not args.yes and not args.json:
        response = input("Proceed with generation? [y/N] ")
        if response.lower() not in ("y", "yes"):
            print("Aborted.")
            return 0

    # Generate.
    try:
        result = generate_scaffold(config, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
