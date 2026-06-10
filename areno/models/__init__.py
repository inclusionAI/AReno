"""Model adapter API and bundled areno-native model plugins."""

from __future__ import annotations

from areno.models.base import CausalLMOutput, ModelAdapter


def register_models() -> None:
    """Register all bundled model adapters with the global registry."""

    from areno.models.registry import register_adapter
    from areno.models.bailing import BailingMoeLinearV2Adapter
    from areno.models.gemma4 import Gemma4Adapter
    from areno.models.llama import LlamaAdapter
    from areno.models.minicpmv46 import MiniCPMV46Adapter
    from areno.models.qwen3 import Qwen3Adapter, Qwen3MoeAdapter
    from areno.models.qwen3_5 import Qwen35Adapter, Qwen35MoeAdapter

    # Order here is not semantically meaningful; adapters are keyed by name
    # and matched against HF config JSON at runtime.
    register_adapter(LlamaAdapter())
    register_adapter(Qwen3Adapter())
    register_adapter(Qwen3MoeAdapter())
    register_adapter(Qwen35MoeAdapter())
    register_adapter(Qwen35Adapter())
    register_adapter(BailingMoeLinearV2Adapter())
    register_adapter(Gemma4Adapter())
    register_adapter(MiniCPMV46Adapter())


__all__ = ["CausalLMOutput", "ModelAdapter", "register_models"]
