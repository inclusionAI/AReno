"""Qwen3.5 plugin."""

from areno.models.qwen3_5.model import (
    Qwen35Adapter,
    Qwen35ForCausalLM,
    Qwen35MoeAdapter,
    Qwen35MoeForCausalLM,
    Qwen35MoeVLAdapter,
    Qwen35MoeVLForConditionalGeneration,
    Qwen35VLAdapter,
    Qwen35VLForConditionalGeneration,
)

__all__ = [
    "Qwen35Adapter",
    "Qwen35ForCausalLM",
    "Qwen35MoeAdapter",
    "Qwen35MoeForCausalLM",
    "Qwen35MoeVLAdapter",
    "Qwen35MoeVLForConditionalGeneration",
    "Qwen35VLAdapter",
    "Qwen35VLForConditionalGeneration",
]
