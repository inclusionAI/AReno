"""Qwen3 plugin.

Re-exports the adapter and the causal LM module so the registry can pick them
up. The implementation is intentionally thin and reuses the generic
``CausalSelfAttention`` / ``GatedMLP`` building blocks because Qwen3 is a
standard GQA decoder with no architecture peculiarities beyond optional Q/K
RMSNorm.
"""

from __future__ import annotations

from areno.models.qwen3.model import Qwen3Adapter, Qwen3ForCausalLM, Qwen3MoeAdapter, Qwen3MoeForCausalLM

__all__ = ["Qwen3Adapter", "Qwen3ForCausalLM", "Qwen3MoeAdapter", "Qwen3MoeForCausalLM"]
