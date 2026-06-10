"""Generic Llama-family plugin.

MiniCPM5-1B publishes a standard ``LlamaForCausalLM`` text config, so this
adapter intentionally reuses the same dense decoder blocks as Qwen3 while
disabling Qwen-specific Q/K RMSNorm.
"""

from __future__ import annotations

from areno.models.llama.model import LlamaAdapter
from areno.models.qwen3.model import Qwen3ForCausalLM as LlamaForCausalLM

__all__ = ["LlamaAdapter", "LlamaForCausalLM"]
