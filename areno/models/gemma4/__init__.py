"""Gemma4 plugin.

Re-exports the adapter and causal LM for registration. The model implements
Google Gemma4 / Gemma4 Unified text trunks, KV-shared tail layers, optional
per-layer input embeddings (PLE), interleaved full/sliding attention, and
per-layer-type RoPE settings.
"""

from __future__ import annotations

from areno.models.gemma4.model import Gemma4Adapter, Gemma4ForCausalLM

__all__ = ["Gemma4Adapter", "Gemma4ForCausalLM"]
