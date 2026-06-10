"""Bailing MoE Linear v2 plugin.

Re-exports the adapter and causal LM module so they can be registered without
forcing callers to know the submodule layout. See ``model.py`` for the
architecture details (hybrid softmax + lightning linear attention, grouped
top-k sigmoid router, optional shared experts).
"""

from __future__ import annotations

from areno.models.bailing.model import BailingMoeLinearV2Adapter, BailingMoeLinearV2ForCausalLM

__all__ = ["BailingMoeLinearV2Adapter", "BailingMoeLinearV2ForCausalLM"]
