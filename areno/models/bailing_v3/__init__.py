"""Bailing MoE V3 plugin.

Re-exports the adapter and causal LM module so they can be registered without
forcing callers to know the submodule layout. See ``model.py`` for the
architecture details (hybrid softmax + lightning linear attention, grouped
top-k sigmoid router, optional shared experts).
"""

from __future__ import annotations

from areno.models.bailing_v3.model import BailingMoeV3Adapter, BailingMoeV3ForCausalLM

__all__ = ["BailingMoeV3Adapter", "BailingMoeV3ForCausalLM"]
