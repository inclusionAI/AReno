"""OLMo 2 model adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from areno.models.olmo2.model import Olmo2Adapter, Olmo2ForCausalLM


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from areno.models.olmo2 import model

    return getattr(model, name)


__all__ = ["Olmo2Adapter", "Olmo2ForCausalLM"]
