"""Model adapter API and bundled areno-native model plugins."""

from __future__ import annotations

from areno.models.base import CausalLMOutput, ModelAdapter

_REGISTERED_GROUPS: set[str] = set()


def _register_qwen3() -> None:
    from areno.models.qwen3 import Qwen3Adapter, Qwen3MoeAdapter
    from areno.models.registry import register_adapter

    register_adapter(Qwen3Adapter())
    register_adapter(Qwen3MoeAdapter())


def _register_qwen35() -> None:
    from areno.models.qwen3_5 import Qwen35Adapter, Qwen35MoeAdapter, Qwen35MoeVLAdapter, Qwen35VLAdapter
    from areno.models.registry import register_adapter

    register_adapter(Qwen35MoeVLAdapter())
    register_adapter(Qwen35MoeAdapter())
    register_adapter(Qwen35VLAdapter())
    register_adapter(Qwen35Adapter())


def _register_bailing() -> None:
    from areno.models.bailing import BailingMoeLinearV2Adapter
    from areno.models.registry import register_adapter

    register_adapter(BailingMoeLinearV2Adapter())


def _register_bailing_v3() -> None:
    from areno.models.bailing_v3 import BailingMoeV3Adapter
    from areno.models.registry import register_adapter

    register_adapter(BailingMoeV3Adapter())


def _register_llama() -> None:
    from areno.models.llama import LlamaAdapter
    from areno.models.registry import register_adapter

    register_adapter(LlamaAdapter())


def _register_gemma4() -> None:
    from areno.models.gemma4 import Gemma4Adapter
    from areno.models.registry import register_adapter

    register_adapter(Gemma4Adapter())


def _register_minicpmv46() -> None:
    from areno.models.minicpmv46 import MiniCPMV46Adapter
    from areno.models.registry import register_adapter

    register_adapter(MiniCPMV46Adapter())


def _register_olmo2() -> None:
    from areno.models.olmo2 import Olmo2Adapter
    from areno.models.registry import register_adapter

    register_adapter(Olmo2Adapter())


_GROUPS = {
    "llama": _register_llama,
    "qwen3": _register_qwen3,
    "qwen3_5": _register_qwen35,
    "bailing": _register_bailing,
    "bailing_v3": _register_bailing_v3,
    "gemma4": _register_gemma4,
    "minicpmv46": _register_minicpmv46,
    "olmo2": _register_olmo2,
}
_MODEL_GROUPS = {
    "bailing_moe_linear": "bailing",
    "bailing_moe_linear_v2": "bailing",
    "bailing_hybrid": "bailing_v3",
    "bailing_moe_v3": "bailing_v3",
    "gemma4_unified": "gemma4",
    "minicpmv4_6": "minicpmv46",
    "qwen3_moe": "qwen3",
    "qwen3_5_moe": "qwen3_5",
    "qwen3_5_vl": "qwen3_5",
    "qwen3_5_vision": "qwen3_5",
    "qwen3_5_vl_moe": "qwen3_5",
    "qwen3_5_moe_vl": "qwen3_5",
}


def register_models(model_type: str | None = None) -> bool:
    """Register one matching bundled adapter group, or all groups.

    Bailing imports FLA, whose module-level decorators can initialize
    ``torch.compile``. Keeping it out of a Qwen process avoids unrelated
    Inductor compilation during Qwen startup.
    """

    key = str(model_type).lower() if model_type is not None else None
    group = _MODEL_GROUPS.get(key, key)
    groups = tuple(_GROUPS) if group is None else (group,)
    if any(name not in _GROUPS for name in groups):
        return False
    for name in groups:
        if name not in _REGISTERED_GROUPS:
            _GROUPS[name]()
            _REGISTERED_GROUPS.add(name)
    return True


__all__ = ["CausalLMOutput", "ModelAdapter", "register_models"]
