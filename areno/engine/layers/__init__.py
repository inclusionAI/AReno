"""Public layer re-exports.

Layer submodules are imported lazily so CPU-only utilities can import
``areno.engine.layers.linear`` without also importing attention kernels.
"""


def __getattr__(name: str):
    if name == "CausalSelfAttention":
        from areno.engine.layers.attention import CausalSelfAttention

        return CausalSelfAttention
    if name == "GatedMLP":
        from areno.engine.layers.mlp import GatedMLP

        return GatedMLP
    if name == "RMSNorm":
        from areno.engine.layers.norm import RMSNorm

        return RMSNorm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CausalSelfAttention", "GatedMLP", "RMSNorm"]
