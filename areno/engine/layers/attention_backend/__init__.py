"""FlashAttention backends used by attention layers.

Backend exports are lazy because importing the package itself should not load
flash-attn in CPU-only tests or tooling.
"""


def __getattr__(name: str):
    if name in {"FlashAttnInferBackend", "build_infer_attention_backend"}:
        from areno.engine.layers.attention_backend import infer

        return getattr(infer, name)
    if name in {"TrainAttentionBackend", "build_train_attention_backend"}:
        from areno.engine.layers.attention_backend import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FlashAttnInferBackend",
    "TrainAttentionBackend",
    "build_infer_attention_backend",
    "build_train_attention_backend",
]
