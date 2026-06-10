"""Engine package public surface.

Keep package import light: many CPU-only helpers live under ``areno.engine``
and should not require CUDA-only dependencies such as flash-attn just because
the package was imported.
"""


def __getattr__(name: str):
    if name == "ArenoEngine":
        from areno.engine.api import ArenoEngine

        return ArenoEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ArenoEngine"]
