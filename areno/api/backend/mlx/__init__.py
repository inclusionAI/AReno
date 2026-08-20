"""MLX backend package with lazy runtime imports."""


def __getattr__(name: str):
    if name == "MlxBackend":
        from areno.api.backend.mlx.backend import MlxBackend

        return MlxBackend
    raise AttributeError(name)


__all__ = ["MlxBackend"]
