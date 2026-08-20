"""CUDA backend package with lazy runtime imports."""


def __getattr__(name: str):
    if name == "CudaBackend":
        from areno.api.backend.cuda.backend import CudaBackend

        return CudaBackend
    raise AttributeError(name)


__all__ = ["CudaBackend"]
