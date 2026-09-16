"""Lazy HPU backend export, symmetric with CUDA and MLX."""


def __getattr__(name):
    if name == "HpuBackend":
        from areno.api.backend.hpu.backend import HpuBackend

        return HpuBackend
    raise AttributeError(name)
