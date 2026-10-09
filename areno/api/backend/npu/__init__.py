"""Lazy NPU backend export, symmetric with CUDA and MLX."""


def __getattr__(name):
    if name == "NpuBackend":
        from areno.api.backend.npu.backend import NpuBackend

        return NpuBackend
    raise AttributeError(name)
