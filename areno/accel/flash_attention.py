"""Lazy imports for the platform's FlashAttention package."""

from importlib import import_module


def flash_attention(device):
    name = "flash_attn_npu" if device.type == "npu" else "flash_attn"
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name or device.type != "npu":
            raise
        raise RuntimeError(
            "Ascend attention requires flash-attn-npu. Install it in the existing "
            "CANN/TorchNPU environment with `python -m pip install flash-attn-npu==0.3.0 "
            "--no-deps --no-build-isolation`."
        ) from exc
