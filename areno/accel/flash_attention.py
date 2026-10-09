"""Lazy imports for the platform's FlashAttention package."""

from importlib import import_module


class FlashAttentionUnavailable(RuntimeError):
    """The optional platform library is absent or rejects this device."""


def flash_attention(device):
    name = "flash_attn_npu" if device.type == "npu" else "flash_attn"
    try:
        return import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name or device.type != "npu":
            raise
        raise FlashAttentionUnavailable(
            "flash-attn-npu is not installed. Install it in the existing "
            "CANN/TorchNPU environment with `python -m pip install flash-attn-npu==0.3.0 "
            "--no-deps --no-build-isolation`."
        ) from exc
    except RuntimeError as exc:
        if device.type != "npu" or not str(exc).startswith("Unsupported Ascend device:"):
            raise
        raise FlashAttentionUnavailable(str(exc)) from exc
