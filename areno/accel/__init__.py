"""Public entry point for the ARENO acceleration shims.

Re-exports thin Python wrappers around the ``areno.accel._areno_accel`` C++/CUDA
extension. Each submodule defines a ``torch.autograd.Function`` (where backward
is needed) and a ``@torch._dynamo.disable``-decorated user-facing function that
performs argument validation, dispatches to the fused CUDA kernel and exposes a
PyTorch-friendly signature. The kernels themselves live in ``csrc/`` and are
built by ``setup.py``; if the extension is not built the first call into the
shim raises ``ModuleNotFoundError`` at import time.
"""
from areno.accel.activations import areno_gelu_tanh_and_mul, areno_sigmoid, areno_silu, areno_silu_and_mul, areno_softplus
from areno.accel.conv import areno_depthwise_causal_conv1d_silu, areno_depthwise_causal_conv1d_silu_decode, areno_packed_depthwise_causal_conv1d_silu
from areno.accel.embedding import areno_vocab_embedding
from areno.accel.linear import areno_grouped_linear, areno_linear
from areno.accel.moe import areno_moe_permute, areno_moe_topk_permute, areno_moe_unpermute
from areno.accel.normalization import areno_optional_scale_rmsnorm, areno_rmsnorm, areno_rmsnorm_silu_gate
from areno.accel.router import areno_grouped_topk_router
from areno.accel.routing import areno_moe_align
from areno.accel.topk import areno_topk_softmax

__all__ = [
    "areno_depthwise_causal_conv1d_silu",
    "areno_depthwise_causal_conv1d_silu_decode",
    "areno_packed_depthwise_causal_conv1d_silu",
    "areno_gelu_tanh_and_mul",
    "areno_grouped_topk_router",
    "areno_grouped_linear",
    "areno_linear",
    "areno_moe_align",
    "areno_moe_permute",
    "areno_moe_topk_permute",
    "areno_moe_unpermute",
    "areno_optional_scale_rmsnorm",
    "areno_rmsnorm",
    "areno_rmsnorm_silu_gate",
    "areno_sigmoid",
    "areno_silu",
    "areno_silu_and_mul",
    "areno_softplus",
    "areno_topk_softmax",
    "areno_vocab_embedding",
]
