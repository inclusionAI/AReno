"""CUDA kernels and real-model integration for saved-activation compression."""

import gc

import pytest
import torch

from areno.engine.config import ModelConfig
from areno.engine.layers.attention_backend.train import deterministic_flash_backward
from areno.engine.parallel.context import TPContext, get_tp_context, set_tp_context
from areno.engine.runtime.activation_compression import (
    UNCOMPRESSED,
    ActivationCompressor,
    dequantize,
    quantize,
    relative_error,
)
from areno.engine.runtime.logprobs import packed_next_token_logprobs_from_hidden
from areno.engine.runtime.metadata import TrainMeta
from areno.models.registry import build_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="activation compression kernels require CUDA")


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("bits", [8, 4, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(3, 64), (5, 70), (4097, 1088), (7, 3, 8192)])
def test_kernel_matches_torch_path_bit_for_bit_cuda(stochastic, bits, dtype, shape):
    generator = torch.Generator(device="cuda").manual_seed(bits)
    x = torch.randn(shape, device="cuda", generator=generator) * 3 + 0.5
    x[..., 5] *= 300
    x = x.to(dtype)
    kernel = quantize(x, bits, stochastic=stochastic, seed=99, use_kernel=True)
    reference = quantize(x, bits, stochastic=stochastic, seed=99, use_kernel=False)
    for name in ("codes", "minimum", "scale", "outlier_index", "outliers"):
        assert torch.equal(getattr(kernel, name), getattr(reference, name)), name
    assert torch.equal(dequantize(kernel, use_kernel=True), dequantize(reference, use_kernel=False))


def _tiny_qwen3(attn_backend: str):
    config = ModelConfig(
        model_type="qwen3",
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        qk_norm=True,
        dtype=torch.bfloat16,
        sequence_parallel=False,
        attn_backend=attn_backend,
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = build_model(config).cuda().train()
    finally:
        torch.set_default_dtype(previous)
    torch.manual_seed(5)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name and param.ndim == 1:
                param.fill_(1.0)
            else:
                param.normal_(0, 0.02)
    return model


def _attn_backends():
    # Evaluated at collection, so it must not touch CUDA on hosts without GPUs.
    backends = ["native"]
    if not torch.cuda.is_available():
        return backends
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return backends
    if torch.cuda.get_device_capability() >= (8, 0):
        backends.append("flash")
    return backends


@pytest.fixture(autouse=True)
def single_rank():
    previous = get_tp_context()
    set_tp_context(TPContext(0, 1, torch.device("cuda", 0), None))
    yield
    set_tp_context(previous)


def _gradients(model, compressor, plan, *, checkpointing, mst_chunk=256):
    tokens = torch.randint(512, (1, 1200), device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
    cu = torch.tensor([0, 1, 500, 1200], device="cuda", dtype=torch.int32)
    meta = TrainMeta(
        cu_seqlens=cu, max_seqlen=700, packed=True, activation_checkpointing=checkpointing, mst_chunk_size=mst_chunk
    )
    model.zero_grad(set_to_none=True)
    with compressor.capture(plan):
        out = model(tokens, train_meta=meta, defer_lm_head=True)
    logps = packed_next_token_logprobs_from_hidden(out.hidden_states, tokens, cu, model.lm_head, chunk_size=256)
    upstream = torch.linspace(-1, 1, logps.numel(), device="cuda")
    (logps * upstream).mean().backward()
    return [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]


@pytest.mark.parametrize("attn_backend", _attn_backends())
@pytest.mark.parametrize("checkpointing", [True, False])
def test_real_model_backward_with_compressed_slots_cuda(attn_backend, checkpointing):
    model = _tiny_qwen3(attn_backend)
    compressor = ActivationCompressor(model)
    with deterministic_flash_backward():
        exact = _gradients(model, compressor, None, checkpointing=checkpointing)
        record = compressor.last_capture
        slots = len(record.signatures)
        if checkpointing:
            # One boundary per decoder layer plus the final norm input.
            assert slots == model.config.num_hidden_layers + 1
        else:
            assert slots > 2 * model.config.num_hidden_layers
        compressor.signatures = list(record.signatures)
        again = _gradients(model, compressor, [UNCOMPRESSED] * slots, checkpointing=checkpointing)
        eight = _gradients(model, compressor, [8] * slots, checkpointing=checkpointing)
        four = _gradients(model, compressor, [4] * slots, checkpointing=checkpointing)
    stats = compressor.last_capture.stats
    assert stats.compressed_slots == slots and stats.plan_misses == 0
    assert stats.stored_bytes < stats.saved_bytes
    assert relative_error(again, exact) < 2e-3
    assert relative_error(eight, exact) < 0.05
    assert relative_error(eight, exact) < relative_error(four, exact)


def test_compressed_forward_saves_less_and_backward_releases_everything_cuda():
    model = _tiny_qwen3("native")
    compressor = ActivationCompressor(model)
    tokens = torch.randint(512, (1, 4096), device="cuda")
    cu = torch.tensor([0, 2048, 4096], device="cuda", dtype=torch.int32)

    def forward_memory(plan):
        meta = TrainMeta(cu_seqlens=cu, max_seqlen=2048, packed=True, activation_checkpointing=True, mst_chunk_size=256)
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with compressor.capture(plan):
            out = model(tokens, train_meta=meta, defer_lm_head=True)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before
        out.hidden_states.float().square().mean().backward()
        del out
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.synchronize()
        return held, torch.cuda.memory_allocated() - before

    forward_memory(None)  # allocate lazy library workspaces first
    exact_held, exact_left = forward_memory(None)
    compressor.signatures = list(compressor.last_capture.signatures)
    slots = len(compressor.signatures)
    for _ in range(2):  # the first compressed pass also compiles the kernels
        held, left = forward_memory([8] * slots)
    stats = compressor.last_capture.stats
    saved, stored = stats.saved_bytes, stats.stored_bytes
    assert stored < 0.6 * saved
    # Live forward memory falls by the saved-activation bytes compression removed.
    assert held <= exact_held - 0.9 * (saved - stored)
    assert left == exact_left == 0


@pytest.mark.skipif("flash" not in _attn_backends(), reason="flash-attn is not available")
def test_deterministic_flash_backward_repeats_gradients_exactly_cuda():
    model = _tiny_qwen3("flash")
    compressor = ActivationCompressor(model)
    with deterministic_flash_backward():
        first = _gradients(model, compressor, None, checkpointing=True)
        second = _gradients(model, compressor, None, checkpointing=True)
    # Embedding and RMSNorm weight gradients still accumulate with atomics;
    # they are leaves, so every projection gradient must repeat exactly.
    nondeterministic = {i for i, (name, _) in enumerate(model.named_parameters()) if "embed" in name or "norm" in name}
    for index, (a, b) in enumerate(zip(first, second)):
        if index not in nondeterministic and a is not None:
            assert torch.equal(a, b)
