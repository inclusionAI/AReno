"""Check native dispatch with metadata-only tensors; no device kernels run here."""

import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import areno.accel as accel
from areno.accel import _extension, ops
from areno.accel.utils import on_kernel_device
from tests.npu_stub import register_npu_device


class NativeReached(Exception):
    """Stop at the native boundary before any numerical work is performed."""


@pytest.fixture(scope="module", autouse=True)
def register_npu_device_name():
    register_npu_device()


@pytest.fixture
def native_tensors(monkeypatch):
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_initialized=lambda: False, is_available=lambda: False), raising=False
    )
    mode = FakeTensorMode()

    def tensor(shape, dtype=torch.float32, device="npu:0"):
        return FakeTensor(mode, torch.empty(shape, device="meta", dtype=dtype), torch.device(device))

    return mode, tensor


def test_extension_caches_cuda_and_npu_independently(monkeypatch):
    modules = {name: object() for name in ("areno.accel._areno_accel", "areno.accel._areno_accel_npu")}
    imports = []

    def load(name):
        if name == "torch_npu":
            return object()
        imports.append(name)
        return modules[name]

    monkeypatch.setattr(_extension, "_EXT", None)
    monkeypatch.setattr(_extension, "_NPU_EXT", None)
    monkeypatch.setattr(_extension.importlib, "import_module", load)
    for _ in range(2):
        assert _extension.extension(torch.device("npu:0")) is modules["areno.accel._areno_accel_npu"]
        assert _extension.extension(torch.device("cuda:1")) is modules["areno.accel._areno_accel"]
    assert imports == ["areno.accel._areno_accel_npu", "areno.accel._areno_accel"]


def test_missing_npu_extension_does_not_fall_back_to_cuda(monkeypatch):
    imports = []

    def load(name):
        if name == "torch_npu":
            return object()
        imports.append(name)
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(_extension, "_EXT", object())
    monkeypatch.setattr(_extension, "_NPU_EXT", None)
    monkeypatch.setattr(_extension.importlib, "import_module", load)
    with pytest.raises(RuntimeError, match="native NPU kernels are not installed"):
        _extension.extension("npu")
    assert imports == ["areno.accel._areno_accel_npu"]


def test_npu_extension_dependency_error_is_preserved(monkeypatch):

    def load(name):
        raise ModuleNotFoundError(name="torch_npu")

    monkeypatch.setattr(_extension, "_NPU_EXT", None)
    monkeypatch.setattr(_extension.importlib, "import_module", load)
    with pytest.raises(ModuleNotFoundError) as exc:
        _extension.extension("npu")
    assert exc.value.name == "torch_npu"


def test_device_guard_rejects_mixed_devices():
    def tensor(device):
        return SimpleNamespace(device=torch.device(device))

    assert on_kernel_device(tensor("npu:0"), tensor("npu:0"), None)
    assert not on_kernel_device(tensor("npu:0"), tensor("cuda:0"))
    assert not on_kernel_device(tensor("cuda:0"), tensor("cuda:1"))
    assert not on_kernel_device(tensor("npu:0"), tensor("cpu"))
    with pytest.raises(RuntimeError, match="require CUDA or NPU"):
        _extension.extension("cpu")


FORWARD_OPS = [
    "silu",
    "sigmoid",
    "softplus",
    "silu_and_mul",
    "gelu_tanh_and_mul",
    "linear_forward",
    "grouped_linear_forward",
    "grouped_linear_forward_counts",
    "rmsnorm_forward",
    "optional_scale_rmsnorm_forward",
    "rmsnorm_silu_gate_forward",
    "vocab_embedding_forward",
    "causal_attention_forward",
    "varlen_causal_attention_forward",
    "paged_causal_attention_decode_forward",
    "depthwise_causal_conv1d_silu_forward",
    "packed_depthwise_causal_conv1d_silu_forward",
    "depthwise_causal_conv1d_silu_decode",
    "topk_softmax_forward",
    "grouped_topk_router",
    "moe_permute_forward",
    "moe_topk_permute_forward",
    "moe_unpermute_forward",
    "moe_align",
]


def forward_calls(t):
    x, w = t((2, 4)), t((4, 4))
    i, counts = t((2, 2), torch.long), t((2,), torch.int32)
    q, kv = t((2, 2, 4)), t((2, 2, 4))
    cache, table = t((2, 4, 2, 4)), t((2, 1), torch.int32)
    conv_x, conv_w = t((1, 2, 4)), t((4, 1, 3))
    return {
        "silu": lambda: accel.areno_silu(x),
        "sigmoid": lambda: accel.areno_sigmoid(x),
        "softplus": lambda: accel.areno_softplus(x),
        "silu_and_mul": lambda: accel.areno_silu_and_mul(x),
        "gelu_tanh_and_mul": lambda: accel.areno_gelu_tanh_and_mul(x, out=t((2, 2))),
        "linear_forward": lambda: accel.areno_linear(x, w),
        "grouped_linear_forward": lambda: accel.areno_grouped_linear(x, t((2, 4, 4)), [1, 1]),
        "grouped_linear_forward_counts": lambda: accel.areno_grouped_linear(x, t((2, 4, 4)), counts),
        "rmsnorm_forward": lambda: accel.areno_rmsnorm(x, t((4,)), 1e-6),
        "optional_scale_rmsnorm_forward": lambda: accel.areno_optional_scale_rmsnorm(x, None, 1e-6),
        "rmsnorm_silu_gate_forward": lambda: accel.areno_rmsnorm_silu_gate(x, x, t((4,)), 1e-6),
        "vocab_embedding_forward": lambda: accel.areno_vocab_embedding(i, w, 0, 4),
        "causal_attention_forward": lambda: accel.areno_causal_attention(
            t((1, 2, 2, 4)), t((1, 2, 2, 4)), t((1, 2, 2, 4))
        ),
        "varlen_causal_attention_forward": lambda: accel.areno_varlen_causal_attention(q, kv, kv, counts),
        "paged_causal_attention_decode_forward": lambda: accel.areno_paged_causal_attention_decode(
            q, kv, kv, cache, cache, table, counts
        ),
        "depthwise_causal_conv1d_silu_forward": lambda: accel.areno_depthwise_causal_conv1d_silu(conv_x, conv_w),
        "packed_depthwise_causal_conv1d_silu_forward": lambda: accel.areno_packed_depthwise_causal_conv1d_silu(
            conv_x, conv_w, counts
        ),
        "depthwise_causal_conv1d_silu_decode": lambda: accel.areno_depthwise_causal_conv1d_silu_decode(
            x, t((2, 4, 2)), conv_w
        ),
        "topk_softmax_forward": lambda: accel.areno_topk_softmax(x, 2),
        "grouped_topk_router": lambda: accel.areno_grouped_topk_router(x, t((4,)), 2, 2, 1),
        "moe_permute_forward": lambda: accel.areno_moe_permute(x, w, t((2, 4), torch.bool), 2),
        "moe_topk_permute_forward": lambda: accel.areno_moe_topk_permute(x, i, t((2, 2)), 0, 4),
        "moe_unpermute_forward": lambda: accel.areno_moe_unpermute(x, i, (2, 4)),
        "moe_align": lambda: accel.areno_moe_align(i, 4, 16, counts, counts, counts, counts),
    }


@pytest.mark.parametrize("device", ["cuda:0", "npu:0"])
@pytest.mark.parametrize("op", FORWARD_OPS)
def test_public_operators_select_the_input_device(monkeypatch, native_tensors, device, op):
    mode, make_tensor = native_tensors

    class Native:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                raise NativeReached(name)

            return call

    native = Native()
    monkeypatch.setattr(_extension, "_EXT", native if device.startswith("cuda") else None)
    monkeypatch.setattr(_extension, "_NPU_EXT", native if device.startswith("npu") else None)
    if device.startswith("npu") and "attention" in op:
        from areno.accel.npu import attention

        entry = op.removesuffix("_forward")
        monkeypatch.setattr(attention, entry, getattr(native, f"areno_{op}"))
    calls = forward_calls(lambda shape, dtype=torch.float32: make_tensor(shape, dtype, device))
    with mode, pytest.raises(NativeReached, match=f"^areno_{op}$"):
        calls[op]()


@pytest.mark.parametrize("device", ["cuda:0", "npu:0"])
@pytest.mark.parametrize("activation", ["Silu", "Sigmoid", "Softplus", "SiluMul", "GeluTanhMul"])
def test_activation_backward_selects_the_gradient_device(monkeypatch, native_tensors, device, activation):
    mode, tensor = native_tensors
    x = tensor((2, 4), device=device)
    grad = tensor((2, 2) if activation.endswith("Mul") else (2, 4), device=device)
    module = importlib.import_module("areno.accel.activations")
    seen = []

    class Native:
        def __getattr__(self, name):
            def call(*args):
                seen.append(name)
                return x

            return call

    monkeypatch.setattr(_extension, "_EXT", Native() if device.startswith("cuda") else None)
    monkeypatch.setattr(_extension, "_NPU_EXT", Native() if device.startswith("npu") else None)
    with mode:
        result = getattr(module, f"_{activation}").backward(SimpleNamespace(saved_tensors=(x,)), grad)
    assert result[0].device == x.device
    assert len(seen) == 1 and seen[0].startswith("areno_d_")


@pytest.mark.parametrize("device", ["cuda:0", "npu:0"])
@pytest.mark.parametrize(
    "variant", ["fp32_master", "fp32_state", "8bit", "4bit", "4bit_factored_stats", "4bit_factored"]
)
def test_all_adam_native_entries_select_the_parameter_device(monkeypatch, native_tensors, device, variant):
    from areno.accel import optimizer

    mode, make_tensor = native_tensors

    def t(size, dtype=torch.float32):
        return make_tensor((size,), dtype, device)

    model, grad = t(4, torch.bfloat16), t(4, torch.bfloat16)
    kwargs = dict(
        beta1=0.9, beta2=0.99, effective_lr=0.01, weight_decay=0.1, eps=1e-8, step_size=0.1, bias_correction2_sqrt=0.1
    )
    if variant == "fp32_master":
        args = (model, t(4, torch.uint16), t(1, torch.uint8), grad, t(4), t(4))
        kwargs["state_offset"] = 0
    elif variant == "fp32_state":
        args = (model, grad, t(4), t(4))
    elif variant == "8bit":
        args = (model, grad, t(4, torch.uint8), t(1), t(4, torch.uint8), t(1), t(256), t(256))
        kwargs["block_size"] = 256
    elif variant == "4bit":
        args = (model, grad, t(2, torch.uint8), t(1), t(2, torch.uint8), t(1))
        kwargs.update(
            moment_packed_offset=0,
            moment_scale_offset=0,
            variance_packed_offset=0,
            variance_scale_offset=0,
            quant_block_size=32,
        )
    elif variant == "4bit_factored_stats":
        args = (grad, t(4), t(1, torch.int32))
        kwargs = dict(parameter_shard_start=0, rows=2, columns=2)
    else:
        args = (model, grad, t(2, torch.uint8), t(1), t(4), t(1), t(1, torch.int32))
        kwargs.pop("beta2")
        kwargs.update(
            moment_packed_offset=0,
            moment_scale_offset=0,
            parameter_shard_start=0,
            quant_block_size=32,
            rows=2,
            columns=2,
        )
    name = f"areno_adamw_{variant}" + ("" if variant.endswith("stats") else "_step")

    def call(*args):
        raise NativeReached(name)

    native = SimpleNamespace(**{name: call})
    monkeypatch.setattr(_extension, "_EXT", native if device.startswith("cuda") else None)
    monkeypatch.setattr(_extension, "_NPU_EXT", native if device.startswith("npu") else None)
    with mode, pytest.raises(NativeReached, match=f"^{name}$"):
        getattr(optimizer, name)(*args, **kwargs)


def test_npu_training_attention_uses_the_shared_native_wrapper(monkeypatch, native_tensors):
    from areno.accel.npu import attention
    from areno.engine.layers.attention_backend.train import FlashAttnTrainAttentionBackend

    mode, tensor = native_tensors
    q = tensor((1, 2, 2, 4))

    def call(*args, force_native):
        assert force_native
        raise NativeReached("areno_varlen_causal_attention_forward")

    monkeypatch.setattr(attention, "varlen_causal_attention", call)
    backend = FlashAttnTrainAttentionBackend("native")
    with mode, pytest.raises(NativeReached, match="areno_varlen_causal_attention_forward"):
        backend(q, q, q, None)


@pytest.mark.parametrize("mode_name", ["packed", "train", "prefill", "decode"])
@pytest.mark.parametrize(
    "dtype, dim, block_size",
    [
        (torch.bfloat16, 128, 256),
        (torch.float16, 256, 256),
        (torch.float32, 64, 256),
        (torch.bfloat16, 512, 256),
        (torch.float16, 31, 256),
        (torch.bfloat16, 128, 4),
    ],
)
def test_npu_attention_selects_library_or_native_by_shape(
    monkeypatch, native_tensors, mode_name, dtype, dim, block_size
):
    from areno.accel.npu import attention
    from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend
    from areno.engine.layers.attention_backend.train import FlashAttnTrainAttentionBackend

    mode, tensor = native_tensors

    class Provider:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, entry):
            def call(*args, **kwargs):
                raise NativeReached(f"{self.name}:{entry}")

            return call

    monkeypatch.setattr(_extension, "_NPU_EXT", Provider("native"))
    monkeypatch.setattr(attention, "_UNAVAILABLE_FLASH_DEVICES", set())
    monkeypatch.setattr(attention, "flash_attention", lambda device: Provider("flash"))
    loader = importlib.import_module("areno.accel.flash_attention")
    monkeypatch.setattr(loader, "flash_attention", lambda device: Provider("flash"))
    native = dtype == torch.float32 or dim > 256
    if mode_name == "decode":
        native |= bool(dim % 8 or block_size % 256)
    expected_provider = "native" if native else "flash"
    entry = {"decode": "paged_causal_attention_decode"}.get(mode_name, "varlen_causal_attention")
    expected_entry = (
        f"areno_{entry}_forward"
        if native
        else {"decode": "flash_attn_with_kvcache", "train": "flash_attn_func"}.get(mode_name, "flash_attn_varlen_func")
    )
    q = tensor((1, 1 if mode_name == "decode" else 2, 2, dim), dtype)
    cu = tensor((2,), torch.int32)
    cache = tensor((2, block_size, 2, dim), dtype)
    meta = SimpleNamespace(
        mode=mode_name,
        cu_seqlens=cu,
        max_seqlen=2,
        block_table=tensor((1, 2), torch.int32),
        cache_seqlens=tensor((1,), torch.int32),
    )
    with mode, pytest.raises(NativeReached, match=f"^{expected_provider}:{expected_entry}$"):
        if mode_name == "packed":
            flat = q.reshape(2, 2, dim)
            accel.areno_varlen_causal_attention(flat, flat, flat, cu)
        elif mode_name == "train":
            FlashAttnTrainAttentionBackend("flash")(q, q, q, None)
        else:
            FlashAttnInferBackend("flash")(q, q, q, cache, cache, meta, update_cache=mode_name == "decode")


def test_accel_metadata_imports_do_not_require_triton_or_device_extensions():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import areno.accel; import areno.accel.ops; import areno.accel.kda; import sys; "
            "assert 'triton' not in sys.modules; "
            "assert 'areno.accel._areno_accel' not in sys.modules; "
            "assert 'areno.accel._areno_accel_npu' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("selection", ["native", "unsupported_device", "missing_package"])
def test_npu_native_selection_covers_train_prefill_and_decode(monkeypatch, native_tensors, selection):
    import warnings

    from areno.accel.npu import attention
    from areno.engine.layers.attention_backend.infer import FlashAttnInferBackend
    from areno.engine.layers.attention_backend.train import FlashAttnTrainAttentionBackend

    mode, tensor = native_tensors
    loader = importlib.import_module("areno.accel.flash_attention")
    imports = []
    entries = []
    cache = tensor((2, 256, 1, 64), torch.bfloat16)
    cu = tensor((2,), torch.int32)
    table = tensor((1, 2), torch.int32)
    lengths = tensor((1,), torch.int32)

    def load(name):
        imports.append(name)
        if selection == "native":
            pytest.fail("explicit native must not import FlashAttention")
        if selection == "missing_package":
            raise ModuleNotFoundError(name=name)
        raise RuntimeError("Unsupported Ascend device: Ascend910_9382")

    class Native:
        def __getattr__(self, entry):
            def call(*args):
                entries.append(entry)
                if "paged" in entry:
                    assert args[3] is cache and args[4] is cache
                    assert args[5] is table and args[6] is lengths
                raise NativeReached(entry)

            return call

    monkeypatch.setattr(loader, "import_module", load)
    monkeypatch.setattr(attention, "_UNAVAILABLE_FLASH_DEVICES", set())
    monkeypatch.setattr(_extension, "_NPU_EXT", Native())
    backend = "native" if selection == "native" else "flash"
    with mode, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for phase in ("train", "prefill", "decode", "decode"):
            q = tensor((1, 1 if phase == "decode" else 2, 2, 64), torch.bfloat16)
            kv = tensor((*q.shape[:2], 1, 64), torch.bfloat16)
            meta = SimpleNamespace(
                mode=phase, cu_seqlens=cu, max_seqlen=2, block_table=table, cache_seqlens=lengths
            )
            with pytest.raises(NativeReached):
                if phase == "train":
                    FlashAttnTrainAttentionBackend(backend)(q, kv, kv, meta)
                else:
                    FlashAttnInferBackend(backend)(q, kv, kv, cache, cache, meta, update_cache=phase == "decode")
    assert entries == ["areno_varlen_causal_attention_forward"] * 2 + [
        "areno_paged_causal_attention_decode_forward"
    ] * 2
    assert imports == ([] if selection == "native" else ["flash_attn_npu"])
    fallback_warnings = [w for w in caught if "falling back to attn_backend='native'" in str(w.message)]
    assert len(fallback_warnings) == (0 if selection == "native" else 1)


def test_npu_triton_equivalents_route_to_their_ascend_providers(monkeypatch, native_tensors):
    mode, tensor = native_tensors
    x = tensor((2, 2, 4))
    calls = []

    class Native:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                calls.append(name)
                return x

            return call

    monkeypatch.setattr(_extension, "_NPU_EXT", Native())
    monkeypatch.setitem(sys.modules, "areno.accel.npu.seg_la", Native())
    meta = ops.SegLaMeta(2, 2, x, x, x, x)
    with mode:
        ops.rms_norm_gate_fwd(x, x, x, 1e-6)
        ops.areno_fused_experts(x, x, x, x, x, ops.FusedMoeConfig(2, 4, 8, 2))
        ops.seg_la_fwd(x, x, x, x, x, meta)
    assert calls == ["rms_norm_gate_fwd", "areno_fused_experts", "seg_la_fwd"]


@pytest.mark.parametrize("device", ["cuda:0", "npu:0"])
@pytest.mark.parametrize("variant", ["chunk", "recurrent_update"])
def test_kda_keeps_the_same_native_arguments(monkeypatch, native_tensors, device, variant):
    from areno.accel import kda

    mode, tensor = native_tensors
    x = tensor((1, 2, 2, 4), device=device)
    state = tensor((1, 2, 4, 4), device=device)
    indices = tensor((1,), torch.int32, device)
    cu_seqlens = tensor((2,), torch.int32, device)
    called = {}
    output = (x, state) if variant == "chunk" else x

    def kernel(**kwargs):
        called.update(kwargs)
        return output

    name = "chunk_kda" if variant == "chunk" else "fused_sigmoid_gating_delta_rule_update"
    native = SimpleNamespace(**{name: kernel})
    if device.startswith("npu"):
        monkeypatch.setitem(sys.modules, "areno.accel.npu.kda", native)
    else:
        module = "kda" if variant == "chunk" else "fused_sigmoid_gating_recurrent"
        monkeypatch.setitem(sys.modules, f"areno.accel.kernels.kda_fla.{module}", native)
    kwargs = dict(state_indices=indices, scale=0.5, cu_seqlens=cu_seqlens, a_log=x, dt_bias=x, lower_bound=-6.0)
    if variant == "chunk":
        kwargs.update(initial_state=state, output_final_state=True)
    else:
        kwargs["state"] = state
    with mode:
        result = getattr(kda, f"areno_kda_{variant}")(x, x, x, x, x, **kwargs)
    assert result is output
    assert called["q"] is x and called["A_log"] is x
    assert called["initial_state_indices"] is indices
    assert called["cu_seqlens"] is cu_seqlens
    assert called["lower_bound"] == -6.0
    assert called["use_qk_l2norm_in_kernel"] is True
