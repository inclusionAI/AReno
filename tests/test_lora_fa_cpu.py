"""LoRA-FA (frozen A) adapter semantics, gradient correction, and PEFT metadata round-trip."""

from __future__ import annotations

import json
import multiprocessing as mp
import socket

import pytest
import torch
import torch.distributed as dist
from torch import nn

from areno.adapters import LoraConfig
from areno.adapters.config import LORA_FA_CONFIG_NAME
from areno.adapters.lora import AdapterRegistry, LoraSlot, RoutedExpertLoraSlot, _AdapterRuntimeState
from areno.adapters.peft import export_peft_adapter
from areno.engine.parallel.context import TPContext, set_tp_context


def _set_cpu_tp() -> None:
    set_tp_context(TPContext(rank=0, world_size=1, device=torch.device("cpu"), group=None))


def _dense_registry(config: LoraConfig, *, base_dtype: torch.dtype = torch.float32) -> AdapterRegistry:
    _set_cpu_tp()
    state = _AdapterRuntimeState()
    slot = LoraSlot(
        logical_name="layers.0.self_attn.q_proj",
        base_weight=nn.Parameter(torch.zeros(1, dtype=base_dtype)),
        global_in_features=16,
        global_out_features=12,
        local_in_features=16,
        local_out_features=12,
        row_parallel=False,
        config=config,
        seed=1,
        runtime_state=state,
    )
    with torch.no_grad():
        slot.lora_B.normal_(generator=torch.Generator().manual_seed(2))
    return AdapterRegistry({slot.logical_name: slot}, config, state)


def _slot(registry: AdapterRegistry) -> LoraSlot:
    return next(iter(registry.slots.values()))


def _backward(registry: AdapterRegistry, x: torch.Tensor, grad_out: torch.Tensor) -> None:
    _slot(registry)(x).backward(grad_out)


def test_lora_fa_requires_lora_fa_for_grad_correction() -> None:
    with pytest.raises(ValueError, match="lora_fa_grad_correction requires lora_fa"):
        LoraConfig(lora_fa_grad_correction=True)


def test_lora_fa_freezes_a_and_matches_lora_b_and_input_gradients() -> None:
    lora = _dense_registry(LoraConfig(rank=4, alpha=8.0, target_modules=("q_proj",)))
    lora_fa = _dense_registry(LoraConfig(rank=4, alpha=8.0, target_modules=("q_proj",), lora_fa=True))
    torch.testing.assert_close(_slot(lora_fa).lora_A, _slot(lora).lora_A)
    assert [p.shape for p in lora.parameters()] == [(4, 16), (12, 4)]
    assert lora_fa.parameters() == (_slot(lora_fa).lora_B,)
    assert len(dict(lora_fa.named_parameters())) == 2

    x = torch.randn(5, 16, generator=torch.Generator().manual_seed(3))
    grad_out = torch.randn(5, 12, generator=torch.Generator().manual_seed(4))
    inputs = []
    for registry in (lora, lora_fa):
        inputs.append(x.clone().requires_grad_(True))
        _backward(registry, inputs[-1], grad_out)
    assert _slot(lora_fa).lora_A.grad is None
    torch.testing.assert_close(_slot(lora_fa).lora_B.grad, _slot(lora).lora_B.grad)
    torch.testing.assert_close(inputs[1].grad, inputs[0].grad)
    # Reference dB = scale * dY^T (x A^T) for the [out, r] layout.
    slot = _slot(lora_fa)
    torch.testing.assert_close(slot.lora_B.grad, slot.scale * grad_out.T @ (x @ slot.lora_A.T))


def test_lora_fa_does_not_save_full_width_adapter_input() -> None:
    def saved_numel(registry: AdapterRegistry) -> int:
        total = 0

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            nonlocal total
            total += tensor.numel()
            return tensor

        x = torch.randn(64, 16)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            _slot(registry)(x)
        return total

    lora = saved_numel(_dense_registry(LoraConfig(rank=2, alpha=4.0, target_modules=("q_proj",))))
    lora_fa = saved_numel(_dense_registry(LoraConfig(rank=2, alpha=4.0, target_modules=("q_proj",), lora_fa=True)))
    # Ordinary LoRA keeps x [64, 16] for dA; LoRA-FA keeps only x A^T [64, 2].
    assert lora - lora_fa >= 64 * 16 - 64 * 2


def test_lora_fa_grad_correction_matches_theorem_4_1_and_accumulates_linearly() -> None:
    config = LoraConfig(rank=4, alpha=8.0, target_modules=("q_proj",), lora_fa=True, lora_fa_grad_correction=True)
    corrected = _dense_registry(config)
    plain = _dense_registry(LoraConfig(rank=4, alpha=8.0, target_modules=("q_proj",), lora_fa=True))
    corrected.install_lora_fa_grad_correction()
    corrected.install_lora_fa_grad_correction()  # idempotent: must not stack a second hook
    plain.install_lora_fa_grad_correction()  # no-op without the correction flag
    assert not hasattr(_slot(plain), "lora_fa_correction")

    generator = torch.Generator().manual_seed(5)
    microbatches = [
        (torch.randn(3, 16, generator=generator), torch.randn(3, 12, generator=generator)) for _ in range(2)
    ]
    for registry in (corrected, plain):
        for x, grad_out in microbatches:
            _backward(registry, x, grad_out)

    A = _slot(corrected).lora_A.detach().double()
    expected = (4.0 / 8.0) ** 2 * _slot(plain).lora_B.grad.double() @ torch.linalg.inv(A @ A.T)
    torch.testing.assert_close(_slot(corrected).lora_B.grad.double(), expected, rtol=1e-5, atol=1e-6)


def test_lora_fa_grad_correction_preserves_bf16_gradient_dtype() -> None:
    config = LoraConfig(rank=2, alpha=4.0, target_modules=("q_proj",), lora_fa=True, lora_fa_grad_correction=True)
    registry = _dense_registry(config, base_dtype=torch.bfloat16)
    registry.install_lora_fa_grad_correction()
    _backward(registry, torch.randn(3, 16, dtype=torch.bfloat16), torch.randn(3, 12, dtype=torch.bfloat16))
    assert _slot(registry).lora_B.grad.dtype == torch.bfloat16


def test_routed_expert_lora_fa_freezes_a_and_corrects_per_expert() -> None:
    _set_cpu_tp()
    config = LoraConfig(rank=2, alpha=4.0, target_modules=("down_proj",), lora_fa=True, lora_fa_grad_correction=True)
    state = _AdapterRuntimeState()
    slot = RoutedExpertLoraSlot(
        logical_name="layers.0.mlp.experts.{expert}.down_proj",
        base_weight=nn.Parameter(torch.zeros(1)),
        local_num_experts=2,
        local_expert_start=0,
        in_features=6,
        out_features=4,
        config=config,
        seed=1,
        runtime_state=state,
    )
    registry = AdapterRegistry({slot.logical_name: slot}, config, state)
    assert registry.parameters() == (slot.lora_B,)
    registry.install_lora_fa_grad_correction()
    assert slot.lora_fa_correction.shape == (2, 2, 2)

    grad = torch.randn(2, 4, 2, generator=torch.Generator().manual_seed(6))
    (slot.lora_B * grad).sum().backward()
    A = slot.lora_A.detach().double()
    expected = (2.0 / 4.0) ** 2 * grad.double() @ torch.linalg.inv(A @ A.transpose(-1, -2))
    torch.testing.assert_close(slot.lora_B.grad.double(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("grad_correction", (False, True))
def test_lora_fa_metadata_round_trips_through_peft_export(tmp_path, grad_correction: bool) -> None:
    config = LoraConfig(
        rank=4, alpha=8.0, target_modules=("q_proj",), lora_fa=True, lora_fa_grad_correction=grad_correction
    )
    exported = export_peft_adapter(_dense_registry(config), tmp_path, base_model_name_or_path="base")
    assert exported == str(tmp_path)
    metadata = json.loads((tmp_path / LORA_FA_CONFIG_NAME).read_text(encoding="utf-8"))
    assert metadata == {"frozen": "lora_A", "grad_correction": grad_correction}
    peft_config = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert peft_config["peft_type"] == "LORA"  # the artifact stays standard PEFT for serving

    reloaded = LoraConfig(adapter_path=str(tmp_path))
    assert reloaded.lora_fa is True
    assert reloaded.lora_fa_grad_correction is grad_correction


def test_ordinary_lora_export_removes_stale_lora_fa_metadata(tmp_path) -> None:
    (tmp_path / LORA_FA_CONFIG_NAME).write_text('{"frozen": "lora_A", "grad_correction": false}\n')
    config = LoraConfig(rank=4, alpha=8.0, target_modules=("q_proj",))
    export_peft_adapter(_dense_registry(config), tmp_path, base_model_name_or_path="base")
    assert not (tmp_path / LORA_FA_CONFIG_NAME).exists()
    assert LoraConfig(adapter_path=str(tmp_path)).lora_fa is False


def test_lora_fa_rejects_unknown_metadata(tmp_path) -> None:
    export_peft_adapter(
        _dense_registry(LoraConfig(rank=4, alpha=8.0, target_modules=("q_proj",))),
        tmp_path,
        base_model_name_or_path="base",
    )
    (tmp_path / LORA_FA_CONFIG_NAME).write_text('{"frozen": "lora_B", "grad_correction": false}\n')
    with pytest.raises(ValueError, match="LoRA-FA adapter metadata"):
        LoraConfig(adapter_path=str(tmp_path))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _row_parallel_correction(rank: int, world_size: int, group) -> torch.Tensor:
    set_tp_context(TPContext(rank=rank, world_size=world_size, device=torch.device("cpu"), group=group))
    config = LoraConfig(rank=4, alpha=8.0, target_modules=("o_proj",), lora_fa=True, lora_fa_grad_correction=True)
    state = _AdapterRuntimeState()
    slot = LoraSlot(
        logical_name="layers.0.self_attn.o_proj",
        base_weight=nn.Parameter(torch.zeros(1)),
        global_in_features=16,
        global_out_features=12,
        local_in_features=16 // world_size,
        local_out_features=12,
        row_parallel=True,
        config=config,
        seed=1,
        runtime_state=state,
    )
    AdapterRegistry({slot.logical_name: slot}, config, state).install_lora_fa_grad_correction()
    return slot.lora_fa_correction


def _gloo_row_parallel_worker(rank: int, port: int, output_queue) -> None:
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        output_queue.put((rank, _row_parallel_correction(rank, 2, dist.group.WORLD).tolist()))
    finally:
        dist.destroy_process_group()


def test_row_parallel_grad_correction_reduces_sharded_gram_across_tp() -> None:
    reference = _row_parallel_correction(0, 1, None)
    spawn = mp.get_context("spawn")
    output_queue = spawn.Queue()
    port = _free_port()
    processes = [spawn.Process(target=_gloo_row_parallel_worker, args=(rank, port, output_queue)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        results = dict(output_queue.get(timeout=60) for _ in processes)
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes)
    for rank in range(2):
        torch.testing.assert_close(torch.tensor(results[rank]), reference)
