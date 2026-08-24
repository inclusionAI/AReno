from __future__ import annotations

import torch

from areno.engine.optim.master_storage import decode_fp32_master, encode_fp32_master


def test_bf16_master_storage_round_trips_every_fp32_bit() -> None:
    generator = torch.Generator().manual_seed(20260821)
    random_values = torch.randn(4099, generator=generator, dtype=torch.float32)
    # Exercise exact halfway cases where BF16 ties-to-even can either increment
    # or retain the truncated high word.  Those are the cases that require the
    # packed carry bit in addition to the low 16 bits.
    tie_words = torch.tensor(
        [0x3F808000, 0x3F818000, -1082097664, -1082032128],
        dtype=torch.int32,
    ).view(torch.float32)
    master = torch.cat((random_values, tie_words))

    model, storage = encode_fp32_master(master)
    restored = decode_fp32_master(model, storage)

    torch.testing.assert_close(restored, master, rtol=0.0, atol=0.0)
    assert torch.equal(restored.view(torch.int32), master.view(torch.int32))


def test_bf16_master_storage_uses_about_two_bytes_per_parameter() -> None:
    master = torch.linspace(-2.0, 2.0, 1024, dtype=torch.float32)

    model, storage = encode_fp32_master(master)

    assert model.element_size() == 2
    assert storage.low_bits.element_size() == 2
    assert storage.round_up_bits.numel() == 128
    assert storage.nbytes == 2 * master.numel() + master.numel() // 8
    assert storage.nbytes < master.numel() * master.element_size()
