"""Load-time MXFP4 E2M1/E8M0 decoding; no quantized training dependency."""

import torch

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequantize_mxfp4(blocks, scales, dtype=torch.bfloat16, device=None):
    if (
        blocks.dtype != torch.uint8
        or scales.dtype != torch.uint8
        or blocks.shape[:-1] != scales.shape
        or blocks.shape[-1] != 16
    ):
        raise ValueError("MXFP4 requires uint8 blocks (...,16) and matching E8M0 scales")
    shape = (*blocks.shape[:-2], blocks.shape[-2] * 32)
    blocks = blocks.to(device=device)
    scales = scales.to(device=device)
    if blocks.is_cuda:
        from areno.accel.kernels.mxfp4 import decode

        return decode(blocks, scales, dtype).reshape(shape)
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=blocks.device)
    codes = torch.stack((blocks & 15, blocks >> 4), -1).long()
    values = lut[codes].flatten(-2)
    return torch.ldexp(values, scales.to(torch.int32)[..., None] - 127).to(dtype).reshape(shape)
