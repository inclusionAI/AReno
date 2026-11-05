"""Pack and unpack cost of the CUDA kernels for one saved tensor.

Times ``quantize`` (including the outlier-channel selection) and
``dequantize`` on a BF16 tensor, by default 65,536 tokens of a 1,024-wide
hidden state, at each precision, and reports milliseconds, the BF16 bytes
processed per second and the stored fraction.

    python examples/activation_compression/microbench.py --output microbench.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    from areno.engine.runtime.activation_compression import dequantize, quantize, stored_bytes

    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    x[:, 7] *= 300  # an outlier channel, as in transformer residual streams
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(fn) -> float:
        for _ in range(5):
            fn()
        start.record()
        for _ in range(args.repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / args.repeats

    rows = []
    for bits in (8, 4, 2):
        packed = quantize(x, bits)
        pack_ms = timed(lambda bits=bits: quantize(x, bits))
        unpack_ms = timed(lambda packed=packed: dequantize(packed))
        rows.append(
            {
                "bits": bits,
                "pack_ms": pack_ms,
                "unpack_ms": unpack_ms,
                "pack_gb_per_s": x.numel() * 2 / pack_ms / 1e6,
                "unpack_gb_per_s": x.numel() * 2 / unpack_ms / 1e6,
                "stored_fraction": stored_bytes(x.numel(), 2, args.hidden, bits) / (x.numel() * 2),
            }
        )
        print(json.dumps(rows[-1]))
    report = {"tokens": args.tokens, "hidden": args.hidden, "gpu": torch.cuda.get_device_name(0), "rows": rows}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
