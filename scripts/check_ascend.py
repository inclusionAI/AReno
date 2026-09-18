#!/usr/bin/env python3
"""Probe Ascend hardware and BF16 autograd without importing/installing AReno.

Run in the node's existing CANN/torch_npu environment:
    python scripts/check_ascend.py
    python scripts/check_ascend.py --all-devices

PASS covers the hardware target and basic TorchNPU execution only. It does
not compile or validate AReno kernels, HCCL, or training/serving workflows.
"""

import argparse
import ctypes
import os
import platform
import re
import sys
from pathlib import Path


def runtime_soc() -> str:
    """Ask the initialized runtime; never trust a build-time SoC override."""
    candidates = ["libascendcl.so"]
    for name in ("ASCEND_HOME_PATH", "ASCEND_CANN_PACKAGE_PATH"):
        root = os.environ.get(name)
        if root:
            candidates.append(str(Path(root).expanduser() / "lib64" / "libascendcl.so"))
    errors = []
    for path in dict.fromkeys(candidates):
        try:
            library = ctypes.CDLL(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        query = library.aclrtGetSocName
        query.argtypes = []
        query.restype = ctypes.c_char_p
        value = query()
        if not value:
            raise RuntimeError("aclrtGetSocName returned an empty SoC name")
        return value.decode("ascii")
    raise RuntimeError("Cannot load CANN libascendcl.so; source the toolkit's set_env.sh. " + " | ".join(errors))


def check_device(torch, index: int) -> None:
    torch.npu.set_device(index)
    torch.npu.init()
    soc = runtime_soc()
    print(f"[npu:{index}] runtime SoC: {soc}", flush=True)
    # Match the current native builder's A2/A3 target. The generic name printed
    # by npu-smi is insufficient, and ARENO_NPU_SOC is deliberately ignored.
    if not re.fullmatch(r"Ascend910(?:B[1-4](?:C|-1)?|_93[0-9]{2})", soc, re.IGNORECASE):
        raise RuntimeError(
            f"{soc!r} does not match the current Ascend A2/A3 kernel target; adaptation must be reviewed"
        )
    device = torch.device("npu", index)
    x = torch.ones((16, 16), dtype=torch.bfloat16, device=device, requires_grad=True)
    y = x @ x.T
    y.float().sum().backward()
    torch.npu.synchronize()
    if x.device != device or y.device != device or x.grad is None or x.grad.device != device:
        raise RuntimeError("Input, output and gradient must all reside on the selected NPU")
    # Every output is 16; differentiating sum(X @ X.T) at X=1 gives 32.
    torch.testing.assert_close(y.detach().float().cpu(), torch.full((16, 16), 16.0), atol=0, rtol=0)
    torch.testing.assert_close(x.grad.float().cpu(), torch.full((16, 16), 32.0), atol=0, rtol=0)
    print(f"[npu:{index}] PASS: BF16 matmul forward=16, gradient=32", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--device", type=int, default=0, help="Visible NPU index (default: 0)")
    selection.add_argument(
        "--all-devices", action="store_true", help="Check each visible NPU sequentially; does not test HCCL"
    )
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"Platform: {platform.system()} / {platform.machine()}", flush=True)
    try:
        import torch
        import torch_npu

        print(f"PyTorch: {torch.__version__}; torch_npu: {torch_npu.__version__}", flush=True)
        count = torch.npu.device_count()
        print(f"Visible NPU devices: {count}", flush=True)
        if count == 0 or not torch.npu.is_available():
            raise RuntimeError("No usable NPU is visible to this Python environment")
        if not args.all_devices and not 0 <= args.device < count:
            raise ValueError(f"--device must be between 0 and {count - 1}")
        for index in range(count) if args.all_devices else (args.device,):
            check_device(torch, index)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print("PASS: Ascend A2/A3 target and TorchNPU BF16 execution verified. AReno kernels and HCCL are NOT tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
