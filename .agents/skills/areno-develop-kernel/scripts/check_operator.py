#!/usr/bin/env python3
"""Compare unary tensor operators in forward and backward."""

from __future__ import annotations

import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import Result, build_parser, skill_main


def resolve(spec: str):
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("callable must be module:function")
    return getattr(importlib.import_module(module), name)


@skill_main
def main() -> Result:
    parser = build_parser("Compare unary tensor operators in forward and backward.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    shape = tuple(int(item) for item in args.shape.split(","))
    dtype = getattr(torch, args.dtype)
    base = torch.randn(shape, device=args.device, dtype=dtype)
    left_input = base.detach().clone().requires_grad_(True)
    right_input = base.detach().clone().requires_grad_(True)
    left = resolve(args.reference)(left_input)
    right = resolve(args.candidate)(right_input)
    gradient = torch.randn_like(left)
    left.backward(gradient)
    right.backward(gradient)
    forward_ok = torch.allclose(left, right, atol=args.atol, rtol=args.rtol)
    if left_input.grad is not None and right_input.grad is not None:
        backward_ok = torch.allclose(left_input.grad, right_input.grad, atol=args.atol, rtol=args.rtol)
        backward_max_abs = float((left_input.grad - right_input.grad).abs().max())
    else:
        backward_ok = False
        backward_max_abs = None
    return Result(
        ok=bool(forward_ok and backward_ok),
        data={
            "forward_ok": bool(forward_ok),
            "backward_ok": bool(backward_ok),
            "forward_max_abs": float((left - right).abs().max()),
            "backward_max_abs": backward_max_abs,
            "shape": shape,
            "dtype": args.dtype,
            "device": args.device,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())