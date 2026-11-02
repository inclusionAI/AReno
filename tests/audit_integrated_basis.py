"""Audit every saved projected basis with the unchanged strict Gram criterion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def audit(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer = payload["optimizer"]
    names = payload["parameter_names"]
    results = []
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        for name, shape, state in zip(names, optimizer["shapes"], optimizer["state"], strict=True):
            basis = state.get("basis")
            if basis is None:
                continue
            gram = basis @ basis.T if shape[0] >= shape[1] else basis.T @ basis
            identity = torch.eye(gram.shape[0], dtype=gram.dtype)
            error = gram - identity
            off_diagonal = error.clone()
            off_diagonal.diagonal().zero_()
            failure = None
            try:
                torch.testing.assert_close(gram, identity, atol=1e-6, rtol=1e-4)
            except AssertionError as exc:
                failure = str(exc)
            results.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "basis_shape": list(basis.shape),
                    "max_absolute_gram_error": float(error.abs().max()),
                    "max_off_diagonal_error": float(off_diagonal.abs().max()),
                    "max_diagonal_error": float(error.diagonal().abs().max()),
                    "passed": failure is None,
                    "failure": failure,
                }
            )
    finally:
        torch.set_float32_matmul_precision(previous)
    if not results:
        raise AssertionError("checkpoint has no projected bases to audit")
    return {
        "checkpoint": str(checkpoint),
        "basis_count": len(results),
        "criterion": {"atol": 1e-6, "rtol": 1e-4},
        "projector_policy": optimizer.get("projector_policy"),
        "backend_policy": optimizer.get("backend_policy"),
        "all_passed": all(row["passed"] for row in results),
        "bases": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("checkpoint", "basis_count", "all_passed")}), flush=True)
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
