"""Compare real local HF safetensors weights before/after OPD, tensor by tensor.

Requires two actual model checkpoint directories with an identical state dict.
Never use on Stage 6 fake-backend fixture checkpoints. All tensors are scanned
unless --max-tensors explicitly requests a partial diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def checkpoint_files(directory: Path) -> list[Path]:
    path = Path(directory)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"not an HF checkpoint directory: {path}")
    json.loads((path / "config.json").read_text(encoding="utf-8"))
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        d = json.loads(idx.read_text(encoding="utf-8"))
        if not isinstance(d.get("weight_map"), dict) or not d["weight_map"]:
            raise ValueError("checkpoint index has no weight map")
        names = sorted(set(d["weight_map"].values()))
        if any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
            raise ValueError("unsafe or invalid shard name")
        files = [path / name for name in names]
    else:
        files = sorted(path.glob("*.safetensors"))
    if not files or any(not file.is_file() or file.stat().st_size == 0 for file in files):
        raise ValueError("checkpoint has no nonempty safetensors weights")
    return files


def _tensor_locations(directory: Path, *, open_safe):
    result = {}
    for file in checkpoint_files(directory):
        with open_safe(str(file), framework="pt", device="cpu") as sf:
            for name in sf.keys():
                if name in result:
                    raise ValueError(f"duplicate tensor {name!r}")
                result[name] = file
    return result


def compare_weight_directories(before: Path, after: Path, *, max_tensors: int | None = None) -> dict:
    if max_tensors is not None and max_tensors <= 0:
        raise ValueError("--max-tensors must be positive")
    if Path(before).resolve() == Path(after).resolve():
        raise ValueError("before/after paths must be different")
    try:
        from safetensors import safe_open
        import torch
    except ImportError as exc:
        raise RuntimeError("Install safetensors and torch for weight verification") from exc
    a = _tensor_locations(Path(before), open_safe=safe_open)
    b = _tensor_locations(Path(after), open_safe=safe_open)
    if a.keys() != b.keys():
        raise ValueError(f"checkpoint tensor keys differ: missing_after={sorted(a.keys()-b.keys())[:5]}, extra_after={sorted(b.keys()-a.keys())[:5]}")
    all_names = sorted(a)
    selected = all_names[:max_tensors] if max_tensors is not None else all_names
    changed = 0
    squared_sum = 0.0
    max_abs = 0.0
    elems = 0
    for name in selected:
        with safe_open(str(a[name]), framework="pt", device="cpu") as fa:
            ta = fa.get_tensor(name)
        with safe_open(str(b[name]), framework="pt", device="cpu") as fb:
            tb = fb.get_tensor(name)
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            raise ValueError(f"shape/dtype mismatch for {name}")
        if not torch.isfinite(ta).all() or not torch.isfinite(tb).all():
            raise ValueError(f"nonfinite weights in {name}")
        if not torch.equal(ta, tb):
            changed += 1
            delta = ta.float() - tb.float()
            # Avoid overflowing float32 accumulation for large parameters.
            squared_sum += float(delta.double().square().sum().item())
            max_abs = max(max_abs, float(delta.abs().max().item()))
        elems += ta.numel()
    return {"schema_version": 1, "evidence_kind": "real_safetensors_comparison",
            "before": str(before), "after": str(after),
            "compared_tensors": len(selected), "total_tensors": len(all_names),
            "complete_scan": len(selected) == len(all_names),
            "changed_tensors": changed, "total_parameters_checked": elems,
            "l2_parameter_difference": math.sqrt(squared_sum), "max_abs_parameter_difference": max_abs,
            "nonzero_update_verified": changed > 0 if len(selected) == len(all_names) else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tensors", type=int)
    args = p.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Weight report already exists; choose a fresh --output")
    report = compare_weight_directories(args.before, args.after, max_tensors=args.max_tensors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
