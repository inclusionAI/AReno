#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass
class TensorRef:
    key: str
    filename: str
    shape: tuple[int, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare same-name tensors between two HF safetensors checkpoints.")
    parser.add_argument("base", type=Path, help="Reference checkpoint directory.")
    parser.add_argument("other", type=Path, help="Checkpoint directory to compare against the reference.")
    parser.add_argument("--top-k", type=int, default=30, help="Number of largest-difference tensors to print.")
    parser.add_argument(
        "--pattern", action="append", default=[], help="fnmatch pattern for keys to include; can be repeated."
    )
    parser.add_argument(
        "--device", default="auto", help="Device used for diff computation: auto, cpu, cuda, or cuda:N."
    )
    parser.add_argument(
        "--max-elements", type=int, default=0, help="Sample at most this many elements per tensor; 0 means full tensor."
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    base_index = load_index(args.base)
    other_index = load_index(args.other)
    patterns = tuple(args.pattern)
    base_keys = filter_keys(set(base_index), patterns)
    other_keys = filter_keys(set(other_index), patterns)
    common = sorted(base_keys & other_keys)
    missing_in_other = sorted(base_keys - other_keys)
    extra_in_other = sorted(other_keys - base_keys)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    comparable: list[str] = []
    for key in common:
        a_ref = base_index[key]
        b_ref = other_index[key]
        if a_ref.shape != b_ref.shape:
            shape_mismatches.append((key, a_ref.shape, b_ref.shape))
            continue
        comparable.append(key)

    rows = compare_tensors(args.base, args.other, base_index, other_index, comparable, device, args.max_elements)
    rows.sort(key=lambda item: (item["rel_l2"], item["max_abs"], item["mean_abs"]), reverse=True)
    print(f"base={args.base}")
    print(f"other={args.other}")
    print(f"device={device}")
    print(
        f"common={len(common)} missing_in_other={len(missing_in_other)} extra_in_other={len(extra_in_other)} shape_mismatch={len(shape_mismatches)}"
    )
    print_section("missing_in_other", missing_in_other[: args.top_k])
    print_section("extra_in_other", extra_in_other[: args.top_k])
    if shape_mismatches:
        print("\nshape_mismatch:")
        for key, a_shape, b_shape in shape_mismatches[: args.top_k]:
            print(f"  {key}: base={a_shape} other={b_shape}")
    if rows:
        print("\nlargest_numeric_diff:")
        for row in rows[: args.top_k]:
            print(
                f"  rel_l2={row['rel_l2']:.6e} max_abs={row['max_abs']:.6e} "
                f"mean_abs={row['mean_abs']:.6e} shape={row['shape']} dtype={row['dtype']} key={row['key']}"
            )


def load_index(path: Path) -> dict[str, TensorRef]:
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text())
        weight_map = data["weight_map"]
    else:
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors files found under {path}")
        weight_map = {}
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    weight_map[key] = file.name
    out: dict[str, TensorRef] = {}
    keys_by_file: dict[str, list[str]] = defaultdict(list)
    for key, filename in weight_map.items():
        keys_by_file[filename].append(key)
    for filename, file_keys in keys_by_file.items():
        with safe_open(path / filename, framework="pt", device="cpu") as handle:
            for key in file_keys:
                tensor_slice = handle.get_slice(key)
                out[key] = TensorRef(key=key, filename=filename, shape=tuple(tensor_slice.get_shape()))
    return out


def filter_keys(keys: set[str], patterns: tuple[str, ...]) -> set[str]:
    if not patterns:
        return keys
    return {key for key in keys if any(fnmatch.fnmatch(key, pattern) for pattern in patterns)}


def compare_tensors(
    base_root: Path,
    other_root: Path,
    base_index: dict[str, TensorRef],
    other_index: dict[str, TensorRef],
    keys: list[str],
    device: str,
    max_elements: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    key_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in keys:
        key_groups[(base_index[key].filename, other_index[key].filename)].append(key)
    with torch.inference_mode():
        for (base_file, other_file), group_keys in key_groups.items():
            with safe_open(base_root / base_file, framework="pt", device=device) as base_handle:
                with safe_open(other_root / other_file, framework="pt", device=device) as other_handle:
                    for key in group_keys:
                        a = base_handle.get_tensor(key)
                        b = other_handle.get_tensor(key)
                        rows.append(compare_one(key, base_index[key].shape, a, b, max_elements))
    return rows


def compare_one(
    key: str, shape: tuple[int, ...], a: torch.Tensor, b: torch.Tensor, max_elements: int
) -> dict[str, object]:
    if max_elements > 0 and a.numel() > max_elements:
        stride = max(1, a.numel() // max_elements)
        a = a.reshape(-1)[::stride][:max_elements]
        b = b.reshape(-1)[::stride][:max_elements]
    af = a.float()
    bf = b.float()
    diff_abs = (af - bf).abs()
    diff_norm = torch.linalg.vector_norm(diff_abs)
    base_norm = torch.linalg.vector_norm(af)
    return {
        "key": key,
        "shape": shape,
        "dtype": f"{a.dtype}/{b.dtype}",
        "max_abs": float(diff_abs.max().item()) if diff_abs.numel() else 0.0,
        "mean_abs": float(diff_abs.mean().item()) if diff_abs.numel() else 0.0,
        "rel_l2": float((diff_norm / base_norm.clamp_min(1e-12)).item()),
    }


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_section(name: str, values: list[str]) -> None:
    if not values:
        return
    print(f"\n{name}:")
    for value in values:
        print(f"  {value}")


if __name__ == "__main__":
    main()
