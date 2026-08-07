#!/usr/bin/env python3
"""Inventory HF-style config and safetensors without importing the model."""

from __future__ import annotations

import fnmatch
import json
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import build_parser, skill_main


def tensor_files(root: Path) -> list[Path]:
    index_files = sorted(root.glob("*.safetensors.index.json"))
    if index_files:
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        return [root / name for name in sorted(set(index.get("weight_map", {}).values()))]
    return sorted(root.glob("*.safetensors"))


@skill_main
def main() -> dict:
    parser = build_parser("Inventory HF-style config and safetensors without importing the model.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    from safetensors import safe_open

    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    tensors = []
    files = tensor_files(args.checkpoint)
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not fnmatch.fnmatch(key, args.pattern):
                    continue
                tensor = handle.get_slice(key)
                tensors.append(
                    {
                        "key": key,
                        "shape": list(tensor.get_shape()),
                        "dtype": str(tensor.get_dtype()),
                        "file": path.name,
                    }
                )
    return {
        "ok": True,
        "checkpoint": str(args.checkpoint),
        "config": config,
        "shard_count": len(files),
        "matched_tensor_count": len(tensors),
        "tensors": tensors[: max(args.limit, 0)],
        "truncated": len(tensors) > args.limit,
    }


if __name__ == "__main__":
    raise SystemExit(main())
