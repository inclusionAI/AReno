#!/usr/bin/env python3
"""Inspect a Qwen3.5-VL-style checkpoint for vision/projector implementation work.

Usage:
    python ci/inspect_qwen35_vl_checkpoint.py /path/to/Qwen3.5-VL

The script does not load tensors into GPU memory. It reads config.json and
safetensors metadata, then prints the config branches and likely vision /
projector tensor keys with shapes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

VISION_PATTERNS = (
    "vision",
    "visual",
    "image",
    "patch",
    "merger",
    "projector",
    "mm_projector",
    "multi_modal",
    "multimodal",
    "resampler",
)


def main() -> None:
    args = _parse_args()
    path = Path(args.checkpoint)
    config = _read_json(path / "config.json")
    print("== config top-level ==")
    for key in sorted(config):
        value = config[key]
        if isinstance(value, dict):
            print(f"{key}: dict[{len(value)}]")
        elif isinstance(value, list):
            print(f"{key}: list[{len(value)}] {value[:5]!r}")
        else:
            print(f"{key}: {value!r}")
    print()
    for key in ("text_config", "vision_config", "visual", "vision_model_config", "mm_projector_config"):
        if key in config:
            print(f"== config.{key} ==")
            _print_nested(config[key])
            print()
    print("== likely tensor keys ==")
    keys = _safetensor_keys(path)
    matched = [key for key in keys if any(pattern in key.lower() for pattern in VISION_PATTERNS)]
    if not matched:
        print("(none matched)")
    for key in matched[: args.limit]:
        shape = _tensor_shape(path, key)
        print(f"{key}: {shape}")
    if len(matched) > args.limit:
        print(f"... truncated {len(matched) - args.limit} more keys")
    print()
    print("== language prefix candidates ==")
    for prefix in ("model", "model.language_model", "language_model", "model.model"):
        count = sum(1 for key in keys if key.startswith(prefix + "."))
        if count:
            print(f"{prefix}: {count} tensors")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Local HF-format checkpoint directory")
    parser.add_argument("--limit", type=int, default=200, help="Maximum matching tensor keys to print")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _print_nested(value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    if not isinstance(value, dict):
        print(f"{prefix}{value!r}")
        return
    for key in sorted(value):
        item = value[key]
        if isinstance(item, dict):
            print(f"{prefix}{key}: dict[{len(item)}]")
            _print_nested(item, indent + 2)
        else:
            print(f"{prefix}{key}: {item!r}")


def _safetensor_keys(path: Path) -> list[str]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        return sorted(_read_json(index).get("weight_map", {}))
    keys: list[str] = []
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="numpy", device="cpu") as handle:
            keys.extend(handle.keys())
    return sorted(keys)


def _tensor_shape(path: Path, key: str) -> tuple[int, ...] | str:
    files = _files_for_key(path, key)
    for file in files:
        with safe_open(file, framework="numpy", device="cpu") as handle:
            if key in handle.keys():
                return tuple(handle.get_tensor(key).shape)
    return "shape unavailable"


def _files_for_key(path: Path, key: str) -> list[Path]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        filename = _read_json(index).get("weight_map", {}).get(key)
        if filename:
            return [path / filename]
    return sorted(path.glob("*.safetensors"))


if __name__ == "__main__":
    main()
