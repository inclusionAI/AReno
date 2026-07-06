#!/usr/bin/env python3
"""Compare Qwen3.5-VL HF reference outputs with AReno outputs.

Run this in a GPU environment with the same checkpoint used by serve:

    python ci/compare_qwen35_vl_reference.py \
      --ckpt /home/admin/Qwen3.5-VL \
      --image /home/admin/josh.jpg \
      --prompt "Describe this image in one sentence."

The script compares:
  1. processor token/image-grid alignment,
  2. vision tower merged image embeddings,
  3. one-step language logits after image embedding replacement.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None

from areno.api.multimodal import (
    expand_image_tokens,
    image_token_counts_from_features,
    mrope_position_ids_from_image_grid,
)
from areno.models.registry import build_model, config_from_hf, load_model_weights


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    ckpt = Path(args.ckpt)
    image = Image.open(args.image).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(args.image)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    print("== load processor ==")
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=[image], return_tensors="pt")
    input_ids = encoded["input_ids"]
    features = {key: value for key, value in dict(encoded).items() if key not in {"input_ids", "attention_mask"}}
    image_token_id = _image_token_id(processor)
    if image_token_id is not None:
        features["image_token_id"] = image_token_id
    counts = image_token_counts_from_features(features)
    areno_tokens, _ = expand_image_tokens(
        input_ids[0].tolist(), image_token_id=image_token_id, image_token_counts=counts
    )
    mrope = mrope_position_ids_from_image_grid(areno_tokens, image_token_id=image_token_id, features=features)
    if mrope is not None:
        features["mrope_position_ids"] = mrope
    print(f"text={text!r}")
    print(f"hf_input_ids={tuple(input_ids.shape)} areno_tokens={len(areno_tokens)}")
    print(f"image_token_id={image_token_id} image_token_counts={counts}")
    if "pixel_values" in features:
        print(f"pixel_values={tuple(features['pixel_values'].shape)} {features['pixel_values'].dtype}")
    if "image_grid_thw" in features:
        print(f"image_grid_thw={features['image_grid_thw'].tolist()}")
    if mrope is not None:
        print(f"mrope_position_ids={tuple(mrope.shape)}")

    print("\n== load hf model ==")
    hf_model = _load_hf_model(ckpt, dtype=dtype).to(device).eval()
    hf_inputs = {key: _to_device(value, device, dtype) for key, value in encoded.items()}

    print("\n== load areno model ==")
    config = config_from_hf(ckpt)
    areno_model = build_model(config).to(device).eval()
    load_model_weights(areno_model, config, ckpt)
    areno_features = {key: _to_device(value, device, dtype) for key, value in features.items()}
    areno_input_ids = torch.tensor([areno_tokens], device=device, dtype=torch.long)

    print("\n== compare visual embeddings ==")
    with torch.inference_mode():
        hf_visual = _hf_visual(hf_model, hf_inputs)
        areno_visual = areno_model.visual(
            areno_features["pixel_values"].to(dtype=dtype),
            areno_features.get("image_grid_thw"),
        )
    _print_compare("visual", hf_visual, areno_visual)

    print("\n== compare full logits ==")
    with torch.inference_mode():
        hf_logits = hf_model(**hf_inputs).logits
        areno_logits = areno_model(areno_input_ids, features=areno_features).logits_shard
    _print_compare("logits_last", hf_logits[:, -1, :], areno_logits[:, -1, :])
    _print_topk("hf", hf_logits[0, -1], processor)
    _print_topk("areno", areno_logits[0, -1], processor)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Local Qwen3.5-VL checkpoint directory")
    parser.add_argument("--image", required=True, type=Path, help="Image file")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    return parser.parse_args()


def _load_hf_model(path: Path, dtype: torch.dtype):
    last_error: Exception | None = None
    classes = [cls for cls in (AutoModelForImageTextToText, AutoModelForCausalLM) if cls is not None]
    for cls in classes:
        try:
            return cls.from_pretrained(path, trust_remote_code=True, torch_dtype=dtype)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError("failed to load HF reference model") from last_error


def _image_token_id(processor: Any) -> int | None:
    for obj in (processor, getattr(processor, "tokenizer", None)):
        if obj is None:
            continue
        for attr in ("image_token_id", "image_token_index"):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return int(value)
        token = getattr(obj, "image_token", None)
        convert = getattr(getattr(processor, "tokenizer", None), "convert_tokens_to_ids", None)
        if isinstance(token, str) and callable(convert):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                return token_id
    return None


def _to_device(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if value.is_floating_point():
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def _hf_visual(model: torch.nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
    visual = _nested_attr(model, ("visual", "model.visual"))
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs.get("image_grid_thw")
    try:
        out = visual(pixel_values, grid_thw=image_grid_thw)
    except TypeError:
        out = visual(pixel_values, image_grid_thw=image_grid_thw)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _nested_attr(obj: Any, candidates: tuple[str, ...]) -> Any:
    for candidate in candidates:
        current = obj
        try:
            for part in candidate.split("."):
                current = getattr(current, part)
            return current
        except AttributeError:
            continue
    raise AttributeError(f"none of {candidates!r} exists on {type(obj)!r}")


def _print_compare(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
    left = left.detach().float().cpu()
    right = right.detach().float().cpu()
    print(f"{name}: hf={tuple(left.shape)} areno={tuple(right.shape)}")
    if left.shape != right.shape:
        print(f"{name}: shape mismatch")
        return
    diff = (left - right).abs()
    denom = left.norm().clamp_min(1e-6)
    rel = diff.norm() / denom
    cosine = torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0)
    print(
        f"{name}: max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g} "
        f"rel_l2={rel.item():.6g} cosine={cosine.item():.6g}"
    )


def _print_topk(name: str, logits: torch.Tensor, processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", processor)
    values, indices = torch.topk(logits.detach().float().cpu(), k=10)
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    pieces = []
    for value, idx in zip(values.tolist(), indices.tolist(), strict=True):
        token = convert(idx) if callable(convert) else str(idx)
        pieces.append(f"{idx}:{token}:{value:.3f}")
    print(f"{name}_top10: " + " | ".join(pieces))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
