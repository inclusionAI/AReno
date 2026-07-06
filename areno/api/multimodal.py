"""Helpers for multimodal token/feature alignment."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any

import torch

from areno.api.tokenizer import apply_chat_template_with_options, normalize_token_ids


def record_has_image(record: dict[str, Any]) -> bool:
    """Return true when a loader row contains raw image input."""

    return record.get("image_base64") is not None or record.get("images_base64") is not None


def encode_multimodal_prompt(
    tokenizer: Any,
    processor: Any,
    record: dict[str, Any],
    *,
    prompt_key: str = "prompt",
) -> tuple[list[int], dict[str, Any] | None]:
    """Encode a loader row with base64 image fields into tokens and features.

    Dataset loaders stay model-agnostic and return raw ``image_base64`` plus
    text fields. This helper is the model boundary: it uses the current
    checkpoint processor to produce token ids, image grids, pixel values, and
    Qwen-style expanded image-token slots.
    """

    if processor is None:
        raise ValueError("image_base64 rows require a checkpoint processor")
    images = _load_record_images(record)
    if isinstance(record.get("messages"), list):
        messages = _normalize_image_messages(record["messages"])
    else:
        prompt = str(record.get(prompt_key, ""))
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}],
            }
        ]
    text = _processor_chat_text(processor, messages, tools=record.get("tools"))
    encoded = _encode_text_and_images(tokenizer, processor, text, images)
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise ValueError("processor did not return input_ids for image row")
    features = {
        key: value
        for key, value in dict(encoded).items()
        if key not in {"input_ids", "attention_mask", "token_type_ids"}
    }
    image_token_id = _image_token_id(tokenizer, processor)
    if image_token_id is not None:
        features["image_token_id"] = image_token_id
    tokens = normalize_token_ids(input_ids[0].tolist())
    counts = image_token_counts_from_features(features)
    if counts:
        if image_token_id is None:
            raise ValueError("image rows require an image token id from tokenizer or processor")
        tokens, _ = expand_image_tokens(tokens, image_token_id=image_token_id, image_token_counts=counts)
        mrope_position_ids = mrope_position_ids_from_image_grid(
            tokens,
            image_token_id=image_token_id,
            features=features,
        )
        if mrope_position_ids is not None:
            features["mrope_position_ids"] = mrope_position_ids
    return tokens, features or None


def _load_record_images(record: dict[str, Any]) -> list[Any]:
    values = record.get("images_base64")
    if values is None:
        values = record.get("image_base64")
    if values is None:
        raise ValueError("multimodal row must contain image_base64 or images_base64")
    if isinstance(values, str):
        values = [values]
    return [_load_base64_image(value) for value in values]


def _load_base64_image(value: str) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("image_base64 rows require Pillow") from exc
    payload = str(value)
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def _normalize_image_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [_normalize_image_content_part(part) for part in content]
        normalized.append(item)
    return normalized


def _normalize_image_content_part(part: Any) -> Any:
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return part
    image_url = part.get("image_url")
    if not isinstance(image_url, dict) or "url" not in image_url:
        raise ValueError("image_url content must be an object with a url field")
    normalized = dict(part)
    normalized["type"] = "image"
    normalized["image"] = image_url["url"]
    normalized.pop("image_url", None)
    return normalized


def _processor_chat_text(processor: Any, messages: list[dict[str, Any]], *, tools: Any = None) -> str:
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if callable(apply_chat_template):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        rendered = apply_chat_template_with_options(processor, messages, **kwargs)
        if isinstance(rendered, str):
            return rendered
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        return apply_chat_template_with_options(
            tokenizer,
            _messages_for_text_fallback(messages),
            **kwargs,
        )
    if tools:
        raise ValueError("image input with tools requires a processor or tokenizer chat template that supports tools")
    return _messages_fallback_text(_messages_for_text_fallback(messages))


def _messages_for_text_fallback(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image":
                        parts.append({"type": "image"})
                    elif part.get("type") == "text":
                        parts.append({"type": "text", "text": str(part.get("text", ""))})
                else:
                    parts.append({"type": "text", "text": str(part)})
            item["content"] = parts
        out.append(item)
    return out


def _messages_fallback_text(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        else:
            text = str(content or "")
        lines.append(f"{message['role']}: {text}")
    lines.append("assistant:")
    return "\n".join(lines)


def _encode_text_and_images(tokenizer: Any, processor: Any, text: str, images: list[Any]) -> dict[str, Any]:
    try:
        return dict(processor(text=[text], images=images, return_tensors="pt"))
    except TypeError as exc:
        if "images" not in str(exc):
            raise
    image_processor = _image_processor_from_processor(processor)
    text_encoded = tokenizer([text], return_tensors="pt")
    image_encoded = image_processor(images=images, return_tensors="pt")
    encoded = dict(image_encoded)
    encoded["input_ids"] = text_encoded["input_ids"]
    if text_encoded.get("attention_mask") is not None:
        encoded["attention_mask"] = text_encoded["attention_mask"]
    return encoded


def _image_processor_from_processor(processor: Any):
    nested = getattr(processor, "image_processor", None)
    if nested is not None:
        return nested
    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ValueError("image_base64 rows require transformers AutoImageProcessor") from exc
    name_or_path = getattr(processor, "name_or_path", None)
    if not name_or_path:
        raise ValueError("image_base64 rows require an image processor")
    return AutoImageProcessor.from_pretrained(name_or_path, trust_remote_code=True)


def _image_token_id(tokenizer: Any, processor: Any) -> int | None:
    for obj in (processor, tokenizer):
        for attr in ("image_token_id", "image_token_index"):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return int(value)
        token = getattr(obj, "image_token", None)
        if isinstance(token, str):
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                token_id = convert(token)
                if isinstance(token_id, int) and token_id >= 0:
                    return int(token_id)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        for token in ("<|image_pad|>", "<|image|>", "<image>"):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                return int(token_id)
    return None


def image_token_counts_from_features(features: dict[str, Any] | None) -> list[int]:
    """Return merged visual-token counts for each image in ``features``.

    Qwen-style image processors place one placeholder token per image in the
    rendered text, while the vision tower emits one embedding per merged patch
    group. The language input must therefore repeat each image placeholder by
    the corresponding merged visual-token count before the model can replace
    those token embeddings with visual embeddings.
    """

    if not features:
        return []
    grid = features.get("image_grid_thw")
    if grid is None:
        return []
    if not isinstance(grid, torch.Tensor):
        grid = torch.as_tensor(grid)
    grid = grid.detach().cpu().to(dtype=torch.long).reshape(-1, 3)
    merge = int(features.get("spatial_merge_size", features.get("merge_size", 2)) or 2)
    merge_unit = merge * merge
    counts: list[int] = []
    for t, h, w in grid.tolist():
        patches = int(t) * int(h) * int(w)
        if patches % merge_unit:
            raise ValueError(f"image_grid_thw patches={patches} is not divisible by spatial_merge_size**2={merge_unit}")
        counts.append(patches // merge_unit)
    return counts


def expand_image_tokens(
    tokens: Sequence[int],
    *,
    image_token_id: int | None,
    image_token_counts: Sequence[int],
    aligned_sequences: dict[str, Sequence[Any]] | None = None,
) -> tuple[list[int], dict[str, list[Any]]]:
    """Expand one image placeholder per image into merged visual-token slots.

    ``aligned_sequences`` may contain masks or per-token arrays with the same
    length as ``tokens``; each value at an expanded image-token position is
    repeated by the same visual-token count.
    """

    out_tokens: list[int] = []
    out_aligned = {name: [] for name in (aligned_sequences or {})}
    count_idx = 0
    image_token_id = int(image_token_id) if image_token_id is not None else None
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        repeat = 1
        if image_token_id is not None and int(token) == image_token_id and count_idx < len(image_token_counts):
            count = int(image_token_counts[count_idx])
            if _has_existing_image_span(tokens, image_token_id, idx, count):
                repeat = count
                count_idx += 1
                out_tokens.extend([int(token)] * repeat)
                for name, values in (aligned_sequences or {}).items():
                    out_aligned[name].extend(values[idx : idx + repeat])
                idx += repeat
                continue
            repeat = count
            count_idx += 1
        out_tokens.extend([int(token)] * repeat)
        for name, values in (aligned_sequences or {}).items():
            out_aligned[name].extend([values[idx]] * repeat)
        idx += 1
    if count_idx != len(image_token_counts):
        raise ValueError(
            "image feature count does not match prompt image token count: "
            f"features={len(image_token_counts)} prompt_tokens={count_idx}"
        )
    return out_tokens, out_aligned


def _has_existing_image_span(tokens: Sequence[int], image_token_id: int, start: int, count: int) -> bool:
    """Return true if a processor already expanded this image token span."""

    end = start + count
    if count <= 1 or end > len(tokens):
        return False
    return all(int(token) == image_token_id for token in tokens[start:end])


def mrope_position_ids_from_image_grid(
    tokens: Sequence[int],
    *,
    image_token_id: int | None,
    features: dict[str, Any] | None,
) -> torch.Tensor | None:
    """Build Qwen3.5-VL MRoPE ids for tokens after image-token expansion.

    This follows the image-only fast path used by SGLang: text spans advance a
    scalar position, image spans use (t, h, w) grid positions at merged-patch
    resolution, and the following text resumes at max(t, h, w) after the image.
    """

    if image_token_id is None or not features:
        return None
    grid = features.get("image_grid_thw")
    if grid is None:
        return None
    if not isinstance(grid, torch.Tensor):
        grid = torch.as_tensor(grid)
    grid = grid.detach().cpu().to(dtype=torch.long).reshape(-1, 3)
    merge = int(features.get("spatial_merge_size", features.get("merge_size", 2)) or 2)
    image_token_id = int(image_token_id)
    token_list = [int(token) for token in tokens]
    segments: list[torch.Tensor] = []
    st = 0
    next_pos = 0
    for t_tensor, h_tensor, w_tensor in grid:
        t, h, w = int(t_tensor), int(h_tensor), int(w_tensor)
        if h % merge or w % merge:
            raise ValueError("image_grid_thw height/width must be divisible by spatial_merge_size")
        llm_t, llm_h, llm_w = t, h // merge, w // merge
        count = llm_t * llm_h * llm_w
        try:
            start = _find_image_span(token_list, image_token_id, st, count)
        except ValueError as exc:
            raise ValueError("image_grid_thw count does not match expanded image tokens") from exc
        text_len = start - st
        if text_len > 0:
            segments.append(torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + next_pos)
            next_pos += text_len
        end = start + count
        t_index = torch.arange(llm_t, dtype=torch.long).view(-1, 1).expand(llm_t, llm_h * llm_w).reshape(-1)
        h_index = torch.arange(llm_h, dtype=torch.long).view(1, -1, 1).expand(llm_t, llm_h, llm_w).reshape(-1)
        w_index = torch.arange(llm_w, dtype=torch.long).view(1, 1, -1).expand(llm_t, llm_h, llm_w).reshape(-1)
        segments.append(torch.stack([t_index, h_index, w_index]) + next_pos)
        next_pos += max(llm_t, llm_h, llm_w)
        st = end
    if st < len(token_list):
        text_len = len(token_list) - st
        segments.append(torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + next_pos)
    return torch.cat(segments, dim=1) if segments else None


def _find_image_span(tokens: list[int], image_token_id: int, start: int, count: int) -> int:
    if count <= 0:
        raise ValueError("image token count must be positive")
    for idx in range(start, len(tokens) - count + 1):
        if all(token == image_token_id for token in tokens[idx : idx + count]):
            return idx
    raise ValueError("expanded image token span does not match image_grid_thw")
