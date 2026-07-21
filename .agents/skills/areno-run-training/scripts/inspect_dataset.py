#!/usr/bin/env python3
"""Load a bounded AReno dataset sample and validate its coarse algorithm contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def default_loader(path: str, *, model_hub: str) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_file() and source.suffix in {".jsonl", ".json"}:
        text = source.read_text(encoding="utf-8")
        if source.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("install datasets or provide a JSON/JSONL file") from exc
    if source.is_dir():
        return load_from_disk(str(source))
    name, _, config = path.partition(":")
    if model_hub == "modelscope":
        try:
            from modelscope.msdatasets import MsDataset
        except ImportError as exc:
            raise RuntimeError("ModelScope dataset loading requires modelscope") from exc
        kwargs = {"subset_name": config} if config else {}
        dataset = MsDataset.load(name, split="train", trust_remote_code=True, **kwargs)
        to_hf_dataset = getattr(dataset, "to_hf_dataset", None)
        return to_hf_dataset() if callable(to_hf_dataset) else dataset
    return load_dataset(name, config or None, split="train")


def load_rows(path: str, loader_path: str | None, *, model_hub: str) -> Any:
    def load_default(dataset_path: str):
        return default_loader(dataset_path, model_hub=model_hub)

    if not loader_path:
        return load_default(path)
    file_path = Path(loader_path)
    spec = importlib.util.spec_from_file_location("areno_skill_dataset_loader", file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import loader {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = getattr(module, "load_training_dataset")
    return loader(path, default_loader=load_default)


def classify(row: dict[str, Any], algo: str) -> list[str]:
    keys = set(row)
    errors: list[str] = []
    if algo == "dpo":
        if not ({"chosen", "rejected"} <= keys):
            errors.append("DPO row requires chosen and rejected")
    elif algo == "sft":
        alternatives = ("messages", "text", "response", "output", "answer")
        if not any(key in keys for key in alternatives):
            errors.append("SFT row has no supervised response field")
    elif "prompt" not in keys and "messages" not in keys:
        if {"question", "answer"} <= keys:
            errors.append(
                "rollout row requires prompt or messages; rerun with "
                "--loader examples/math/dataset_loader.py for GSM8K-style rows"
            )
        else:
            errors.append("rollout row requires prompt or messages; select and verify a dataset loader")
    return errors


def summarize(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("data:"):
            return f"<data-url chars={len(value)}>"
        return value[:240] + ("..." if len(value) > 240 else "")
    if isinstance(value, dict):
        return {str(key): summarize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [summarize(item) for item in value[:8]]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--loader")
    parser.add_argument("--model-hub", choices=("modelscope", "hf"), default="modelscope")
    parser.add_argument("--algo", choices=("sft", "dpo", "gspo", "grpo", "ppo"), required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    try:
        rows = load_rows(args.dataset_path, args.loader, model_hub=args.model_hub)
        count = len(rows)
        samples = [dict(rows[index]) for index in range(min(count, max(args.limit, 0)))]
        errors = [error for row in samples for error in classify(row, args.algo)]
        result = {
            "ok": not errors and bool(samples),
            "count": count,
            "sample_keys": [sorted(row) for row in samples],
            "samples": [summarize(row) for row in samples],
            "errors": errors or ([] if samples else ["dataset is empty"]),
        }
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
