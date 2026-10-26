"""Preflight: teacher and student must interpret every token ID identically.

Only tokenizer/config metadata is downloaded; full model weights are not loaded.
Run from the AReno root: python examples/opd/check_models.py --student ... --teacher ...
"""
from __future__ import annotations

import argparse
import json
from typing import Any


def validate_compatibility(student_tokenizer: Any, teacher_tokenizer: Any,
                           student_config: Any, teacher_config: Any) -> dict:
    """Check semantic tokenizer compatibility, not just identical vocab size."""
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if not student_vocab or student_vocab != teacher_vocab:
        raise ValueError("student/teacher token-to-ID vocabularies differ; basic OPD requires identical token IDs")

    special_ids = ("bos_token_id", "eos_token_id", "pad_token_id")
    for name in special_ids:
        if getattr(student_tokenizer, name, None) != getattr(teacher_tokenizer, name, None):
            raise ValueError(f"student/teacher {name} differs")

    s_type = getattr(student_config, "model_type", None)
    t_type = getattr(teacher_config, "model_type", None)
    if s_type != t_type:
        raise ValueError(f"student/teacher model types differ: {s_type!r} vs {t_type!r}")
    if s_type not in {"qwen3", "qwen2", "llama"}:
        raise ValueError(f"unsupported/unverified model type {s_type!r}; check an AReno adapter before running")

    s_size = getattr(student_config, "vocab_size", None)
    t_size = getattr(teacher_config, "vocab_size", None)
    if s_size != t_size:
        raise ValueError("student/teacher model vocabulary sizes differ")

    # Ensure both checkpoints' embeddings can represent all shared IDs.
    max_id = max(student_vocab.values())
    if s_size is None or max_id >= s_size:
        raise ValueError("token IDs exceed the model's embedding vocabulary")

    return {
        "model_type": s_type,
        "vocab_size": s_size,
        "tokenizer_entries": len(student_vocab),
        "eos_token_id": getattr(student_tokenizer, "eos_token_id", None),
        "compatible": True,
    }


def check_model_refs(student: str, teacher: str) -> dict:
    """Download only Hugging Face tokenizer/config metadata and compare it."""
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to run tokenizer preflight: pip install transformers") from exc

    student_tokenizer = AutoTokenizer.from_pretrained(student, trust_remote_code=False)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher, trust_remote_code=False)
    student_cfg = AutoConfig.from_pretrained(student, trust_remote_code=False)
    teacher_cfg = AutoConfig.from_pretrained(teacher, trust_remote_code=False)
    return validate_compatibility(student_tokenizer, teacher_tokenizer, student_cfg, teacher_cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher", default="Qwen/Qwen3-1.7B")
    args = parser.parse_args(argv)
    try:
        report = check_model_refs(args.student, args.teacher)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Model compatibility check failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
