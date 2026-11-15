from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def native_request(value: dict[str, Any]) -> dict[str, Any]:
    request = value.get("request", value)
    if set(request) - {"state", "questions", "instructions"}:
        raise ValueError("native request may only contain state, questions and instructions")
    return request


def render(request: dict[str, Any], tokenizer_path: str, revision: str) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm.entrypoints.generate.structured_decisions.question_types import build_question
    from vllm.entrypoints.generate.structured_decisions.strategies import (
        NextTokenStrategy,
        type_label_ids,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    context = SimpleNamespace(
        default_chat_template_kwargs={},
        chat_template=None,
        online_renderer=SimpleNamespace(
            renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer),
        ),
    )
    strategy = object.__new__(NextTokenStrategy)
    strategy.context = context
    generation_prompt = strategy._generation_prompt()
    all_label_ids = type_label_ids(tokenizer, generation_prompt)
    state = strategy.render_state(request["state"])
    reads = []
    for question_id, spec in request["questions"].items():
        question = build_question(
            question_id,
            spec["type"],
            spec.get("instructions", ""),
            spec.get("criteria"),
            max_options=26,
        )
        messages = [{"role": "user", "content": strategy._content(state, question)}]
        if request.get("instructions"):
            messages.insert(0, {"role": "system", "content": request["instructions"]})
        prompt_token_ids = tokenizer.apply_chat_template(
            messages,
            chat_template=None,
            add_generation_prompt=True,
            continue_final_message=False,
            enable_thinking=False,
            tokenize=True,
            return_dict=False,
        )
        if isinstance(prompt_token_ids, str):
            prompt_token_ids = tokenizer.encode(prompt_token_ids, add_special_tokens=False)
        reads.append(
            {
                "question_id": question_id,
                "prompt_token_ids": list(prompt_token_ids),
                "labels": list(question.labels),
                "label_token_ids": [all_label_ids[label] for label in question.labels],
            }
        )
    chat_template = tokenizer.chat_template or ""
    return {
        "request": request,
        "reads": reads,
        "metadata": {
            "source": "vllm_native_reference",
            "vllm_version": version("vllm"),
            "transformers_version": version("transformers"),
            "tokenizer_path": tokenizer_path,
            "requested_revision": revision,
            "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
            "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
            "enable_thinking": False,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "default_chat_template_kwargs": {},
            "truncation": None,
            "generation_prompt_token_ids": generation_prompt,
        },
    }


def compare(areno: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    comparisons = []
    areno_reads = {read["question_id"]: read for read in areno["reads"]}
    native_reads = {read["question_id"]: read for read in native["reads"]}
    question_ids_match = tuple(areno_reads) == tuple(native_reads)
    request_match = native_request(areno) == native_request(native)
    ordered_ids = list(areno_reads) + [question_id for question_id in native_reads if question_id not in areno_reads]
    for question_id in ordered_ids:
        areno_read = areno_reads.get(question_id)
        native_read = native_reads.get(question_id)
        item = {"question_id": question_id}
        for field in ("prompt_token_ids", "labels", "label_token_ids"):
            expected = None if native_read is None else native_read[field]
            actual = None if areno_read is None else areno_read[field]
            item[field] = {
                "match": actual == expected,
                "areno": actual,
                "vllm_native": expected,
            }
        item["match"] = all(item[field]["match"] for field in ("prompt_token_ids", "labels", "label_token_ids"))
        comparisons.append(item)
    passed = request_match and question_ids_match and bool(comparisons) and all(item["match"] for item in comparisons)
    return {
        "status": "PASS" if passed else "FAIL",
        "request_match": request_match,
        "question_order_match": question_ids_match,
        "questions": comparisons,
        "areno_metadata": areno.get("metadata"),
        "native_metadata": native.get("metadata"),
    }


def write(output: Path, value: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--request-json", type=Path, required=True)
    render_parser.add_argument("--tokenizer", required=True)
    render_parser.add_argument("--revision", required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--areno-encoded", type=Path, required=True)
    compare_parser.add_argument("--native-encoded", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "render":
        request = native_request(json.loads(args.request_json.read_text()))
        result = render(request, args.tokenizer, args.revision)
    else:
        areno = json.loads(args.areno_encoded.read_text())
        native = json.loads(args.native_encoded.read_text())
        result = compare(areno, native)
    write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
