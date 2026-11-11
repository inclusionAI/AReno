from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from areno.api.jev_vllm import LabelScore, VLLMSystemOneAdapter

DEFAULT_STATE = {"ticket": "My card was charged twice and I cannot access the refund page."}
DEFAULT_QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "payments and refunds",
            "shipping": "deliveries",
            "security": "account access",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does the customer need a reply within the hour?",
        "criteria": {"true": "money is moving or the account is locked", "false": "it can wait"},
    },
    "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}


def compare_scores(decision: dict[str, Any], scores: list[LabelScore]) -> dict[str, Any]:
    if tuple(decision["answers"]) != tuple(score.question_id for score in scores):
        raise ValueError("encoded read question IDs do not match the native response")
    comparisons = []
    for score in scores:
        answer = decision["answers"][score.question_id]
        native_probabilities = list(answer["probabilities"].values())
        supplemental_probabilities = list(score.restricted_probabilities)
        if len(native_probabilities) != len(supplemental_probabilities):
            raise ValueError(f"label count mismatch for {score.question_id!r}")
        probability_errors = [
            abs(native - supplemental) for native, supplemental in zip(native_probabilities, supplemental_probabilities)
        ]
        native_mass = decision["diagnostics"][score.question_id]["label_mass"]
        native_argmax = max(range(len(native_probabilities)), key=native_probabilities.__getitem__)
        supplemental_argmax = max(range(len(supplemental_probabilities)), key=supplemental_probabilities.__getitem__)
        comparisons.append(
            {
                "question_id": score.question_id,
                "answer_keys": list(answer["probabilities"]),
                "labels": list(score.labels),
                "label_token_ids": list(score.label_token_ids),
                "native_restricted_probabilities": native_probabilities,
                "supplemental_restricted_probabilities": supplemental_probabilities,
                "max_probability_abs_error": max(probability_errors),
                "native_label_mass": native_mass,
                "supplemental_label_mass": score.label_mass,
                "label_mass_abs_error": abs(native_mass - score.label_mass),
                "native_raw_logprobs_present": False,
                "supplemental_raw_logprobs": list(score.raw_logprobs),
                "native_argmax_index": native_argmax,
                "supplemental_argmax_index": supplemental_argmax,
                "decision_agreement": native_argmax == supplemental_argmax,
            }
        )
    return {
        "questions": comparisons,
        "max_probability_abs_error": max(item["max_probability_abs_error"] for item in comparisons),
        "max_label_mass_abs_error": max(item["label_mass_abs_error"] for item in comparisons),
        "decision_agreement": sum(item["decision_agreement"] for item in comparisons) / len(comparisons),
    }


def benchmark(adapter: VLLMSystemOneAdapter, request: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    def call():
        return adapter.decide(
            request["state"],
            request["questions"],
            instructions=request.get("instructions"),
            adapter_name=args.adapter_name,
        )

    for _ in range(args.warmup):
        call()
    responses = []
    latencies = []
    for _ in range(args.measured):
        started = time.perf_counter()
        responses.append(call())
        latencies.append(time.perf_counter() - started)
    elapsed = sum(latencies)
    question_count = len(request["questions"])
    return {
        "warmup": args.warmup,
        "measured": args.measured,
        "latency_seconds": latencies,
        "latency_mean_seconds": statistics.fmean(latencies),
        "latency_min_seconds": min(latencies),
        "latency_max_seconds": max(latencies),
        "requests_per_second": args.measured / elapsed,
        "questions_per_second": args.measured * question_count / elapsed,
        "input_tokens_per_second": sum(response["usage"]["input_tokens"] for response in responses) / elapsed,
        "output_tokens_per_second": sum(response["usage"]["output_tokens"] for response in responses) / elapsed,
        "native_responses": responses,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-name")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--encoded-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--measured", type=int, default=1)
    parser.add_argument("--probability-atol", type=float, default=1e-2)
    parser.add_argument("--label-mass-atol", type=float, default=1e-2)
    args = parser.parse_args()
    if args.warmup < 0 or args.measured < 1:
        parser.error("--warmup must be >= 0 and --measured must be >= 1")
    if not math.isfinite(args.probability_atol) or args.probability_atol < 0:
        parser.error("--probability-atol must be finite and >= 0")
    if not math.isfinite(args.label_mass_atol) or args.label_mass_atol < 0:
        parser.error("--label-mass-atol must be finite and >= 0")
    return args


def main() -> None:
    args = parse_args()
    encoded = None
    if args.encoded_json:
        encoded = json.loads(args.encoded_json.read_text())
        request = encoded["request"]
        if set(request) - {"state", "questions", "instructions"}:
            raise ValueError("encoded request may only contain state, questions and instructions")
    else:
        request = {
            "state": DEFAULT_STATE,
            "questions": DEFAULT_QUESTIONS,
            "instructions": "Answer each question using only its supplied labels.",
        }
    adapter = VLLMSystemOneAdapter(args.base_url, args.model, timeout=args.timeout, api_key=args.api_key)
    measurements = benchmark(adapter, request, args)
    result = {
        "status": "PASS",
        "selected_model": args.adapter_name or args.model,
        "request": adapter.build_decision_payload(
            request["state"],
            request["questions"],
            instructions=request.get("instructions"),
            adapter_name=args.adapter_name,
        ),
        "measurements": measurements,
    }
    if encoded is not None:
        scores = adapter.score_labels(encoded["reads"], adapter_name=args.adapter_name)
        comparison = compare_scores(measurements["native_responses"][-1], scores)
        passed = (
            comparison["max_probability_abs_error"] <= args.probability_atol
            and comparison["max_label_mass_abs_error"] <= args.label_mass_atol
            and comparison["decision_agreement"] == 1.0
        )
        result.update(
            status="PASS" if passed else "FAIL",
            encoded_metadata=encoded.get("metadata"),
            label_scores=[score.as_dict() for score in scores],
            comparison=comparison,
            tolerances={
                "probability_atol": args.probability_atol,
                "label_mass_atol": args.label_mass_atol,
            },
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
