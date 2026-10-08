"""Evaluate daily sandwich-feedback outputs for the competition example."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_generator import generate_records  # noqa: E402
from dataset_loader import make_prompt  # noqa: E402
from game import (  # noqa: E402
    GENERATE_CONTENT_TOOL,
    check_content_relevance,
    check_sandwich_structure,
    load_profile,
    simulate_user_score,
)
from reward import reward_fn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or summarize competition feedback evaluation.")
    parser.add_argument("--dataset-path", help="Optional JSONL diary evaluation set.")
    parser.add_argument("--count", type=int, default=16, help="Generated evaluation record count.")
    parser.add_argument("--seed", type=int, default=9001, help="Generated evaluation seed.")
    parser.add_argument("--output-jsonl", help="Where to write evaluated outputs.")
    parser.add_argument("--report-md", help="Optional Markdown summary report path.")
    parser.add_argument("--compare-jsonl", help="Optional baseline JSONL for before/after report deltas.")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint, usually from `areno serve`.")
    parser.add_argument("--model", default="policy", help="Model name served by AReno.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "areno-agentic"))
    parser.add_argument("--label", default="eval", help="Run label, for example before or after.")
    parser.add_argument("--candidates", type=int, default=2, help="Candidates generated per diary.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N records for quick smoke checks.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens per candidate.")
    parser.add_argument("--request-timeout", type=float, default=180.0, help="Endpoint request timeout in seconds.")
    parser.add_argument("--strip-reasoning", action="store_true", help="Strip Qwen-style <think> traces before scoring.")
    parser.add_argument("--no-think", action="store_true", help="Ask reasoning models to suppress thinking traces.")
    parser.add_argument("--from-jsonl", help="Skip generation and summarize an existing eval JSONL file.")
    args = parser.parse_args()

    if args.from_jsonl:
        rows = read_jsonl(Path(args.from_jsonl))
    else:
        if not args.output_jsonl:
            parser.error("--output-jsonl is required unless --from-jsonl is used")
        records = load_eval_records(args.dataset_path, count=args.count, seed=args.seed)
        if args.limit is not None:
            records = records[: max(args.limit, 0)]
        rows = run_evaluation(
            records,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            label=args.label,
            candidates=args.candidates,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            strip_reasoning=args.strip_reasoning,
            no_think=args.no_think,
        )
        write_jsonl(Path(args.output_jsonl), rows)

    if args.report_md:
        baseline_rows = read_jsonl(Path(args.compare_jsonl)) if args.compare_jsonl else None
        write_report(Path(args.report_md), rows, label=args.label, baseline_rows=baseline_rows)


def load_eval_records(dataset_path: str | None, *, count: int, seed: int) -> list[dict[str, Any]]:
    """Load a fixed diary eval set or generate one deterministically."""

    if dataset_path is None:
        return generate_records(count, seed=seed)
    return read_jsonl(Path(dataset_path))


def run_evaluation(
    records: list[dict[str, Any]],
    *,
    base_url: str | None,
    model: str,
    api_key: str,
    label: str,
    candidates: int,
    max_tokens: int = 128,
    request_timeout: float = 180.0,
    strip_reasoning: bool = False,
    no_think: bool = False,
) -> list[dict[str, Any]]:
    """Generate candidates when an endpoint is supplied, otherwise score placeholders."""

    profile = load_profile()
    rows = []
    for record in records:
        prepared = dict(record)
        prepared["user_profile"] = profile
        prompt = make_prompt(prepared, profile)
        for candidate_index in range(max(int(candidates), 1)):
            content = ""
            error = None
            if base_url:
                try:
                    content = generate_feedback(
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                        prompt=prompt,
                        candidate_index=candidate_index,
                        max_tokens=max_tokens,
                        request_timeout=request_timeout,
                        no_think=no_think,
                    )
                except Exception as exc:  # noqa: BLE001 - surface endpoint failures in the eval artifact.
                    error = str(exc)
            raw_content = content
            if strip_reasoning:
                content = strip_reasoning_traces(content)
            metrics = score_feedback(prepared, content)
            row = {
                "label": label,
                "id": prepared.get("id"),
                "candidate_index": candidate_index,
                "diary": prepared.get("diary", ""),
                "mood": prepared.get("mood", ""),
                "content": content,
                "metrics": metrics,
                "error": error,
            }
            if strip_reasoning and raw_content != content:
                row["raw_content"] = raw_content
            rows.append(row)
    return rows


def generate_feedback(
    *,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    candidate_index: int,
    max_tokens: int = 128,
    request_timeout: float = 180.0,
    no_think: bool = False,
) -> str:
    """Call an OpenAI-compatible model and extract the generate_content tool payload."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install `openai` to call a served model for evaluation.") from exc

    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=request_timeout)
    user_prompt = prompt
    if no_think:
        user_prompt = f"/no_think\n{prompt}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are competing to write helpful daily sandwich feedback. "
                    f"You are candidate {candidate_index}. "
                    "If the model supports it, do not emit private reasoning or <think> tags."
                ),
            },
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": "Call generate_content with the feedback you would show the user."},
        ],
        tools=[GENERATE_CONTENT_TOOL],
        tool_choice={"type": "function", "function": {"name": "generate_content"}},
        max_tokens=max_tokens,
        stream=False,
    )
    message = response.choices[0].message
    calls = [call for call in (message.tool_calls or []) if call.function.name == "generate_content"]
    if not calls:
        return message.content or ""
    try:
        arguments = json.loads(calls[0].function.arguments or "{}")
    except json.JSONDecodeError:
        return ""
    return str(arguments.get("content", ""))


def strip_reasoning_traces(content: str) -> str:
    """Remove Qwen-style reasoning traces from model output before scoring."""

    text = str(content)
    lower = text.lower()
    while "<think>" in lower and "</think>" in lower:
        start = lower.find("<think>")
        end = lower.find("</think>", start)
        if end < start:
            break
        text = text[:start] + text[end + len("</think>") :]
        lower = text.lower()
    start = lower.find("<think>")
    if start >= 0:
        text = text[:start]
    return text.strip()


def score_feedback(record: dict[str, Any], content: str) -> dict[str, float]:
    """Score one generated feedback item with the same rule components used by reward."""

    content = str(content)
    diary = str(record.get("diary", ""))
    profile = record.get("user_profile") or load_profile()
    reward_record = SimpleNamespace(
        source_record=record,
        tool_calls=[
            {"name": "generate_content", "arguments": json.dumps({"content": content}, ensure_ascii=False)},
            {"name": "self_score", "arguments": json.dumps({"score": 0.5, "reason": "eval default"})},
        ],
        metadata={"sample_index": 0},
    )
    return {
        "reward": float(reward_fn(reward_record)),
        "user_score": float(simulate_user_score(content, diary, profile)),
        "structure_score": float(check_sandwich_structure(content)),
        "relevance_score": float(check_content_relevance(content, diary)),
        "specificity_score": float(min(len(content) / 300, 1.0)),
        "length": float(len(content)),
    }


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    label: str,
    baseline_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Write a compact Markdown report for contributor discussion."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summarize(rows)
    examples = rows[: min(5, len(rows))]
    lines = [
        f"# Competition Feedback Evaluation: {label}",
        "",
        "## Summary",
        "",
    ]
    for name in ["reward", "user_score", "structure_score", "relevance_score", "specificity_score", "length"]:
        lines.append(f"- {name}: {metrics.get(name, 0.0):.4f}")
    if baseline_rows is not None:
        baseline_metrics = summarize(baseline_rows)
        lines.extend(["", "## Delta Vs Baseline", ""])
        for name in ["reward", "user_score", "structure_score", "relevance_score", "specificity_score", "length"]:
            delta = metrics.get(name, 0.0) - baseline_metrics.get(name, 0.0)
            lines.append(f"- {name}: {delta:+.4f}")
    errors = [row for row in rows if row.get("error")]
    lines.extend(
        [
            f"- evaluated_outputs: {len(rows)}",
            f"- endpoint_errors: {len(errors)}",
            "",
            "## Review Questions",
            "",
            "1. Does the output mention concrete diary events?",
            "2. Does it include affirmation, improvement, suggestion, and final affirmation?",
            "3. Is the suggestion specific enough to act on tomorrow?",
            "4. Would the user accept the criticism without feeling attacked?",
            "",
            "## Sample Outputs",
            "",
        ]
    )
    for row in examples:
        metrics_text = ", ".join(f"{key}={value:.3f}" for key, value in row["metrics"].items())
        lines.extend(
            [
                f"### Record {row.get('id')} / Candidate {row.get('candidate_index')}",
                "",
                f"Diary: {row.get('diary', '')}",
                "",
                f"Output: {row.get('content', '') or '(empty)'}",
                "",
                f"Metrics: {metrics_text}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average numeric metrics across evaluated rows."""

    if not rows:
        return {}
    metric_names = sorted({name for row in rows for name in row.get("metrics", {})})
    summary = {}
    for name in metric_names:
        values = [float(row["metrics"][name]) for row in rows if name in row.get("metrics", {})]
        summary[name] = statistics.fmean(values) if values else 0.0
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
