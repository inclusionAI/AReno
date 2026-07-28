"""Trainable-token statistics across trainable_turns modes x mask_tool_call_args.

CPU-only: no GPU, no network.  Builds deterministic multi-tool trajectory
fixtures, runs them through AReno's real ``RolloutSession._train_rows_from_samples``
pipeline, and tabulates trainable_tokens / masked_response_tokens /
total_response_tokens for each configuration.

This is the research artifact for issue #199 G8: shows how supervision density
varies across trainable-turn selection modes on the same trajectory data.

Base mask note: all fixtures use ``loss_mask_override = [True]*n`` as the
starting point (no ``_tool_call_loss_mask`` pre-suppression). This isolates the
effect of trainable_turns mode and mask_tool_call_args. In live rollout the
tool-result region inside tool-call spans may be pre-zeroed by
``_tool_call_loss_mask``, further reducing counts; this script reports the
mode/args effect in isolation.

Usage::

    python examples/agentic/trainable_turns_stats.py
    python examples/agentic/trainable_turns_stats.py --json stats.json
    python examples/agentic/trainable_turns_stats.py --csv stats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from areno.api.agentic import (
    AgentBatch,
    LossMaskPolicy,
    ResponseSpan,
    RolloutSession,
    _AgentSample,
    RewardEvent,
)

MODES = ("all_assistant", "last_assistant", "final_answer")

# -- Tokenizer ---------------------------------------------------------------


class _CharTokenizer:
    """Round-trip character tokenizer: token id = ord(char)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class _FakeTrainer:
    """Minimal trainer with a tokenizer for _apply_trainable_turn_mode."""

    def __init__(self) -> None:
        self.tokenizer = _CharTokenizer()

    def get_tokenizer(self) -> _CharTokenizer:
        return self.tokenizer

    def dp_size(self) -> int:
        return 1


# -- Fixtures ---------------------------------------------------------------


def _tc(name: str, **kwargs: object) -> str:
    """Generate a compact tool-call JSON string with *name* and *kwargs*."""
    args = json.dumps(kwargs, separators=(",", ":"))
    return f'{{"name":"{name}","arguments":{args}}}'


FIXTURES: list[dict] = [
    {
        "name": "simple",
        "desc": "text -> tool_call -> text (single tool)",
        "turns": [
            ("assistant_text", "Let me search that for you."),
            ("assistant_tool_call", _tc("search", query="rlhf")),
            ("assistant_text", "Found 3 relevant papers on this topic."),
        ],
        "tool_results": 1,
    },
    {
        "name": "double_tool",
        "desc": "text -> tool_call -> tool_call -> text (consecutive tools)",
        "turns": [
            ("assistant_text", "I'll look this up."),
            ("assistant_tool_call", _tc("lookup", id="42")),
            ("assistant_tool_call", _tc("format", style="yaml")),
            ("assistant_text", "Here is the formatted output."),
        ],
        "tool_results": 2,
    },
    {
        "name": "multi_step",
        "desc": "text -> tool_call -> text -> tool_call -> text",
        "turns": [
            ("assistant_text", "I need to investigate this step by step."),
            ("assistant_tool_call", _tc("search", query="rlhf papers 2024")),
            ("assistant_text", "Good, the first search confirms my hypothesis."),
            ("assistant_tool_call", _tc("fetch", url="arxiv.org", limit=3)),
            ("assistant_text", "Here is the final summary."),
        ],
        "tool_results": 2,
    },
    {
        "name": "no_tools",
        "desc": "text only (degenerate, no tool calls)",
        "turns": [
            ("assistant_text", "The answer involves considering multiple factors carefully."),
        ],
        "tool_results": 0,
    },
    {
        "name": "bare_trailing",
        "desc": "text -> tool_call (trailing bare call, zero signal for final_answer)",
        "turns": [
            ("assistant_text", "Starting the computation now."),
            ("assistant_tool_call", _tc("execute", command="run --args")),
        ],
        "tool_results": 0,
    },
    {
        "name": "deep_trajectory",
        "desc": "text -> tool -> text -> tool -> text -> tool -> text (3 tools)",
        "turns": [
            ("assistant_text", "Beginning analysis of the problem."),
            ("assistant_tool_call", _tc("query", table="users", limit=10)),
            ("assistant_text", "Data retrieved. Running correlation analysis."),
            ("assistant_tool_call", _tc("compute", method="pearson")),
            ("assistant_text", "Correlation done. Testing significance."),
            ("assistant_tool_call", _tc("test", alpha=0.05)),
            ("assistant_text", "Results are statistically significant."),
        ],
        "tool_results": 3,
    },
]


# -- Sample builder ---------------------------------------------------------


def _build_sample(fixture: dict, tokenizer: _CharTokenizer) -> _AgentSample:
    turns: list[tuple[str, str]] = fixture["turns"]
    span_texts = [text for _, text in turns]
    response_text = "".join(span_texts)
    response_tokens = tokenizer.encode(response_text)
    spans = [ResponseSpan(kind, len(text)) for kind, text in turns]

    trace: list[RewardEvent] = []
    for kind, text in turns:
        if kind == "assistant_tool_call":
            trace.append(RewardEvent(type="assistant_tool_call", name="tool", arguments="{}"))
        else:
            trace.append(RewardEvent(type="assistant_text", text=text))

    messages: list[dict] = [{"role": "user", "content": "task"}]
    for i in range(fixture["tool_results"]):
        messages.append({"role": "tool", "content": f"result-{i}"})

    item = next(
        AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples()
    )
    return _AgentSample(
        item=item,
        messages=messages,
        response_text=response_text,
        last_response_text=span_texts[-1] if span_texts else "",
        response_tokens=response_tokens,
        response_logprobs=[0.0] * len(response_tokens),
        trace=trace,
        response_spans=spans,
        loss_mask_override=[True] * len(response_tokens),
    )


# -- Statistics --------------------------------------------------------------


def _run_config(fixture: dict, mode: str, mask_args: bool) -> dict:
    trainer = _FakeTrainer()
    session = RolloutSession(
        trainer,
        sampling_params=None,
        loss_mask_policy=LossMaskPolicy(trainable_turns=mode, mask_tool_call_args=mask_args),
    )
    sample = _build_sample(fixture, trainer.get_tokenizer())
    rows = session._train_rows_from_samples([sample])
    total_response = sum(sum(rm) for rm in rows.response_masks)
    return {
        "fixture": fixture["name"],
        "desc": fixture["desc"],
        "mode": mode,
        "mask_tool_call_args": mask_args,
        "total_response_tokens": total_response,
        "trainable_tokens": rows.trainable_tokens,
        "masked_response_tokens": rows.masked_response_tokens,
    }


def _print_table(results: list[dict]) -> None:
    by_fixture: dict[str, list[dict]] = {}
    for r in results:
        by_fixture.setdefault(r["fixture"], []).append(r)

    print("=" * 100)
    print("Trainable-Token Statistics -- issue #199 G8 (CPU-only, real RolloutSession pipeline)")
    print("=" * 100)

    for fixture_name in [f["name"] for f in FIXTURES]:
        rows = by_fixture[fixture_name]
        desc = rows[0]["desc"]
        total = rows[0]["total_response_tokens"]
        if total == 0:
            print(f"\n  [{fixture_name}] {desc}  (total=0, skipped)")
            continue
        print(f"\n  [{fixture_name}] {desc}")
        print(f"  total_response_tokens = {total}")
        print(
            f"  {'mode':16s} {'mask_args':10s} {'trainable':>10s} {'masked':>8s}"
            f" {'ratio':>8s} {'vs_all':>8s}"
        )
        print(f"  {'-' * 16} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8}")

        baseline = next(
            r["trainable_tokens"]
            for r in rows
            if r["mode"] == "all_assistant" and not r["mask_tool_call_args"]
        )
        for r in rows:
            ratio = r["trainable_tokens"] / total if total else 0.0
            vs_all = f"{r['trainable_tokens'] / baseline:.2f}" if baseline else "N/A"
            ma = "Y" if r["mask_tool_call_args"] else "-"
            print(
                f"  {r['mode']:16s} {ma:10s} {r['trainable_tokens']:>10d}"
                f" {r['masked_response_tokens']:>8d} {ratio:>8.1%} {vs_all:>8s}"
            )

    # Summary
    grand_total = sum(r["total_response_tokens"] for r in results) // len(MODES) // 2
    print(f"\n{'=' * 100}")
    print("Summary: aggregate trainable ratio across all fixtures")
    print(f"{'=' * 100}")
    for mode in MODES:
        for mask_args in (False, True):
            subset = [r for r in results if r["mode"] == mode and r["mask_tool_call_args"] == mask_args]
            total_pool = sum(r["total_response_tokens"] for r in subset)
            trainable_pool = sum(r["trainable_tokens"] for r in subset)
            ratio = trainable_pool / total_pool if total_pool else 0.0
            print(f"  {mode:16s} mask_args={'Y' if mask_args else '-':3s}  "
                  f"total={total_pool:5d}  trainable={trainable_pool:5d}  "
                  f"ratio={ratio:6.1%}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", metavar="PATH", help="Write results as JSON")
    parser.add_argument("--csv", metavar="PATH", help="Write results as CSV")
    args = parser.parse_args()

    results: list[dict] = []
    for fixture in FIXTURES:
        for mode in MODES:
            for mask_args in (False, True):
                results.append(_run_config(fixture, mode, mask_args))

    _print_table(results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON written: {args.json}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV written: {args.csv}")


if __name__ == "__main__":
    main()