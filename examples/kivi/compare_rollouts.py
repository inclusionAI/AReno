"""Compare two greedy rollout dumps from ``rollout_eval.py`` prompt by prompt.

Greedy decoding makes the two runs token-identical until the first position
where the candidate's KV cache changes the argmax. Up to that point both runs
scored the same token after the same context, so the absolute logprob
difference isolates the KV-cache error in the behavior-policy logprobs.

    python examples/kivi/compare_rollouts.py runs/auto runs/kivi2 --output kivi2_vs_auto.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BUCKETS = (128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 32768)


def _load(path: Path) -> list[dict]:
    with (path / "rollouts.jsonl").open() as handle:
        return [json.loads(line) for line in handle]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _bucket(position: int) -> str:
    for edge in BUCKETS:
        if position < edge:
            return f"<{edge}"
    return f">={BUCKETS[-1]}"


def compare(baseline: list[dict], candidate: list[dict]) -> dict:
    if len(baseline) != len(candidate):
        raise ValueError(f"prompt count mismatch: {len(baseline)} vs {len(candidate)}")
    identical = 0
    agreed_tokens = 0
    baseline_tokens = 0
    divergence = []
    abs_diff_by_bucket: dict[str, list[float]] = {}
    for base, cand in zip(baseline, candidate, strict=True):
        if base["index"] != cand["index"] or base["prompt_tokens"] != cand["prompt_tokens"]:
            raise ValueError(f"rows are not aligned at index {base['index']}")
        base_tokens, cand_tokens = base["resp_tokens"], cand["resp_tokens"]
        common = 0
        for left, right in zip(base_tokens, cand_tokens, strict=False):
            if left != right:
                break
            common += 1
        identical += int(base_tokens == cand_tokens)
        agreed_tokens += common
        baseline_tokens += len(base_tokens)
        if base_tokens != cand_tokens:
            divergence.append(common)
        for offset in range(common):
            diff = abs(base["resp_logprobs"][offset] - cand["resp_logprobs"][offset])
            abs_diff_by_bucket.setdefault(_bucket(base["prompt_tokens"] + offset), []).append(diff)
    all_diffs = [value for values in abs_diff_by_bucket.values() for value in values]
    accuracy_base = sum(row["reward"] for row in baseline) / len(baseline)
    accuracy_cand = sum(row["reward"] for row in candidate) / len(candidate)
    return {
        "prompts": len(baseline),
        "accuracy_baseline": accuracy_base,
        "accuracy_candidate": accuracy_cand,
        "accuracy_delta": accuracy_cand - accuracy_base,
        "identical_response_rate": identical / len(baseline),
        "agreed_token_fraction": agreed_tokens / max(1, baseline_tokens),
        "mean_first_divergence_token": sum(divergence) / len(divergence) if divergence else None,
        "logprob_abs_diff_mean_on_agreed_prefix": sum(all_diffs) / len(all_diffs) if all_diffs else None,
        "logprob_abs_diff_by_absolute_position": {
            bucket: {"tokens": len(abs_diff_by_bucket[bucket]), "mean_abs_diff": _mean(abs_diff_by_bucket[bucket])}
            for bucket in [f"<{edge}" for edge in BUCKETS] + [f">={BUCKETS[-1]}"]
            if bucket in abs_diff_by_bucket
        },
        "mean_response_tokens_baseline": sum(len(row["resp_tokens"]) for row in baseline) / len(baseline),
        "mean_response_tokens_candidate": sum(len(row["resp_tokens"]) for row in candidate) / len(candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = compare(_load(args.baseline), _load(args.candidate))
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
