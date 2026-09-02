"""Held-out single-turn evaluation for the Bash game (巴什博弈).

Evaluates a served policy checkpoint against a fixed JSONL split and reports
mean reward, exact-solve rate (optimal move / resign), legal-move rate, and a
bucketed breakdown by ``m`` (max take, the difficulty axis).

Usage (run an ``areno serve`` on some port first, then):

    python examples/agentic/bash_game/eval.py \
      --dataset /new/bash_data/test.jsonl \
      --base-url http://127.0.0.1:8000/v1 \
      --api-key token --model policy \
      --n-samples 1 --temperature 1.0 --seed 7 \
      --out /new/eval_test_seed7.json

It reuses the exact ``reward_fn`` used in training so baseline and post-training
numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import format_prompt, normalize_record, parse_move, tool_schema  # noqa: E402
from reward import reward_fn  # noqa: E402

SYSTEM_PROMPT = (
    "You are a perfect Bash-game (取石子游戏) strategist. Output exactly one "
    "submit_move tool call: take k stones for a winning move, or resign in a "
    "losing position. Do not narrate; only call the tool once."
)


class FakeRecord:
    def __init__(self, source: dict, tool_calls) -> None:
        self.source_record = source
        self.tool_calls = tool_calls


def load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(normalize_record(json.loads(line)))
    return records


def evaluate(*, records, base_url, api_key, model, n_samples, temperature, seed, workers=8) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=3)
    tool = tool_schema()

    def run_record(rec) -> dict:
        prompt = format_prompt(rec)
        oracle = rec["oracle_move"]
        rewards = []
        solved = []
        legal = []
        for _ in range(n_samples):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=[tool],
                tool_choice="required",
                temperature=temperature,
                seed=seed,
            )
            message = response.choices[0].message
            tool_calls = []
            for call in message.tool_calls or []:
                tool_calls.append({"name": call.function.name, "arguments": call.function.arguments})
            fake = FakeRecord(rec, tool_calls)
            r = reward_fn(fake)
            rewards.append(r)
            move = parse_move(fake)
            solved.append(move is not None and move == oracle)
            legal.append(move is not None)
        return {
            "m": rec["m"],
            "n": rec["n"],
            "winning": rec["winning"],
            "mean_reward": sum(rewards) / len(rewards),
            "solved": any(solved),
            "legal": any(legal),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_record, records))

    n = len(rows)
    mean_reward = sum(r["mean_reward"] for r in rows) / n if n else 0.0
    solved_frac = sum(1 for r in rows if r["solved"]) / n if n else 0.0
    legal_frac = sum(1 for r in rows if r["legal"]) / n if n else 0.0

    by_m = defaultdict(lambda: {"n": 0, "reward": 0.0, "solved": 0})
    for r in rows:
        b = by_m[r["m"]]
        b["n"] += 1
        b["reward"] += r["mean_reward"]
        b["solved"] += int(r["solved"])
    by_m_out = {}
    for m, b in sorted(by_m.items()):
        by_m_out[str(m)] = {
            "n": b["n"],
            "mean_reward": round(b["reward"] / b["n"], 4),
            "solved_frac": round(b["solved"] / b["n"], 4),
        }

    return {
        "n": n,
        "mean_reward": round(mean_reward, 4),
        "solved_frac": round(solved_frac, 4),
        "legal_frac": round(legal_frac, 4),
        "by_m": by_m_out,
        "eval_seed": seed,
        "temperature": temperature,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a served Bash-game policy.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="token")
    p.add_argument("--model", default="policy")
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    records = load_records(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    result = evaluate(
        records=records,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        n_samples=args.n_samples,
        temperature=args.temperature,
        seed=args.seed,
        workers=args.workers,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
