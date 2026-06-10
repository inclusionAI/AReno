"""Math verifier reward: 1.0 when `math_verify` accepts the completion.

The math example uses Hugging Face's `math_verify` to symbolically compare
the model's boxed answer against the ground-truth solution. Each completion
is scored independently and returned in input order so the trainer can pair
rewards with rollout sequences.
"""

from __future__ import annotations

from math_verify import parse, verify


def reward_fn(example: dict, completions: list[str]) -> list[float]:
    """Score completions by verifying them against the first math solution."""

    # `solutions` is the canonical field produced by the math dataset loader;
    # fall back to raw `solution` or GSM8K `answer` fields for ad-hoc datasets.
    solutions = example.get("solutions")
    if solutions is None:
        raw = example.get("solution", example.get("answer"))
        if raw is None:
            raise KeyError("math reward expects `solutions`, `solution`, or GSM8K `answer`")
        solutions = [_extract_gsm8k_final_answer(str(raw))]
    ground_truth = solutions[0] if isinstance(solutions, list) else solutions
    gt_parsed = parse(ground_truth)
    rewards = []
    for completion in completions:
        pred_parsed = parse(completion)
        try:
            # Binary reward: 1.0 if the symbolic comparison succeeds, else 0.
            rewards.append(1.0 if verify(gt_parsed, pred_parsed) else 0.0)
        except Exception:
            # `verify` is intentionally tolerant; any unexpected failure is
            # treated as an incorrect answer rather than crashing the trainer.
            rewards.append(0.0)
    return rewards


def _extract_gsm8k_final_answer(answer: str) -> str:
    # GSM8K stores final answers after `####`; non-GSM8K answers pass through.
    return answer.rsplit("####", 1)[-1].strip() if "####" in answer else answer.strip()
