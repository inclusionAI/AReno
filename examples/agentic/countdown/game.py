"""Small Countdown arithmetic helpers for agentic examples.

Countdown is a numbers game: given a set of numbers and a target, the
player picks two numbers and an operation (+, -, *, /) to produce a result
as close to the target as possible.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable

# Allowed operations.
OPERATIONS = ("+", "-", "*", "/")

# Regex helpers for XML-style fallback (when tools are not used).
_XML_CALC_RE = re.compile(
    r"<calc>\s*(\d+)\s*([+\-*/])\s*(\d+)\s*</calc>", re.IGNORECASE | re.DOTALL
)
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_CHAT_SPECIAL_RE = re.compile(r"<\|[^>]+?\|>|</?s>", re.IGNORECASE)


def normalize_numbers(numbers: Iterable[int]) -> list[int]:
    """Return a validated list of positive integers for the puzzle."""

    nums = [int(n) for n in numbers]
    if len(nums) < 2:
        raise ValueError("Countdown puzzle needs at least 2 numbers")
    if any(n <= 0 for n in nums):
        raise ValueError("Countdown numbers must be positive")
    return nums


def format_prompt(numbers: list[int], target: int) -> str:
    """Build the one-step prompt for the tool-call agent."""

    nums_str = ", ".join(str(n) for n in numbers)
    return (
        "You are playing the Countdown numbers game.\n\n"
        f"Available numbers: {nums_str}\n"
        f"Target: {target}\n\n"
        "Rules:\n"
        "- Pick exactly TWO numbers from the list above.\n"
        "- Choose one operation: +, -, *, or /.\n"
        "- The result should be as close to the target as possible.\n"
        "- Each number can only be used once.\n"
        "- Call the calculate tool with your choice.\n\n"
        "Move:"
    )


def format_xml_prompt(numbers: list[int], target: int) -> str:
    """Build the one-step prompt for the XML no-tool agent."""

    nums_str = ", ".join(str(n) for n in numbers)
    return (
        "You are playing the Countdown numbers game.\n\n"
        f"Available numbers: {nums_str}\n"
        f"Target: {target}\n\n"
        "Rules:\n"
        "- Pick exactly TWO numbers from the list above.\n"
        "- Choose one operation: +, -, *, or /.\n"
        "- The result should be as close to the target as possible.\n"
        "- Each number can only be used once.\n"
        '- Answer with exactly one XML tag such as <calc>25 * 10</calc>.\n\n'
        "Move:"
    )


def calculate(a: int, b: int, op: str) -> int | float | None:
    """Execute a single arithmetic operation.

    Returns the result, or ``None`` if the operation is invalid (e.g.
    division by zero or unknown operator).
    """

    if op not in OPERATIONS:
        return None
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            return None
        result = a / b
        # Only allow integer results for division (classic Countdown rule).
        if result != int(result):
            return None
        return int(result)
    return None


def score_move(
    numbers: list[int], target: int, a: int | None, b: int | None, op: str | None
) -> float:
    """Score one calculate move.

    Scoring:
    - Result exactly equals target → 1.0
    - Result is close to target → scaled by proximity
    - Invalid operation or numbers not in the list → -1.0
    - Valid but result is None → -1.0
    """

    if a is None or b is None or op is None:
        return -1.0

    nums = normalize_numbers(numbers)

    # Check that a and b are available in the number list.
    # If a == b, the number must appear at least twice.
    available = list(nums)
    for val in (a, b):
        if val not in available:
            return -1.0
        available.remove(val)

    result = calculate(a, b, op)
    if result is None:
        return -1.0

    if result == target:
        return 1.0

    # Scale by proximity: closer to target = higher score.
    distance = abs(result - target)
    if target == 0:
        return 0.0 if result == 0 else -0.5
    proximity = max(0.0, 1.0 - distance / target)
    # Clamp to [0, 0.8] so only exact match gets 1.0.
    return min(0.8, proximity)


def best_score(numbers: list[int], target: int) -> float:
    """Return the best possible score across all valid two-number moves.

    Used as a reference baseline (not required for reward, but useful
    for diagnostics and dataset generation).
    """

    nums = normalize_numbers(numbers)
    best = -1.0
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if i == j:
                continue
            for op in OPERATIONS:
                s = score_move(nums, target, a, b, op)
                if s > best:
                    best = s
    return best


def parse_xml_calc(text: str) -> tuple[int | None, int | None, str | None]:
    """Extract the final XML calculation from a model response.

    Returns ``(a, b, op)`` or ``(None, None, None)`` if not found.
    """

    text = strip_chat_special_tokens(strip_think_tags(text)).strip()
    matches = list(_XML_CALC_RE.finditer(text))
    if not matches:
        return None, None, None
    match = matches[-1]
    return int(match.group(1)), int(match.group(3)), match.group(2)


def strip_think_tags(text: str) -> str:
    """Remove reasoning spans before parsing the policy action."""

    return _THINK_RE.sub(" ", text)


def strip_chat_special_tokens(text: str) -> str:
    """Remove chat-template sentinels that may trail generated text."""

    return _CHAT_SPECIAL_RE.sub(" ", text)


# ---------------------------------------------------------------------------
# Random-policy baseline, trace replay, and evaluation metrics.
# ---------------------------------------------------------------------------


def random_baseline(
    numbers: list[int], target: int, *, seed: int = 0
) -> tuple[int, int, str]:
    """Pick two random numbers and a random operation.

    Returns ``(a, b, op)``.  Useful as a baseline to compare against
    a trained policy.
    """

    rng = random.Random(seed)
    nums = normalize_numbers(numbers)
    idx_a, idx_b = rng.sample(range(len(nums)), 2)
    a, b = nums[idx_a], nums[idx_b]
    op = rng.choice(OPERATIONS)
    return a, b, op


def random_baseline_score(
    numbers: list[int], target: int, *, seed: int = 0, trials: int = 100
) -> float:
    """Average reward of a random policy over *trials* samples."""

    scores = []
    for i in range(trials):
        a, b, op = random_baseline(numbers, target, seed=seed + i)
        scores.append(score_move(numbers, target, a, b, op))
    return sum(scores) / len(scores) if scores else 0.0


def format_trace(
    numbers: list[int], target: int, a: int, b: int, op: str
) -> str:
    """Render one move as a human-readable trace line.

    Example output::

        Puzzle: numbers=[1, 5, 10, 25], target=525
        Move:   25 * 10 = 250
        Score:  0.4762 (distance=275)
    """

    result = calculate(a, b, op)
    score = score_move(numbers, target, a, b, op)
    if result is None:
        result_str = "invalid"
        distance_str = "n/a"
    else:
        result_str = str(result)
        distance_str = str(abs(result - target))
    return (
        f"Puzzle: numbers={numbers}, target={target}\n"
        f"Move:   {a} {op} {b} = {result_str}\n"
        f"Score:  {score:.4f} (distance={distance_str})"
    )


def evaluate_moves(
    numbers: list[int],
    target: int,
    moves: list[tuple[int | None, int | None, str | None]],
) -> dict:
    """Evaluate a batch of moves and return aggregate metrics.

    Returns a dict with:
    - ``total``: number of moves
    - ``exact_solves``: moves that exactly hit the target
    - ``invalid_actions``: moves with invalid operation or unavailable numbers
    - ``valid_actions``: moves that are legal but may not hit target
    - ``exact_solve_rate``: exact_solves / total
    - ``invalid_action_rate``: invalid_actions / total
    - ``mean_reward``: average reward across all moves
    - ``best_reward``: best reward across all moves
    - ``excess_steps``: extra steps the policy took beyond the oracle solver.
      For single-step Countdown the oracle always uses exactly one move,
      so ``excess_steps`` is ``max(0, actual_steps - 1)``.  Since each entry
      in *moves* represents one policy step, ``excess_steps`` equals
      ``max(0, len(moves) - 1)`` — 0 when the policy makes a single move,
      >0 when it wastes additional steps.
    """

    total = len(moves)
    exact = 0
    invalid = 0
    valid = 0
    rewards: list[float] = []

    for a, b, op in moves:
        score = score_move(numbers, target, a, b, op)
        rewards.append(score)
        if score == 1.0:
            exact += 1
        if score < 0:
            invalid += 1
        else:
            valid += 1

    return {
        "total": total,
        "exact_solves": exact,
        "invalid_actions": invalid,
        "valid_actions": valid,
        "exact_solve_rate": exact / total if total else 0.0,
        "invalid_action_rate": invalid / total if total else 0.0,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "best_reward": max(rewards) if rewards else 0.0,
        # Oracle solver always uses exactly 1 step in single-step Countdown.
        # excess_steps measures how many extra moves the policy wasted.
        "excess_steps": max(0, total - 1),
    }


def oracle_solve(numbers: list[int], target: int) -> float:
    """Return the best achievable score (oracle solver).

    For single-step Countdown, the oracle simply tries all valid
    two-number combinations and returns the highest score.
    """

    return best_score(numbers, target)