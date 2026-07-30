"""Balance-scale helpers for the odd-ball agentic example.

The example is intentionally small: a set of visually identical balls where
exactly one is heavier or lighter. The agent uses a balance-scale tool
(``weigh``) to compare two equal-size disjoint groups and a final-answer
action (``submit_answer``) to identify the odd ball and its weight direction.

The environment is deterministic and self-contained -- no network services,
no sandbox, no external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

HEAVIER = "heavier"
LIGHTER = "lighter"
DIRECTIONS = (HEAVIER, LIGHTER)


@dataclass(frozen=True)
class BallSet:
    """One odd-ball puzzle instance.

    ``num_balls`` visually identical balls numbered ``0 .. num_balls - 1``
    have exactly one odd ball at ``odd_ball_index`` that is either heavier
    or lighter (``direction``) than the rest. The agent has at most
    ``max_weighings`` calls to :func:`weigh` before it must submit an answer.
    """

    num_balls: int
    odd_ball_index: int
    direction: str
    max_weighings: int

    def __post_init__(self) -> None:
        if self.num_balls < 2:
            raise ValueError("num_balls must be at least 2")
        if not (0 <= self.odd_ball_index < self.num_balls):
            raise ValueError(
                f"odd_ball_index {self.odd_ball_index} out of range [0, {self.num_balls})"
            )
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.max_weighings < 1:
            raise ValueError("max_weighings must be at least 1")

    @property
    def min_weighings(self) -> int:
        """Information-theoretic minimum weighings to solve this puzzle."""

        import math
        return max(1, math.ceil(math.log(self.num_balls * 2, 3)))


def make_ball_set(
    num_balls: int = 12,
    *,
    odd_ball_index: int | None = None,
    direction: str | None = None,
    max_weighings: int = 3,
    seed: int | None = None,
) -> BallSet:
    """Create a puzzle instance, randomly choosing the odd ball if not given.

    When ``seed`` is provided the choice is deterministic so that dataset
    generation is reproducible.
    """

    rng = random.Random(seed)
    if odd_ball_index is None:
        odd_ball_index = rng.randint(0, num_balls - 1)
    if direction is None:
        direction = rng.choice(DIRECTIONS)
    return BallSet(
        num_balls=num_balls,
        odd_ball_index=odd_ball_index,
        direction=direction,
        max_weighings=max_weighings,
    )


def validate_group(group: list[int], num_balls: int) -> None:
    """Validate one group of ball indices for a weighing."""

    if not isinstance(group, list):
        raise ValueError("ball group must be a list")
    if not group:
        raise ValueError("ball group must not be empty")
    for ball in group:
        if not isinstance(ball, int) or isinstance(ball, bool):
            raise ValueError(f"ball index must be an int, got {ball!r}")
        if not (0 <= ball < num_balls):
            raise ValueError(f"ball index {ball} out of range [0, {num_balls})")


def weigh(
    ball_set: BallSet,
    left: list[int],
    right: list[int],
    *,
    weighings_used: int,
) -> str:
    """Simulate one balance-scale weighing.

    Returns ``"left_heavy"``, ``"right_heavy"``, or ``"balanced"``.

    Raises ``ValueError`` if the groups are invalid, overlap, have unequal
    sizes, or the weighing budget has been exhausted.
    """

    if weighings_used >= ball_set.max_weighings:
        raise ValueError(
            f"weighing budget exhausted: used {weighings_used}/{ball_set.max_weighings}"
        )

    validate_group(left, ball_set.num_balls)
    validate_group(right, ball_set.num_balls)

    if len(left) != len(right):
        raise ValueError(
            f"groups must have equal size: left={len(left)}, right={len(right)}"
        )

    left_set = set(left)
    right_set = set(right)
    if left_set & right_set:
        raise ValueError(
            f"groups must be disjoint: overlap={sorted(left_set & right_set)}"
        )

    odd = ball_set.odd_ball_index
    direction = ball_set.direction

    left_has_odd = odd in left_set
    right_has_odd = odd in right_set

    if not left_has_odd and not right_has_odd:
        return "balanced"

    # The odd ball is on one side. heavier → that side goes down; lighter → the other side.
    if left_has_odd:
        return "left_heavy" if direction == HEAVIER else "right_heavy"
    return "right_heavy" if direction == HEAVIER else "left_heavy"


def check_answer(
    ball_set: BallSet, ball_index: int, direction: str
) -> dict[str, bool | str]:
    """Verify a submitted answer against the ground truth.

    Returns a dict with ``ball_correct``, ``direction_correct``, and ``full_correct``.
    """

    ball_correct = ball_index == ball_set.odd_ball_index
    direction_correct = direction == ball_set.direction
    return {
        "ball_correct": ball_correct,
        "direction_correct": direction_correct,
        "full_correct": ball_correct and direction_correct,
    }


def format_prompt(ball_set: BallSet) -> str:
    """Build the user-facing prompt for one puzzle instance."""

    min_w = ball_set.min_weighings
    return (
        f"There are {ball_set.num_balls} balls numbered 0 to {ball_set.num_balls - 1}. "
        "They look identical, but exactly one ball is either heavier or lighter "
        "than all the others.\n\n"
        "You have a balance scale. Call the weigh tool with two equal-size "
        "disjoint lists of ball indices to compare them. The scale will return "
        "'left_heavy', 'right_heavy', or 'balanced'.\n\n"
        f"Theoretically, this puzzle can be solved in {min_w} weighings. "
        "Try to use as few weighings as possible. When you are ready, call "
        "submit_answer with the ball index and direction ('heavier' or 'lighter').\n\n"
        "Example tool calls:\n"
        '  weigh: {"left": [0, 1], "right": [2, 3]}  → "balanced"\n'
        '  weigh: {"left": [4], "right": [5]}  → "left_heavy"\n'
        '  submit_answer: {"ball_index": 4, "direction": "heavier"}\n'
    )


def format_system_prompt() -> str:
    """Build the system prompt for the agent."""

    return (
        "You are a logical reasoner solving an odd-ball balance-scale puzzle. "
        "Use the weigh tool to compare two equal-size disjoint groups of balls. "
        "After each weighing, reason about which balls could still be the odd one. "
        "When you have narrowed it down, call submit_answer with the ball index "
        "and direction (heavier or lighter). Minimise the number of weighings.\n\n"
        "IMPORTANT: Do not use thinking mode. Output your tool call directly "
        "without any /think tags. Respond ONLY with a tool call in this exact format:\n"
        '{"left": [0, 1], "right": [2, 3]}  for weighing, or\n'
        '{"ball_index": 5, "direction": "heavier"}  for submitting your answer.'
    )
