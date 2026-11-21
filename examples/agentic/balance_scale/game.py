"""Balance-scale odd-ball game logic for agentic RL.

A set of visually identical balls contains exactly one odd ball that is
either heavier or lighter than the rest.  The agent may call ``weigh`` to
compare two disjoint equal-size groups on a balance scale, and must
eventually call ``answer`` to identify the odd ball and its weight direction.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

ODD_DIRECTIONS = ("heavier", "lighter")
WEIGH_RESULTS = ("left_heavy", "right_heavy", "balanced")


class BalanceGame:
    """One odd-ball balance-scale puzzle instance.

    Attributes:
        num_balls: Total number of balls (0-indexed internally).
        odd_ball_index: Index of the odd ball.
        odd_ball_direction: ``"heavier"`` or ``"lighter"``.
        max_weighings: Maximum allowed weighings before the agent must answer.
    """

    __slots__ = ("num_balls", "odd_ball_index", "odd_ball_direction", "max_weighings", "_weighings_used")

    def __init__(
        self,
        num_balls: int,
        odd_ball_index: int,
        odd_ball_direction: str,
        max_weighings: int,
    ) -> None:
        if num_balls < 3:
            raise ValueError("num_balls must be at least 3")
        if not (0 <= odd_ball_index < num_balls):
            raise ValueError(f"odd_ball_index {odd_ball_index} out of range [0, {num_balls})")
        if odd_ball_direction not in ODD_DIRECTIONS:
            raise ValueError(f"odd_ball_direction must be one of {ODD_DIRECTIONS}")
        if max_weighings < 1:
            raise ValueError("max_weighings must be at least 1")

        self.num_balls = num_balls
        self.odd_ball_index = odd_ball_index
        self.odd_ball_direction = odd_ball_direction
        self.max_weighings = max_weighings
        self._weighings_used = 0

    # -- public API ---------------------------------------------------------

    @property
    def weighings_used(self) -> int:
        """Number of weighings consumed so far."""

        return self._weighings_used

    @property
    def weighings_remaining(self) -> int:
        """Remaining weighings before the agent must answer."""

        return self.max_weighings - self._weighings_used

    def weigh(self, left_group: Sequence[int], right_group: Sequence[int]) -> str:
        """Compare two disjoint equal-size groups on the balance scale.

        Returns one of ``"left_heavy"``, ``"right_heavy"``, ``"balanced"``.
        Raises ``ValueError`` on invalid input or when the budget is exhausted.
        """

        left = list(left_group)
        right = list(right_group)
        self._validate_groups(left, right)
        if self._weighings_used >= self.max_weighings:
            raise ValueError(
                f"weighing budget exhausted ({self.max_weighings} weighings used)"
            )
        self._weighings_used += 1

        odd = self.odd_ball_index
        heavier = self.odd_ball_direction == "heavier"
        odd_in_left = odd in left
        odd_in_right = odd in right

        if not odd_in_left and not odd_in_right:
            return "balanced"
        if odd_in_left:
            return "left_heavy" if heavier else "right_heavy"
        return "right_heavy" if heavier else "left_heavy"

    def check_answer(self, ball_index: int, direction: str) -> tuple[bool, bool]:
        """Verify the agent's final answer.

        Returns ``(identity_correct, direction_correct)``.
        """

        if not (0 <= ball_index < self.num_balls):
            raise ValueError(f"ball_index {ball_index} out of range [0, {self.num_balls})")
        if direction not in ODD_DIRECTIONS:
            raise ValueError(f"direction must be one of {ODD_DIRECTIONS}")
        identity_correct = ball_index == self.odd_ball_index
        direction_correct = direction == self.odd_ball_direction
        return (identity_correct, direction_correct)

    # -- internal helpers ---------------------------------------------------

    def _validate_groups(self, left: list[int], right: list[int]) -> None:
        for group, label in ((left, "left"), (right, "right")):
            for idx in group:
                if not isinstance(idx, int) or isinstance(idx, bool):
                    raise ValueError(f"{label}_group contains non-integer index: {idx!r}")
                if not (0 <= idx < self.num_balls):
                    raise ValueError(f"{label}_group index {idx} out of range [0, {self.num_balls})")
        if len(left) != len(right):
            raise ValueError(
                f"groups must have equal size: left={len(left)}, right={len(right)}"
            )
        if not left:
            raise ValueError("groups must not be empty")
        overlap = set(left) & set(right)
        if overlap:
            raise ValueError(f"groups must be disjoint: overlap={sorted(overlap)}")
        duplicates = {idx for idx in left if left.count(idx) > 1}
        if duplicates:
            raise ValueError(f"left_group has duplicate indices: {sorted(duplicates)}")
        duplicates = {idx for idx in right if right.count(idx) > 1}
        if duplicates:
            raise ValueError(f"right_group has duplicate indices: {sorted(duplicates)}")


def generate_game(
    num_balls: int = 9,
    *,
    seed: int | None = None,
    max_weighings: int = 3,
) -> BalanceGame:
    """Create a random puzzle with a seeded RNG."""

    if num_balls < 3:
        raise ValueError("num_balls must be at least 3")
    rng = random.Random(seed)
    odd_index = rng.randint(0, num_balls - 1)
    direction = rng.choice(ODD_DIRECTIONS)
    return BalanceGame(
        num_balls=num_balls,
        odd_ball_index=odd_index,
        odd_ball_direction=direction,
        max_weighings=max_weighings,
    )


def format_prompt(num_balls: int, max_weighings: int) -> str:
    """Build the user-facing prompt for the tool-call variant."""

    return (
        f"You have {num_balls} visually identical balls numbered 0 to {num_balls - 1}. "
        f"Exactly one ball is odd — it is either heavier or lighter than the rest. "
        f"You have a balance scale and may use it at most {max_weighings} time(s). "
        f"Call the weigh tool to compare two equal-size disjoint groups of balls. "
        f"When you know the answer, call the answer tool with the odd ball index "
        f"and whether it is heavier or lighter."
    )


def format_xml_prompt(num_balls: int, max_weighings: int) -> str:
    """Build the user-facing prompt for the XML no-tool variant."""

    return (
        f"You have {num_balls} visually identical balls numbered 0 to {num_balls - 1}. "
        f"Exactly one ball is odd — it is either heavier or lighter than the rest. "
        f"You have a balance scale and may use it at most {max_weighings} time(s).\n\n"
        f"To weigh, output a tag like: <weigh left=\"0,1\" right=\"2,3\"/>\n"
        f"You will receive the result (left_heavy, right_heavy, or balanced) as the next message.\n\n"
        f"To submit your answer, output a tag like: "
        f"<answer ball=\"3\" direction=\"heavier\"/>\n"
        f"Direction must be \"heavier\" or \"lighter\".\n\n"
        f"You may reason step by step before each tag. "
        f"Use at most {max_weighings} weigh tags, then use one answer tag."
    )


# ---------------------------------------------------------------------------
# XML parsing helpers for the no-tool variant
# ---------------------------------------------------------------------------

_WEIGH_RE = re.compile(
    r'<weigh\s+left\s*=\s*"([^"]*)"\s+right\s*=\s*"([^"]*)"\s*/?>',
    re.IGNORECASE | re.DOTALL,
)
_ANSWER_RE = re.compile(
    r'<answer\s+ball\s*=\s*"(\d+)"\s+direction\s*=\s*"(heavier|lighter)"\s*/?>',
    re.IGNORECASE | re.DOTALL,
)


def parse_xml_weigh(text: str) -> tuple[list[int], list[int]] | None:
    """Extract the last <weigh> tag from *text*.

    Returns ``(left_group, right_group)`` or ``None`` when no tag is found.
    """

    matches = list(_WEIGH_RE.finditer(text))
    if not matches:
        return None
    left_str, right_str = matches[-1].group(1), matches[-1].group(2)
    left = _parse_int_list(left_str)
    right = _parse_int_list(right_str)
    if left is None or right is None:
        return None
    return (left, right)


def parse_xml_answer(text: str) -> tuple[int, str] | None:
    """Extract the last <answer> tag from *text*.

    Returns ``(ball_index, direction)`` or ``None`` when no tag is found.
    """

    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    ball_index = int(matches[-1].group(1))
    direction = matches[-1].group(2).lower()
    return (ball_index, direction)


def _parse_int_list(text: str) -> list[int] | None:
    """Parse a comma-separated integer list like '0,1,2'."""

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None
