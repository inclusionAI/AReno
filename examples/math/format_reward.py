"""Format reward: 1.0 when a completion wraps its final answer in \\boxed{}.

The companion to ``accuracy_reward.py`` in the multi-component reward demo.
It rewards *well-formed* output independent of correctness, so a weighted
combination (e.g. 0.7 accuracy + 0.3 format) can shape both behavior and
presentation. Deterministic, offline, no network or sandbox.
"""

from __future__ import annotations

import re

# Match a non-empty \boxed{...} anywhere in the completion.
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def reward_fn(record) -> float:
    """Score one completion 1.0 if it contains a non-empty ``\\boxed{...}``."""

    return 1.0 if _BOXED_RE.search(record.completion) else 0.0
