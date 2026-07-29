"""Accuracy reward: 1.0 when a completion's final answer matches the gold answer.

This is one half of the multi-component reward demo in
`docs/ISSUE-组合奖励函数-分析与方案-CN.md`. Combine it with
``format_reward.py`` via repeatable ``--reward-fn-path``, e.g.::

    --reward-fn-path examples/math/accuracy_reward.py:0.7 \
    --reward-fn-path examples/math/format_reward.py:0.3
"""

from __future__ import annotations

import re

# A loose "boxed answer" extractor: capture the contents of the last \boxed{...}
# in the text, falling back to the trailing number/word when no box is present.
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _boxed_answer(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def reward_fn(record) -> float:
    """Score one completion 1.0 if its boxed answer matches the gold solution.

    ``record.answer`` is the normalized gold answer supplied by the dataset
    loader; we compare it symbolically against the contents of ``\boxed{}``
    in the completion so the reward is deterministic and needs no network or
    sandbox service.
    """

    ground_truth = record.answer
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0] if ground_truth else None
    prediction = _boxed_answer(record.completion)
    if ground_truth is None or prediction is None:
        return 0.0
    return 1.0 if str(ground_truth).strip() == prediction else 0.0
