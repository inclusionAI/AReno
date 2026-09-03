"""Deterministic single-turn reward for the Bash game (巴什博弈).

Reads only the model's single ``submit_move`` tool call and the hidden oracle
move in ``source_record``. It never calls an external model and never trusts a
stored answer: the optimal move is recomputed from ``n`` and ``m``.

Reward scale (in ``[0, 1]``):

* malformed / missing / multi-call / wrong tool name / illegal ``take`` -> 0.0
* winning position, ``take k`` == optimal ``k*`` -> 1.0
* winning position, legal but suboptimal ``take k``   -> dense closeness to ``k*``
* winning position, ``resign``                         -> 0.0 (gave away a win)
* losing position, ``resign``                          -> 1.0
* losing position, ``take k``                          -> 0.0

The dense closeness term for winning positions gives GSPO a monotone signal:
as ``k`` approaches the unique optimal ``k* = n % (m+1)`` the reward rises
smoothly, so gradient is informative even when no sample in a group is exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import parse_move  # noqa: E402


def _closeness(k: int, k_star: int, m: int) -> float:
    """Monotone score for a legal-but-wrong take, peaking at ``k == k_star``."""
    if k == k_star:
        return 1.0
    # floor keeps a small credit for any legal move so the group is not all-zero
    return max(0.2, 1.0 - abs(k - k_star) / m)


def reward_fn(record) -> float:
    """Score one completion via its submit_move tool call."""
    source = record.source_record
    n = int(source["n"])
    m = int(source["m"])
    move = parse_move(record)
    if move is None:
        return 0.0

    k_star = n % (m + 1)  # 0 iff losing position
    if k_star != 0:
        # winning position: the unique optimal move is take k_star
        if "take" in move:
            k = int(move["take"])
            if k == k_star:
                return 1.0
            if 1 <= k <= m and k <= n:
                return _closeness(k, k_star, m)
            return 0.0  # illegal take
        return 0.0  # resigned a won position

    # losing position: resign is the only correct answer
    if move.get("resign"):
        return 1.0
    return 0.0  # taking stones from a lost position still loses
