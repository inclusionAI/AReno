"""Reward function loading and group-relative advantage normalisation.

GRPO/GSPO compute advantages by standardising rewards within the group of
`n_samples` rollouts that share a prompt; that helper lives here. The other
function lets algorithm scripts plug in a user-defined reward function from an
arbitrary Python file, which keeps the trainer agnostic about scoring logic.
"""

from __future__ import annotations

import importlib.util
import numpy as np
from pathlib import Path
from typing import Callable

def compute_group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Normalize rewards within one prompt group for GRPO/GSPO training.

    For a group with rewards r_1..r_n the advantage is
    ``A_i = (r_i - mean(r)) / (std(r) + eps)``. The small `eps` avoids
    division-by-zero when all rollouts return the same reward.
    """

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    return ((rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + eps)).tolist()


def load_reward_fn(path: str) -> Callable[[dict, list[str]], list[float]]:
    """Load a user reward function from a Python file.

    The file must define `reward_fn(example, completions)`. Keeping rewards as
    a loaded callable lets algorithm scripts swap verifiers without changing
    backend or training-loop code.
    """

    # spec_from_file_location lets us import a module whose path is supplied
    # at runtime without polluting `sys.modules` with a stable name.
    module_path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load reward function from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reward_fn = getattr(module, "reward_fn", None)
    if not callable(reward_fn):
        raise ValueError(f"{module_path} must define callable reward_fn(example, completions)")
    return reward_fn
