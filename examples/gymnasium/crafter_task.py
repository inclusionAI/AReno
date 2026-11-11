"""Crafter RGB task file for the Gymnasium rollout interface.

Drives Crafter's RGB observations through the same ``make_gymnasium_rollout``
API as the CartPole example. Crafter ships a classic Gym API, so this file
adapts it to the Gymnasium contract. Load with both hooks pointing at this
file:

    areno train \
      --agent-fn examples/gymnasium/crafter_task.py \
      --reward-fn-path examples/gymnasium/crafter_task.py \
      --dataset-loader-fn examples/gymnasium/dataset_loader.py ...
"""

from __future__ import annotations

import base64
import io
import re

import crafter
from PIL import Image

from areno.integrations.gymnasium import make_gymnasium_rollout

MAX_EPISODE_STEPS = 8

ACTION_NAMES = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]

INSTRUCTION = (
    "You are playing Crafter, a survival crafting game. The image shows your surroundings. "
    "Reply with one action number:\n" + "\n".join(f"{index}: {name}" for index, name in enumerate(ACTION_NAMES))
)


class CrafterGymnasiumAdapter:
    """Map Crafter's classic Gym API onto the Gymnasium contract.

    Crafter only seeds at construction, so ``reset(seed=...)`` rebuilds the
    environment. Crafter's ``done`` maps to ``terminated``; truncation comes
    from the rollout's ``max_episode_steps`` cap.
    """

    def __init__(self):
        self._env = crafter.Env()
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self._env = crafter.Env(seed=int(seed))
        self._steps = 0
        return self._env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._steps += 1
        return obs, float(reward), bool(done), False, info

    def close(self):
        return None


def encode_observation(obs):
    return [
        {"type": "text", "text": INSTRUCTION},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_base64(obs)}"}},
    ]


def decode_action(model_text):
    # Vision-language models often echo the action label ("7: place_stone"),
    # so accept any reply containing an action number.
    match = re.search(r"\d+", str(model_text))
    if match is None:
        raise ValueError(f"no action number in model reply: {model_text!r}")
    action = int(match.group())
    if not 0 <= action < len(ACTION_NAMES):
        raise ValueError(f"action {action} outside 0-{len(ACTION_NAMES) - 1}")
    return action


def _png_base64(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


_run_episode, _score_episode = make_gymnasium_rollout(
    make_env=CrafterGymnasiumAdapter,
    encode_observation=encode_observation,
    decode_action=decode_action,
    max_episode_steps=MAX_EPISODE_STEPS,
)


# The CLI preflight validates task hooks by finding top-level `def` nodes, so
# the factory pair is exposed through explicit wrappers.
async def run_agent(ctx, batch):
    return await _run_episode(ctx, batch)


def reward_fn(record):
    return _score_episode(record)
