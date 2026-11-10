"""CartPole-v1 task file for the Gymnasium rollout interface.

Load with both hooks pointing at this file:

    areno train \
      --agent-fn examples/gymnasium/cartpole_task.py \
      --reward-fn-path examples/gymnasium/cartpole_task.py \
      --dataset-loader-fn examples/gymnasium/dataset_loader.py ...
"""

import gymnasium as gym

from areno.integrations.gymnasium import make_gymnasium_rollout


def encode_observation(obs):
    position, velocity, angle, angular_velocity = obs
    return (
        f"Cart position: {position:.3f}, velocity: {velocity:.3f}. "
        f"Pole angle: {angle:.3f}, angular velocity: {angular_velocity:.3f}. "
        "Reply with 0 to push left or 1 to push right."
    )


def decode_action(model_text):
    return int(model_text.strip())


run_agent, reward_fn = make_gymnasium_rollout(
    make_env=lambda: gym.make("CartPole-v1"),
    encode_observation=encode_observation,
    decode_action=decode_action,
    max_episode_steps=128,
)
