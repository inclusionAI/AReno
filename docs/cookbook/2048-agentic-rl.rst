2048 agentic RL
===============

2048 is an episode-based agentic AReno recipe. Each prompt is one seeded 2048
starting board; the policy emits a bounded sequence of directions in a single
``choose_moves`` tool call, and a pure-Python engine replays the whole episode
deterministically at reward time. It exercises the agent function, the local
OpenAI-compatible proxy, the seeded environment loop, and a reward function that
reports episode score, max tile, invalid-move rate, and trained-vs-baseline
improvement.

.. code-block:: bash

   areno train \
     --agent-fn examples/agentic/2048/run_agent.py \
     --reward-fn-path examples/agentic/2048/reward.py \
     --dataset-loader-fn examples/agentic/2048/dataset_loader.py \
     --algo gspo

Use it when you want a multi-step agentic environment with a random-action
baseline and observable episode metrics. The same engine also powers a CPU-only
``baseline.py`` evaluation harness and a local ``web_ui.py`` demo (Human /
Random / LLM modes), both runnable without a GPU.

The random baseline (in ``reward_fn``, ``baseline.py``, and the web UI's Random
mode) is a uniform-random direction over all four directions, not a legal-only
policy — so it carries a nonzero invalid-move rate that any legal-direction
policy should beat.

Key adaptation points:

* Replace the seeded environment loop in ``game.py`` with your own dynamics.
* Keep the agent function responsible for emitting the action sequence.
* Keep replay deterministic under an explicit seed so reward and evaluation
  agree.
* Add reward diagnostics (episode score, max tile, invalid-rate, improvement)
  before increasing concurrency.

See :doc:`/reference/agentic-rollout-api` for the agentic rollout API contract.