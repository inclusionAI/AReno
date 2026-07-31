Wordle agentic RL recipe
========================

This recipe runs a small Wordle word-guessing RL task with a bundled
word list, a ``guess_word`` tool with exact/present/absent feedback,
and a multi-turn agent loop.

.. code-block:: bash

   python examples/agentic/wordle/dataset_generator.py \
     --output /tmp/wordle.jsonl --count 256 --seed 2026

   areno train \
     --ckpt Qwen/Qwen3-1.7B \
     --dataset-path /tmp/wordle.jsonl \
     --dataset-loader-fn examples/agentic/wordle/dataset_loader.py \
     --reward-fn-path examples/agentic/wordle/reward.py \
     --agent-fn examples/agentic/wordle/run_agent.py \
     --algo gspo --tp-size 2 --world-size 2

Key files:

* ``examples/agentic/wordle/game.py`` implements Wordle rules and scoring,
  including repeat-letter quota handling.
* ``examples/agentic/wordle/run_agent.py`` runs the multi-turn agent loop.
* ``examples/agentic/wordle/reward.py`` scores trajectories.
* :doc:`/cli/training` documents rollout, loss, and checkpoint flags.

