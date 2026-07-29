Othello agentic RL
===================

Othello is a two-player board game on a 6x6 grid.  This recipe trains a policy
to choose one Othello move from a rendered board using AReno's agentic
file-callback contracts.

.. code-block:: bash

   areno train \
     --agent-fn examples/agentic/othello/run_agent.py \
     --reward-fn-path examples/agentic/othello/reward.py \
     --dataset-loader-fn examples/agentic/othello/dataset_loader.py \
     --algo gspo

The environment implements 8-direction disc flipping, pass handling (including
double-pass termination), and terminal disc scoring.  The dataset generator
produces deterministic reachable opening positions from the standard 6x6
starting board.

Key adaptation points:

* Replace the game logic in ``game.py`` with your own environment rules.
* The reward function scores moves: -1.0 for illegal, 0.0 for legal
  non-terminal, 1.0 for a terminal winning move.
* The ``play_episode`` helper runs full games against a seeded random opponent
  and reports win rate and illegal-move rate.
* All new code lives under ``examples/agentic/othello/`` — no core changes
  required, preserving backward compatibility.

See :doc:`/reference/agentic-rollout-api` for the agentic rollout API contract.