:orphan:

Agentic rollout API
===================

Agentic rollout uses an agent function to run task-specific interaction and
return trajectory turns for training.

The training CLI accepts the agent entry point through ``--agent-fn``:

.. code-block:: bash

   areno train \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --algo gspo

The agent function can use the local OpenAI-compatible proxy to call the model
and can execute tools or environment logic between model turns.

See :doc:`/cli/training` for current flags and
:doc:`/cookbook/tictactoe-agentic-rl` for the smallest runnable recipe.
