TicTacToe agentic RL
====================

TicTacToe is the smallest agentic AReno recipe. It exercises an agent function,
the local OpenAI-compatible proxy, a task rule loop, and a reward function.

.. code-block:: bash

   areno train \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --algo gspo

Use it when you want to learn the shape of an agentic task before moving to a
larger environment.

Key adaptation points:

* Replace the rule loop with your environment.
* Keep the agent function responsible for external actions.
* Return trajectory turns that AReno can tokenize and score.
* Add reward diagnostics before increasing concurrency.

See :doc:`/reference/agentic-rollout-api` for the agentic rollout API contract.
