Reward functions
================

Reward functions turn generated completions or trajectories into numeric
scores. They are task-specific Python files loaded by AReno's training path
and should be deterministic while you are debugging a run.

The public shape is:

.. code-block:: python

   def reward_fn(record) -> float:
       ...

``record`` is a :doc:`RewardRecord </reference/reward-function-api>` and the
function returns one scalar score for that one prompt/sample record. AReno
calls ``reward_fn`` once per rollout sample, so there is no batching inside
the reward function itself.

Prompt RL
---------

For rollout algorithms such as GSPO, GRPO, and PPO, each generated completion
becomes one record:

* ``record.prompt`` is the rendered prompt string.
* ``record.completion`` is the decoded completion text.
* ``record.answer`` carries the ground truth when the dataset row provides
  ``solutions``.
* ``record.source_record`` is the full dataset row returned by your dataset
  loader.

The math demo verifies each completion against the stored solution; see
``examples/math/math_verify_reward.py``.

Agentic RL
----------

For agentic rollouts, the record additionally carries the multi-turn
trajectory:

* ``record.messages`` is the OpenAI-style message list, including tool
  results.
* ``record.trace`` is the normalized event list (assistant text, tool calls,
  tool results, finish, errors).
* ``record.tool_calls`` and ``record.tool_results`` summarize what the agent
  invoked and what the environment answered.
* ``record.rendered_completion`` renders the whole trajectory for display and
  ``record.final_answer`` is the last assistant text.

Keep enough task state in the dataset row for the reward function to explain
why a trajectory passed or failed; it arrives intact as
``record.source_record``. The tic-tac-toe demo scores the chosen tool call
against the board stored in the row; see
``examples/agentic/tictactoe/reward.py``.

How dataset metadata reaches the reward
---------------------------------------

The dataset loader returns one record dict per row. AReno keeps that row on
the rollout item and copies it into ``record.source_record`` when it builds
the reward record. A ``solutions`` field also arrives as ``record.answer``.
If the reward needs extra metadata, add it in the dataset loader record
rather than through global state. See :doc:`/cli/dataset_loaders` for the
loader contract.

Practical rules
---------------

* Keep parsing and scoring explicit.
* Return one scalar for each call; never return a list.
* Score empty or malformed completions as ``0.0`` instead of raising.
* Avoid network calls in the hot path unless the task requires them.
* Log enough context to debug wrong scores.

Where to go next
----------------

* :doc:`/cli/training` documents the training CLI flag.
* :doc:`/troubleshooting/reward-function` covers debugging workflow.
* :doc:`/reference/reward-function-api` documents the API contract.
