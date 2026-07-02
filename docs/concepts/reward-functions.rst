Reward functions
================

Reward functions turn generated completions or trajectories into numeric scores.
They are task-specific Python files loaded by AReno's training path and should
be deterministic while you are debugging a run.

The public shape is:

.. code-block:: python

   def reward_fn(example, completions) -> list[float]:
       ...

The function receives the original example plus generated completions, then
returns one score per completion. For agentic workflows, keep enough task state
in the dataset row for the reward function to explain why a trajectory passed or
failed.

Practical rules
---------------

* Keep parsing and scoring explicit.
* Return one score for each completion.
* Avoid network calls in the hot path unless the task requires them.
* Log enough context to debug wrong scores.

Where to go next
----------------

* :doc:`/cli/training` documents the training CLI flag.
* :doc:`/troubleshooting/reward-function` covers debugging workflow.
* :doc:`/reference/reward-function-api` documents the API contract.
