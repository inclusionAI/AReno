Training Loop
=============

AReno models post-training as one local loop:

.. code-block:: text

   Dataset or task
     -> Prompt or initial state
     -> Rollout engine
         -> Model inference
         -> Environment or tools
         -> Multi-turn trajectory
     -> Reward function
     -> Advantage or loss
     -> Trainer
     -> Checkpoint, metrics, and logs

This shape matters because most failures happen at the boundaries. A dataset
can produce the wrong prompt, an agent can return a malformed trajectory, a
reward function can score the wrong completion, or a trainer can receive rows
that do not match the selected algorithm.

AReno keeps these boundaries visible. The CLI wires them together for common
tasks, while the SDK lets you control rollout, reward calculation, training,
and checkpoint cadence directly.

Where to go next
----------------

* :doc:`chat-templates` explains how messages become model prompts.
* :doc:`dataset-formats` explains the fields expected by each training mode.
* :doc:`reward-functions` explains how task outcomes become scores.
