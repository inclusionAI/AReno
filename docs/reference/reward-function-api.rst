:orphan:

Reward function API
===================

Reward files should expose a callable named ``reward_fn``:

.. code-block:: python

   def reward_fn(record) -> float:
       ...

Parameters:

``record``
   A :class:`RewardRecord` describing one prompt/sample rollout. Prompt RL
   and agentic RL share this input type.

Return value:

``float``
   One scalar reward for this record. AReno calls ``reward_fn`` once per
   rollout sample and collects the scalars itself.

RewardRecord fields
-------------------

``prompt`` / ``completion``
   The rendered prompt string and the decoded completion text. Always set.

``rendered_completion`` / ``final_answer``
   Display-oriented views of the output. For prompt RL they equal
   ``completion``; for agentic rollouts ``rendered_completion`` renders the
   whole message list and ``final_answer`` is the last assistant text.

``answer``
   Ground truth copied from the dataset row's ``solutions`` field when
   present, else ``None``.

``messages``
   The OpenAI-style message list for agentic rollouts, including tool
   results and the final assistant message. Empty for prompt RL.

``trace``
   Normalized trajectory events for agentic rollouts. Each event has a
   ``type`` (``request``, ``assistant_text``, ``assistant_tool_call``,
   ``tool_result``, ``finish``, ``error``) plus optional ``text``, ``name``,
   ``arguments``, ``content``, ``messages``, and ``metadata``.

``tool_calls`` / ``tool_results``
   Parsed assistant tool calls and environment tool results extracted from
   the agentic trajectory. Empty for prompt RL.

``tokens`` / ``logprobs`` / ``loss_mask``
   Token ids, rollout logprobs, and per-token loss-mask bits for the full
   row. Prompt positions are masked out of the loss.

``source_record``
   The dataset row dict returned by your dataset loader, unchanged. Use it
   to read task metadata such as boards, expected outputs, or difficulty.

``metadata``
   Extra trainer metadata such as ``prompt_index`` and ``sample_index``.

Loading
-------

``--reward-fn-path`` loads the file with ``areno.api.rewards.load_reward_fn``
and expects a module-level ``reward_fn`` callable taking exactly one
positional argument. Startup validation fails fast with
``<path> must define callable reward_fn(record)`` when the symbol is missing
or not callable.

Keep this API stable for task code. If the reward needs extra metadata, add it
through the dataset loader record (it arrives as ``record.source_record``)
rather than through global state. See ``examples/math/math_verify_reward.py``
and ``examples/agentic/tictactoe/reward.py`` for working reward files.
