:orphan:

Agentic rollout issues
======================

Agentic failures can come from model output, tool execution, environment
state, reward scoring, timeout limits, or context length.

Start with a tiny run:

* One task or environment seed.
* Low batch size.
* Verbose trajectory diagnostics.
* Deterministic tool and environment behavior.
* A reward function that logs the final state and score.

Common symptoms:

* Trajectories are dropped: check context length and message formatting.
* Rewards are missing: check the reward function and final trajectory state.
* Runs hang: check external tool or environment calls before model calls.
* Tool calls fail to parse: inspect raw assistant turns and schema format.

Use :doc:`/cookbook/tictactoe-agentic-rl` as the smallest reference recipe.

Overlength handling
-------------------

A multi-turn trajectory can blow past the context cap in three distinct ways.
AReno classifies them so you can act on the real cause instead of a generic
"trajectory too long":

* ``generation_limit`` — the model hit ``--max-new-tokens`` mid tool call (the
  engine reports ``finish_reason="length"``).
* ``context_limit`` — the next turn's rendered prompt exceeds
  ``--max-context-len`` because the accumulated trajectory grew too long.
* ``oversized_tool_result`` — a single ``role == "tool"`` message *on its own*
  already exceeds ``--max-context-len``.

Select a policy with ``--agent-overlength-policy`` (default ``off``):

* ``off`` preserves today's behavior: parsed tool calls are kept even when
  generation hit the token limit, and a post-rollout whole-trajectory filter
  drops samples that still exceed the cap (with a warning). The
  ``termination_reason`` is still recorded for observability only.
* ``safe-stop`` treats each overlength as terminal for that item:

  * ``generation_limit`` — the half-finished tool call is **dropped** (never
    recorded or executed) and the proxy surfaces ``finish_reason="length"``.
  * ``context_limit`` — the turn returns an empty response and the item stops.
  * ``oversized_tool_result`` — the oversized result is **not appended**; the
    item stops before corrupting the trajectory.

  The trajectory up to and including the terminal turn is kept for training
  (partial credit) instead of being discarded wholesale.

Observable output:

* Per-turn: the proxy ``areno`` metadata block carries
  ``termination_reason``; ``RewardRecord.metadata`` and the trace ``finish``
  event carry it too. The OpenAI ``finish_reason`` surface is ``"length"`` for
  overlength turns under ``safe-stop`` and unchanged under ``off``.
* Per-step metrics: ``rollout/overlength_generation_limit``,
  ``rollout/overlength_context_limit``,
  ``rollout/overlength_oversized_tool_result``, and ``rollout/overlength_total``.
* Diagnostics: the whole-trajectory filter "top" list includes
  ``termination_reason`` per dropped sample.

Copyable minimal example (CPU-only, no sandbox)::

    areno train --ckpt Qwen/Qwen3-0.6B --dataset-path myagent:main \
      --reward-fn-path examples/agentic/myagent/reward.py \
      --algo gspo --agent-fn examples/agentic/myagent/run_agent.py \
      --max-context-len 8192 --max-new-tokens 2048 \
      --agent-overlength-policy safe-stop

Contract for ``run_agent``: a response with ``finish_reason == "length"`` is
terminal. Stop the item, record the turn, and do **not** nudge the model to
retry or execute a (half) tool call — otherwise the loop spins to the turn
limit and the post-rollout backstop discards the trajectory anyway. The
bundled ``areno/agent/agent_loop.py`` example already honors this signal.
