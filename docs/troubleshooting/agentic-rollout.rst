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

Trainable-turn selection
------------------------

When using ``--trainable-turns last_assistant`` or ``--trainable-turns
final_answer``, only a subset of assistant spans contribute to policy loss. If
``trainable_tokens`` in the batch log is unexpectedly zero:

* **Bare trailing tool call**: the trajectory ends with a tool call and no
  following assistant text. ``final_answer`` yields zero trainable signal by
  design. Use ``all_assistant`` or ensure the agent produces a final text
  answer.
* **No tool calls at all**: ``final_answer`` degenerates to ``last_assistant``,
  so the last assistant span should be trainable. If it is not, check that
  ``response_spans`` is populated (the agent function must return explicit
  trajectory turns).
* **Tool call without matching tool result**: a mid-trajectory tool call with
  no corresponding ``tool`` message raises ``ValueError`` before worker
  initialization. Trailing tool calls are exempt.

The ``mask_tool_call_args`` flag zeroes JSON-argument tokens within tool-call
spans. If ``trainable_tokens`` drops more than expected, verify that the
tokenizer decode/encode round-trip preserves argument boundaries — the
localization is approximate and should be pinned with per-token CPU tests.
