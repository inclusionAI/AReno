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

Controlling which turns receive policy loss
-------------------------------------------

By default every assistant span in an agentic trajectory contributes to policy
loss (``--trainable-turns all_assistant``). To concentrate supervision on the
final decision, use ``--trainable-turns final_answer`` (only the
``assistant_text`` span following the last tool result) or
``--trainable-turns last_assistant`` (only the final assistant span regardless
of kind). Check the rollout log for ``trainable_tokens`` and
``masked_response_tokens`` to verify how many tokens are actually trained.

A mid-trajectory tool call without a matching tool result is rejected with
``ValueError`` before training begins. A trajectory ending in a bare tool call
(result never received) is allowed; under ``final_answer`` this yields zero
trainable signal — if that is not intended, switch to ``last_assistant``.

``--mask-tool-call-args`` masks JSON-argument tokens within tool-call spans
while keeping tool-name tokens trainable. This is a research ablation that
diverges from industry tool-use training (which trains the full tool-call
span); argument localization is approximate. See :doc:`/cli/training` for the
full option reference.
