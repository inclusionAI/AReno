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
