:orphan:

Reward function issues
======================

Reward issues often look like training instability. Debug the reward function
before changing algorithms.

Check:

* The file passed to ``--reward-fn-path`` exports ``reward_fn``.
* The reward list length matches the completions list length.
* Parsing handles empty, malformed, or unexpected completions.
* Scores match a hand-checked example.
* Logged examples include enough context to explain wrong scores.

If rewards are always zero, inspect answer parsing first. If rewards are
always one, verify the checker actually reads the completion.

See :doc:`/concepts/reward-functions` and :doc:`/reference/reward-function-api`.
