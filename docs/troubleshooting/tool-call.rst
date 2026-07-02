:orphan:

Tool call issues
================

Tool-call failures usually happen at the schema, parser, or trajectory
boundary.

Check:

* The model output matches the expected tool-call format.
* Tool arguments are validated before execution.
* Tool results are recorded in the message history.
* Parser failures are logged with the raw assistant turn.
* Dropped trajectories include enough context to reproduce the bad turn.

For serving-side tool-call behavior, see :doc:`/cli/inference`. For agentic
diagnostics, see :doc:`/cli/observability`.
