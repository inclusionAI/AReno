DuelGrid visual agent
=====================

DuelGrid is a browser-game agentic example with multi-action turns, reward
curves, and replayable behavior before and after training.

The example lives under ``examples/agentic/duelgrid`` and includes:

* A rule engine.
* A fixed-path dataset loader.
* A reward function.
* An OpenAI-compatible agent.
* A browser UI for replaying the task.

Before GSPO/RLVR post-training, the model often moves back and forth without
progress. After training, it learns to collect pickups, chase the user, attack
when in range, and avoid trap tiles.

Use this recipe after TicTacToe when you need richer environment state or a
visual replay loop.
