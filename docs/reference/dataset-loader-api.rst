:orphan:

Dataset loader API
==================

Dataset loaders normalize external records for the selected training path.
The exact record shape depends on the algorithm family.

Current loader shapes are documented in :doc:`/cli/dataset_loaders`:

* SFT loaders provide prompts and target responses.
* DPO loaders provide prompt, chosen, and rejected responses.
* Prompt-based RL loaders provide prompts and task metadata for reward
  scoring.
* Agentic RL loaders provide task state for agent functions.

Keep loaders small, deterministic, and testable outside the GPU training loop.
