Dataset formats
===============

AReno loaders normalize external datasets into small dictionaries consumed by
the selected algorithm. Keep tokenization out of the dataset layer; trainers
own tokenizer rendering, sequence limits, and chat-template behavior.

SFT rows
--------

SFT rows provide a supervised prompt and target response:

.. code-block:: python

   {"prompt": "Instruction: ...", "response": "..."}

DPO rows
--------

DPO rows provide one shared prompt and two ranked answers:

.. code-block:: python

   {"prompt": "...", "chosen": "...", "rejected": "..."}

Prompt-based RL rows
--------------------

GSPO, GRPO, and PPO prompt datasets provide ``prompt``. They can also preserve
task metadata such as ``solutions`` for reward functions.

Agentic rows
------------

Agentic datasets provide ``prompt`` plus any task metadata consumed by the agent
and reward files.

Where to go next
----------------

* :doc:`/cli/dataset_loaders` documents loader shapes and examples.
* :doc:`reward-functions` explains how preserved metadata is used for scoring.
