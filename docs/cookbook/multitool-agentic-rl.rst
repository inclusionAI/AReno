Multi-tool agentic RL
=====================

The multi-tool recipe trains a policy on tasks that require two or more
correctly ordered tool calls. All tools are side-effect-free and operate on
in-memory data — no network, database, or sandbox is needed. Use it when you
want to evaluate tool selection, argument accuracy, call ordering, and final
answer correctness as separate reward signals.

Tools
-----

All state lives in module-level constants inside ``game.py``.

==================== ===========================================================
Tool                 Description
==================== ===========================================================
``lookup_contact``   Look up a contact by partial name (case-insensitive)
``read_note``        Read a note by exact key
``calculate``        Evaluate a safe arithmetic expression (``+ - * /`` and
                     parentheses; uses a custom parser, not ``eval``)
``unit_convert``     Convert between length (``m``, ``cm``, ``mm``, ``km``)
                     or weight (``g``, ``kg``, ``mg``) units
``lookup_parcel``    Look up parcel tracking info by tracking id
``search_notes``     Search notes by keyword (returns matching keys + snippets)
``list_contacts_by_city``
                     List all contacts in a given city
==================== ===========================================================

Task types
----------

Each task requires at least two tool calls in a fixed order. The dataset
loader validates every record's tool names and expected fields before model
initialization, so schema errors surface immediately.

============================ ========================= ======================================
Task id                      Steps                     Example
============================ ========================= ======================================
``contact-meeting``          2                         Find Alice's phone, then read meeting note
``budget-shipping``          2                         Read budget note, then read shipping note
``parcel-city``              2                         Look up parcel P002, then find a contact in that city
``calc-shipping``            2                         Calculate ``3 * 15``, then read shipping note
``convert-parcel``           2                         Convert 100 cm to m, then look up parcel P003
``search-meeting-contact``   3                         Search notes, read meeting note, list Shanghai contacts
``parcel-calc-note``         3                         Look up parcel, calculate ETA, read shipping note
``convert-search-contact-parcel`` 4                    Convert, search notes, list contacts, look up parcel
============================ ========================= ======================================

Reward dimensions
-----------------

The reward function scores each trajectory across four independent dimensions
and tracks failure classes separately. Per-dimension scores and failure
classes are emitted via structured log lines at INFO level (logger
``areno.multitool.reward``).

================== ===================================================
Dimension          What it checks
================== ===================================================
``tool_selection`` All required tools appear in the trajectory
``arguments``      Tool call arguments match expected values
``order``          Tools appear in the correct relative order
``final_answer``   The last relevant tool call produces a valid result
================== ===================================================

Generate tasks and train
------------------------

.. code-block:: bash

   python examples/agentic/multitool/dataset_generator.py \
     --output /tmp/areno-multitool.jsonl \
     --count 2048 \
     --seed 2026

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-1.7B \
     --dataset-path /tmp/areno-multitool.jsonl \
     --dataset-loader-fn examples/agentic/multitool/dataset_loader.py \
     --reward-fn-path examples/agentic/multitool/reward.py \
     --agent-fn examples/agentic/multitool/run_agent.py \
     --algo gspo \
     --batch-size 8 \
     --n-samples 4 \
     --max-new-tokens 128

Observable output
-----------------

* **Logs**: The agent logs task count and max running prompts at start. The
  reward function emits per-dimension scores and failure classes via
  structured log lines at INFO level.
* **Metrics**: Overall reward is emitted per sample as a scalar.
* **Artifacts**: Trajectories and tool results are captured in standard
  AReno rollout artifacts via ``AgentTrajectory``.

Limitations
-----------

* All data is in-memory and deterministic — no external state.
* The calculator supports ``+``, ``-``, ``*``, ``/`` and parentheses only.
* Unit conversion supports length (``m``, ``cm``, ``mm``, ``km``) and weight
  (``g``, ``kg``, ``mg``). Cross-category conversion returns an error.

See :doc:`/reference/agentic-rollout-api` for the agentic rollout API contract.