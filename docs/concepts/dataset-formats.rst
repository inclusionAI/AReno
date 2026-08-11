Dataset formats
===============

This guide answers one question: **"What columns must my dataset have for each
AReno algorithm?"**

Mental model
------------

AReno separates three concerns:

1. **Raw datasets** — upstream files from Hugging Face, ModelScope, or local
   storage. Columns like ``question``/``answer`` (GSM8K) or
   ``instruction``/``output`` (Alpaca) belong here.
2. **Loader functions** — a Python file you write (``--dataset-loader-fn``)
   that normalizes raw columns into the AReno training schema.
3. **Training schemas** — the dict shape each algorithm expects. These are the
   target of your loader function and the contract described in this document.

.. code-block:: text

   Raw dataset          Loader function          Training schema
   ───────────          ──────────────          ───────────────
   {"question": "2+2?",  ──►  load_training_  ──►  {"prompt": "Problem: 2+2?",
    "answer": "4"}              dataset()            "solutions": ["4"]}

SFT schemas
-----------

SFT requires rows with ``prompt`` and ``response``. The trainer tokenizes both,
masks the prompt prefix, and computes next-token loss only on response
positions.

.. code-block:: python

   {"prompt": "Instruction: Translate to French\nHello\nResponse:", "response": "Bonjour"}

Pre-tokenized rows
~~~~~~~~~~~~~~~~~~

When you need full control over tokenization, return ``tokens`` and
``prompt_mask`` instead:

.. code-block:: python

   {
       "tokens": [101, 2054, 2003, 102, 2043, 2051, 103],
       "prompt_mask": [True, True, True, True, False, False, False],
   }

Rows with an optional ``loss_mask`` can further exclude tokens (e.g. padding
regions within multi-turn chat sequences). ``eos_token_id`` and ``features``
(multimodal side-input) are also recognized when present. See
``areno/api/trainers/sft.py`` for the full ``_record_to_train_sequence`` logic.

Example loader and CLI
~~~~~~~~~~~~~~~~~~~~~~

``examples/sft/alpaca/dataset_loader.py`` normalizes Alpaca
``instruction``/``input``/``output`` rows:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path yahma/alpaca-cleaned \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --algo sft \
     --tp-size 1 \
     --world-size 1

Prompt-based RL schema
----------------------

GSPO, GRPO, and PPO require at least ``prompt``. Reward functions typically
need a ``solutions`` field — this is **reward metadata, not an SFT target**.
The trainer never feeds ``solutions`` to the model; it is passed to
``reward_fn`` via ``record.answer``.

.. code-block:: python

   {
       "prompt": "Solve the following math problem. Show your reasoning and put the final answer in \\boxed{}.\n\nProblem: Natalia sold 48 clips in May. She sold half as many in April. How many did she sell in April and May?\nSolution:",
       "solutions": ["72"],
   }

GSM8K example
~~~~~~~~~~~~~

Raw GSM8K rows look like this:

.. code-block:: python

   {"question": "Natalia sold 48 clips...", "answer": "Natalia sold 48/2 = 24 clips in April... #### 72"}

Raw GSM8K is **not RL-ready** (no ``prompt`` column, answer is a rationale
string rather than a machine-comparable value) and **not SFT-ready** (the
``answer`` field contains chain-of-thought reasoning mixed with the result,
not a clean supervised response). The loader at
``examples/math/dataset_loader.py`` converts it for RL:

.. code-block:: python

   # Loader extracts the final answer from "#### 72" and wraps the
   # question in a chain-of-thought instruction.
   {"prompt": "Problem: Natalia sold 48 clips...\nSolution:", "solutions": ["72"]}

Working CLI:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --batch-size 1

DPO schema
----------

DPO pairs a preferred ("chosen") answer against a less-preferred ("rejected")
answer. Two sub-formats are supported.

Prompt/response pairs
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {"prompt": "Explain gravity in one sentence.", "chosen": "Gravity is the force that attracts objects with mass toward each other.", "rejected": "Gravity is when things fall."}

The ``prompt`` may be a plain string or a list of chat messages. See
``areno/api/data_utils.py`` for the ``encode_prompt_value`` path.

Chat message pairs
~~~~~~~~~~~~~~~~~~

When ``chosen`` and ``rejected`` are both lists of chat messages, the trainer
automatically detects their common prefix as the shared prompt context. Only
the divergent suffixes contribute to the DPO objective.

.. code-block:: python

   {
       "chosen": [
           {"role": "user", "content": "Explain gravity."},
           {"role": "assistant", "content": "Gravity is the force that attracts objects with mass toward each other."}
       ],
       "rejected": [
           {"role": "user", "content": "Explain gravity."},
           {"role": "assistant", "content": "Gravity is when things fall."}
       ]
   }

No separate ``prompt`` key is needed for this sub-format. See
``areno/api/trainers/dpo.py`` for the full ``_record_to_train_pair`` logic.

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path ./my_dpo_data.jsonl \
     --dataset-loader-fn ./my_dpo_loader.py \
     --algo dpo \
     --tp-size 1 \
     --world-size 1

Agentic schema
--------------

Agentic datasets provide ``prompt`` plus any task metadata consumed by the
agent function (``--agent-fn``) and reward function (``--reward-fn-path``).

.. code-block:: python

   {
       "id": "board-00001",
       "prompt": "You are playing TicTacToe. Choose your next move.\n\n X | O |   \n---+---+---\n   | X |   \n---+---+---\n O |   |   \n",
       "board": ["X", "O", "", "", "X", "", "O", "", ""],
       "best_moves": [5, 3, 7],
       "valid_moves": [3, 5, 6, 7, 8],
   }

See ``examples/agentic/tictactoe/dataset_loader.py`` for the full example.

.. code-block:: bash

   areno train \
     --agent-fn examples/agentic/tictactoe/run_agent.py \
     --reward-fn-path examples/agentic/tictactoe/reward.py \
     --algo gspo

Loader function contract
------------------------

When raw columns do not match a training schema, write a loader file with one
function:

.. code-block:: python

   def load_training_dataset(dataset_path: str, *, default_loader, **_: object):
       dataset = default_loader(dataset_path)
       records = []
       for row in dataset:
           records.append({
               "prompt": f"Problem: {row['question']}\nAnswer:",
               "response": str(row["answer"]),
           })
       return records

**Rules:**

* Use ``default_loader`` — it handles JSONL, Parquet, CSV, HF datasets, and
  more.
* Return a list of dicts whose keys match one of the schemas above.
* Keep tokenization in the loader only when using the pre-tokenized
  ``tokens``/``prompt_mask`` format; otherwise trainers own tokenization.
* Loaders execute on the main process before GPU workers start. Keep them fast
  and deterministic.

Schema summary
--------------

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Required keys
     - Optional keys
     - Example file
   * - SFT
     - ``prompt``, ``response``
     - ``tokens``, ``prompt_mask``, ``loss_mask``, ``features``, ``eos_token_id``
     - ``examples/sft/alpaca/dataset_loader.py``
   * - GSPO / GRPO / PPO
     - ``prompt``
     - ``solutions``, task metadata
     - ``examples/math/dataset_loader.py``
   * - DPO
     - | prompt/response: ``prompt``, ``chosen``, ``rejected``
       | chat messages: ``chosen``, ``rejected``
     - —
     - —
   * - Agentic RL
     - ``prompt``
     - task state for agent and reward
     - ``examples/agentic/tictactoe/dataset_loader.py``

Where to go next
----------------

* :doc:`/cookbook/writing-loaders-and-rewards` — step-by-step tutorial.
* :doc:`/cli/dataset_loaders` — CLI flag reference and loader shapes.
* :doc:`reward-functions` — how ``solutions`` and metadata are consumed.
* :doc:`/reference/dataset-loader-api` — full API contract.
