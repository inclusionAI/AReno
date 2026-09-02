Writing dataset loaders and reward functions
============================================

This tutorial walks through writing a custom dataset loader and reward function
for AReno. You will learn the function signatures, the record shapes each
training mode expects, and how to wire everything together with the CLI.

Prerequisites
-------------

You should be comfortable with Python and have read the concept guides:

* :doc:`/concepts/dataset-formats` — record shapes per algorithm family.
* :doc:`/concepts/reward-functions` — how scores flow into the trainer.

The reference pages :doc:`/reference/dataset-loader-api` and
:doc:`/reference/reward-function-api` provide the full API contract; keep them
handy as you read.

.. _tutorial-dataset-loader:

Writing a dataset loader
------------------------

A dataset loader is a Python file that defines one function:

.. code-block:: python

   def load_training_dataset(dataset_path: str, *, default_loader, **_: object):
       ...

**Parameters**

``dataset_path``
   The value of ``--dataset-path`` from the CLI. It can be a local file, a
   directory, or a Hugging Face / ModelScope dataset reference.

``default_loader``
   A callable provided by AReno that can load CSV, TSV, JSON/JSONL, Parquet,
   Arrow, ``datasets.save_to_disk(...)`` directories, and Hugging Face dataset
   references. Always use this instead of writing your own I/O.

**Return value**

A list of dicts. The required keys depend on the training mode:

.. list-table::
   :header-rows: 1

   * - Training mode
     - Required keys
     - Optional keys
   * - SFT
     - ``prompt``, ``response``
     - —
   * - DPO
     - ``prompt``, ``chosen``, ``rejected``
     - —
   * - Prompt-based RL (GSPO/GRPO/PPO)
     - ``prompt``
     - ``solutions``, task metadata
   * - Agentic RL
     - ``prompt``
     - task state for agent and reward

Step-by-step example: a custom math loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Imagine you have a JSONL file where each row looks like this:

.. code-block:: json

   {"problem": "What is 2 + 2?", "ground_truth": "4"}

You want to use it for GSPO training. Here is the loader, built up one piece at
a time.

**1. Load the raw data with default_loader**

.. code-block:: python

   def load_training_dataset(dataset_path: str, *, default_loader, **_: object):
       dataset = default_loader(dataset_path)
       # dataset is a list of dicts — one per JSONL line
       return dataset

At this point every row still has the original ``problem`` / ``ground_truth``
keys. The trainer does not understand those.

**2. Normalize rows to the expected schema**

Prompt-based RL requires at least a ``prompt`` key. Reward functions often need
a ``solutions`` key. Add both:

.. code-block:: python

   def load_training_dataset(dataset_path: str, *, default_loader, **_: object):
       dataset = default_loader(dataset_path)
       records = []
       for row in dataset:
           records.append({
               "prompt": f"Problem: {row['problem']}\nAnswer:",
               "solutions": [str(row["ground_truth"])],
           })
       return records

**3. Make it robust**

Real-world datasets are rarely clean. Add a sniffing step so the loader
passes through rows that are already in the right shape:

.. code-block:: python

   def load_training_dataset(dataset_path: str, *, default_loader, **_: object):
       dataset = default_loader(dataset_path)
       if len(dataset) == 0:
           return dataset
       if "prompt" in dataset[0]:
           return dataset  # already normalized — pass through
       records = []
       for row in dataset:
           records.append({
               "prompt": f"Problem: {row['problem']}\nAnswer:",
               "solutions": [str(row["ground_truth"])],
           })
       return records

That is a complete, production-style dataset loader. The full version is about
20 lines and handles three input schemas; see ``examples/math/dataset_loader.py``
for the final form.

.. _tutorial-reward-function:

Writing a reward function
-------------------------

A reward file must expose a callable named ``reward_fn``. The signature depends
on the training mode.

Prompt-based RL reward
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def reward_fn(record) -> float:
       ...

The ``record`` object has these attributes:

* ``record.prompt`` — the prompt string.
* ``record.completion`` — the model-generated completion string.
* ``record.answer`` — the ``solutions`` list from the dataset loader (or
  ``None`` if you did not provide one).

Return a ``float``. The trainer calls this once per (example, completion) pair.

Agentic RL reward
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def reward_fn(record) -> float:
       ...

The ``record`` object also carries:

* ``record.source_record`` — the original dataset row dict.
* ``record.tool_calls`` — the list of tool calls the model made during the
  trajectory.

Return a ``float``.

Step-by-step example: a custom math reward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuing the math example from the loader tutorial, you have rows with a
``problem`` prompt and a ``solutions`` list containing the ground-truth answer.

**1. Extract the ground truth**

.. code-block:: python

   def reward_fn(record) -> float:
       solutions = record.answer
       if solutions is None:
           raise KeyError("reward expects record.answer; check your dataset loader")
       ground_truth = solutions[0] if isinstance(solutions, list) else solutions
       ...

**2. Compare prediction to ground truth**

The simplest approach is an exact-match check after normalizing whitespace:

.. code-block:: python

   def reward_fn(record) -> float:
       solutions = record.answer
       if solutions is None:
           return 0.0
       ground_truth = str(solutions[0]).strip()
       prediction = record.completion.strip()
       return 1.0 if prediction == ground_truth else 0.0

**3. Add tolerance for real model output**

Models rarely output just the answer. They produce reasoning chains. A robust
reward function parses the final answer from the completion. For math tasks,
extract the last boxed expression:

.. code-block:: python

   import re


   def _extract_final_answer(text: str) -> str | None:
       # Look for \boxed{...} patterns and return the last one.
       matches = re.findall(r"\\boxed\{([^}]*)\}", text)
       return matches[-1].strip() if matches else None


   def reward_fn(record) -> float:
       solutions = record.answer
       if solutions is None:
           return 0.0
       ground_truth = str(solutions[0]).strip()
       prediction = _extract_final_answer(record.completion)
       if prediction is None:
           return 0.0
       return 1.0 if prediction == ground_truth else 0.0

This is the pattern used by the built-in math verifier at
``examples/math/math_verify_reward.py``, which additionally uses symbolic
comparison so that ``1/2`` and ``0.5`` are treated as equal.

Wiring them together
--------------------

Once both files are written, pass them to the CLI:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path ./my_dataset.jsonl \
     --dataset-loader-fn ./my_loader.py \
     --reward-fn-path ./my_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1

The flags work independently: you can pair your loader with an existing reward
function, or your reward function with an existing loader.

Debugging tips
--------------

**Test the loader in isolation**

.. code-block:: python

   # test_loader.py
   from my_loader import load_training_dataset

   # default_loader is just datasets.load_dataset for HF paths,
   # or a JSONL reader for local files — approximate it:
   import json

   def fake_default_loader(path):
       with open(path) as f:
           return [json.loads(line) for line in f if line.strip()]

   records = load_training_dataset("./my_dataset.jsonl", default_loader=fake_default_loader)
   print(records[0])
   # Check that every record has the keys your training mode requires.

**Test the reward function in isolation**

.. code-block:: python

   # test_reward.py
   from types import SimpleNamespace

   from my_reward import reward_fn

   fake_record = SimpleNamespace(
       prompt="Problem: 2+2\nAnswer:",
       completion="The answer is \\boxed{4}",
       answer=["4"],
   )
   score = reward_fn(fake_record)
   print(f"Score: {score}")  # Should be 1.0

   fake_wrong = SimpleNamespace(
       prompt="Problem: 2+2\nAnswer:",
       completion="The answer is \\boxed{5}",
       answer=["4"],
   )
   score = reward_fn(fake_wrong)
   print(f"Score: {score}")  # Should be 0.0

**Common pitfalls**

* **Loader not called**: Make sure ``--dataset-loader-fn`` points to the file,
  not just the directory. The path must end in ``.py``.
* **Wrong keys**: Print ``records[0].keys()`` from your loader to verify the
  keys match what your training mode expects.
* **Non-deterministic rewards**: Avoid random number generators, timestamps, or
  network calls inside ``reward_fn``. Non-deterministic rewards make training
  dynamics hard to reproduce.
* **Reward always zero**: Check that your reward function handles the model's
  actual output format. Models often add markdown, extra whitespace, or
  unexpected tokens.

Where to go next
----------------

* Browse the shipped examples under ``examples/`` for more patterns:
  ``examples/math/``, ``examples/agentic/tictactoe/``, ``examples/sft/alpaca/``.
* :doc:`/cookbook/math-rlvr` — runnable recipe for the math RLVR path.
* :doc:`/cookbook/tictactoe-agentic-rl` — agentic recipe with tool calls.
* :doc:`/troubleshooting/reward-function` — debugging reward functions.
* :doc:`/troubleshooting/tool-call` — debugging tool-call extraction.
