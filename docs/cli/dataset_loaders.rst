:orphan:

Dataset loaders
===============

``--dataset-loader-fn`` points to a Python function that normalizes raw dataset
rows before the trainer sees them. The function shape is the same across
examples:

.. code-block:: python

   def load_training_dataset(dataset_path: str, *, default_loader, **_: object):
       dataset = default_loader(dataset_path)
       ...
       return normalized_rows

``default_loader`` understands the same ``--dataset-path`` values as the CLI:
JSON/JSONL, Parquet, CSV/TSV, Arrow, ``datasets.save_to_disk(...)`` directories,
and Hugging Face dataset references. Loaders should keep tokenization out of the
dataset layer; trainers own tokenizer-specific rendering and token limits.

SFT
---

SFT always requires ``--dataset-loader-fn``. The loader must return rows with
``prompt`` and ``response`` keys:

.. code-block:: python

   def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
       rows = default_loader(dataset_path)
       records = []
       for row in rows:
           record = dict(row)
           records.append(
               {
                   "prompt": f"Instruction: {record['instruction']}\nAnswer:",
                   "response": str(record["answer"]),
               }
           )
       return records

For a concrete example, use ``--dataset-path yahma/alpaca-cleaned`` with
``examples/sft/alpaca/dataset_loader.py``.

DPO
---

DPO loaders should return ``prompt``, ``chosen``, and ``rejected``. ``prompt``
is the shared context, ``chosen`` is the preferred answer, and ``rejected`` is
the lower-ranked answer.

Prompt-based RL
---------------

GSPO, GRPO, and PPO prompt datasets should return ``prompt``. Reward functions
may require additional fields such as ``solutions`` or task metadata, so loaders
usually preserve those fields while adding the canonical prompt. The math loader
in ``examples/math/dataset_loader.py`` follows this pattern.

Agentic RL
----------

Agentic loaders also return ``prompt`` plus task metadata consumed by
``run_agent.py`` and ``reward.py``. Examples include
``examples/agentic/coding/dataset_loader.py`` and
``examples/agentic/shopping/dataset_loader.py``.
