:orphan:

Data contract validation
========================

``areno data inspect``

Load a dataset and optionally validate it against the per-mode data contract —
before any model or worker initialization. This is useful for catching data
issues during dataset preparation, without spinning up a training run.

.. code-block:: bash

   areno data inspect \
     --dataset-path /path/to/dataset.jsonl \
     --model-hub hf \
     --contract \
     --mode sft

Inspect without contract
------------------------

Without ``--contract``, the command loads the dataset and prints a summary:

.. code-block:: bash

   areno data inspect --dataset-path /path/to/data.jsonl --model-hub hf

Output:

.. code-block:: text

   Dataset: /path/to/data.jsonl
     records: 42
     loader:  (default)

   First record keys:
     prompt: str
     response: str

Add ``--json`` for machine-readable output:

.. code-block:: bash

   areno data inspect --dataset-path /path/to/data.jsonl --model-hub hf --json

Contract validation
-------------------

With ``--contract --mode {sft,dpo,online_rl,agentic}``, the command validates
the loaded dataset against the mode-specific contract. It scans up to
``--max-samples`` records (default 100) and collects up to ``--max-errors``
errors (default 20) before truncating.

.. code-block:: bash

   areno data inspect \
     --dataset-path /path/to/sft.jsonl \
     --model-hub hf \
     --contract \
     --mode sft

Output on failure:

.. code-block:: text

   Contract validation: mode=sft  scanned=4  total=4  errors=3  warnings=0  status=failed

   ERROR sample=0  field='response'  expected=str  got=missing
       hint: add a 'response' field to the dataset record
   ERROR sample=1  field='prompt'  expected=str  got=int
       hint: 'prompt' must be of type str

   Failed: 3 error(s) found.  Fix the listed fields or adjust the dataset loader.

Add ``--json`` for structured output suitable for CI pipelines:

.. code-block:: bash

   areno data inspect \
     --dataset-path /path/to/data.jsonl \
     --model-hub hf \
     --contract \
     --mode sft \
     --json

The JSON report has this shape:

.. code-block:: json

   {
     "mode": "sft",
     "total_scanned": 4,
     "ok": false,
     "errors": [
       {
         "sample_index": 0,
         "field_path": "response",
         "expected": "str",
         "actual": "missing",
         "hint": "add a 'response' field to the dataset record"
       }
     ],
     "warnings": []
   }

The command exits with code 0 on success and 1 on contract failure.

Per-mode contracts
------------------

SFT
~~~

Accepts either:

* ``prompt`` (str) + ``response`` (str), or
* ``messages`` (list of dicts with ``role`` and ``content``).

DPO
~~~

Accepts two variants, auto-detected per row:

* String variant: ``prompt`` (str) + ``chosen`` (str) + ``rejected`` (str)
* Chat-list variant: ``chosen`` (list[dict]) + ``rejected`` (list[dict]);
  ``prompt`` is optional.

Online RL (GSPO / GRPO / PPO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``prompt`` (str) — required.
* ``solutions`` (list[str]) — optional, nullable; preserved for reward
  functions.

Agentic
~~~~~~~

Accepts either:

* ``prompt`` (str) — for prompt-only agentic rollout, or
* ``messages`` (list of dicts) — for chat-format agentic input.
  Each message must have ``role`` in {system, user, assistant, tool} and
  ``content`` (str or null). Assistant messages may include ``tool_calls``
  (list of dicts with ``function.name`` and ``function.arguments``).

Error messages
--------------

Every error includes:

* **sample_index**: the row position in the dataset (0-based).
* **field_path**: dotted path, e.g. ``messages[2].role`` or ``chosen``.
* **expected**: the expected type or constraint, e.g. ``str`` or
  ``one of {'assistant', 'system', 'tool', 'user'}``.
* **actual**: the observed type, e.g. ``int``, ``NoneType``, or ``missing``.
* **hint**: a concrete one-line fix suggestion.

Error messages never include raw field values, so they are safe to share in
bug reports and CI logs.

Integration with ``areno train``
--------------------------------

The same contract validation can run automatically before training by passing
``--validate-data-contract``:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /path/to/sft.jsonl \
     --dataset-loader-fn /path/to/loader.py \
     --algo sft \
     --tp-size 1 \
     --world-size 1 \
     --validate-data-contract

The validation runs after the dataset loader but before model/worker
initialization. On failure, the command exits with a summary of all errors.
Pass ``--no-validate-data-contract`` to skip (this is the default).

Python API
----------

The validator is also available as a library function:

.. code-block:: python

   from areno.api import validate_contract

   report = validate_contract(dataset, mode="sft")
   if not report.ok:
       for err in report.errors:
           print(f"sample {err.sample_index}: {err.field_path} — {err.hint}")

See :doc:`/concepts/dataset-formats` for the field requirements of each
training mode.

Help
----

.. code-block:: bash

   areno data inspect --help
