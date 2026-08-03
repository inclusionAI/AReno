:orphan:

Loss-mask explainer CLI reference
==================================

``areno explain-mask`` maps the token-level training mask produced by the SFT
packer back to human-readable roles and statistics. It runs entirely on
CPU — no model weights or GPU workers are loaded.

Use it to inspect which parts of a packed sequence contribute to training loss
before starting an expensive training run.

.. code-block:: bash

   areno explain-mask \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /path/to/data.jsonl \
     --dataset-loader-fn examples/sft/loader.py \
     --max-rows 5

Example output:

.. code-block:: text

   Row 0: total=52 tokens, loss=12 tokens

     Span  Role      Start  End  Tokens  Loss
     ─────────────────────────────────────────
     0     prompt    0      40   40      No
     1     response  40     52   12      Yes

     Summary:
       prompt:   40 tokens, 0 loss tokens
       response: 12 tokens, 12 loss tokens

Options
-------

``--ckpt`` (required)
    Tokenizer or model checkpoint path. Loaded on CPU only; no CUDA or model
    weights are initialized.

``--dataset-path`` (required)
    Dataset path in the same format as ``areno train``. Supports local JSONL,
    JSON, Parquet, CSV, and HuggingFace ``save_to_disk`` directories, as well as
    remote dataset references like ``gsm8k:main``.

``--dataset-loader-fn`` (required)
    Python file containing a loader function, same format as
    ``areno train --dataset-loader-fn``. The function receives the raw dataset
    and must return an iterable of records with ``prompt`` and ``response``
    string fields.

``--max-rows`` (default: 5)
    Number of dataset rows to process.

``--show-text``
    Decode and display the text of each span. By default, span text is hidden
    to avoid exposing full training samples.

``--json``
    Emit machine-readable JSON output instead of a terminal table.

Output fields
-------------

Terminal output
~~~~~~~~~~~~~~~

Each row produces a table with one line per span:

* **Span** — zero-based span index
* **Role** — ``prompt`` or ``response`` (SFT packer)
* **Start / End** — half-open token offsets within the packed sequence
* **Tokens** — number of tokens in this span
* **Loss** — ``Yes`` if the span contributes to training loss, ``No`` otherwise
* **Text** — decoded span text (only when ``--show-text`` is passed)

A summary section lists per-role token counts and loss token counts.

JSON output
~~~~~~~~~~~~

When ``--json`` is passed, the output is a JSON object with a ``rows`` array.
Each row contains:

* ``row`` — zero-based row index
* ``total_tokens`` — total number of tokens in the packed sequence
* ``loss_tokens`` — number of tokens that contribute to loss
* ``spans`` — array of span objects with ``role``, ``start``, ``end``,
  ``loss``, ``turn``, and ``token_count``
* ``summary`` — array of per-role statistics with ``role``, ``token_count``,
  and ``loss_tokens``
* ``text_preview`` — dictionary mapping span indices to decoded text, or
  ``null`` when ``--show-text`` is not passed

Minimal example
---------------

1. Create a small JSONL dataset:

.. code-block:: bash

   cat > /tmp/sft_demo.jsonl << 'EOF'
   {"prompt": "What is 2+2?", "response": "4"}
   {"prompt": "What is 3+3?", "response": "6"}
   EOF

2. Create a loader function:

.. code-block:: bash

   cat > /tmp/loader.py << 'EOF'
   def load_training_dataset(dataset):
       return dataset
   EOF

3. Run the explainer:

.. code-block:: bash

   areno explain-mask \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /tmp/sft_demo.jsonl \
     --dataset-loader-fn /tmp/loader.py

SDK API
-------

For programmatic use, including Agentic packer output that cannot be run from
the CLI without a model, use the SDK API:

.. code-block:: python

   from areno.api import explain_loss_mask, spans_from_prompt_mask, LossSpan

   # SFT path: derive spans from prompt_mask
   prompt_mask = [True, True, True, True, False, False, False]
   tokens = [101, 102, 103, 104, 105, 106, 107]
   loss_mask = [not m for m in prompt_mask]
   spans = spans_from_prompt_mask(prompt_mask)

   explanation = explain_loss_mask(tokens, loss_mask, spans)
   print(f"total={explanation.total_tokens}, loss={explanation.loss_tokens}")
   for span in explanation.spans:
       print(f"  {span.role}: [{span.start}:{span.end}] loss={span.loss}")

   # Agentic path: spans come from AgentTrainBatch.spans
   # (populated by the agentic packer during rollout)
   explanation = explain_loss_mask(
       batch.token_rows[0],
       batch.loss_masks[0],
       batch.spans[0],
   )

Limitations
-----------

* The CLI supports **SFT packer** output only. Agentic packer output requires
  model rollout and is accessible via the SDK API.
* SFT prompt spans are labeled ``"prompt"`` as a whole. The SFT packer receives
  plain text, so system-prompt vs. user-prompt segmentation is not available.
* Agentic prompt spans are labeled ``"prompt"`` by default. Finer-grained
  ``system_prompt`` / ``user_prompt`` segmentation requires per-message
  tokenization and is not performed by default.
