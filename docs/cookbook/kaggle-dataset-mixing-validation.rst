Kaggle dataset-mixing validation
================================

This page will document the reproducible Kaggle GPU validation for
deterministic weighted SFT dataset mixing introduced by Issue #198. It is
intended to capture the environment, exact command, observable artifacts, and
acceptance evidence without embedding dataset samples or credentials.

.. note::

   The validation has been run successfully. The detailed report below is a
   placeholder and will be finalized after the test-report scope and
   presentation are agreed.

Environment
-----------

``[TODO: Record the Kaggle accelerator, GPU count, CUDA version, PyTorch
version, Python version, and AReno commit.]``

Setup
-----

``[TODO: Add the minimal repository checkout, dependency installation, CUDA
extension build, and environment-check commands.]``

Usage
-----

Use ``--dataset-source NAME=PATH:WEIGHT`` once per source. The following
single-GPU example requests a 60/30/10 sample mix and trains 1,000 accepted
rows over 500 optimizer steps:

.. code-block:: bash

   areno train \
     --algo sft \
     --ckpt /kaggle/working/qwen3-0.6b-fp16 \
     --model-hub hf \
     --dataset-source alpaca=yahma/alpaca-cleaned:0.6 \
     --dataset-source stanford=tatsu-lab/alpaca:0.3 \
     --dataset-source gpt4=vicgalle/alpaca-gpt4:0.1 \
     --dataset-mix-samples-per-epoch 5000 \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --world-size 1 --tp-size 1 \
     --batch-size 2 --mini-bs 1 \
     --max-steps 500 \
     --max-prompt-tokens 256 --max-new-tokens 256 \
     --attn-backend native \
     --metrics-log-dir /kaggle/working/mixed-sft-metrics

Weights are normalized automatically. Inline sources use seed ``42``,
``cycle`` exhaustion, and per-source shuffling by default. Override the
defaults with ``--dataset-mix-seed`` and ``--dataset-mix-exhaustion`` when the
validation requires another contract.

``--dataset-mix-samples-per-epoch`` bounds the scheduled rows for one epoch; it
does not limit source download or loading. ``--max-steps`` and ``--batch-size``
bound accepted training rows, while tokenizer and length filtering can require
additional scheduled rows to fill each batch.

The command prints a sample-free ``AReno dataset mix`` summary before model
initialization. During training, inspect ``stage=dataset_mix_progress`` for
scheduled, filtered, and trained row counts plus target-token contribution.
The structured plan is written as
``dataset_mix_plan.<pid>.json`` under ``--metrics-log-dir``.

Test configuration
------------------

``[TODO: Record the model checkpoint, named dataset sources, normalized sample
weights, seed, exhaustion policy, samples-per-epoch budget, token limits,
batch sizes, and step count.]``

Validation procedure
--------------------

``[TODO: Describe the deterministic replay check, schedule-plan inspection,
training-progress inspection, and checkpoint or artifact checks.]``

Results
-------

``[TODO: Add the schedule and mix-spec hashes, planned and observed source
proportions, scheduled/filtered/trained row counts, target-token counts,
training-loss summary, step timing, and completion status.]``

Artifacts
---------

``[TODO: List the sample-free dataset-mix plan, metrics directory, relevant
log excerpts, and checkpoint location when checkpoint saving is enabled.]``

Known limitations
-----------------

``[TODO: Document that weights are sample-based, post-tokenization filtering
can change effective row proportions, token contribution can differ by source,
and exact mid-epoch resume is outside the current contract.]``
