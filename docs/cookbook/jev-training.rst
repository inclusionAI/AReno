Train and serve a JEV model
===========================

Fine-tune **Qwen3-4B** to route support requests on an Apple Silicon Mac, then
serve typed decisions. The frozen base is quantized to 4-bit in memory while
LoRA and a shared scalar decision head train together. Each candidate gets
one score; softmax over all candidates of the question produces probabilities
summing to one. See :doc:`../concepts/jev` for the encoding and objective.

1. Install
----------

Use native arm64 Python 3.10+ on Apple Silicon. Run commands from the
repository root in an environment with AReno installed:

.. code-block:: bash

   git clone https://github.com/inclusionAI/AReno.git
   cd AReno
   python -m pip install -e .

AReno downloads and caches ``Qwen/Qwen3-4B`` from ModelScope on first use.
The source checkpoint stays unchanged. For CUDA installation, see
:doc:`../getting-started/installation`. On CUDA, replace ``--backend mlx``
with ``--backend cuda`` and remove ``--q4``; CUDA QLoRA uses ``--qlora``
instead. Quantized artifacts are backend-specific. Full-parameter CUDA
training omits the LoRA and quantization flags; this Mac recipe uses LoRA.

2. Prepare data
---------------

Use :download:`train.jsonl <jev/train.jsonl>` as a small, inspectable example.
It contains four requests with three questions each. No custom dataset loader
is needed. Inspect it with:

.. code-block:: bash

   python -m json.tool --json-lines docs/cookbook/jev/train.jsonl

Each line is a request with target probabilities:

.. code-block:: json

   {
     "id": "example", "split": "train",
     "request": {
       "state": "Please send a refund.",
       "questions": {
         "route": {
           "type": "choice", "instructions": "Choose the support team.",
           "criteria": {"billing": "Refunds", "technical": "Login problems"}
         }
       }
     },
     "targets": {"route": {"billing": 1.0, "technical": 0.0}}
   }

* ``choice``: ``criteria`` maps candidate IDs to descriptions; targets use
  those IDs, as above.
* ``noul``: a yes/no question; targets use ``"false"`` and ``"true"``.
* ``score``: ``criteria`` lists ordered levels; target keys are their zero-based
  indices. ``{"0": 0.1, "1": 0.8, "2": 0.1}`` is a valid soft target.

Every question needs a finite, nonnegative target for every candidate, with
sum one. Training expands requests into questions and selects the ``train``
split. Keep held-out records in separate files. Typed Parquet is also supported;
see :doc:`../concepts/jev` for its schema.

3. Train
--------

.. code-block:: bash

   areno train --algo jev --backend mlx --ckpt Qwen/Qwen3-4B --model-hub modelscope \
     --dataset-path docs/cookbook/jev/train.jsonl --save-path outputs/jev \
     --q4 --world-size 1 --tp-size 1 --lora-rank 128 \
     --batch-size 32 --mini-bs 2 --max-context-len 4096 --lr 2e-5 \
     --seed 17 --epochs 2 --save-interval 50

``batch-size`` counts questions per optimizer update; ``mini-bs`` bounds
candidate sequences per backbone microbatch. All candidates still share one
softmax. Reduce ``mini-bs`` first for memory pressure. ``max-context-len``
limits each candidate sequence; AReno truncates the prefix while preserving
the full candidate/answer suffix. ``lr`` controls the backbone adapters; the
decision head uses its default learning rate ``2e-4``.

Check ``outputs/jev/metrics.jsonl`` for ``jev_ce``, ``jev_brier`` and
``jev_top1``. Lower cross-entropy/Brier is better. These four examples exercise
the pipeline; two epochs do not establish model quality or long-context capacity.

Checkpoints are saved every 50 updates as ``step_000050``, etc., and at
completion as ``final``. This tiny dataset gives one update per epoch, so the
command above produces only ``outputs/jev/final``. It contains the decision
head, LoRA adapter, JEV configuration, quantization metadata and training-data
position; the base model is not copied.

To continue later, rerun the same command with
``--lora-adapter-path outputs/jev/final``, raise ``--epochs`` to ``3``, and
use a new ``--save-path``. Keep the same base, dataset, seed and batch size.
Weights and data position restore, but optimizer moments restart.

4. Serve
--------

In another terminal, use the same environment and repository root:

.. code-block:: bash

   areno serve --backend mlx --model-path Qwen/Qwen3-4B --model-hub modelscope \
     --decision-checkpoint outputs/jev/final \
     --world-size 1 --tp-size 1 --host 127.0.0.1 --port 8001

Keep the checkpoint intact, including its quantization metadata; saved Q4
mode restores automatically. Wait for ``Application startup complete``, then
send the bundled :download:`request <jev/request.json>`:

.. code-block:: bash

   cat docs/cookbook/jev/request.json
   curl --fail-with-body http://127.0.0.1:8001/v1/systemone \
     -H 'Content-Type: application/json' \
     --data-binary @docs/cookbook/jev/request.json

``answers.route`` contains the predicted ``choice`` and candidate
``probabilities``. ``answers.urgent.noul`` is P(true); P(false) is one minus
that value. ``answers.priority`` contains level probabilities and their
expected index ``score``. These are candidate probabilities, not generated
tokens. Keep question definitions consistent with training. Stop the server
with Ctrl-C when finished.
