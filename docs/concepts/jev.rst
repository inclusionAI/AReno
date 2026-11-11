JEV
===

JEV trains a model to return probabilities over the candidates of a typed
decision. The backbone produces a hidden representation for each candidate;
a shared scalar decision head scores it. AReno supports ``choice``, ``noul``
(yes/no), and ordered ``score`` questions through the same model-independent
training and serving path.

Objective and backbone updates
------------------------------

Each candidate is encoded as an independent causal sequence containing the
state, question, candidate, and answer marker. The head reads the last valid
token and applies a linear--GELU--linear projection to produce one score.
These sequences use the JEV markers ``<|oj:state|>``,
``<|oj:question|>``, and ``<|oj:answer|>`` directly, without a chat template.

All candidates of a question participate in one softmax. For target
distribution :math:`t` and candidate scores :math:`s`, the default loss is:

.. math::

   p_i = \operatorname{softmax}(s)_i, \qquad
   L = -\sum_i t_i \log p_i + 0.5 \sum_i (p_i - t_i)^2.

The optimizer averages this objective over questions. Both one-hot and soft
targets are supported. With LoRA enabled, the decision head and backbone
adapters train together while base weights stay frozen. CUDA also supports
full-parameter training by omitting LoRA. The MLX training path requires LoRA.

The shared trainer, encoding, candidate packing, loss, and checkpoint metadata
are reused across backends. Model execution stays in the backend's
``decision.py`` module. CUDA uses the existing native optimizer, gradient
synchronization, clipping, and tensor-parallel runtime.

Training data
-------------

``--dataset-path`` accepts prepared Open-Jev request/target JSONL records or
typed-decision parquet files. Each JSONL line contains one record with this
shape:

.. code-block:: json

   {
     "id": "support-1",
     "split": "train",
     "request": {
       "state": "The customer requests a refund.",
       "questions": {
         "route": {
           "type": "choice",
           "instructions": "Choose the team responsible for this request.",
           "criteria": {"billing": "Billing", "technical": "Technical support"}
         }
       }
     },
     "targets": {"route": {"billing": 1.0, "technical": 0.0}}
   }

Choice targets use the criteria IDs. Noul targets use ``false`` and ``true``;
Score targets use level indices such as ``0``, ``1``, and ``2``. Targets must
be finite, nonnegative distributions with total probability one. A record
may contain several questions, which become separate training examples.

Typed parquet records contain ``id``, ``workflow``, ``state``, ``questions``,
and ``gold`` columns. Questions and gold can be JSON-encoded objects. Gold
contains either a candidate ID or a ``probabilities`` mapping per question.
Original Open-Jev rows using ``kind``, ``options``, and ``target`` need to be
converted to the request/target format above before training.

Training selects the ``train`` split of a split mapping and excludes records
explicitly marked with another split. Keep calibration, validation, and test
files separate from the training path.

Train
-----

Select ``--algo jev`` and configure LoRA using the existing projection-target
options. For example, on one CUDA GPU:

.. code-block:: bash

   areno train \
     --algo jev --backend cuda \
     --ckpt /path/to/base-model \
     --dataset-path data/train.jsonl \
     --save-path outputs/jev \
     --world-size 1 --tp-size 1 \
     --lora-rank 32 --lora-alpha 64 \
     --batch-size 32 --mini-bs 4 \
     --max-context-len 1024 \
     --epochs 100 --max-steps 1000 --save-interval 100

Use projection targets supported by the backbone; see :doc:`native-lora`.
For CUDA QLoRA, add ``--qlora``. Native ``--adam-4bit`` or ``--adam-8bit``
can be added to select compressed AdamW state. Omitting these optimizer flags
uses the native default AdamW. For full-parameter CUDA training, omit all
LoRA and QLoRA options. Set ``--world-size 4 --tp-size 4`` for TP4.

On Apple Silicon, use ``--backend mlx --world-size 1 --tp-size 1`` with LoRA.
Add ``--q4`` for online 4-bit loading. This conversion happens in memory and
preserves the source checkpoint files.

Batching controls have distinct units:

* ``--batch-size`` counts questions per optimizer update.
* ``--mini-bs`` bounds candidate sequences in a backbone microbatch.
* ``--max-context-len`` limits each candidate sequence. AReno preserves the
  complete candidate and answer suffix, truncating the prefix when needed.

Candidate microbatches preserve whole-question normalization. If a question
has more candidates than fit in one microbatch, AReno scores all candidates
first and replays bounded forwards for the exact joint gradient.

The CLI uses decision-head learning rate ``2e-4``, Brier weight ``0.5``, and
dataset seed ``17``. SDK callers can set ``head_lr``, ``brier_weight``, and
``seed`` on ``JevTrainerConfig``. ``--lr`` controls the backbone or adapter
learning rate.

Checkpoint and continue
-----------------------

``--save-interval`` writes ``step_000100``-style directories, and the trainer
writes ``final`` when it finishes. A JEV checkpoint contains:

* ``score_head.safetensors`` and ``jev_config.json`` for the decision head,
  pooling, input length, and probability temperature.
* PEFT adapter files, or full backbone weights for full-parameter training.
* Quantization metadata when applicable.
* ``jev_training_state.json`` with dataset identity, seed, batch size, epoch,
  next position, and completed optimizer steps.

Continue an adapter checkpoint by passing its base model and
``--lora-adapter-path``:

.. code-block:: bash

   areno train \
     --algo jev --backend cuda \
     --ckpt /path/to/base-model \
     --lora-adapter-path outputs/jev/step_000100 \
     --dataset-path data/train.jsonl \
     --save-path outputs/jev-continued \
     --world-size 1 --tp-size 1 \
     --batch-size 32 --mini-bs 4 --max-context-len 1024 \
     --epochs 100 --max-steps 100 --save-interval 100

Adapter metadata restores rank, alpha, targets, and quantization mode. To
continue a full-parameter CUDA checkpoint, set ``--ckpt`` to its directory
and omit adapter options. The head and dataset state are restored alongside
the backbone or adapters. Dataset content, seed, and batch size must match
the saved iterator state; incompatible resumes raise an error.

``--max-steps`` bounds additional updates in this invocation. The restored
epoch must also remain below ``--epochs``. Optimizer moments are initialized
again on restart; current checkpoints restore weights, step scheduling, and
data position rather than complete optimizer state.

Serve typed decisions
---------------------

Pass the JEV checkpoint with ``--decision-checkpoint``:

.. code-block:: bash

   areno serve \
     --backend cuda \
     --model-path /path/to/base-model \
     --decision-checkpoint outputs/jev/final \
     --world-size 1 --tp-size 1 --port 8000

For full-backbone checkpoints, AReno loads backbone weights from the JEV
checkpoint directory. Saved adapters include their own adapter and
quantization metadata, so a separate ``--lora-adapter-path`` is unnecessary
and rejected in this serving mode. CUDA serving accepts the chosen TP and
world sizes. MLX serving requires both sizes to be one and accepts ``--q4``.

Send a request to ``POST /v1/systemone``; ``/api/alpha/decisions`` is an alias:

.. code-block:: bash

   curl http://127.0.0.1:8000/v1/systemone \
     -H 'Content-Type: application/json' \
     --data '{"state":"The customer requests a refund.","questions":{"route":{"type":"choice","instructions":"Choose the team.","criteria":{"billing":"Billing","technical":"Technical support"}}}}'

``choice`` returns a candidate ID and a probability mapping. ``noul`` returns
the probability of true. ``score`` returns a probability mapping over level
indices and their expected numeric score. Serving scores candidate paths
directly and serializes independent HTTP requests on one owning worker.

The saved positive temperature scales logits before softmax; the default is
``1.0``. It is a probability-calibration setting, with no sampling step.
Fit calibration temperatures on a separate calibration split.
