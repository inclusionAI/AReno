Replay training from saved rollout records
==========================================

Replay training lets you skip rollout and train directly from previously saved
rollout records. This is useful for debugging loss spikes, comparing algorithms
on the same data, or iterating on hyperparameters without burning GPU hours on
repeated inference.

.. note::

   Replay is a **debugging and comparison path**, not a general checkpoint
   replacement. It reproduces the *training batch* but does not restore
   optimizer state, model weights, or critic training.

Saving rollout records
----------------------

Add ``--save-replay-path`` to your normal training command to write one
``.jsonl`` file per step:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --save-replay-path /tmp/areno-replay \
     --max-steps 5

Each file (``step_000000.jsonl``, ``step_000001.jsonl``, ...) contains one JSON
object per ``TrainSequence`` with these fields:

================== =========================== =================================
Field             Type                        Description
================== =========================== =================================
``format_version``  ``int``                   Schema version (currently 1)
``epoch``           ``int``                   Epoch when the record was saved
``step``            ``int``                   Global step when saved
``prompt_index``    ``int``                   Index of the prompt in the batch
``sample_index``    ``int``                   Index of the sample in its group
``tokens``          ``list[int]``             Full token sequence (prompt + response)
``prompt_mask``     ``list[bool]``            ``True`` = prompt position
``loss_mask``       ``list[bool]``            ``True`` = contributes to loss
``logprobs``        ``list[float]``           Per-token rollout logprobs
``advantages``      ``list[float]``           Per-token advantages
``reward``          ``float``                 Scalar reward for the sequence
``eos_token_id``    ``int``                   EOS token ID at save time
``metadata``        ``dict``                  Optional algorithm-specific fields
================== =========================== =================================

Replaying from saved records
-----------------------------

Use ``--replay-path`` to load records instead of running rollout:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1 \
     --replay-path /tmp/areno-replay \
     --max-steps 5

Observable output
------------------

When saving rollout records, the logs include:

* ``stage=replay_saved path=<file>`` — a batch was written to disk

When replaying from saved records, the logs include:

* ``stage=replay_loaded path=<file>`` — a batch was loaded from disk
* ``stage=replay_exhausted`` — no more replay files found; the loop terminates

These stages also appear in the dashboard state file when metrics recording
is enabled.

Validation
-----------

AReno validates replay files **before** model initialization:

* **Version mismatch**: records with an incompatible ``format_version`` are
  rejected with a message showing the expected and actual versions.
* **Missing fields**: each required field is checked; the error names the
  missing field and line number.
* **Length alignment**: ``tokens``, ``prompt_mask``, ``loss_mask``,
  ``logprobs``, and ``advantages`` must all have the same length.
* **Empty or missing files**: an explicit ``ValueError`` is raised.

Invalid records are never silently coerced.

Limitations
-----------

* Replay does not restore optimizer state. The replayed training step starts
  from the model checkpoint at the beginning of the run, not from the optimizer
  state at the original step.
* PPO replay uses saved ``advantages``/``returns``/``values``/``ref_logprobs``
  and skips critic training.
* Files are JSON Lines format; large batches may produce files in the tens of
  MB range per step.

SDK usage
---------

.. code-block:: python

   from areno import Trainer
   from areno.api.models import TrainSequence

   trainer = Trainer(world_size=1, model_path="Qwen/Qwen3-0.6B")
   trainer.init()

   # Save a batch
   trainer.save_rollout_batch(
       "/tmp/replay/step_000000.jsonl",
       epoch=0, step=0,
       train_batch=my_batch,
   )

   # Load it back
   replayed = trainer.load_rollout_batch("/tmp/replay/step_000000.jsonl")

   # Train on the replayed batch
   result = trainer.train(replayed, loss_fn, mini_bs=8)
   trainer.close()