:orphan:

Reward function API
===================

Reward files should expose a callable named ``reward_fn``:

.. code-block:: python

   def reward_fn(example, completions) -> list[float]:
       ...

Parameters:

``example``
   The source record returned by the dataset loader.

``completions``
   The generated completions to score.

Return value:

``list[float]``
   One reward score per completion.

Keep this API stable for task code. If the reward needs extra metadata, add it
through the dataset loader record rather than through global state.

Runtime validation
------------------

Enable reward hook validation with the ``--validate-reward`` CLI flag:

.. code-block:: bash

   areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo --tp-size 4 --validate-reward

When enabled, AReno checks the reward function before and during training:

1. **Signature check** (at load time): warns if the parameter or return
   type annotation does not match ``RewardRecord`` / ``float``.
2. **Dry-run** (at load time): calls the function once with a minimal
   ``RewardRecord`` to catch runtime errors early.  Failures produce a
   warning, not a hard error, because many valid hooks require specific
   field values (e.g. ``record.answer``).
3. **Output validation** (per call): rejects non-numeric return values
   (``None``, ``str``, ``list``, ``dict``), non-finite values (``NaN``,
   ``Inf``), and multi-dimensional tensors.  Each failure message includes
   the hook name, sample index (from ``record.metadata["sample_index"]``),
   and a truncated prompt preview (100 chars).

Validation is **off by default** and does not change existing behavior.

Limitations:

- The reward function contract is one ``RewardRecord`` in, one ``float``
  out.  Batch-mode reward functions (returning a list of floats) are not
  supported; a list return value is rejected with the list length.
- ``sample_index`` is read from ``record.metadata["sample_index"]``, which
  is populated by AReno's trainer.  When unavailable (e.g. dry-run or
  direct SDK calls without metadata), the index is reported as ``unknown``.
- Serializability of the return value is not checked explicitly, but the
  validated output is always a plain ``float``, which is serializable.

Environment variables:

``ARENO_REWARD_VALIDATION``
   Set to ``1`` to enable validation (equivalent to ``--validate-reward``).

``ARENO_REWARD_VALIDATION_DRY_RUN``
   Set to ``0`` to skip the dry-run step while keeping output validation active.
