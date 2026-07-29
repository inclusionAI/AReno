Reward clipping and batch normalization
=======================================

Raw reward scores produced by user-supplied ``reward_fn`` calls can have extreme
magnitudes or unstable variance, which propagates into advantage estimates and
can destabilise training.  AReno provides three opt-in reward transformation
modes that sit between raw reward calculation and advantage computation:

* **disabled** (default) -- rewards pass through unchanged; full backward
  compatibility.
* **clip** -- clamps each reward to ``[reward_clip_min, reward_clip_max]``.
* **standardize** -- subtracts the batch mean and divides by the batch std
  (population std, matching the convention used by
  ``compute_group_advantages``).

The transform is applied **per prompt group** (the same granularity as GRPO/GSPO
advantage normalization) before ``compute_group_advantages`` is called.

Configuration
--------------

Set the mode and parameters on a ``PolicyTrainerConfig`` (or any subclass such
as ``PPOTrainerConfig``):

.. code-block:: python

   from areno.api.trainer_config import PolicyTrainerConfig

   config = PolicyTrainerConfig(
       algo="gspo",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       reward_fn_path="examples/math/math_verify_reward.py",
       # Enable reward clipping:
       reward_transform_mode="clip",
       reward_clip_min=-5.0,
       reward_clip_max=5.0,
   )

For standardization instead:

.. code-block:: python

   config = PolicyTrainerConfig(
       algo="gspo",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       reward_fn_path="examples/math/math_verify_reward.py",
       reward_transform_mode="standardize",
       reward_standardize_eps=1e-8,
   )

To disable (default, no change from existing behaviour):

.. code-block:: python

   config = PolicyTrainerConfig(
       algo="gspo",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       reward_fn_path="examples/math/math_verify_reward.py",
       reward_transform_mode="disabled",  # this is the default
   )

Standalone usage
----------------

The transform functions can be called directly without running a full training
loop, which is useful for testing and debugging:

.. code-block:: python

   from areno.api.reward_transforms import transform_rewards

   rewards = [1.0, 5.0, -3.0, 10.0]

   # Clip mode
   clipped, stats = transform_rewards(rewards, mode="clip", clip_min=-2.0, clip_max=8.0)
   print(clipped)  # [1.0, 5.0, -2.0, 8.0]
   print(stats["raw_max"], stats["transformed_max"])  # 10.0 8.0

   # Standardize mode
   standardized, stats = transform_rewards(rewards, mode="standardize")
   print(sum(standardized) / len(standardized))  # ~0.0

   # Disabled mode (no-op, returns a copy)
   passthrough, stats = transform_rewards(rewards, mode="disabled")
   assert passthrough == rewards

Observable output
-----------------

When a transform is active (``clip`` or ``standardize``), the trainer logs a
``reward_transform`` metric line containing:

* ``mode`` -- the active transform mode.
* ``raw_mean``, ``raw_std`` -- distribution of raw rewards before transform.
* ``transformed_mean``, ``transformed_std`` -- distribution after transform.

The ``transform_rewards`` function also returns a stats dict with full
``raw_*`` and ``transformed_*`` fields (mean, std, min, max, count) for
programmatic consumption.

Distributed statistics semantics
--------------------------------

The transform operates on the reward list available to a single
``_materialize_train_batch`` call.  In single-device training this is the full
batch.  In multi-device training where rewards are not gathered across ranks,
the statistics are **per-shard** -- standardization normalizes within each
rank's local shard, not globally.  This is an explicit design choice: it
avoids an all-reduce communication round-trip on the critical path.

Clipping is element-wise and has no distributed-statistics implications.

Error handling
---------------

* **NaN rewards** -- both ``clip`` and ``standardize`` raise ``ValueError``
  immediately, preventing NaN from silently propagating into advantages.
* **Empty reward list** -- ``clip`` returns an empty list (safe no-op);
  ``standardize`` raises ``ValueError`` because standardization is undefined
  for an empty set.
* **Invalid mode** -- ``transform_rewards`` raises ``ValueError`` listing the
  valid modes.
* **clip_min > clip_max** -- raises ``ValueError``.
