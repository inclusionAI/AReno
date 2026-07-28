Duplicate resampling within a bounded budget
============================================

.. versionadded:: 0.0.7

During RL post-training, a single prompt may produce multiple identical or
near-identical completions across its ``n_samples`` rollouts.  When every
sample in a GRPO/GSPO group is the same, the group-relative advantage
collapses to zero and the training signal is wasted.

AReno can optionally detect normalized duplicate completions and request
replacement samples up to a bounded budget.  The feature is **disabled by
default** so existing behavior is unchanged.

How it works
------------

1. After rollout, each prompt's ``n_samples`` completions are decoded and
   normalized (whitespace-stripped and lowercased).
2. Completions that match an earlier one in the group are flagged as
   duplicates.  The first occurrence is always kept.
3. Replacement samples are requested for duplicate positions, up to
   ``dedup_max_resample`` extra requests or until ``dedup_min_unique``
   unique completions are obtained.
4. If the budget is exhausted before all duplicates are replaced, the
   remaining duplicates stay in the batch so downstream training always
   receives a full ``n_samples`` set per prompt.

Configuration
-------------

The following fields on ``RolloutTrainerConfig`` (and its subclasses
``PolicyTrainerConfig``, ``PPOTrainerConfig``) control the feature:

============================ ========================================== ====================
Field                        Description                                Default
============================ ========================================== ====================
``dedup_enabled``            Enable duplicate resampling.               ``False``
``dedup_min_unique``         Target number of unique completions per    ``None`` (= ``n_samples``)
                             group.  Resampling stops once this many
                             unique samples exist.
``dedup_max_resample``       Hard cap on extra rollout requests per     ``None`` (= ``n_samples``)
                             group.  Prevents infinite loops.
============================ ========================================== ====================

Usage via CLI
-------------

Pass the options when constructing the trainer config in a custom script:

.. code-block:: python

   from areno.api.trainer_config import PolicyTrainerConfig

   config = PolicyTrainerConfig(
       algo="gspo",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       n_samples=8,
       dedup_enabled=True,
       dedup_min_unique=6,      # stop once 6 unique samples exist
       dedup_max_resample=4,    # at most 4 extra requests per group
   )

Usage via SDK
-------------

The detection logic is also available as a standalone helper for custom
training loops:

.. code-block:: python

   import areno.api

   completions = ["42", "42", "43", "42"]
   result = areno.api.detect_duplicates(
       completions,
       target_unique=3,
       max_resample=2,
   )
   print(result.duplicate_count)       # 2
   print(result.duplicate_indices)     # [1, 3]
   print(result.resample_requested)    # 2
   print(result.duplicate_ratio)       # 0.5

Observable output
-----------------

When the feature is enabled, the trainer logs a summary line per batch:

.. code-block:: text

   dedup duplicates=12 unique=20 total=32 ratio=0.3750 resample_requests=8

The ``DedupResult`` dataclass exposes the following fields for programmatic
use:

============================ ==========================================
Field                        Description
============================ ==========================================
``duplicate_count``          Number of duplicate completions detected.
``unique_count``             Number of unique completions.
``total_count``              Total completions in the group.
``duplicate_ratio``          ``duplicate_count / total_count``.
``resample_requested``       Bounded number of extra requests to issue.
``duplicate_indices``        Positions of duplicates within the group.
============================ ==========================================

Limitations
-----------

* Normalization is conservative: only exact matches after whitespace
  stripping and lowercasing are treated as duplicates.  Semantic similarity
  is not considered.
* Resampling is not recursive: a replacement that is itself a duplicate of
  an existing unique completion still counts against the budget.
* The feature applies to prompt-based rollout (GSPO/GRPO/PPO).  Agentic
  rollout is not affected.

Backward compatibility
----------------------

When ``dedup_enabled`` is ``False`` (the default), no detection or
resampling occurs and the training loop behaves exactly as before.