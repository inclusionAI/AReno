:orphan:

Non-finite value detection
==========================

When AReno training encounters anomalous values (NaN/Inf), a human-readable
report is generated to help users quickly identify the root cause.

This feature was implemented as part of Issue #238.

Design
------

Two-layer detection strategy:

* **Fast check** — loss-based detection every step, zero overhead.
* **Deep check** — parameter sweep every 100 steps (or when loss is anomalous).

The detector reuses existing infrastructure (``_param_grad`` / ``_grad_norm``)
rather than introducing parallel machinery.

Implementation
--------------

============================ ==================================================
File                         Content
============================ ==================================================
``non_finite.py``            Detector core module (detection + report
                             formatting + JSON output + NonFiniteTrainingError
                             + cross-rank all_reduce_non_finite_flag +
                             emit_non_finite_report)
``training.py``              Actor training injection point (skip update +
                             terminate + cross-rank all_reduce)
``roles.py``                 Critic training injection point (skip update +
                             terminate + cross-rank all_reduce)
``policy_only.py``           Trainer-layer rewards/advantages detection
                             (``_check_non_finite_values``)
``config.py``                ``RuntimeConfig``: ``non_finite_skip_update`` /
                             ``non_finite_terminate`` fields (default False)
``trainer_config.py``        ``TrainerConfig``: same fields propagated to
                             ``ArenoConfig`` via ``areno_config()``
``train.py``                 CLI flags ``--non-finite-skip-update`` and
                             ``--non-finite-terminate``
``tests/test_non_finite_cpu.py`` Unit tests (CPU-only, follows project
                             naming convention)
============================ ==================================================

Environment
-----------

* Kaggle Notebook, Tesla T4 x2 (TP=2)
* Model: Qwen/Qwen3-0.6B (HuggingFace cache)
* Dataset: AI-MO/NuminaMath-CoT
* Algorithm: GSPO

Issues encountered and resolved
-------------------------------

=== =============================== ========================================= =========================================
 #  Problem                         Cause                                     Solution
=== =============================== ========================================= =========================================
 1  ``--model-hub modelscope``     ModelScope slow on Kaggle network         Use ``--model-hub hf`` with HF cache
    download very slow (~700 KB/s)
 2  ``--model-hub huggingface``    Only ``hf`` or ``modelscope`` are valid   Use ``--model-hub hf``
    error
 3  Tesla T4 does not support      cc 7.5 does not meet requirements         Auto-fallback to native attention
    flash-attn
 4  Tesla T4 does not support BF16 Hardware limitation                       Auto-fallback to eager execution
 5  async SDK reward_fn mismatch   SDK interface differs from docs           Use ``areno train`` CLI command
 6  ``_param.data[0, 0] = NaN``    1D params (e.g. bias) cannot use 2D       Use ``.flatten()[0]``
    index out of bounds             index
 7  Both ranks inject NaN          TP=2: both rank 0 and rank 1 execute     Add ``if _ctx.rank == 0`` guard
                                    injection
 8  Optimizer has no               AReno uses custom ``AdamWFP32Master``     Add ``_safe_optimizer_state``
    ``param_groups``                                                         compatibility function
 9  Report outputs all 28 layers   200+ events after full NaN propagation   Truncate to first 5 + summary
 10 ``_merge_metrics`` crashes     ``to_dict`` includes string ``"actor"``  ``to_dict`` outputs numeric fields only
                                    which cannot convert to float
 11 ``__init__.py`` missing        Omission                                  Add import
    ``NonFiniteReport`` import
 12 ``to_json_dict`` accesses      Field is actually ``gpu_memory_gb``       Fix attribute name
    ``self.gpu_memory``
 13 JSON file cannot be parsed     NaN is not a valid JSON value             Add ``_sanitize`` to recursively
                                                                            replace NaN/Inf with ``null``
 14 ``to_json_file``: ``json``     Missing import                            Add ``import json`` inside method
    undefined
 15 JSON report not identifiable   No explicit marker                        Add ``alert`` / ``alert_type`` /
                                                                            ``severity`` fields
=== =============================== ========================================= =========================================

Verification
------------

Steps 0-4: normal training, ``reward_mean`` 0-0.25, ``grad_norm`` 0-2.8.

Step 5 (first detection after NaN injection):

Terminal report:

.. code-block:: text

   ========================================================
    WARNING Non-Finite Value Training Report
   ========================================================

   LOCATION
    Step: 5 | Phase: actor
    Last checkpoint: N/A

   ANOMALIES DETECTED
    [GRAD] embed_tokens.weight.grad
    -> 475,136 NaN (0.61%)
    -> grad_norm = nan
    [PARAM] layers.0.input_layernorm.weight
    -> 1 NaN (0.10%)
    -> max=1.0469e+00 min=1.2158e-01
    [GRAD] layers.0.input_layernorm.weight.grad
    -> 1024 NaN (100.00%)
    -> grad_norm = nan
    [GRAD] layers.0.self_attn.qkv_proj.weight.grad
    -> 2097152 NaN (100.00%)
    -> grad_norm = nan
    [GRAD] layers.0.self_attn.o_proj.weight.grad
    -> 1048576 NaN (100.00%)
    -> grad_norm = nan

   CONTEXT
    Loss: nan
    LR: 1.00e-06
    Global grad_norm: nan
    GPU memory: 4.07 GB
    ... and 223 more events (showing first 5)
    SUMMARY: 227 gradient + 1 parameter events
    Total NaN: 298,532,865  Total Inf: 0
    Affected layers: 31 (layers.21, layers.2, layers.27...)

   LIKELY CAUSES
    1. [MID] Single-layer anomaly -> layers.0

   SUGGESTED FIXES
    1. Check input data range/distribution for that layer
    2. Consider adding LayerNorm or reducing init variance

   JSON REPORT: non_finite_reports/step_5_actor.json
   ========================================================

JSON report (``non_finite_reports/step_5_actor.json``):

.. code-block:: json

   {
     "alert": true,
     "alert_type": "non_finite_values_detected",
     "severity": "critical",
     "step": 5,
     "phase": "actor",
     "loss": null,
     "global_grad_norm": null,
     "learning_rate": 1e-06,
     "gpu_memory_gb": 4.07,
     "events": [
       {
         "name": "embed_tokens.weight.grad",
         "layer": "embed_tokens",
         "is_gradient": true,
         "nan_count": 475136,
         "nan_ratio": 1.0,
         "grad_norm": null
       },
       {
         "name": "layers.0.input_layernorm.weight",
         "layer": "layers.0",
         "is_gradient": true,
         "nan_count": 1024,
         "nan_ratio": 1.0,
         "grad_norm": null
       },
       {
         "name": "layers.0.self_attn.qkv_proj.weight.grad",
         "layer": "layers.0",
         "nan_count": 2097152,
         "nan_ratio": 1.0
       },
       {
         "name": "layers.0.self_attn.o_proj.weight.grad",
         "layer": "layers.0",
         "nan_count": 1048576,
         "nan_ratio": 1.0
       },
       {
         "name": "layers.0.mlp.gate_up_proj.weight.grad",
         "layer": "layers.0",
         "nan_count": 3145728,
         "nan_ratio": 1.0
       },
       {
         "name": "layers.0.mlp.down_proj.weight.grad",
         "layer": "layers.0",
         "nan_count": 1572864,
         "nan_ratio": 1.0
       }
     ],
     "causes": ["Single-layer anomaly -> layers.0"],
     "suggestions": ["Check input data range/distribution for that layer"],
     "total_nan": 8341760,
     "total_inf": 0,
     "affected_layers": ["embed_tokens", "layers.0"]
   }

Step 8 (global NaN propagation):

.. code-block:: text

   ========================================================
    WARNING Non-Finite Value Training Report
   ========================================================

   LOCATION
    Step: 8 | Phase: actor
    Last checkpoint: N/A

   ANOMALIES DETECTED
    [PARAM] embed_tokens.weight
    -> 77791232 NaN (100.00%)
    [GRAD] embed_tokens.weight.grad
    -> 6144 NaN (0.01%)
    -> grad_norm = nan
    [PARAM] layers.0.input_layernorm.weight
    -> 1024 NaN (100.00%)
    [GRAD] layers.0.input_layernorm.weight.grad
    -> 1024 NaN (100.00%)
    -> grad_norm = nan
    [PARAM] layers.0.self_attn.qkv_proj.weight
    -> 2097152 NaN (100.00%)

   CONTEXT
    Loss: nan
    LR: 1.00e-06
    Global grad_norm: nan
    GPU memory: 4.14 GB
    ... and 449 more events (showing first 5)
    SUMMARY: 227 gradient + 227 parameter events
    Total NaN: 673,912,832  Total Inf: 0
    Affected layers: 31 (layers.6, layers.18, layers.15...)

   LIKELY CAUSES
    1. [UNKNOWN] Cannot auto-diagnose; check data and model

   SUGGESTED FIXES
    1. Check input data for NaN/Inf values
    2. Try reducing learning rate and batch size

   JSON REPORT: non_finite_reports/step_8_actor.json
   ========================================================

NaN propagation comparison
--------------------------

================ ======= ============================
Metric         Step 5 Step 8
================ ======= ============================
Affected       2      31 (all)
layers
Event count    228    454
Total NaN      299M   674M
Anomaly type   Grad   Param + Grad
Root cause     Single Unknown
               layer
GPU memory     4.07GB 4.14GB
================ ======= ============================

This validates the core value of #238: **detect early, intervene early**.
At step 5 only 2 layers are affected and the root cause can be pinpointed;
by step 8 all 31 layers are corrupted beyond recovery.

Lessons learned
---------------

1. **Environment first**: GPU model determines attn_backend and precision.
   Tesla T4 only supports native attention + FP32.
2. **Verify interfaces**: Do not trust docs — run it to find out (``--model-hub``
   valid values, SDK parameters, etc.).
3. **TP multi-rank caution**: Injection/modification should only happen on one
   rank, otherwise duplicate operations or shape mismatch.
4. **JSON is strict**: NaN/Inf are not valid JSON values; must sanitize to null.
5. **Truncate reports**: 200+ event reports are unreadable; truncation + summary
   is essential.
6. **``_merge_metrics`` only accepts float**: Putting strings in the metrics dict
   will crash.
7. **Reports need explicit markers**: Adding ``alert`` / ``severity`` fields to
   JSON makes it immediately clear this is an anomaly alert, not a regular log.

Configuration
-------------

Issue #238 adds two opt-in flags. Both default to ``False`` so existing
behavior is preserved when they are not set.

======================================== ========================================
Flag                                     Effect
======================================== ========================================
``--non-finite-skip-update``             When NaN/Inf is detected, skip
                                         ``optimizer.step()`` and discard
                                         polluted gradients. The global step
                                         counter does not advance.
``--non-finite-terminate``               After reporting, raise
                                         ``NonFiniteTrainingError`` to
                                         terminate training in a controlled
                                         manner.
======================================== ========================================

These flags can also be set programmatically:

.. code-block:: python

   from areno.api.trainer_config import TrainerConfig

   config = TrainerConfig(
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       non_finite_skip_update=True,
       non_finite_terminate=False,
   )

Detection coverage
------------------

The detector checks four metric categories as required by Issue #238:

==================== ====================================================
Metric               Detection location
==================== ====================================================
Loss                 Engine layer: every step via
                     ``check_loss_non_finite``
Gradients            Engine layer: every 100 steps (or on NaN loss)
                     via ``detect_non_finite``
Parameters           Engine layer: same schedule as gradients
Optimizer state      Engine layer: same schedule as gradients
Rewards              Trainer layer: ``_check_non_finite_values``
                     in ``policy_only.py``
Advantages           Trainer layer: ``_check_non_finite_values``
                     in ``policy_only.py``
==================== ====================================================

Cross-rank coordination
-----------------------

When running with tensor-parallel or data-parallel, each rank performs
local detection independently. ``all_reduce_non_finite_flag`` uses a
``MAX`` all-reduce so that if *any* rank detects NaN/Inf, *all* ranks
treat the step as non-finite. This ensures consistent skip/terminate
behavior across the cluster — no rank advances ``optimizer.step()``
while another discards gradients.
