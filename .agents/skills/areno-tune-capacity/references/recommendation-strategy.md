# Recommendation Strategy

The capacity recommender (``scripts/recommend_capacity.py``) generates three
override sets from measured or estimated memory data **without starting a
training run**. Each override set is validated against AReno's
``RolloutTrainerConfig`` constraints so the output can be applied directly to
``areno train`` command-line arguments.

## Recommendation Modes

| Mode | Target mem_frac | When to use |
| --- | --- | --- |
| conservative | min(0.7, peak * 0.75) | Sharing a node, first run on unfamiliar hardware, or recovering from OOM |
| balanced | min(0.85, peak * 0.9) | Steady-state operation with moderate safety margin |
| throughput | peak (typically 0.9) | Stable workload where maximizing samples/s is the goal |

## Adjustment Rules

| Parameter | Conservative | Balanced | Throughput |
| --- | --- | --- | --- |
| max_running_prompts | floor_pow2(value * 0.5) | floor_pow2(value * 0.75) | ceil_pow2(value), capped at batch*n_samples |
| mini_bs | floor_pow2(value * 0.5) | unchanged or *0.75 if peak > 0.7 | ceil_pow2(value), capped at batch*n_samples |
| activation_checkpointing | True | True | False if peak < 0.7 |
| keep_rollout_state | False (drop) | False (drop) | True (keep) |
| adam_8bit | True | unchanged | unchanged |
| max_new_tokens | **never modified** | **never modified** | **never modified** |
| max_context_len | **never modified** | **never modified** | **never modified** |
| max_prompt_tokens | **never modified** | **never modified** | **never modified** |

Power-of-two rounding follows the same logic as ``areno/cli/auto_tune.py``:
``floor_power_of_two`` rounds down to the nearest power of two (minimum 1),
``ceil_power_of_two`` rounds up.

## Fallback Estimation (no profile data)

When ``--peak-mem-frac`` is not provided the recommender estimates memory usage
from ``--gpu-memory-gb`` and ``--model-params-billions``:

```
weights_per_gpu = model_params_billions * 2 (FP16) / tp_size
optimizer_per_gpu = model_params_billions * (8 if 32-bit Adam else 2) / tp_size
base_mem_frac = (weights_per_gpu + optimizer_per_gpu) / gpu_memory_gb
```

This estimate covers weights and optimizer state but not activation, KV cache,
or gradient memory. Factor in those costs by lowering ``--mem-frac`` or
running ``--smoke-infer`` / ``--smoke-train`` for a measured profile.

If neither profile data nor GPU/model inputs are provided, the recommender uses
a default peak fraction of 0.6 and labels the profile source as ``"default"``.

## Validation

Each recommendation passes two checks:

1. **Static checks**: positive integers for ``max_running_prompts`` and
   ``mini_bs``; ``world_size % tp_size == 0``; positive rollout demand.
2. **Config construction**: builds a ``RolloutTrainerConfig`` with the
   recommended values and verifies ``__post_init__`` does not raise. This
   catches invalid ``attn_backend`` or ``model_hub`` combinations. If the
   ``areno`` package is not installed, this check is skipped and static-only
   validation is reported.

## Difference from ``areno train --tune-params``

| Aspect | ``--tune-params`` | ``recommend_capacity.py`` |
| --- | --- | --- |
| GPU probing | Yes (dummy-load + synthetic rows) | No |
| Output | Modified ``TrainerConfig`` (single result) | Three override sets (JSON + text) |
| Training run | Starts after tuning | Never starts |
| Use case | Find one safe configuration | Compare trade-offs before deciding |

Run ``--smoke-infer`` or ``--tune-params`` first to obtain measured profile
data, then feed the ``--peak-mem-frac`` to the recommender for precise
recommendations.