# Failure Triage

| First failing stage | Inspect first |
| --- | --- |
| Dataset | loader import, normalized row, model-hub access |
| Dataset cache | `stage=dataset_cache_config` path/mode; `dataset_cache_rejected` (fingerprint/version mismatch); `dataset_cache_skip` (non-serializable record) |
| Rollout | active sequences, generation bound, agent timeout, worker stack |
| Reward | decoded answer, source solution, parser, sample grouping |
| Rollout OOM | context/cache capacity, concurrency, TP, retained state |
| Train OOM | `mini_bs`, packed length, recompute, optimizer state |
| NaN | rollout logprob, reward/advantage, train logprob, ratio/loss, gradients |
| NCCL watchdog | earliest distinct worker error before watchdog |
| Compile/capture | warmup, graph capture boundary, unsupported mutation or dtype |

Do not add fallback behavior, loosen tolerances, or shorten semantic limits before explaining the mismatch.
