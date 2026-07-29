# Parameter Relations

| Parameter | Main effect |
| --- | --- |
| `tp_size` | supported tensor/cache sharding; constrained by model dimensions |
| `world_size` | total workers and data-parallel replicas |
| `batch_size` | prompts or pairs per step |
| `n_samples` | rollout samples per prompt |
| `max_running_prompts` | simultaneous scheduler/cache occupancy |
| `mini_bs` | training rows per microbatch |
| `score_micro_bs` | auxiliary role scoring rows per microbatch |
| `max_context_len` | prompt plus generated context capacity |
| `max_new_tokens` | generation semantics and worst-case decode work |
| `drop_rollout_state` | lifecycle memory release before training |

Total rollout demand is `batch_size * n_samples`. It may exceed active concurrency and execute in continuous-batching waves.
