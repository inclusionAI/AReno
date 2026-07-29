# AReno Code Map

Use this map to choose the first files to read for a small change or review.
It maps behavior to its owner; [AGENTS.md](AGENTS.md) remains the source for
setup, working rules, and repository conventions.

## Main execution paths

- CLI dispatch starts at `ArenoCli` and `main` in `areno/cli/main.py`.
- Training enters `train_command` and then `run` in `areno/cli/train.py`.
  `run` resolves the algorithm, dataset, reward, loss, and concrete trainer
  before calling `fit`.
- Serving enters `serve_command` and `create_app` in `areno/cli/serve.py`.
  `create_app` constructs `ArenoEngine` and the OpenAI-compatible FastAPI
  routes.
- SDK callers start with `Trainer` in `areno/api/trainer.py`. Its `init`,
  `rollout_batch`, `train`, and `close` methods own the public lifecycle.

The SDK training runtime crosses these ownership boundaries:

```text
Trainer
  -> Backend
    -> ArenoBackend
      -> ArenoEngine
        -> ArenoWorker
```

`Backend` in `areno/api/backend/base.py` defines the execution contract.
`ArenoBackend` in `areno/api/backend/areno/backend.py` packs public API data for
AReno. `ArenoEngine` in `areno/engine/api.py` dispatches rollout and train
operations to `ArenoWorker` in `areno/engine/worker.py`. Rank-side rollout is
owned by `InferenceManager` in `areno/engine/inference.py`; rank-side training
is owned by `TrainingManager` in `areno/engine/training.py`, with shared helpers
under `areno/engine/runtime/`. See
[Backend Topology](docs/concepts/backend-topology.rst) for the shorter
backend-only view.

## Registries and extension points

- Algorithms: `AlgorithmSpec` and `register_algorithm` in
  `areno/api/algorithms.py`; concrete loops in `areno/api/trainers/`;
  construction in `areno/api/trainer_factory.py`.
- Losses: `sft_loss_fn`, `dpo_loss_fn`, `gspo_loss_fn`, `grpo_loss_fn`, and
  `ppo_loss_fn` under `areno/api/loss_fns/`.
- Models: the `ModelAdapter` contract in `areno/models/base.py`,
  `register_adapter` and checkpoint dispatch in `areno/models/registry.py`, and
  one implementation directory per model family under `areno/models/`.

## Where to start

| Task | Start here | Nearby verification |
| --- | --- | --- |
| Change train or serve CLI behavior | `areno/cli/train.py` or `areno/cli/serve.py` | `tests/test_train_cli_config_cpu.py` or `tests/test_serve_cli_cpu.py` |
| Change SDK rollout or training behavior | `areno/api/trainer.py`, then follow the `Backend` call | `tests/test_trainer_api_cpu.py` and `tests/test_protocol_cpu.py` |
| Add or change an algorithm or loss | `areno/api/algorithms.py`, `areno/api/trainers/`, and `areno/api/loss_fns/` | `tests/test_algorithms_cpu.py` and `tests/test_losses_rewards_cpu.py` |
| Add or change a model family | `areno/models/base.py`, `areno/models/registry.py`, then the closest family under `areno/models/` | `tests/test_registry_cpu.py` and `tests/test_registry_discovery_cpu.py` |
| Change agentic rollout behavior | `areno/api/agentic.py`, then a matching task under `examples/agentic/` | `tests/test_agentic_cpu.py` and the matching example test |
| Cache tokenized dataset samples | `areno/api/dataset_cache.py`, then `areno/api/trainer.py::Trainer.load_prompt_batches` | `tests/test_dataset_cache_cpu.py` and `tests/test_trainer_api_cpu.py::TrainerDatasetCacheTest` |

## Tests, examples, and skills

Runnable context lives in `examples/`: math RLVR under `examples/math/`, SFT
under `examples/sft/`, and multi-turn agents under `examples/agentic/`.
CPU-safe tests use the `*_cpu.py` suffix in `tests/`. Repeatable
model-adaptation guidance lives in `.agents/skills/areno-model-adaptation/SKILL.md`.
