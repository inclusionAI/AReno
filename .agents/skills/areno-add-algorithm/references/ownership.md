# Algorithm Ownership

- `areno/api/algorithms.py`: declarative registration and loss binding.
- `areno/api/trainer_config.py`: minimal public configuration.
- `areno/api/trainers/`: data materialization and lifecycle orchestration.
- `areno/api/loss_fns/`: pure tensor loss mathematics.
- `areno/api/advantages.py`: reusable advantage calculations.
- `areno/api/roles.py` and backend role APIs: reference, critic, and reward ownership.
- `areno/experimental/`: unstable algorithms loaded through registration.

Avoid algorithm-name conditionals in CLI, trainer factory, engine workers, or model adapters when metadata can express the behavior.
