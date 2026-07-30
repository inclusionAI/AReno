# Early Stopping Callback for AReno

This module provides early stopping functionality to automatically stop training when validation metrics stop improving.

## Features

- **Configurable Monitor**: Track any metric (default: `eval_loss`)
- **Patience Control**: Number of evaluations to wait before stopping
- **Mode Support**: `"min"` for loss metrics, `"max"` for accuracy/reward metrics
- **Minimum Delta**: Ignore insignificant improvements below threshold
- **Warmup Evaluations**: Skip initial unstable evaluations
- **Baseline Comparison**: Stop if model doesn't beat specified baseline
- **NaN Handling**: Gracefully handles NaN metric values
- **State Persistence**: Save/restore state for checkpoint resume
- **Transformers Compatible**: Works with Hugging Face Trainer via callback wrapper

## Installation

```bash
# Copy the areno/callbacks directory to your AReno installation
cp -r areno/callbacks /path/to/areno/
```

## Quick Start

### Basic Usage

```python
from areno.callbacks import EarlyStopping

# Monitor eval_loss, stop after 3 rounds without improvement
es = EarlyStopping(
    monitor="eval_loss",
    patience=3,
    mode="min",
    verbose=True
)

# Training loop
for epoch in range(100):
    metrics = trainer.evaluate()
    if es(metrics):
        print(f"Early stopping at epoch {epoch}")
        break
```

### With Transformers Trainer

```python
from transformers import Trainer, TrainerCallback
from areno.callbacks import EarlyStopping

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self):
        self.es = EarlyStopping(
            monitor="eval_loss",
            patience=3,
            mode="min"
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.es(metrics):
            control.should_training_stop = True
        return control

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[EarlyStoppingCallback()]
)
trainer.train()
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monitor` | str | `"eval_loss"` | Metric name to monitor |
| `patience` | int | `3` | Evaluations to wait before stopping |
| `mode` | str | `"min"` | `"min"` (lower better) or `"max"` (higher better) |
| `min_delta` | float | `0.0` | Minimum change to qualify as improvement |
| `verbose` | bool | `True` | Print status messages |
| `warmup` | int | `0` | Initial evaluations to skip |
| `baseline` | float | `None` | Stop if not beating baseline |

## Examples

### Monitor Accuracy (Max Mode)

```python
es = EarlyStopping(
    monitor="eval_accuracy",
    patience=5,
    mode="max",
    min_delta=0.001
)
```

### With Warmup

```python
es = EarlyStopping(
    monitor="eval_loss",
    patience=3,
    warmup=2  # Skip first 2 evaluations
)
```

### With Baseline

```python
es = EarlyStopping(
    monitor="eval_loss",
    patience=2,
    baseline=2.0  # Stop if loss doesn't go below 2.0
)
```

## Running Tests

```bash
python tests/test_early_stopping.py
```

Expected output:
```
==================================================
Running EarlyStopping Comprehensive Unit Tests
==================================================
✅ test_min_mode_basic passed
✅ test_max_mode_basic passed
...
🎉 All tests passed!
```

## Running Example

```bash
# Install dependencies
pip install transformers datasets torch

# Run training example
python examples/train_with_early_stopping.py
```

## API Reference

### `EarlyStopping.__call__(metrics) -> bool`

Check if early stopping should be triggered.

**Args:**
- `metrics` (dict): Dictionary containing metric values

**Returns:**
- `bool`: True if training should stop

### `EarlyStopping.state_dict() -> dict`

Get state for checkpoint saving.

### `EarlyStopping.load_state_dict(state_dict)`

Restore state from checkpoint.

### `EarlyStopping.reset()`

Reset to initial state.

### `EarlyStopping.get_status() -> dict`

Get current status information.

## Output Format

When verbose=True, the following messages are printed:

```
[EarlyStopping] Warmup 1/2: eval_loss=2.5000
[EarlyStopping] eval_loss initialized to 2.0000
[EarlyStopping] eval_loss improved from 2.0000 to 1.8000 (Δ0.2000)
[EarlyStopping] eval_loss did not improve. Current: 1.9000, Best: 1.8000. Counter: 1/3
[EarlyStopping] Stopping! Best eval_loss: 1.8000
```

## Limitations

- Currently designed for single-device training
- Distributed training coordination not yet implemented
- Requires explicit metric key in evaluation output

## License

Same as AReno project.
