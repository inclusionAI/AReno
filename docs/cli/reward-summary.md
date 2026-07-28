# Reward Summary

Summarise reward distributions from local training artifacts in the terminal.

## Synopsis

```bash
areno reward-summary --metrics-log-dir <dir> [options]
```

## Description

The `reward-summary` command reads `reward_metrics.*.jsonl` files produced
during training and prints summary statistics for reward distributions:
mean, standard deviation, min/max, zero fraction, missing fraction, and a
configurable outlier fraction.

Statistics are computed for both the **total** reward and any **named
reward components** that the reward function produced. A reward function
that returns a plain `float` yields only the `total` row; one that returns
a `dict[str, float]` additionally produces one row per component key.

## Options

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--metrics-log-dir` | path | `/tmp/areno/tfevent` | Directory containing `reward_metrics.*.jsonl` files. |
| `--outlier-threshold` | float | `3.0` | A value is counted as an outlier when its absolute deviation from the mean exceeds this many standard deviations. |
| `--step` | int | _none_ | Only summarise records from this training step. |
| `--json` | flag | off | Emit machine-readable JSON instead of a table. |

## Output Fields

Each row in the table (or each entry in the JSON output) contains:

| Field | Description |
| --- | --- |
| Count | Total number of samples (including missing/non-finite). |
| Mean | Arithmetic mean of finite reward values. |
| Std | Standard deviation of finite reward values. |
| Min | Minimum finite reward value. |
| Max | Maximum finite reward value. |
| Zero% | Fraction of samples whose reward is exactly 0.0. |
| Missing% | Fraction of samples that are missing, NaN, or infinite. |
| Outlier% | Fraction of samples deviating more than `--outlier-threshold` std devs from the mean. |

## Named Reward Components

A reward function may return a `dict[str, float]` instead of a plain
`float` to expose individual reward components:

```python
def reward_fn(record) -> dict[str, float]:
    return {
        "correctness": 1.0 if is_correct(record) else 0.0,
        "format": 0.5 if is_well_formatted(record) else 0.0,
    }
```

AReno sums all component values to obtain the scalar total used by the
training loop. The individual component values are persisted to
`reward_metrics.*.jsonl` and appear as separate rows in the summary.

Components that appear in some samples but not others are treated as
*missing* for the samples where they are absent — this distinguishes
"component not produced" (missing) from "component value is 0" (zero).

## Example

After a short training run:

```bash
areno train --algo gspo --ckpt model --dataset-path data.jsonl \
    --reward-fn-path reward.py --metrics-log-dir /tmp/areno/run1
```

Summarise the reward distribution:

```bash
areno reward-summary --metrics-log-dir /tmp/areno/run1
```

Output:

```
                    Reward Distribution Summary
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Component   ┃ Count ┃   Mean ┃    Std ┃    Min ┃    Max ┃  Zero% ┃ Missing% ┃ Outlier% ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ total       │    50 │ 0.6200 │ 0.4521 │ 0.0000 │ 1.5000 │ 20.00% │    0.00% │    10.00% │
│ correctness │    50 │ 0.5000 │ 0.5051 │ 0.0000 │ 1.0000 │ 50.00% │    0.00% │     0.00% │
│ format      │    50 │ 0.1200 │ 0.3279 │ 0.0000 │ 0.5000 │ 80.00% │    0.00% │     0.00% │
└─────────────┴───────┴────────┴────────┴────────┴────────┴────────┴──────────┴──────────┘
Samples: 50
```

JSON output:

```bash
areno reward-summary --metrics-log-dir /tmp/areno/run1 --json
```

```json
{
  "components": {
    "correctness": {
      "count": 50,
      "max": 1.0,
      "mean": 0.5,
      "min": 0.0,
      "missing_fraction": 0.0,
      "outlier_fraction": 0.0,
      "std": 0.5050967595,
      "zero_fraction": 0.5
    },
    "format": {
      "count": 50,
      "max": 0.5,
      "mean": 0.12,
      "min": 0.0,
      "missing_fraction": 0.0,
      "outlier_fraction": 0.0,
      "std": 0.3279257692,
      "zero_fraction": 0.8
    }
  },
  "sample_count": 50,
  "total": {
    "count": 50,
    "max": 1.5,
    "mean": 0.62,
    "min": 0.0,
    "missing_fraction": 0.0,
    "outlier_fraction": 0.1,
    "std": 0.4521219484,
    "zero_fraction": 0.2
  }
}
```

## Implementation Notes

### Data Flow

1. During training, `_materialize_train_batch` / `_materialize_agentic_train_batch`
   calls `reward_fn(record)` and normalises the result via
   `normalize_reward_result()`.
2. If the reward function returns a `dict[str, float]`, values are summed
   for the scalar total and individual components are preserved.
3. Per-sample reward data is persisted to
   `reward_metrics.{pid}.jsonl` in the metrics log directory via
   `MetricsRecorder.record_reward_metrics()`.
4. `areno reward-summary` reads these JSONL files, computes statistics
   via `compute_component_statistics()`, and renders the output.

### Key Files

| File | Role |
| --- | --- |
| `areno/api/reward_stats.py` | Core statistics computation and formatting (pure functions). |
| `areno/api/rewards.py` | `normalize_reward_result()` — unifies `float` and `dict` return types. |
| `areno/api/metrics.py` | `MetricsRecorder.record_reward_metrics()` and `load_reward_samples()`. |
| `areno/cli/reward_summary.py` | CLI command implementation. |
| `areno/api/trainers/policy_only.py` | Reward normalization and persistence in GSPO/GRPO loops. |
| `areno/api/trainers/ppo.py` | Reward normalization and persistence in PPO loop. |

### Backward Compatibility

- Existing `reward_fn` returning `float` continues to work unchanged.
- `reward_metrics.*.jsonl` is only produced when `MetricsRecorder` is
  active (i.e. `metrics_log_dir` is set), which matches the existing
  behavior for `rollout_samples.*.jsonl`.
- The `reward-summary` command gracefully handles empty or missing
  directories with a user-friendly message.