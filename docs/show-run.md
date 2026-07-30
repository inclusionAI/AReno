# Show Detailed Information for One Run

## Overview

The `areno show` command displays detailed information for a single training run — model, dataset, algorithm, resolved key settings, current stage, latest metrics, and last error — directly in the terminal. No dashboard or external service required.

## Usage

```bash
areno show <run_id>                  # human-readable output (default)
areno show <run_id> --format json    # structured JSON
areno show <run_id> --format table   # compact key-value table
```

The `run_id` can be a full ID, a partial ID prefix (if unique), or a PID.

## Example Output

```
$ areno show 5a3660acfbe3

Run: train sft Qwen/Qwen3.5-0.8B  (id=5a3660acfbe3)
Kind:     train
Status:   exited
Step:     268
Created:  2026-07-29T02:30:00+00:00
Updated:  2026-07-29T03:15:00+00:00
Metrics:  /tmp/areno/tfevent_sft

Key settings:
  algo                  sft
  ckpt                  Qwen/Qwen3.5-0.8B
  dataset_path          yahma/alpaca-cleaned
  world_size            2
  batch_size            2
  lr                    1e-06

Latest metrics (29):
  rollout/accuracy            0.0  @step 267
  rollout/prompt_len_mean     31.0  @step 267
  train/loss                  1.2566  @step 267
  train/grad_norm             35.443  @step 267
  ...

Timing (268 steps recorded):
  Avg total / step:    1.07s

Recent logs (last 10 lines):
  process started pid=12345; metrics_dir=/tmp/areno/tfevent_sft
  ...
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `run_id` | (required) | Run ID, partial ID prefix, or PID |
| `--format` | `human` | `human`, `table`, or `json` |
| `--dashboard-url` | `http://127.0.0.1:8765` | Dashboard URL (used when available) |

## Limitations

- Only local dashboard data is used (no external database).
- When the dashboard server is not running, the command falls back to reading local artifact files.
- Secrets (keys, tokens, passwords) in config are automatically redacted from output.
- Full training samples are never printed.

## Testing

CPU tests are in `tests/test_cli_show_cpu.py` (20 tests):

| Category | Tests | Coverage |
|----------|-------|----------|
| Human format | 7 | Basic output, key settings, metrics, error section, timing, secrets redaction |
| JSON format | 1 | Valid JSON with expected fields |
| Table format | 1 | Key fields present in table output |
| Invalid input | 6 | Nonexistent ID, ambiguous partial ID, partial match, empty metrics/config, partial artifacts |
| Active runs | 2 | Running status, stage display, no exit code |
| Resolve job | 3 | Exact ID match, PID match, no match |

Run with: `pytest tests/test_cli_show_cpu.py -v`