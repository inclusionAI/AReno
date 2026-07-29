# Compare Two Training Runs

## Overview

The **Compare** feature lets you select two AReno training runs (active or completed) and display a side-by-side comparison of their configuration, metrics, curves, timing, and throughput—all within the existing local dashboard. No external database or cloud service is required.

## Accessing the Feature

1. Start the dashboard:

   ```bash
   areno dashboard --start
   ```

2. Open <http://127.0.0.1:8765> in your browser.

3. Click the **Compare** tab in the left navigation bar.

## Using the Compare Panel

1. **Select Job A** from the first dropdown (lists all known jobs).
2. **Select Job B** from the second dropdown.
3. Click **Compare**.

The results are displayed in the following sections:

### Run Header Cards

Two cards at the top show each run's name, status badge (exited/running/failed), algorithm type, and current step count. Job A has a blue left border; Job B has an orange left border.

### Key Metric Cards

A grid of cards showing the most important metrics side by side:

| Card | Description |
|------|-------------|
| Loss | Latest loss value (lower is better, shown in green) |
| Reward Mean | Latest reward mean (higher is better) — hidden for SFT |
| Accuracy | Latest accuracy (higher is better) — hidden if absent |
| Learning Rate | Latest learning rate |
| Steps | Total steps completed |
| Duration | Total runtime in seconds |
| Throughput | Steps per second |

Each card shows values for both Job A (blue tag) and Job B (orange tag). Cards for metrics not present in either job are automatically hidden.

### Hyperparameters

Only **changed** settings are shown by default with an orange diff indicator. Identical settings are folded and can be revealed by clicking *Show identical settings*. Fields that only apply to RL algorithms (e.g., `n_samples`, `reward_fn_path`) include a note explaining why they are absent on the SFT side.

### Metric Curve Comparison

Overlaid line charts for every metric, with tabs to switch between metrics:

- **Job A**: blue solid line
- **Job B**: orange dashed line
- Both share the same Y-axis range for direct comparison

When the two jobs have different step counts, a **Normalize X-axis** checkbox (enabled by default) stretches each line to full width by training progress percentage. Disable it to use absolute step numbers.

### Key Differences

An auto-generated summary listing all changed configuration values with arrows (`old -> new`) and ratio annotations (`Nx`) for changes of 2x or more.

### All Metrics Comparison

Each metric present in either job is listed with:

| Column | Meaning |
|--------|---------|
| Metric | Metric name (e.g., `train/loss`, `rollout/rewards_mean`) |
| Job A (latest) | Latest value and step for Job A |
| Job B (latest) | Latest value and step for Job B |
| Diff | `value_a - value_b` (green if negative, red if positive) |
| Note | Explanation if the metric is not comparable |

Metrics only present in one job are marked as non-comparable with an explanatory note.

### Timing Comparison

| Row | Meaning |
|-----|---------|
| Steps completed | Number of timeperf entries recorded |
| Avg total / step | Average `total_s` across all steps |
| Avg rollout / step | Average `rollout_s` (null for SFT) |
| Avg train / step | Average `train_s` |
| Total duration | `updated_at - created_at` in seconds |

If the two jobs ran very different numbers of steps (difference > 3), a warning is shown indicating the timing comparison may be less reliable.

## CLI Usage

```bash
areno compare --job-a <id> --job-b <id>
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--job-a` | (required) | First job ID |
| `--job-b` | (required) | Second job ID |
| `--format` | `human` | Output format: `human` or `json` |
| `--dashboard-url` | `http://127.0.0.1:8765` | Dashboard server URL |

### Example (human output)

```bash
$ areno compare --job-a abc123 --job-b def456

Job A: train gspo Qwen/Qwen3.5-0.8B (id=abc123, status=exited, step=13)
Job B: train sft Qwen/Qwen3.5-0.8B (id=def456, status=exited, step=11)

Config: 2 changed, 4 identical
  Changed settings:
    algo: A='gspo'  B='sft'
    max_new_tokens: A=1024  B=512

Metrics (3):
    loss: A=0.65  B=0.80  diff=-0.150000
    reward_mean: A=0.324  B=None  (metric only in job A (gspo))

Timing:
    Steps: A=14  B=12
    Avg total/step: A=51.5s  B=50.4s
    Step time diff: +1.1s
```

### Example (JSON output)

```bash
$ areno compare --job-a abc123 --job-b def456 --format json | python -m json.tool
```

## API Reference

```
GET /api/compare?job_a=<id>&job_b=<id>
```

### Example

```bash
curl "http://127.0.0.1:8765/api/compare?job_a=abc123&job_b=def456" | python -m json.tool
```

### Response Fields

```json
{
  "job_a": { "id": "...", "name": "...", "status": "...", "kind": "...", "step": 13 },
  "job_b": { "id": "...", "name": "...", "status": "...", "kind": "...", "step": 11 },
  "comparable": true,
  "config": {
    "identical": [ { "key": "ckpt", "value": "Qwen/Qwen3.5-0.8B" } ],
    "different": [ { "key": "algo", "value_a": "gspo", "value_b": "sft", "note": null } ]
  },
  "metrics": [ { "name": "loss", "value_a": 0.65, "value_b": 0.80, "diff": -0.15, "comparable": true } ],
  "metric_charts": {
    "loss": {
      "points_a": [ { "step": 1, "value": 1.5 }, { "step": 2, "value": 1.0 } ],
      "points_b": [ { "step": 1, "value": 2.0 } ]
    }
  },
  "diff_summary": [ "algo: gspo -> sft", "max_new_tokens: 1024 -> 512" ],
  "throughput_a": 0.1,
  "throughput_b": 0.08,
  "timing": {
    "job_a": { "total_steps": 14, "avg_total_s": 51.5, "duration_s": 140.0 },
    "job_b": { "total_steps": 12, "avg_total_s": 50.4, "duration_s": 155.0 },
    "comparison": { "avg_total_diff_s": 1.1, "steps_diff": 2, "note": null }
  }
}
```

### Error Responses

| Condition | HTTP Status | Body |
|-----------|-------------|------|
| Missing `job_a` or `job_b` | 400 | `{"error": "job_a and job_b are required"}` |
| Job not found | 400 | `{"error": "job <id> not found"}` |
| Same job selected | 200 | `{"comparable": false, "reason": "same job"}` |

## Limitations

- Only local dashboard data is used (no external database).
- Jobs started from the command line (not via the Dashboard Launcher) may not have `launch_config` saved, so hyperparameter comparison may show 0 differences.
- SFT jobs do not produce rollout or reward metrics; these fields show as N/A.
- If a job has no timeperf entries, all timing averages are null.
- The comparison is read-only; it does not modify or stop any running job.
- Backward compatible: existing endpoints and dashboard pages are unchanged.

## Testing

CPU tests are in `tests/test_dashboard_compare_cpu.py` (27 tests):

| Category | Tests | Coverage |
|----------|-------|----------|
| Success paths | 3 | Normal comparison, GSPO vs SFT, identical config |
| Invalid input | 3 | Missing job_a, missing job_b, nonexistent job |
| Boundary | 5 | Same job, no metrics, unequal steps, no timeperf, empty config |
| Active & compat | 3 | Running jobs, duration calculation, existing methods unchanged |
| CLI | 3 | Human format, JSON format, invalid job |
| New features | 10 | metric_charts, diff_summary, throughput, non-numeric metrics |

Run with: `pytest tests/test_dashboard_compare_cpu.py -v`