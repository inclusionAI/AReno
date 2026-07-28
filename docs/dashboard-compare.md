# Compare Two Training Runs

## Overview

The **Compare** feature lets you select two AReno training runs (active or completed) and display a side-by-side comparison of their configuration, metrics, and timing—all within the existing local dashboard. No external database or cloud service is required.

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

The results are displayed in three sections:

### Config Differences

Only **changed** settings are shown by default. Identical settings are folded and can be revealed by clicking *Show identical settings*. Fields that only apply to RL algorithms (e.g., `n_samples`, `reward_fn_path`) include a note explaining why they are absent on the SFT side.

### Metrics Comparison

Each metric present in either job is listed with:

| Column | Meaning |
|--------|---------|
| Metric | Metric name (e.g., `loss`, `reward_mean`) |
| Job A (latest) | Latest value and step for Job A |
| Job B (latest) | Latest value and step for Job B |
| Diff | `value_a - value_b` (green if A is lower, red if higher) |
| Note | Explanation if the metric is not comparable |

Metrics only present in one job (e.g., `reward_mean` for a GSPO run but not for SFT) are marked as non-comparable with an explanatory note.

### Timing Comparison

| Row | Meaning |
|-----|---------|
| Steps completed | Number of timeperf entries recorded |
| Avg total / step | Average `total_s` across all steps |
| Avg rollout / step | Average `rollout_s` (null for SFT) |
| Avg train / step | Average `train_s` |
| Total duration | `updated_at - created_at` in seconds |

If the two jobs ran very different numbers of steps (difference > 3), a warning is shown indicating the timing comparison may be less reliable.

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
  "job_a": { "id": "...", "name": "...", "status": "...", "step": 13 },
  "job_b": { "id": "...", "name": "...", "status": "...", "step": 11 },
  "comparable": true,
  "config": {
    "identical": [ { "key": "ckpt", "value": "Qwen/Qwen3.5-0.8B" } ],
    "different": [ { "key": "algo", "value_a": "gspo", "value_b": "sft", "note": null } ]
  },
  "metrics": [ { "name": "loss", "value_a": 0.65, "value_b": 0.80, "diff": -0.15, "comparable": true } ],
  "timing": {
    "job_a": { "total_steps": 14, "avg_total_s": 51.5 },
    "job_b": { "total_steps": 12, "avg_total_s": 156.4 },
    "comparison": { "avg_total_diff_s": -104.9, "steps_diff": 2, "note": null }
  }
}
```

### Error Responses

| Condition | HTTP Status | Body |
|-----------|-------------|------|
| Missing `job_a` or `job_b` | 400 | `{"error": "job_a and job_b are required"}` |
| Job not found | 400 | `{"error": "job <id> not found"}` |

## Limitations

- Only local dashboard data is used (no external database).
- SFT jobs do not produce rollout or reward metrics; these fields show as N/A.
- If a job has no timeperf entries, all timing averages are null.
- The comparison is read-only; it does not modify or stop any running job.
- Backward compatible: existing endpoints and dashboard pages are unchanged.