# Recipe Generation

The `generate_recipe.py` script derives a complete, editable training
configuration from AReno's existing `TrainerConfig` dataclass hierarchy
defaults.  It is the recommended starting point when you know the training
mode, GPU count, context length, and target batch size but need the full set
of AReno options filled in.

## Design rationale

AReno exposes training configuration exclusively through CLI flags and Python
dataclasses — there is no YAML or JSON config file format.  The recipe
generator bridges this gap by producing a structured JSON recipe *and* a
directly runnable `areno train` command, each value annotated with provenance
explaining why it was chosen.

### Derivation rules

| Category | Rule |
| --- | --- |
| Topology | `world_size = gpu_count`; `tp_size = 1` when `gpu_count <= 2`, else `min(4, gpu_count)` |
| Batch sizing | `batch_size = target_batch`; `mini_bs = min(target_batch, 4)` for small GPU, else `min(target_batch, 16)` |
| Context split | `max_prompt_tokens = min(context_length // 2, 1024)`; `max_new_tokens = min(remaining, 3071)` |
| Rollout fields | Included only for `gspo`/`grpo`/`ppo` (`n_samples`, `temperature`, `top_k`, `top_p`, etc.) |
| Algorithm fields | DPO: `dpo_beta`, `ref_ckpt`; GSPO: `gspo_clip_eps`; GRPO: `grpo_clip_eps`; PPO: full `PPOTrainerConfig` set |
| Safe defaults | `activation_checkpointing=True`, `attn_backend="flash"`, `optimizer_lr=1e-6`, `epochs=10` |

### Provenance categories

Each recipe value has a provenance string explaining its source:

- **user request** — the value comes directly from a required input (mode, GPU count, context length, target batch)
- **TrainerConfig default** — the value matches the dataclass field default in `areno/api/trainer_config.py`
- **capacity-derived** — the value was adjusted for small GPU counts (<=2) to avoid OOM
- **user override** — the value was set via `--override key=value` or a named flag like `--tp-size`
- **placeholder** — the value is a placeholder token (e.g. `<ckpt>`) that the user must replace before training

### Placeholder convention

The generated command uses angle-bracket placeholders for user-supplied paths:

| Placeholder | Meaning | Required by |
| --- | --- | --- |
| `<ckpt>` | Model checkpoint path or remote repo ID | All modes |
| `<dataset-path>` | Dataset path, HF save_to_disk dir, or remote ref | All modes |
| `<reward-fn-path>` | Python file defining `reward_fn(record)` | GSPO, GRPO, PPO |
| `<ref-ckpt>` | Frozen reference model checkpoint | DPO, PPO |
| `<reward-ckpt>` | PPO reward model checkpoint | PPO (optional) |
| `<critic-ckpt>` | PPO critic model checkpoint | PPO |
| `<loader-path>` | Dataset loader function file | SFT |

## Per-mode field matrix

| Field | SFT | DPO | GSPO | GRPO | PPO |
| --- | --- | --- | --- | --- | --- |
| `algo` | sft | dpo | gspo | grpo | ppo |
| `ckpt` | yes | yes | yes | yes | yes |
| `dataset_path` | yes | yes | yes | yes | yes |
| `dataset_loader_fn` | required | optional | optional | optional | optional |
| `tp_size` / `world_size` | derived | derived | derived | derived | derived |
| `batch_size` / `mini_bs` | derived | derived | derived | derived | derived |
| `n_samples` | — | — | 8 | 8 | 8 |
| `temperature` / `top_k` / `top_p` | — | — | included | included | included |
| `reward_fn_path` | — | — | required | required | required |
| `ref_ckpt` | — | required | — | — | required |
| `dpo_beta` | — | 0.1 | — | — | — |
| `gspo_clip_eps` | — | — | 3e-4 | — | — |
| `grpo_clip_eps` | — | — | — | 0.2 | — |
| `clip_eps` / `clip_ratio_c` | — | — | — | — | 0.2 / 3.0 |
| `critic_lr` / `critic_warmup_steps` | — | — | — | — | 1e-5 / 20 |
| `use_kl_loss` / `kl_loss_coef` | — | — | — | — | True / 0.001 |
| `gamma` / `lam` | — | — | — | — | 1.0 / 0.95 |
| `reward_ckpt` / `critic_ckpt` | — | — | — | — | optional / required |

## Workflow integration

1. **Generate**: Run `generate_recipe.py` with mode, GPU count, context length, and target batch.
2. **Inspect**: Review the `warnings` list for required-but-unset placeholders.
3. **Validate dataset**: Run `inspect_dataset.py` to confirm the dataset matches the algorithm's data contract.
4. **Check capacity**: Run `check_capacity.py` with the derived topology to verify memory feasibility.
5. **Replace placeholders**: Substitute `<ckpt>`, `<dataset-path>`, and other placeholders with real paths.
6. **Launch**: Copy the `command` field and run it.

## Kaggle T4 considerations

On Kaggle dual-T4 environments (`gpu_count=2`):

- `tp_size` defaults to 1 (data-parallel across 2 GPUs)
- `mini_bs` and `score_micro_bs` are reduced to 4 to avoid OOM
- T4 GPUs do not support FlashAttention 2 — consider `--override attn_backend=native` if you encounter attention backend errors
- Provenance strings explicitly mention the small-GPU downgrade reasoning

## Copyable example

```bash
# Generate a GSPO recipe for a 2-GPU (Kaggle T4) setup with 4096-token context
python .agents/skills/areno-run-training/scripts/generate_recipe.py \
  --mode gspo --gpu-count 2 --context-length 4096 --target-batch 8
```

Sample output (abbreviated):

```json
{
  "ok": true,
  "mode": "gspo",
  "recipe": {
    "algo": "gspo",
    "ckpt": "<ckpt>",
    "dataset_path": "<dataset-path>",
    "tp_size": 1,
    "world_size": 2,
    "batch_size": 8,
    "mini_bs": 4,
    "n_samples": 8,
    "gspo_clip_eps": 0.0003
  },
  "provenance": {
    "tp_size": "set to 1 because gpu_count (2) <= 2 (small-GPU friendly)",
    "mini_bs": "capped to min(target_batch, 4) = 4 for small GPU count (<=2) to avoid OOM"
  },
  "command": "areno train --algo gspo --ckpt <ckpt> --dataset-path <dataset-path> --reward-fn-path <reward-fn-path> --tp-size 1 ...",
  "warnings": [
    "ckpt is a placeholder; replace <ckpt> with your model checkpoint path or repo ID",
    "dataset_path is a placeholder; replace <dataset-path> with your dataset path or ref",
    "reward_fn_path is not set; provide --reward-fn-path before training"
  ]
}
```

## Limitations

- The script does not validate that model or dataset paths actually exist —
  replace placeholders and run `inspect_dataset.py` before training.
- GPU-memory-aware defaults (tp_size, mini_bs reductions) are heuristics for
  <=2 GPU setups; always run `check_capacity.py` to confirm feasibility.
- The generated command uses real AReno CLI flag names but does not execute
  `areno train --help` to validate against the current CLI surface.
