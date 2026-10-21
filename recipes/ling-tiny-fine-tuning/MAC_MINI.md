# Fine-Tuning LLMs on Mac mini with AReno

Use a local Jupyter notebook to fine-tune **Ling-3.0-tiny** for structured
payment responses with **AReno's MLX backend**. This walkthrough takes you
from a fresh Python environment to a saved LoRA adapter, with the same
validation prompts evaluated before training and after reloading the adapter.

[Open the Mac notebook](Ling-Tiny-SFT-Mac.ipynb) ·
[AReno](https://github.com/inclusionAI/AReno) ·
[Base model](https://modelscope.cn/models/inclusionAI/Ling-3.0-tiny) ·
[Dataset](https://modelscope.cn/datasets/AsystemAReno/Ling-Tiny-SFT)

The setup-to-notebook structure follows
[Unsloth's DGX Station tutorial](https://unsloth.ai/docs/blog/dgx-station).
Here, computation runs on Apple Silicon through MLX and Metal. AReno manages
training, evaluation, and adapter saving through its public `Trainer` API.

> **Experimental recipe:** Ling attention LoRA is implemented on this branch,
> but end-to-end Mac mini execution remains unverified. There is no established
> minimum unified-memory capacity, runtime, or accuracy improvement for this
> workload. Treat the defaults below as an experiment to validate on your machine.

## What you will build

The model receives payment rules and a request, then returns a direct JSON
answer. The synthetic dataset covers scenarios such as promotions, rounding,
payment-channel recommendations, split payments, and error responses.

You will load the published dataset, measure the base model, train an
attention LoRA adapter, save it, and evaluate it in a new process. LoRA
updates small adapter matrices while keeping the base model frozen. This
recipe uses **unquantized base weights**; their full memory footprint still
matters.

| Setting | Notebook default |
| --- | --- |
| Model | `inclusionAI/Ling-3.0-tiny` |
| Runtime | AReno MLX, one local worker |
| Task | Supervised fine-tuning, response-only loss |
| Data | 1,400 training rows and 100 validation rows |
| Training | One epoch, 350 optimizer steps |
| Batch / microbatch | 4 rows per update / 1 row per microbatch |
| LoRA | Rank 8, alpha 16, selected attention projections |
| Learning rate | Constant `1e-4` |
| Token budgets | 4,096 prompt + 1,024 response; 5,120 total |
| Evaluation | Greedy decoding, one request at a time |
| Output | PEFT-style adapter, metrics, completions, and run metadata |

## Quickstart

### 1. Prepare your Mac mini

Use an **Apple Silicon Mac mini** with a native `arm64` Python interpreter,
Git, and enough free disk space for the base checkpoint, package caches, and
run artifacts. MLX's installation guide lists Python 3.10+ and macOS 14+
as baseline requirements; installed package versions may impose additional
constraints. See the [MLX installation guide](https://ml-explore.github.io/mlx/build/html/install.html).

Check your interpreter before creating the environment:

```bash
python3 -c "import platform, sys; print(platform.machine()); print(sys.version)"
git --version
```

The architecture should be `arm64`. An Intel Mac or a Python process running
under Rosetta is outside this notebook's supported path.

Unified memory is shared with macOS and other applications. Allow headroom
for model weights, activations, optimizer state, and evaluation caches.
The notebook reports stored weight size when the checkpoint index provides
it, but that number alone cannot establish whether training will fit.

### 2. Start JupyterLab from the recipe checkout

Use the AReno checkout containing this guide, `recipe.py`, and
`Ling-Tiny-SFT-Mac.ipynb`. This guide accompanies the
`chore/ling-tiny-sft-recipe` development branch; availability of that branch
on the public remote has not been verified. A checkout containing only an
older release may not include this recipe or its Ling support.

From the repository root, create a dedicated environment:

```bash
python3 -m venv recipes/ling-tiny-fine-tuning/.venv
source recipes/ling-tiny-fine-tuning/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel
python -m ipykernel install --sys-prefix \
  --name areno-mac --display-name "AReno Mac"
python -m jupyterlab
```

Open the local URL printed by JupyterLab, navigate to
`recipes/ling-tiny-fine-tuning/Ling-Tiny-SFT-Mac.ipynb`, and select the
**AReno Mac** kernel. Keep the terminal running while using the notebook.

### 3. Install AReno in the notebook

Run **Section 1: Installation**. Its first code cell installs the current
AReno checkout and notebook dependencies using the selected kernel's Python.
AReno's platform-specific dependencies select MLX on Apple Silicon; CUDA,
FlashAttention, and the Linux CUDA installation scripts are unnecessary here.

The notebook requests MLX-LM revision
`1840cbba9cb3f8125b8cf7e137606256a2f162d1`, which the repository's Ling
implementation targets. The older MLX-LM 0.31.3 package in the development
environment lacks the required `bailing_moe_v3` module. Keep the notebook's
pin while validating this recipe.

Use a fresh kernel. If you imported MLX or AReno before changing packages,
restart the kernel and rerun from the installation cell.

## Run the fine-tuning tutorial

Run the cells in order for the first experiment. **Run → Run All Cells**
also starts downloads, both validation passes, and the full training epoch,
so review the configuration first.

### Configure the experiment

In **Section 2**, keep `DATA_HUB = "modelscope"` for this walkthrough.
Both the model and dataset have pinned revisions in the notebook. A failed
download stops the workflow; it does not silently switch providers.

Leave `MODEL_PATH = ""` to download the base checkpoint from ModelScope.
To reuse a downloaded checkpoint, set it to your local directory containing
the unquantized Ling weights, tokenizer files, and `config.json`. The recipe
checks for `model_type = "bailing_hybrid"` and rejects quantized checkpoints.

`RUN_DIR` defaults to a new timestamped directory under `runs/`.
Each execution of the configuration cell creates a new directory. Keep that
directory consistent across all subsequent stages of an experiment.

The environment cell verifies Metal availability and the Ling module in a
separate Python process, then writes `hardware.json`. A successful check
establishes runtime availability, not model compatibility or memory capacity.

### Inspect the dataset

**Section 3** downloads the fixed `train.jsonl` and `valid.jsonl` files.
It validates split sizes, checks targets against their ground truth, and
rejects duplicate cases or overlapping requests. No dataset generation or
random re-splitting occurs.

Each row supplies:

| Field | Purpose |
| --- | --- |
| `prompt` | Payment rules and the request given to the model |
| `response` | The supervised JSON answer |
| `ground_truth` | The structured reference used for scoring |
| `case_id`, `bucket` | Case identity and source-scenario grouping |
| `input` | Structured request used in duplicate detection |

Read the displayed prompt and target before proceeding. These are synthetic
examples, and scenario labels are not independent coverage guarantees.

### Load the checkpoint and check token lengths

**Section 4** downloads model metadata and then the weights, or uses
`MODEL_PATH`. AReno applies the checkpoint's chat template with thinking
disabled. Prompt tokens are masked from the SFT loss; response tokens,
including EOS, contribute to training.

Preflight checks every training and validation row against the configured
token budgets. An oversized example stops execution with its case ID and
lengths. No rows are silently dropped or truncated. The length summary is
saved in `preflight.json`.

### Measure the base model

**Section 5** starts a fresh worker and evaluates all 100 validation
prompts. It attaches a newly initialized LoRA adapter with zero initial
effect and performs no optimizer steps.

The scorer requires one complete JSON document. It checks every field,
value, type, and array position. Whitespace and object-key order do not
affect the score. Extra prose, Markdown fences, duplicate keys, missing or
extra fields, and floating-point values in place of integer cents fail
exact match.

The worker saves raw completions to `before.jsonl` and aggregate results to
`before-summary.json`. Its model memory is released when the process exits.

### Train and save the adapter

**Section 6** trains on the 1,400 training rows, shuffling them
deterministically for each epoch. Validation rows never enter training
batches. At the defaults, one epoch produces **350 optimizer steps**.

The implementation in [recipe.py](recipe.py) uses:

```python
metrics = trainer.train(batch, sft_loss_fn, mini_bs=1)
```

AReno updates adapters on selected KDA and MLA attention projections.
Base weights, including experts and routers, remain frozen. The MLX
configuration enables gradient checkpointing, disables compiled training
steps, and limits evaluation concurrency to one request.

Training writes SDK metrics to `training.jsonl` and TensorBoard events
under `events/`. After the last batch, the worker calls
`trainer.save_checkpoint(...)` and checks for both adapter files:

```text
adapter/
  adapter_config.json
  adapter_model.safetensors
```

The notebook displays completed steps and first/last loss. A lower training
loss alone does not establish better validation accuracy.

### Reload and compare

**Section 7** starts another worker, loads the same base checkpoint plus
the saved adapter through `LoraConfig(adapter_path=...)`, and repeats the
same greedy validation evaluation. This exercises the saved artifact.

Read the comparison table alongside individual completions:

| Metric | Interpretation |
| --- | --- |
| `rows` | Should be 100 for each completed validation pass |
| `valid_json` | Number of completions that parse as strict JSON |
| `exact_match` | Number matching the entire typed reference answer |
| `exact_match_rate` | `exact_match / rows` |
| `at_token_limit` | Completions reaching the generation budget; inspect for truncation |

A syntactically valid JSON response can still be wrong. The notebook also
shows per-bucket counts and examples of remaining failures. The reported
percentage-point change is `100 × (after rate − before rate)`; a negative
value is a valid experimental result.

These are validation results on the same 100 prompts used for comparison.
After repeated tuning, use a separate held-out test set for a final quality
claim. Successful reload is an execution check; it does not establish
numerical equivalence or portability to another runtime.

## Keep and reuse your results

**Section 8** prints the run and adapter paths. Save the notebook in
JupyterLab to retain its outputs. Each run directory contains:

| Artifact | Contents |
| --- | --- |
| `config.json` | Model/data references, settings, AReno commit and dirty state |
| `environment.txt` | Installed Python package versions |
| `hardware.json` | Mac model, chip, memory, Python, and MLX runtime information |
| `preflight.json` | Maximum token lengths for both splits |
| `stage-times.json` | Wall time for each successfully completed stage |
| `before.log`, `train.log`, `after.log` | Worker output for diagnosis |
| `training.jsonl`, `events/` | Training metrics and TensorBoard events |
| `before.jsonl`, `after.jsonl` | Generated answers, references, and scores |
| `before-summary.json`, `after-summary.json` | Aggregate and per-bucket scores |
| `adapter/` | Final adapter configuration and weights |

Retain the base checkpoint reference with the adapter. The adapter directory
does not include base weights or optimizer state. Restarting training from
an adapter is not an exact continuation of an interrupted optimizer run.
No model upload, GGUF conversion, or merged-model export is performed.

To repeat only the saved adapter's validation, use the activated environment
from the repository root. Replace `ling-tiny-YOUR-RUN` with the actual run
directory name:

```bash
python recipes/ling-tiny-fine-tuning/recipe.py after \
  runs/ling-tiny-YOUR-RUN/config.json
```

The worker reads the base model, dataset, and adapter location from that
configuration. It overwrites `after.jsonl` and `after-summary.json` in the
configured run directory. Archive a copy first if you need to retain the
original results. This direct invocation prints to the terminal and does
not update the notebook wrapper's stage log or timing file. Stored paths
are absolute: moving a run to another location or machine requires updating
`run_dir`, `model_path`, `train_file`, and `valid_file` in its configuration.

When reporting an experiment, include the chip and unified-memory size,
source commit and local modifications, package versions, model/data
revisions, completed steps, validation counts, and reload outcome.
`stage-times.json` includes worker setup, model loading, and output writing;
it is not a measurement of pure training throughput. This notebook does
not measure peak memory or swap.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Python reports `x86_64` | Create the environment with a native Apple Silicon interpreter and select its kernel. |
| Metal is unavailable | Check the macOS and MLX installation requirements and the kernel's Python path. |
| Ling module is missing | Rerun the installation cell with its pinned MLX-LM revision in the selected environment. |
| Model or dataset download fails | Check access to the specified ModelScope asset and revision. For the model, an explicit local checkpoint is also supported. |
| Token preflight fails | Inspect the case and reported lengths. Increase the relevant budget and total context explicitly only if memory permits. |
| Out of memory or heavy swap | Interrupt the cell, close other memory-heavy apps, and inspect memory pressure. The recipe already uses microbatch 1 and evaluation concurrency 1; lowering `BATCH_SIZE` alone may not reduce the peak enough. |
| Worker fails or training is interrupted | Read the corresponding stage log. Only the final adapter is saved, so partial metrics do not imply a usable completed checkpoint. Start a fresh run for a clean retry. |
| JSON validity rises but exact match does not | Inspect incorrect fields, numeric types, error messages, and array order in `after.jsonl`. |

Change one setting at a time in subsequent experiments. Keep the dataset
revisions, validation examples, and decoding settings fixed for comparisons.
Changing token budgets changes the experiment and should be recorded as such.
The dataset and base model have separate license terms; consult their model
and dataset cards before redistribution.
