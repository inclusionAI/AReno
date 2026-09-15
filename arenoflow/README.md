# ARenoflow

A local, visual workspace for training and serving with AReno on Modal.
Build an SFT → GSPO/GRPO workflow, inspect the exact commands, follow real training
metrics, and deploy a checkpoint behind an authenticated endpoint.

The React interface uses AReno's original repository logo and a parallax home
page, with reduced-motion support. The control plane runs without CUDA; training
uses the published AReno container on your Modal GPUs.

## Run locally

From the **AReno repository root**, with Python 3.10+ and Node.js 22+:

```bash
python3 -m venv arenoflow/.venv
source arenoflow/.venv/bin/activate
pip install -r arenoflow/requirements.txt
npm --prefix arenoflow/web ci
npm --prefix arenoflow/web run build
python -m arenoflow.server
```

Open **http://127.0.0.1:8787**. In Settings, enter your Modal **Token ID** and
**Token Secret**. Alternatively, export `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
before starting the server, then choose the environment connection in Settings.
Credentials stay in server memory; they are not written to SQLite or browser storage.

With Docker Compose:

```bash
docker compose -f arenoflow/compose.yaml up --build
```

The compose service publishes only to localhost and persists local history in a
named volume. This is a single-user local application, not a multi-tenant service.

## Build a training flow

1. Choose a model family and checkpoint. Registered adapters, documented checkpoint
   suggestions, algorithms, and CLI options are read from this checkout automatically.
   Checkpoint suggestions are examples, not a compatibility guarantee for every GPU.
2. Add a dataset in Dataset Manager, then select it by name in each training stage.
   Choose an algorithm and edit its recommended preset. Uploads remain local until
   a referencing job is launched. Model checkpoints default to Hugging Face and
   fill automatically when you choose a model family.
3. Add and reorder stages. Each later stage consumes the preceding stage's final
   checkpoint. Configure each stage's dataset and objective separately.
4. Choose compute and a lifetime limit. Review the generated commands, then launch.
5. Follow stage status, actual TensorBoard scalar metrics, logs and checkpoints.
   Metrics are separated by stage so independent step counters never mix.

### Parameters follow the algorithm

| Algorithm | Objective-specific controls |
| --- | --- |
| SFT | Supervised training; no reward function, rollout sampling, or critic |
| DPO | Preference training, reference settings and DPO beta; no reward function |
| GSPO | Reward, rollout sampling and GSPO clipping |
| GRPO | Reward, rollout sampling and GRPO clipping |
| PPO | Reward, rollout, actor/critic/reference settings and PPO controls |

The full CLI surface is available across the algorithm-specific forms. Applicability
is derived from AReno's config construction, with a small semantic policy for
loss-specific options. The API also rejects incompatible parameters. Switching
algorithms resets objective parameters to that algorithm's preset while retaining
the dataset. Presets are starting points; adjust batch sizes and parallelism to your
checkpoint and available memory.

The initial catalog contains 90 train options, 18 serve options and 12 visible
adapters (Bailing Linear V2 is excluded from the website). These counts are discovered, not UI constants. `ckpt`, output paths and
serving host/port participate in orchestration; reviewed commands show their resolved
values. Dataset and function selectors expose names, with runtime file locations managed internally.

## Dataset Manager and Script Manager

**Script Manager** stores complete Python modules by name and type:

- **Dataset Loader:** `load_training_dataset(dataset_path, *, default_loader, **kwargs)`.
  AReno also passes `load_dataset` and `load_from_disk` keyword helpers. Return records
  in the format expected by the selected algorithm.
- **Reward Function:** `reward_fn(record)` returns a numeric score for one completion.
- **Agentic Function:** `run_agent(ctx, batch)` implements your agent rollout;
  asynchronous functions are supported. The initial template requires implementation.

Write code in the Python editor with syntax highlighting, line numbers, indentation and undo/redo, or import a Python file. The editor loads only when needed. Saving checks syntax and entrypoint
signatures using AST inspection; it never imports or executes user code locally.
Runtime dependencies must exist in the selected AReno image. Functions selected for
training are staged as immutable, content-addressed snapshots, so later edits do not
change an already submitted run. Functions assigned to datasets cannot be deleted
until those datasets select another loader.

**Dataset Manager** defaults to local upload: drag a dataset onto the upload area or choose **Browse local files**. Switch to **Dataset repository** to use remote data. It stores repository references or uploaded data files, their
modalities, media attachments, and a Dataset Loader selection. Function bodies belong
exclusively in Script Manager. The training form selects datasets and, for rollout
algorithms, reward and agent scripts by name. SFT and DPO omit rollout hooks.

Text, image, audio and video datasets are supported as data inputs. For local media,
upload a JSON, JSONL, CSV or TSV manifest and attach the referenced media files.
Use attachment filenames in sample values, including nested messages and lists:

```json
{"prompt":"Describe this scene", "images":["scene.png"], "audio":"speech.wav", "video":"clip.mp4"}
```

Before launch, the manager resolves those filenames to the uploaded media. Your
Dataset Loader is responsible for converting the sample schema into the selected
AReno model/algorithm's input format. Media is staged without transcoding or decoding.
Each file can be up to **16 MiB**, with **128 attachments** per dataset. For larger
media collections or packaged Parquet/Arrow multimodal data, use a dataset repository.
Choosing a modality does not make a text-only model multimodal; use an appropriate
AReno adapter and loader. CPU tests verify reference resolution, not GPU decoding.


## Runtime and deployment

Training and deployment forms automatically fetch published GHCR tags on opening
and every **60 seconds**, selecting and displaying the highest stable semantic
version. Editing the image pins your selection; enable **Follow latest published
tag** to resume automatic updates. A newly selected tag invalidates the command
review. Discovery failures are visible and retried; no version is hardcoded.

Each launch resolves the selected tag to its current immutable
registry digest and starts a Modal Sandbox using that image. If an API caller uses
`ghcr.io/inclusionai/areno:latest` and the registry has
no `latest` alias, it selects the highest published stable semantic-version tag.
Explicit tags never fall back to another version. AReno runs through
its own CLI and public Trainer API. The remote adapter forwards scalar writes and
saves a final successful checkpoint before the Trainer closes.

Artifacts and model caches persist on the `arenoflow-artifacts` Modal Volume.
Stages run sequentially in the same sandbox; failures prevent dependent stages
from starting. A single-stage LoRA run can be deployed with its base model.
Multi-stage adapter-only chaining is currently rejected because it needs an explicit
merge/base-model contract.

Use **Deploy checkpoint**, or create a deployment from an existing model. Set an
endpoint API key of at least 24 characters and retain it yourself. A Modal Secret
passes the key to an authenticated gateway; the underlying AReno server binds to
loopback. The UI exposes the HTTPS `/v1` URL after its health check passes.
Deployments reserve GPUs until stopped or their configured lifetime expires;
they do not scale to zero.

Local server shutdown does **not** stop remote jobs. Restart the server and reconnect
the same Modal workspace to resume monitoring. Stop runs in the UI or Modal console.
If the server exits during submission before saving the sandbox ID, reconcile that
submission in the Modal console. Local structured events are deduplicated on reconnect;
plain stdout may replay. The local event tail is bounded to 20,000 events per job.

The latest published image can lag behind this checkout. The recorded source revision
identifies the UI catalog, while the resolved image digest identifies the runtime.
An image without the selected CLI capabilities will fail visibly in logs; publish a
matching AReno image or choose a compatible checkout/image pair.

## Actual usage and billing

Billing polls Modal every **15 seconds** using the official Python SDK's workspace
billing summary and hourly report. It shows actual metered cost, billed cost,
adjustments, resource breakdown and reported object usage. Actual billing figures are never replaced with estimates or invented pending charges.

This is live polling of reported usage, not a guaranteed instantaneous meter.
Modal can report usage after a delay; the current partial hourly interval may be
absent from reports. Pending usage is labeled explicitly. Figures cover the **entire
connected workspace**, including non-ARenoflow jobs. Access to granular reports can
depend on plan/permissions; an unavailable report displays its error instead of zero.

See [Modal billing](https://modal.com/docs/guide/billing) and the
[Workspace API](https://modal.com/docs/reference/modal.Workspace).

## Run cost estimates

The workflow builder estimates one run and a user-selected number of runs from
**expected runtime**, requested GPUs, physical CPU cores and memory. Runtime covers
the whole workflow; it is a planning assumption, not a throughput prediction.
Launching still submits one workflow regardless of the planning run count.

Rates are fetched from [Modal's public pricing page](https://modal.com/pricing), using
its Sandbox CPU/memory rates and GPU list rates, with a 15-minute cache. No production
price constants are embedded. Missing rates or a changed page format produce an
unavailable estimate instead of a guessed price. Quotes are saved with submitted runs.

Training Runs and Usage & Billing show planned totals and estimated elapsed compute
for all locally recorded training runs. Deployments are excluded from these training
totals. Older runs without a quote use current list rates and their configured lifetime
as the duration assumption. Unknown prices are flagged and excluded from subtotals.

Estimates are shown separately from Modal API billing. Sandbox CPU/memory can burst
above the reservation, and storage, transfer, image builds, credits, negotiated prices
and other billing adjustments are excluded. The displayed lifetime cost is a scenario,
not a guaranteed spending cap. See [Sandbox pricing semantics](https://modal.com/docs/guide/sandbox-resources).

## Development

Keep the Python server running, then start the React development server:

```bash
npm --prefix arenoflow/web run dev
```

Vite proxies `/api` and `/brand` to port 8787. Production serves the built bundle
from `arenoflow/static/`; generated assets and dependencies are gitignored.

```bash
pip install pytest ruff
python -m pytest arenoflow/tests -q
ruff check arenoflow
ruff format --check arenoflow
npm --prefix arenoflow/web run format:check
npm --prefix arenoflow/web run build
```

CPU tests cover metadata and algorithm boundaries, workflow validation, checkpoint
handoff, cancellation, incremental metric decoding, credential redaction, billing
permissions, dataset uploads and local HTTP security. Provider doubles avoid cloud
charges. Real GPU training, endpoint serving and account billing require a separate
Modal integration run with credentials; CPU tests do not establish GPU compatibility.

### Code map

- `catalog.py`, `algorithm_policy.py`: repository metadata and algorithm applicability.
- `workflows.py`: validated, serializable command plans.
- `provider.py`, `remote.py`: Modal boundary and sandbox execution.
- `controller.py`, `store.py`, `events.py`: lifecycle, persistence and event streaming.
- `billing.py`, `assets.py`, `server.py`: actual usage, uploads and local HTTP API.
- `web/src/`: React pages, shared controls, charts and responsive styles.
- `tests/`: isolated CPU contract and lifecycle tests.

The local API enforces a localhost Host, same-origin requests and a per-process CSRF
header on mutations. Do not expose it directly on a public network. Configuration
and datasets are stored under `arenoflow/.data/` (override with `--data-dir`). The
project follows the parent AReno repository license and contribution conventions.

### Interface languages

Use **EN / 中文** on the homepage or workspace toolbar to switch languages.
The initial language follows the browser preference; the selected language is
saved locally. Switching languages preserves in-progress forms and Python code.
Model identifiers, configuration values, source code, and raw provider logs remain
unchanged. Costs are always denominated in USD, with locale-aware formatting.

UI strings use `web/src/i18n.js`; English source strings are translation keys and
Chinese translations live in `web/src/locales/zh.js`. Translate presentation labels
only, never submitted values or user-authored content. New CLI metadata falls back
to its English source until translated. Run the locale checks with
`node --test arenoflow/web/src/i18n.test.js` from the repository root.

Maximum runtime is configured in whole seconds (`timeout_seconds`), from 1 to
86400 s, with a default of 14400 s. Legacy `timeout_hours` configurations are
converted when read. Expected duration for cost estimates remains in hours.

### LLM-assisted Python scripts

Script Manager stores complete Python modules, including imports, helper functions,
classes, and the required AReno entrypoint (`load_training_dataset`, `reward_fn`, or
`run_agent`). Modules are validated statically and executed only in the training
sandbox. Imported third-party packages must already be available in the AReno image.
The existing `/api/functions` and stored function IDs remain compatible.

Configure an OpenAI-compatible Chat Completions **Base URL**, **Model**, and **API
Key** in Settings. For example, a base URL ending in `/v1` receives requests at
`/v1/chat/completions`. Settings are retained in server memory only; the API never
returns the key. Changing the provider clears the previous key unless a replacement
is supplied. Saving settings does not contact the provider.

In Script Manager, select a dataset and algorithm, inspect the sample, enter the
requirements, and check the script types to generate. One LLM request generates the
complete selected set with consistent dataset fields and interfaces. Reward and agent scripts require a rollout algorithm;
SFT and DPO support dataset loader generation only. Uploaded JSON/JSONL/CSV/TSV data
provides up to three sample records where bounded parsing is possible. Supply a
sample manually for repository datasets, Parquet/Arrow, or oversized records.
Only the displayed sample, prompt, dataset metadata, and repository API references
are sent to the configured LLM; media files and Modal credentials are not sent.
Generation may incur charges from that LLM provider, separate from Modal billing.

Every selected script must be returned and pass Python syntax and entrypoint checks
before the batch is shown. Review and edit each script, then save the complete set
in one action. Batch save validates every script before inserting all records in a
single transaction. Existing scripts are not overwritten. Syntax validation is
not a runtime correctness check. Saved generated scripts retain the dataset and
algorithm used for generation; these are provenance, not restrictions on reuse.
