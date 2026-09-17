# Modal jobs in the AReno dashboard

Areno Flow now runs inside the existing dashboard. There is no separate web app
or server to launch. Start the dashboard with `areno dashboard --start`.

Install the optional Modal SDK and dataset preview support in the dashboard's
Python environment with `pip install -r areno/dashboard/flow/requirements.txt`.
GPU training still runs remotely; local CUDA is not required for the control plane.

- **Settings → Modal credentials** (from the dashboard header or Agent tab): connect using Token ID and Token Secret,
  or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment variables. By default,
  validated credentials are remembered in an owner-only (0600) server-side file
  inside the private data directory (0700). They never reach browser storage,
  chat history or job records. Startup automatically reconnects and retries
  transient connection failures. Uncheck Remember for a session-only connection,
  or use Forget saved credentials to remove the saved copy.
- **Launcher → Train / Serve → Run on Modal:** select the model adapter, checkpoint and GPU
  reservation and maximum duration in the extra Modal section. The existing train /
  serve form, presets, advanced settings and preflight stay visible. Estimated
  GPU + CPU + memory fees update as resources or duration change.
- Plans appear inline in Agent chat (including plans prepared from Launcher).
  Edit, add or delete parameters there; saving validates the changes and refreshes
  the fee estimate before execution. Required fields cannot be removed.
- Training defaults to Flash attention and Adam 4-bit, with editable optimizer
  controls. Serving defaults to Flash attention. Plans fetch
  `ghcr.io/inclusionai/areno:latest` and pin its current digest; if no latest tag is
  published, registry resolution falls back to the highest stable version. The
  reviewed digest is retained through execution. No source-package install is used.
- **Agent:** ask for a Modal training or serving task. The agent reads the live
  repository catalog and prepares a validated plan. Only the dashboard confirmation
  executes it. Plans persist across dashboard restarts and can execute once.
  After 30 minutes, edit and save the plan to revalidate it before execution.
- **Agent → + attachment button:** upload a dataset file (16 MiB maximum),
  import a Hugging Face / ModelScope dataset repository URL,
  remove a managed dataset, or attach its ID to chat. Repository downloads run in
  the background; wait for completion before preparing a plan. Removing a dataset
  removes its library entry, not files snapshotted by existing plans/jobs.
- **Jobs:** Modal jobs share the existing metrics and logs views. Usage so far is a
  compute-only estimate based on public rates and elapsed sandbox time, not an
  invoice. Missing rates are shown as unavailable. Workspace billing remains
  available via `/api/modal/billing` and is not attributed to individual jobs.
- **Serve:** the dashboard generates an endpoint API key and shows it once after
  execution. Save it before dismissing the key dialog; it is not persisted locally.

Job records, managed datasets, uploaded files and durable scalar history live in
`.areno-modal/` by default (`ARENOFLOW_DATA_DIR` overrides this location). When
migrating an existing standalone installation, move `arenoflow/.data` to
`.areno-modal` before starting the dashboard. Existing Modal app/Volume names stay
unchanged so running sandboxes and artifacts can be reattached.

All structured scalar tags from Modal are retained independently of the bounded
stdout tail. Multi-stage workflows prefix tags with their stage to distinguish
steps that reset. Charts wait for each poll to finish, preserve the last successful
snapshot on fetch errors, and render at most 1,200 points using per-bucket extrema.
The metrics API still returns the complete stored series by default.

Validation:

```sh
python -m pytest tests/test_dashboard_modal_cpu.py tests/test_dashboard_metrics_path_cpu.py tests/dashboard_flow -q
node --test dashboard/src/metrics.test.js
pnpm --dir dashboard build
```

These checks do not allocate a GPU or submit a live Modal job.

Pricing refresh failures use the last verified public-rate cache (or the bundled
verified snapshot) with a visible cached-rate date. Refresh estimate retries the
quote without clearing the plan. After updating dashboard Python code, restart
the dashboard backend; refreshing the browser only reloads the frontend.


Modal GPU recommendations are shared by the launcher and the agent's
`recommend_modal_gpu` tool. Automatic selection uses the lowest GPU list price
among estimated fits at the configured TP/world size; GPU count alone never
implies model sharding. Full training defaults to Adam 4-bit, including compact
master metadata, packed first moments, and factored variance. Memory allowances
include BF16 weights/gradients, activation/KV estimates, and 20% headroom.
Parameter counts inferred from model names are explicitly approximate; custom
models can supply `model.parameters_billion` (all experts for MoE). These are
planning heuristics, not measured GPU probes. Manual GPU selection remains
available. Auto-selected resources are recalculated when editing a plan.


Modal image setup installs `tilelang` on top of the selected GHCR image before
starting the sandbox. This supplies the FLA gated chunk backward workaround for
Hopper images with Triton >=3.4.0 and <3.7.1. The layer applies to new sandboxes;
existing jobs retain the image they started with.
