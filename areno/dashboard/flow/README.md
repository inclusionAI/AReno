# Modal jobs in the AReno dashboard

Areno Flow now runs inside the existing dashboard. There is no separate web app
or server to launch. Start the dashboard with `areno dashboard --start`.

Install the optional Modal SDK and dataset preview support in the dashboard's
Python environment with `pip install -r areno/dashboard/flow/requirements.txt`.
GPU training still runs remotely; local CUDA is not required for the control plane.

- **Settings → Modal credentials** (from the dashboard header or Agent tab): connect using Token ID and Token Secret,
  or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` environment variables. Credentials stay
  in server memory and must be reconnected after a server restart.
- **Launcher → Train / Serve → Run on Modal:** select the model adapter, checkpoint and GPU
  reservation, configure parameters, review the execution plan, then execute.
- **Agent:** ask for a Modal training or serving task. The agent reads the live
  repository catalog and prepares a validated plan. Only the dashboard confirmation
  executes it. Plans expire after 30 minutes and can execute once.
- **Agent → Upload file / URL · Datasets:** upload a dataset file (16 MiB maximum),
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
