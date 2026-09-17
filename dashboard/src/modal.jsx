import React, { useEffect, useRef, useState } from "react";
import { Database, Plus, Upload } from "lucide-react";
import { modalWorkflow } from "./modal-workflow";

export function ModalUsage({ job, detail = false }) {
  if (job.provider !== "modal") return detail ? null : "—";
  const usage = job.usage || {};
  const cost = usage.accrued_cost;
  return <div className="modalUsage" title={usage.error || "Compute estimate from reserved resources and elapsed sandbox time; excludes storage, network and billing adjustments."}>
    <strong>{cost == null ? "Unavailable" : `$${Number(cost).toFixed(4)}`}</strong>
    <small>Estimated usage so far</small>
    {detail && <><span>{Math.round(usage.seconds || 0)} seconds · {job.modal?.resources?.count} × {job.modal?.resources?.gpu}</span><p>Compute estimate, not an invoice. Storage, networking and billing adjustments are excluded.</p>{usage.error && <p role="status">{usage.error}</p>}{job.modal?.endpoint && <p>Endpoint: <a href={job.modal.endpoint} target="_blank" rel="noreferrer">{job.modal.endpoint}</a></p>}</>}
  </div>;
}

export function ModalSettings({ request }) {
  const [status, setStatus] = useState(null);
  const [tokenId, setTokenId] = useState("");
  const [tokenSecret, setTokenSecret] = useState("");
  const [remember, setRemember] = useState(true);
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState("");
  useEffect(() => {
    let active = true;
    const refresh = () => request("/bootstrap").then(value => { if (active) setStatus(value); }).catch(error => { if (active) setMessage(error.message); });
    refresh(); const timer = setInterval(refresh, 2000);
    return () => { active = false; clearInterval(timer); };
  }, []);
  async function connect(event) {
    event.preventDefault(); setPending(true); setMessage("");
    try {
      await request("/connect", { token_id: tokenId, token_secret: tokenSecret, remember });
      setTokenId(""); setTokenSecret("");
      setStatus(await request("/bootstrap")); setMessage("Modal connected.");
    } catch (error) { setMessage(error.message); } finally { setPending(false); }
  }
  return <section className="modalSettings"><h3>Modal credentials</h3>
    <p>{status?.connected ? "Connected" : status?.reconnecting ? "Reconnecting to Modal…" : "Connect to launch Modal jobs"}. {status?.credentials_saved ? "Credentials saved on this server. Automatic reconnect is enabled." : "Credentials are never stored in browser storage or chat."}</p>
    {status?.connection_error && <p role="status">{status.connection_error}</p>}
    <form onSubmit={connect} className="launcherSections">
      <div className="formGrid"><label className="field"><span>Modal Token ID</span><input type="password" autoComplete="off" value={tokenId} onChange={e => setTokenId(e.target.value)} /></label>
      <label className="field"><span>Modal Token Secret</span><input type="password" autoComplete="new-password" value={tokenSecret} onChange={e => setTokenSecret(e.target.value)} /></label></div>
      <label className="modalExecutionSwitch"><input type="checkbox" checked={remember} onChange={event => setRemember(event.target.checked)} /><span>Remember credentials on this server and reconnect automatically</span></label>
      <div className="detailActions"><button className="primaryButton" disabled={pending || !(tokenId && tokenSecret || (!tokenId && !tokenSecret && (status?.environment_credentials || status?.credentials_saved)))}>{pending ? "Connecting…" : tokenId || tokenSecret ? "Connect Modal" : status?.credentials_saved ? "Reconnect saved credentials" : status?.environment_credentials ? "Use environment credentials" : "Connect Modal"}</button>
      {status?.credentials_saved && <button type="button" className="secondaryButton" disabled={pending} onClick={async () => { setPending(true); try { await request("/forget-credentials", {}); setStatus(await request("/bootstrap")); setMessage("Saved credentials removed. The current connection remains active until restart."); } catch (error) { setMessage(error.message); } finally { setPending(false); } }}>Forget saved credentials</button>}</div>
      {message && <p role="status">{message}</p>}
    </form>
  </section>;
}

export function DatasetManager({ request, onSelect }) {
  const [datasets, setDatasets] = useState([]);
  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState("");
  const fileInput = useRef(null);
  async function refresh() { setDatasets(await request("/datasets")); }
  useEffect(() => {
    refresh().catch(error => setMessage(error.message));
    const timer = setInterval(() => refresh().catch(() => {}), 4000);
    return () => clearInterval(timer);
  }, []);
  async function run(operation) {
    setPending(true); setMessage("");
    try { await operation(); await refresh(); } catch (error) { setMessage(error.message); } finally { setPending(false); }
  }
  async function upload(file) {
    if (!file) return;
    await run(async () => {
      if (!file.size || file.size > 16 * 1024 * 1024) throw new Error("Choose a non-empty dataset up to 16 MiB.");
      const content = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(",")[1]);
        reader.onerror = () => reject(new Error("Could not read this file"));
        reader.readAsDataURL(file);
      });
      const asset = await request("/uploads", { name: file.name, content });
      await request("/datasets", { name: name.trim() || file.name, source: asset.path, source_type: "upload", source_name: file.name });
      setName(""); setMessage("Dataset uploaded. Select Use in chat to attach it.");
    });
    if (fileInput.current) fileInput.current.value = "";
  }
  return <div className="launcherSections">
    <p>Manage datasets for Modal training. Upload JSON, JSONL, CSV, TSV, Parquet or Arrow files (up to 16 MiB), or import a dataset from a Hugging Face or ModelScope repository link.</p>
    <label className="field"><span>Dataset name (optional)</span><input value={name} onChange={e => setName(e.target.value)} /></label>
    <input ref={fileInput} type="file" accept=".json,.jsonl,.csv,.tsv,.parquet,.arrow" hidden onChange={e => upload(e.target.files?.[0])} />
    <button className="secondaryButton" disabled={pending} onClick={() => fileInput.current?.click()}><Upload size={16} /> Upload file</button>
    <form className="datasetUrlForm" onSubmit={event => { event.preventDefault(); run(async () => { await request("/datasets/url", { url, name }); setUrl(""); setName(""); setMessage("Dataset added. Repository download is running in the background."); }); }}>
      <label className="field"><span>Hugging Face / ModelScope dataset URL</span><input type="url" required placeholder="https://huggingface.co/datasets/owner/name" value={url} onChange={e => setUrl(e.target.value)} /></label>
      <button className="secondaryButton" disabled={pending || !url.trim()}><Plus size={16} /> Import dataset</button>
    </form>
    <p className="datasetUrlHint">Hugging Face: <code>https://huggingface.co/datasets/owner/name</code><br />ModelScope: <code>https://modelscope.cn/datasets/owner/name</code></p>
    {pending && <p role="status">Importing dataset…</p>}{message && <p role="status">{message}</p>}
    <div className="datasetList">{datasets.length === 0 && <p>No datasets yet.</p>}{datasets.map(dataset => <div className="datasetRow" key={dataset.id}>
      <div><strong>{dataset.name}</strong><small>{dataset.source_type} · {dataset.sample_status || "saved"}</small><code>{dataset.source}</code></div>
      <div className="detailActions"><button className="secondaryButton" disabled={pending} onClick={() => onSelect(dataset)}>Use in chat</button><button className="secondaryButton" disabled={pending} onClick={() => run(() => request(`/datasets/${dataset.id}/delete`, {}))}>Remove</button></div>
    </div>)}</div>
  </div>;
}

export function ModalResourceForm({ mode, config, request, settings, setSettings, onSettings }) {
  const [bootstrap, setBootstrap] = useState(null);
  const [error, setError] = useState("");
  const [quote, setQuote] = useState(null);
  const [quoting, setQuoting] = useState(false);
  const [quoteError, setQuoteError] = useState("");
  const [quoteRetry, setQuoteRetry] = useState(0);
  const [recommendation, setRecommendation] = useState(null);
  const [recommending, setRecommending] = useState(false);
  const modelInputs = JSON.stringify(config);
  useEffect(() => {
    if (!bootstrap) return;
    let active = true;
    setRecommendation(null); setRecommending(true);
    const timer = setTimeout(async () => {
      try {
        const workflow = modalWorkflow(mode, config, bootstrap.catalog, settings);
        const result = await request("/recommend-gpu", workflow);
        if (!active) return;
        setRecommendation(result);
        if (settings.auto_gpu && result.recommended) {
          setSettings(current => ({ ...current, gpu: result.recommended.gpu, count: result.recommended.count }));
        }
      } catch (error) { if (active) setRecommendation({ available: false, reason: error.message }); }
      finally { if (active) setRecommending(false); }
    }, 400);
    return () => { active = false; clearTimeout(timer); };
  }, [bootstrap, mode, modelInputs, settings.parameters_billion, settings.auto_gpu]);
  useEffect(() => { let active = true; request("/bootstrap").then(value => { if (active) setBootstrap(value); }).catch(error => { if (active) setError(error.message); }); return () => { active = false; }; }, []);
  useEffect(() => {
    let active = true;
    setQuote(null); setQuoteError(""); setQuoting(true);
    const timer = setTimeout(async () => {
      try {
        const result = await request("/estimate", { resources: { gpu: settings.gpu, count: settings.count, cpu: settings.cpu, memory_gib: settings.memory_gib, timeout_seconds: Math.round(settings.duration_hours * 3600) }, hours: settings.duration_hours });
        if (active) setQuote(result);
      } catch (error) { if (active) setQuoteError(error.message); }
      finally { if (active) setQuoting(false); }
    }, 400);
    return () => { active = false; clearTimeout(timer); };
  }, [settings.gpu, settings.count, settings.cpu, settings.memory_gib, settings.duration_hours, quoteRetry]);
  return <section className="modalResourceSection">
    <div className="modalResourceHeader"><strong>Modal resources</strong><button type="button" className="secondaryButton" onClick={onSettings}>Modal settings</button></div>
    <p>Container: <code>ghcr.io/inclusionai/areno:latest</code> · fetched from GHCR and pinned when the plan is prepared.</p>
    {error && <p role="alert">{error}</p>}
    {!bootstrap && !error && <p>Loading Modal catalog…</p>}
    {bootstrap && <>
      <label className="field"><span><input type="checkbox" checked={!!settings.auto_gpu} onChange={e => setSettings(current => ({ ...current, auto_gpu: e.target.checked }))} /> Automatically select GPU from model and task settings</span></label>
      <label className="field"><span>Total model parameters (billions, optional override)</span><input type="number" min="0.001" step="any" value={settings.parameters_billion || ""} placeholder="Infer from model name" onChange={e => setSettings(current => ({ ...current, parameters_billion: e.target.value }))} /></label>
      {recommending ? <p>Estimating GPU memory…</p> : <GpuRecommendation value={recommendation} />}
      <div className="formGrid">
        <label className="field"><span>Model adapter</span><select value={settings.adapter} onChange={e => setSettings(current => ({ ...current, adapter: e.target.value }))}><option value="">Select adapter</option>{bootstrap.catalog.models.map(item => <option key={item.id} value={item.id}>{item.id}</option>)}</select></label>
        <label className="field"><span>GPU type</span><select value={settings.gpu} onChange={e => setSettings(current => ({ ...current, gpu: e.target.value, auto_gpu: false }))}>{bootstrap.gpu_types.map(gpu => <option key={gpu}>{gpu}</option>)}</select></label>
        {[["count", "GPU count", 1, 8, 1], ["duration_hours", "Duration / maximum runtime (hours)", 0.01, 24, 0.01], ["cpu", "CPU cores", 1, 64, 1], ["memory_gib", "Memory (GiB)", 4, 512, 1]].map(([key, label, min, max, step]) => <label className="field" key={key}><span>{label}</span><input type="number" min={min} max={max} step={step} value={settings[key]} onChange={e => setSettings(current => ({ ...current, [key]: Number(e.target.value), ...(key === "count" ? { auto_gpu: false } : {}) }))} /></label>)}
      </div>
      <div className="modalFee" role="status"><span>Estimated fee</span><strong>{settings.auto_gpu && recommending ? "Sizing GPU…" : settings.auto_gpu && !recommendation?.available ? "Model sizing required" : quoting ? "Calculating…" : quote ? `$${Number(quote.planned_cost).toFixed(2)} USD` : "Unavailable"}</strong>
        {quote && <small>${Number(quote.hourly_cost).toFixed(2)} / hour × {settings.duration_hours} hours · GPU + CPU + memory</small>}
        {quoteError && <small>{quoteError}</small>}
        {quote?.rates?.cached && <small>Using cached public rates verified {new Date(quote.rates.fetched_at * 1000).toLocaleDateString()}.</small>}
        <button type="button" className="secondaryButton" disabled={quoting} onClick={() => setQuoteRetry(value => value + 1)}>Refresh estimate</button>
        <small>Compute estimate at public list rates for the full duration. Actual usage may differ; excludes storage and networking.</small>
      </div>
      <p>Uses the training or serving configuration above. Relative output paths are stored under /artifacts. Modal currently requires the Hugging Face model hub; local input files must be uploaded first.</p>
    </>}
  </section>;
}

function GpuRecommendation({ value }) {
  if (!value) return null;
  return <div className="modalFee" role="status">
    <span>GPU memory estimate</span>
    {value.required_gib_per_gpu != null && <strong>{value.required_gib_per_gpu.toFixed(1)} GiB per GPU including headroom</strong>}
    {value.recommended && <small>Recommended: {value.recommended.count} × {value.recommended.gpu} · lowest GPU list cost among estimated fits</small>}
    {value.reason && <small>{value.reason}</small>}
    {value.parameter_source && <small>{value.parameters_billion}B parameters · {value.parameter_source}</small>}
    {value.stages?.map((stage, i) => <small key={i}>{stage.algo}: {stage.optimizer.replaceAll("_", " ")}</small>)}
    {value.assumptions && <details><summary>Memory assumptions</summary>{value.assumptions.map(item => <p key={item}>{item}</p>)}{value.stages?.map((stage, index) => <p key={index}>{Object.entries(stage.components_gib).map(([name, gib]) => `${name.replaceAll("_", " ")}: ${gib.toFixed(1)} GiB`).join(" · ")}</p>)}</details>}
  </div>;
}

export function PlanParameters({ title, value = {}, onChange }) {
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  return <fieldset className="planParameterGroup"><legend>{title}</legend>
    {Object.entries(value).map(([key, item]) => <div className="planParameterRow" key={key}>
      <label className="field"><span>{key}</span><input value={typeof item === "string" ? item : JSON.stringify(item)} onChange={event => onChange({ ...value, [key]: event.target.value })} /></label>
      <button type="button" className="secondaryButton" aria-label={`Delete ${title} ${key}`} onClick={() => { const next = { ...value }; delete next[key]; onChange(next); }}>Delete</button>
    </div>)}
    <div className="planParameterAdd"><label className="field"><span>Parameter name</span><input value={newKey} onChange={event => setNewKey(event.target.value)} /></label><label className="field"><span>Value</span><input value={newValue} onChange={event => setNewValue(event.target.value)} /></label>
      <button type="button" className="secondaryButton" disabled={!newKey.trim() || Object.hasOwn(value, newKey.trim()) || ["__proto__", "constructor", "prototype"].includes(newKey.trim())} onClick={() => { onChange({ ...value, [newKey.trim()]: newValue }); setNewKey(""); setNewValue(""); }}>Add parameter</button>
    </div>
  </fieldset>;
}

export function ModalPlanCard({ plan, request, onConfirm, onUpdate }) {
  const [current, setCurrent] = useState(plan);
  const [draft, setDraft] = useState(plan.workflow || {});
  const [editing, setEditing] = useState(false);
  const [pending, setPending] = useState("");
  const [message, setMessage] = useState("");
  const [executed, setExecuted] = useState(plan.status === "started");
  const [estimating, setEstimating] = useState(false);
  async function refreshEstimate() {
    if (!current.resources) return;
    setEstimating(true);
    try {
      const estimate = await request("/estimate", { resources: current.resources, hours: current.resources.timeout_seconds / 3600 });
      setCurrent(value => ({ ...value, estimate, estimate_error: null }));
    } catch (error) { setCurrent(value => ({ ...value, estimate_error: error.message })); }
    finally { setEstimating(false); }
  }
  useEffect(() => { if (!plan.estimate && plan.resources) refreshEstimate(); }, [plan.id]);
  useEffect(() => { setCurrent(plan); setDraft(plan.workflow || {}); setExecuted(plan.status === "started"); }, [plan]);
  async function save() {
    setPending("Saving…"); setMessage("");
    try {
      const result = await request("/revise", { plan_id: current.id, workflow: draft });
      setCurrent(result.plan); setDraft(result.plan.workflow); onUpdate?.(result.plan); setEditing(false);
    } catch (error) { setMessage(error.message); } finally { setPending(""); }
  }
  async function execute() {
    setPending("Starting…"); setMessage("");
    try {
      const result = await onConfirm(current);
      if (result?.ok === false) throw new Error(result.error || "Execution failed");
      const started = { ...current, status: "started" };
      setCurrent(started); onUpdate?.(started); setExecuted(true); setMessage(`Started job ${result.job.id}`);
    } catch (error) { setMessage(error.message); } finally { setPending(""); }
  }
  const estimate = current.estimate;
  return <section className="agentPlanCard">
    <div className="agentPlanHeader"><div><span>Modal execution plan</span><strong>{current.objective}</strong></div><span>{executed ? "Started" : "Proposed"}</span></div>
    <p>{current.summary}</p>
    {!editing && <GpuRecommendation value={current.gpu_recommendation} />}
    <div className="modalFee"><span>Estimated fee</span><strong>{editing ? "Save changes to refresh" : estimating ? "Calculating…" : estimate ? `$${Number(estimate.planned_cost).toFixed(2)} USD` : "Unavailable"}</strong>
      {!editing && estimate && <small>${Number(estimate.hourly_cost).toFixed(2)} / hour · {(current.resources.timeout_seconds / 3600).toFixed(2)} hours</small>}
      {!editing && current.estimate_error && <small>{current.estimate_error}</small>}
      {!editing && estimate?.rates?.cached && <small>Cached public rates · verified {new Date(estimate.rates.fetched_at * 1000).toLocaleDateString()}</small>}
      {!editing && <button className="secondaryButton" disabled={estimating || !!pending} onClick={refreshEstimate}>Refresh estimate</button>}
      <small>Compute estimate for the full duration; excludes storage and networking.</small>
    </div>
    {editing ? <div className="planEditor">
      <label className="field"><span>Job name</span><input value={draft.name || ""} onChange={e => setDraft({ ...draft, name: e.target.value })} /></label>
      <PlanParameters title="Model" value={draft.model} onChange={model => setDraft({ ...draft, model })} />
      <PlanParameters title="Resources" value={draft.resources} onChange={resources => setDraft({ ...draft, resources })} />
      {(draft.stages || []).map((stage, index) => <div key={index}><label className="field"><span>Stage {index + 1} algorithm</span><input value={stage.algo} onChange={e => setDraft({ ...draft, stages: draft.stages.map((item, i) => i === index ? { ...item, algo: e.target.value } : item) })} /></label><PlanParameters title={`Stage ${index + 1} parameters`} value={stage.params} onChange={params => setDraft({ ...draft, stages: draft.stages.map((item, i) => i === index ? { ...item, params } : item) })} /></div>)}
      {draft.kind === "deployment" && <PlanParameters title="Serving parameters" value={draft.serve} onChange={serve => setDraft({ ...draft, serve })} />}
      <p>Use AReno parameter names. Values are validated when you save; numbers, decimals and true/false are supported. Deleting an optional parameter restores its runtime default.</p>
      <div className="agentPlanActions"><button className="primaryButton" disabled={!!pending} onClick={save}>Save changes &amp; estimate</button><button className="secondaryButton" disabled={!!pending} onClick={() => { setDraft(current.workflow); setEditing(false); setMessage(""); }}>Cancel edits</button></div>
    </div> : <>
      <div className="agentPlanParams">{Object.entries(current.resources || {}).map(([key, value]) => <label key={key}><span>{key.replaceAll("_", " ")}</span><strong>{String(value)}</strong></label>)}</div>
      <p>Image: <code className="planImage">{current.image || "ghcr.io/inclusionai/areno:latest"}</code></p>
      <pre className="agentPlanCommand">{current.command}</pre>
      <div className="agentPlanActions"><button className="primaryButton" disabled={!!pending || executed} onClick={execute}>{executed ? "Started" : pending || "Confirm execution"}</button><button className="secondaryButton" disabled={!!pending || executed || !current.workflow} onClick={() => setEditing(true)}>Edit parameters</button></div>
    </>}
    {message && <p role="status">{message}</p>}
  </section>;
}
