import React, { useEffect, useRef, useState } from "react";
import { Database, Plus, Upload } from "lucide-react";

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
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState("");
  useEffect(() => { request("/bootstrap").then(setStatus).catch(error => setMessage(error.message)); }, []);
  async function connect(event) {
    event.preventDefault(); setPending(true); setMessage("");
    try {
      await request("/connect", { token_id: tokenId, token_secret: tokenSecret });
      setTokenId(""); setTokenSecret("");
      setStatus(await request("/bootstrap")); setMessage("Modal connected.");
    } catch (error) { setMessage(error.message); } finally { setPending(false); }
  }
  return <section className="modalSettings"><h3>Modal credentials</h3>
    <p>{status?.connected ? "Connected" : "Connect to launch Modal jobs"}. Tokens stay in server memory for this session.</p>
    <form onSubmit={connect} className="launcherSections">
      <div className="formGrid"><label className="field"><span>Modal Token ID</span><input type="password" autoComplete="off" value={tokenId} onChange={e => setTokenId(e.target.value)} /></label>
      <label className="field"><span>Modal Token Secret</span><input type="password" autoComplete="new-password" value={tokenSecret} onChange={e => setTokenSecret(e.target.value)} /></label></div>
      <button className="primaryButton" disabled={pending || !(tokenId && tokenSecret || (!tokenId && !tokenSecret && status?.environment_credentials))}>{pending ? "Connecting…" : tokenId || tokenSecret ? "Connect Modal" : status?.environment_credentials ? "Use environment credentials" : "Connect Modal"}</button>
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

export function ModalLauncher({ mode, controls, request, onPlan, onSettings, onDatasets }) {
  const [bootstrap, setBootstrap] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const kind = mode === "serve" ? "deployment" : "training";
  const [adapter, setAdapter] = useState("");
  const [checkpoint, setCheckpoint] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [datasetPath, setDatasetPath] = useState("");
  const [algo, setAlgo] = useState("sft");
  const [resources, setResources] = useState({ gpu: "H100", count: 1, cpu: 4, memory_gib: 32, timeout_seconds: 14400 });
  const [optionsByKind, setOptionsByKind] = useState({ training: "{}", deployment: "{}" });
  const options = optionsByKind[kind];
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([request("/bootstrap"), request("/datasets")]).then(([boot, data]) => { setBootstrap(boot); setDatasets(data); }).catch(error => setError(error.message));
  }, []);
  async function preview(event) {
    event.preventDefault(); setPending(true); setError("");
    try {
      const params = JSON.parse(options);
      if (!params || Array.isArray(params) || typeof params !== "object") throw new Error("Advanced parameters must be a JSON object.");
      const body = { kind, name: `${kind === "training" ? algo.toUpperCase() : "Serve"} ${checkpoint}`, model: { adapter, checkpoint }, resources,
        ...(kind === "training" ? { stages: [{ algo, ...(datasetId ? { dataset_id: datasetId } : {}), params: { world_size: resources.count, tp_size: resources.count, ...(datasetPath && !datasetId ? { dataset_path: datasetPath } : {}), ...params } }] } : { serve: params }) };
      onPlan((await request("/preview", body)).plan);
    } catch (error) { setError(error.message); } finally { setPending(false); }
  }
  return <section className="panel"><div className="panelHeader"><div><h2>Task Launcher</h2><p>Run AReno {mode === "serve" ? "serving" : "training"} on a reserved Modal GPU.</p></div>{controls}</div>
    <div className="modalResourceHeader"><strong>Modal configuration</strong><button className="secondaryButton" onClick={onSettings}>Modal settings</button></div>
    {!bootstrap && !error && <p>Loading Modal catalog…</p>}
    {bootstrap && <form onSubmit={preview} className="launcherSections">
      {!bootstrap.connected && <p role="status">Connect your workspace in dashboard Settings before executing a plan.</p>}
      <div className="formGrid">
        <label className="field"><span>Model adapter</span><select required value={adapter} onChange={e => setAdapter(e.target.value)}><option value="">Select adapter</option>{bootstrap.catalog.models.map(model => <option key={model.id} value={model.id}>{model.id}</option>)}</select></label>
        <label className="field"><span>Checkpoint / repository</span><input required value={checkpoint} onChange={e => setCheckpoint(e.target.value)} /></label>
        {kind === "training" && <><label className="field"><span>Algorithm</span><select value={algo} onChange={e => setAlgo(e.target.value)}>{bootstrap.catalog.algorithms.map(item => <option key={item.id} value={item.id}>{item.id.toUpperCase()}</option>)}</select></label>
        <label className="field"><span>Managed dataset</span><select value={datasetId} onChange={e => setDatasetId(e.target.value)}><option value="">Enter a dataset path below</option>{datasets.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        {!datasetId && <label className="field"><span>Dataset path / repository</span><input required value={datasetPath} onChange={e => setDatasetPath(e.target.value)} /></label>}</>}
        <label className="field"><span>GPU</span><select value={resources.gpu} onChange={e => setResources({ ...resources, gpu: e.target.value })}>{bootstrap.gpu_types.map(gpu => <option key={gpu}>{gpu}</option>)}</select></label>
        {[["count", "GPU count", 1, 8], ["cpu", "CPU cores", 1, 64], ["memory_gib", "Memory (GiB)", 4, 512], ["timeout_seconds", "Maximum lifetime (seconds)", 1, 86400]].map(([key, label, min, max]) => <label className="field" key={key}><span>{label}</span><input required type="number" min={min} max={max} value={resources[key]} onChange={e => setResources({ ...resources, [key]: Number(e.target.value) })} /></label>)}
      </div>
      <label className="field"><span>Advanced {kind === "training" ? "training" : "serving"} parameters (JSON)</span><textarea className="mono" rows={5} value={options} onChange={e => setOptionsByKind(current => ({ ...current, [kind]: e.target.value }))} /></label>
      <p>Use AReno parameter names for dataset loaders, reward functions, batch sizes and other options. Plans are validated against the repository catalog.</p>
      <div className="detailActions"><button className="primaryButton" disabled={pending}>{pending ? "Preparing…" : "Review execution plan"}</button><button type="button" className="secondaryButton" onClick={onDatasets}><Database size={16} /> Manage datasets</button></div>
    </form>}{error && <p role="alert">{error}</p>}
  </section>;
}
