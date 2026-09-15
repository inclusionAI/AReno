import { t } from '../i18n';
import { useEffect, useRef, useState } from 'react';
import {
  Activity,
  ArrowRight,
  Check,
  Copy,
  GitBranch,
  Layers,
  Radio,
  Square,
  Terminal,
} from 'lucide-react';
import { api, duration, money, time } from '../api';
import { Badge, Button, Empty, External, Notice, PageHeader } from './UI';
import { Chart } from './Chart';
import { RunEstimates } from './Estimates';
const terminal = ['succeeded', 'failed', 'cancelled'];
export function JobList({ jobs, deployments = false }) {
  const list = jobs.filter((j) => (j.kind === 'deployment') === deployments);
  return (
    <>
      <PageHeader
        eyebrow={deployments ? t('MODEL SERVING') : t('TRAINING RUNS')}
        title={deployments ? t('Model deployments') : t('Training runs')}
        action={
          <a className="button primary" href={deployments ? '#deploy' : '#workspace'}>
            {t('New')}
            {deployments ? t('deployment') : t('training flow')} <ArrowRight size={16} />
          </a>
        }
      >
        {t('View task status, saved configurations, and execution history.')}
      </PageHeader>
      {!deployments && <RunEstimates compact />}
      {!list.length ? (
        <Empty
          icon={deployments ? Radio : GitBranch}
          title={deployments ? t('No deployments') : t('No training runs')}
          action={
            <a className="button primary" href={deployments ? '#deploy' : '#workspace'}>
              {deployments ? t('Deploy a model') : t('Create a training flow')}
            </a>
          }
        >
          {deployments
            ? t('Deploy a supported checkpoint as an authenticated, OpenAI-compatible endpoint.')
            : t('Create a training workflow. Submitted tasks will appear here.')}
        </Empty>
      ) : (
        <div className="panel table-scroll">
          <table className="jobs-table">
            <thead>
              <tr>
                <th>{deployments ? t('Deployment') : t('Workflow')}</th>
                <th>{t('Status')}</th>
                <th>{t('Model')}</th>
                <th>{t('Compute')}</th>
                <th>{t('Elapsed')}</th>
                <th>{t('Estimated cost')}</th>
              </tr>
            </thead>
            <tbody>
              {list.map((job) => (
                <tr key={job.id}>
                  <td>
                    <a href={`#run/${job.id}`}>
                      <b>{job.name}</b>
                      <small>{time(job.created_at)}</small>
                    </a>
                  </td>
                  <td>
                    <Badge status={job.status} />
                  </td>
                  <td>
                    <span className="model-cell">{job.manifest.model.checkpoint}</span>
                  </td>
                  <td>
                    {job.resources.count} × {job.resources.gpu}
                  </td>
                  <td>{duration(job.started_at, job.finished_at)}</td>
                  <td>{money(job.estimate?.planned_cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
export function JobDetail({ id, onDeploy, notify }) {
  const [job, setJob] = useState(null),
    [events, setEvents] = useState([]),
    [error, setError] = useState('');
  const [busy, setBusy] = useState(false),
    [tab, setTab] = useState('metrics'),
    [metric, setMetric] = useState('');
  const cursor = useRef(0);
  const [metricStage, setMetricStage] = useState(null);
  useEffect(() => {
    let cancelled = false,
      timer;
    const abort = new AbortController();
    cursor.current = 0;
    setEvents([]);
    setJob(null);
    async function poll() {
      try {
        const [record, more] = await Promise.all([
          api(`/jobs/${id}`, undefined, abort.signal),
          api(`/jobs/${id}/events?after=${cursor.current}`, undefined, abort.signal),
        ]);
        if (cancelled) return;
        setJob(record);
        setError('');
        if (more.length) {
          cursor.current = more.at(-1).cursor;
          setEvents((old) => [...old, ...more].slice(-20000));
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
      if (!cancelled) timer = setTimeout(poll, 2500);
    }
    poll();
    return () => {
      cancelled = true;
      abort.abort();
      clearTimeout(timer);
    };
  }, [id]);
  const selectedStage = metricStage ?? job?.stage ?? 0;
  const metrics = events.filter((e) => e.type === 'metric' && (e.index ?? 0) === selectedStage);
  const tags = [...new Set(metrics.map((e) => e.tag))];
  const selected = tags.includes(metric) ? metric : tags.find((t) => t.includes('loss')) || tags[0];
  const latest = metrics.filter((e) => e.tag === selected).at(-1);
  async function stop() {
    setBusy(true);
    try {
      setJob(await api(`/jobs/${id}/stop`, {}));
      notify(t('Stop requested. Waiting for Modal to confirm termination.'));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  if (!job) return <Notice error={!!error}>{error || t('Loading run…')}</Notice>;
  return (
    <>
      <PageHeader
        eyebrow={t('{p0} / {p1}', {
          p0: job.kind.toUpperCase(),
          p1: job.id,
        })}
        title={job.name}
        action={
          <div className="button-group">
            {job.checkpoint && (
              <Button onClick={() => onDeploy(job)}>
                <Radio size={16} />
                {t('Deploy checkpoint')}
              </Button>
            )}
            {!terminal.includes(job.status) && (
              <Button className="danger" busy={busy} onClick={stop}>
                <Square size={14} />
                {t('Stop')}
                {job.kind === 'deployment' ? t('endpoint') : t('run')}
              </Button>
            )}
          </div>
        }
      >
        <Badge status={job.status} />{' '}
        <span className="detail-model">{job.manifest.model.checkpoint}</span>
      </PageHeader>
      {(error || job.error) && <Notice error>{error || job.error}</Notice>}
      {job.estimate && (
        <Notice>
          {t('Planned compute estimate:')}
          <b>{money(job.estimate.planned_cost)}</b> {t('for')} {job.estimate.hours}
          {t(
            'hours, using Modal public list rates. Actual usage and billing are reported separately.',
          )}
        </Notice>
      )}
      <div className="run-stats">
        <div>
          <span>{t('Compute')}</span>
          <b>
            {job.resources.count} × {job.resources.gpu}
          </b>
        </div>
        <div>
          <span>{t('Elapsed')}</span>
          <b>{duration(job.started_at, job.finished_at)}</b>
        </div>
        <div>
          <span>{t('Started')}</span>
          <b>{time(job.started_at)}</b>
        </div>
        <div>
          <span>{t('Costs')}</span>
          <a href="#billing">{t('Live Modal billing ↗')}</a>
        </div>
      </div>
      {job.kind === 'training' && (
        <div className="panel execution-flow">
          {job.manifest.stages.map((stage, i) => {
            const status =
              events.filter((e) => e.type === 'stage' && e.index === i).at(-1)?.status ||
              (i === job.stage && job.status === 'running' ? 'running' : 'queued');
            return (
              <div className="execution-stage" key={i}>
                <div className={`stage-symbol ${status}`}>
                  {status === 'succeeded' ? <Check size={18} /> : <Layers size={18} />}
                </div>
                <div>
                  <small>
                    {t('STAGE')} {i + 1}
                  </small>
                  <b>{stage.algo.toUpperCase()}</b>
                  <Badge status={status} />
                </div>
                {i < job.manifest.stages.length - 1 && <ArrowRight size={18} />}
              </div>
            );
          })}
        </div>
      )}
      {job.kind === 'deployment' && (
        <section className="panel">
          <div className="panel-heading">
            <Radio size={19} />
            <h2>{t('OpenAI-compatible endpoint')}</h2>
          </div>
          {job.endpoint ? (
            <>
              <code className="endpoint-url">{job.endpoint}</code>
              <Button
                onClick={() =>
                  navigator.clipboard
                    .writeText(job.endpoint)
                    .then(() => notify(t('Endpoint URL copied')))
                    .catch(() => notify(t('Copy the URL above')))
                }
              >
                <Copy size={14} />
                {t('Copy URL')}
              </Button>
              <pre className="code-block">{`curl ${job.endpoint}/chat/completions \\
  -H "Authorization: Bearer $ENDPOINT_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'`}</pre>
            </>
          ) : (
            <p className="muted">
              {t('Waiting for the model to load and its health check to pass.')}
            </p>
          )}
          <p className="muted">
            {t('GPU-backed endpoint · maximum lifetime')}
            {job.resources.timeout_seconds ?? job.resources.timeout_hours * 3600}
            {t('s · stop it here when finished.')}
          </p>
        </section>
      )}
      <div className="tabs" role="tablist" aria-label={t('Run views')}>
        {['metrics', 'logs', 'configuration', 'artifacts'].map((view) => (
          <button key={view} role="tab" aria-selected={tab === view} onClick={() => setTab(view)}>
            {t(view)}
          </button>
        ))}
      </div>
      {tab === 'metrics' && job.manifest.stages.length > 1 && (
        <label className="field">
          <span>{t('Training stage')}</span>
          <select value={selectedStage} onChange={(e) => setMetricStage(Number(e.target.value))}>
            {job.manifest.stages.map((stage, index) => (
              <option key={index} value={index}>
                {t('Stage')}
                {index + 1} · {stage.algo.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
      )}
      {tab === 'metrics' &&
        (metrics.length ? (
          <>
            <div className="metric-selector">
              <select
                aria-label={t('Metric to plot')}
                value={selected}
                onChange={(e) => setMetric(e.target.value)}
              >
                {tags.map((tag) => (
                  <option key={tag}>{tag}</option>
                ))}
              </select>
              <span>
                {metrics.length} {t('real scalar samples')}
              </span>
            </div>
            <section className="panel metric-panel">
              <span className="eyebrow">{selected}</span>
              <strong>{latest?.value.toPrecision(5)}</strong>
              <Chart label={selected} points={metrics.filter((e) => e.tag === selected)} />
            </section>
            <div className="metric-grid">
              {tags
                .filter((t) => t !== selected)
                .slice(0, 8)
                .map((tag) => {
                  const points = metrics.filter((e) => e.tag === tag);
                  return (
                    <section className="panel" key={tag}>
                      <span className="eyebrow">{tag}</span>
                      <h2>{points.at(-1).value.toPrecision(4)}</h2>
                      <Chart points={points} label={tag} />
                    </section>
                  );
                })}
            </div>
          </>
        ) : (
          <Empty icon={Activity} title={t('Waiting for the first training step')}>
            {t(
              "Charts populate from AReno's actual TensorBoard scalar writes. No synthetic metrics are shown.",
            )}
          </Empty>
        ))}
      {tab === 'logs' && (
        <div className="log-panel">
          <div>
            <Terminal size={15} /> {t('Sandbox output')}{' '}
            <span>
              {events.length} {t('events retained locally')}
            </span>
          </div>
          <pre>
            {events
              .filter((e) => e.type !== 'metric')
              .map((e) => (e.type === 'log' ? e.message : JSON.stringify(e)))
              .join('\n') || t('Waiting for sandbox output…')}
          </pre>
        </div>
      )}
      {tab === 'configuration' && (
        <section className="panel">
          <h2>{t('Reproducible runtime')}</h2>
          <p className="muted">
            {t('Source:')} {job.manifest.revision}
          </p>
          <p className="mono wrap">{job.manifest.image}</p>
          {job.commands.map((c, i) => (
            <pre className="code-block" key={i}>
              {c}
            </pre>
          ))}
        </section>
      )}
      {tab === 'artifacts' && (
        <section className="panel">
          <h2>{t('Persistent artifacts')}</h2>
          <p>
            {t('Modal Volume:')}
            <code>{t('arenoflow-artifacts')}</code>
          </p>
          {job.checkpoint ? (
            <>
              <p className="mono wrap">{job.checkpoint}</p>
              <Button onClick={() => onDeploy(job)}>
                <Radio size={16} />
                {t('Deploy this checkpoint')}
              </Button>
            </>
          ) : (
            <p className="muted">{t('No checkpoint has been reported yet.')}</p>
          )}
          <p className="muted">
            {t('Checkpoint and TensorBoard files stay on the volume after compute stops.')}
          </p>
          <External href="https://modal.com/apps">{t('Open Modal workspace')}</External>
        </section>
      )}
    </>
  );
}
