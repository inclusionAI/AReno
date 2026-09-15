import { t } from '../i18n';
import { useEffect, useState } from 'react';
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  Box,
  Check,
  ChevronDown,
  Code2,
  Download,
  Layers,
  Plus,
  Rocket,
  Search,
  Trash2,
} from 'lucide-react';
import { api } from '../api';
import { useLibrary } from './Library';
import { RunEstimate } from './Estimates';
import { Badge, Button, Notice, PageHeader } from './UI';
export function Field({ field, value, onChange, idPrefix = '', disabled = false }) {
  const id = `${idPrefix}${field.name}`;
  return (
    <label className="field" htmlFor={id}>
      <span>
        {t(field.name.replaceAll('_', ' '))}
        {value != null && <i className="override-dot" title={t('Explicit override')} />}
      </span>
      {field.type === 'bool' ? (
        <select
          id={id}
          value={value == null ? '' : String(value)}
          onChange={(e) => onChange(e.target.value === '' ? null : e.target.value === 'true')}
          disabled={disabled}
        >
          <option value="">
            {t('Default')}
            {field.default != null
              ? t('({p0})', {
                  p0: field.default,
                })
              : ''}
          </option>
          <option value="true">{t('Enabled')}</option>
          <option value="false">{t('Disabled')}</option>
        </select>
      ) : field.choices ? (
        <select
          id={id}
          value={value ?? ''}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value || null)}
        >
          <option value="">
            {t('Default')}
            {field.default != null
              ? t('({p0})', {
                  p0: field.default,
                })
              : ''}
          </option>
          {field.choices.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          value={Array.isArray(value) ? value.join(',') : (value ?? '')}
          disabled={disabled}
          placeholder={field.default == null ? t('Not set') : String(field.default)}
          type={['int', 'float'].includes(field.type) ? 'number' : 'text'}
          step={field.type === 'int' ? 1 : 'any'}
          onChange={(e) =>
            onChange(
              field.multiple
                ? e.target.value.split(',').map((v) => v.trim())
                : e.target.value === ''
                  ? null
                  : ['int', 'float'].includes(field.type)
                    ? Number(e.target.value)
                    : e.target.value,
            )
          }
        />
      )}
      <small>
        {t(field.help)}
        {field.multiple && t('Separate repeated values with commas.')}
      </small>
    </label>
  );
}
export function Parameters({ schema, values, onChange, idPrefix = '', excluded = [] }) {
  const [search, setSearch] = useState('');
  const filtered = schema.filter(
    (f) =>
      !excluded.includes(f.name) &&
      `${f.name} ${f.help} ${t(f.name.replaceAll('_', ' '))} ${t(f.help)}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  const groups = [...new Set(filtered.map((f) => f.group))];
  return (
    <div className="parameters">
      <div className="search-field">
        <Search size={16} />
        <input
          aria-label={t('Search all parameters')}
          placeholder={t('Search {p0} parameters…', {
            p0: schema.length,
          })}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>
      {groups.map((group) => (
        <details key={t(group)} open={search ? true : undefined}>
          <summary>
            {t(group)}
            <span>{filtered.filter((f) => f.group === group).length}</span>
            <ChevronDown size={16} />
          </summary>
          <div className="field-grid">
            {filtered
              .filter((f) => f.group === group)
              .map((field) => (
                <Field
                  key={field.name}
                  field={field}
                  value={values[field.name]}
                  idPrefix={idPrefix}
                  onChange={(v) => onChange(field.name, v)}
                />
              ))}
          </div>
        </details>
      ))}
      {!filtered.length && <p className="muted">{t('No matching parameters.')}</p>}
    </div>
  );
}
export default function Workflow({
  bootstrap,
  kind = 'training',
  initial,
  onDraft,
  onLaunched,
  notify,
}) {
  const { catalog, gpu_types } = bootstrap;
  const library = useLibrary();
  const [name, setName] = useState(
    initial?.name || (kind === 'training' ? 'My first training flow' : 'My model endpoint'),
  );
  const [model, setModel] = useState(
    initial?.model || {
      adapter: catalog.models.find((m) => m.id === 'qwen3')?.id || catalog.models[0].id,
      checkpoint:
        catalog.models.find((m) => m.id === 'qwen3')?.checkpoint ||
        catalog.models[0].checkpoint ||
        '',
    },
  );
  const [stages, setStages] = useState(
    initial?.stages || [
      {
        algo: 'sft',
        params: {
          ...catalog.presets.sft,
        },
      },
    ],
  );
  const [active, setActive] = useState(0);
  const [resources, setResources] = useState(() => {
    const { timeout_hours = 4, ...saved } = initial?.resources || {};
    return {
      gpu: 'H100',
      count: 1,
      cpu: 4,
      memory_gib: 32,
      ...saved,
      timeout_seconds: saved.timeout_seconds ?? timeout_hours * 3600,
    };
  });
  const [serve, setServe] = useState(initial?.serve || {});
  const [image, setImage] = useState(initial?.image || catalog.image);
  const [autoImage, setAutoImage] = useState(initial?.autoImage ?? true);
  const [imageInfo, setImageInfo] = useState(null);
  const [imageError, setImageError] = useState('');
  const [preparing, setPreparing] = useState('');
  const [endpointKey, setEndpointKey] = useState('');
  const [estimateHours, setEstimateHours] = useState(initial?.estimate_hours ?? 1);
  useEffect(() => {
    if (resources.timeout_seconds >= 1) {
      setEstimateHours((current) =>
        current > resources.timeout_seconds / 3600 ? resources.timeout_seconds / 3600 : current,
      );
    }
  }, [resources.timeout_seconds]);
  const [preview, setPreview] = useState(null),
    [error, setError] = useState(''),
    [busy, setBusy] = useState(false);
  const stage = stages[active];
  const stageSchema = catalog.train.filter((field) => field.algorithms.includes(stage.algo));
  const request = () => ({
    name,
    kind,
    model,
    stages: stages.map((s) => ({ ...s, dataset_loader_id: s.dataset_loader_id ?? '' })),
    resources,
    image,
    serve,
    autoImage,
    estimate_hours: estimateHours,
  });
  // Preparation follows the first stage's effective model settings, including presets.
  const preparationParams =
    kind === 'training' ? { ...catalog.presets[stages[0].algo], ...stages[0].params } : serve;
  const preparationHub =
    preparationParams.model_hub ??
    (kind === 'training'
      ? catalog.train.find((field) => field.name === 'model_hub')?.default
      : 'hf');
  const preparationModel = {
    ...model,
    checkpoint:
      (kind === 'training' ? preparationParams.ckpt : preparationParams.model_path) ||
      model.checkpoint,
  };
  function setPreparationHub(hub) {
    edit(() => {
      if (kind === 'training') {
        setStages((list) =>
          list.map((s, i) => (i === 0 ? { ...s, params: { ...s.params, model_hub: hub } } : s)),
        );
      } else {
        setServe((current) => ({ ...current, model_hub: hub }));
      }
    });
  }
  const reviewed = preview?.request === JSON.stringify(request());
  // Keep navigation drafts in React memory; never persist credentials or endpoint keys.
  useEffect(() => {
    onDraft?.({
      name,
      kind,
      model,
      stages,
      resources,
      image,
      serve,
      autoImage,
      estimate_hours: estimateHours,
    });
  }, [name, kind, model, stages, resources, image, serve, autoImage, estimateHours, onDraft]);
  useEffect(() => {
    if (!autoImage) return;
    const abort = new AbortController();
    let timer;
    async function refreshImage() {
      try {
        const result = await api('/images/latest', undefined, abort.signal);
        if (abort.signal.aborted) return;
        setImageInfo(result);
        setImageError('');
        if (result.reference !== image) {
          setImage(result.reference);
          setPreview(null);
        }
      } catch (e) {
        if (!abort.signal.aborted) setImageError(e.message);
      }
      if (!abort.signal.aborted) timer = setTimeout(refreshImage, 60000);
    }
    refreshImage();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, [autoImage, image]);
  const edit = (action) => {
    action();
    setPreview(null);
    setError('');
  };
  const setParam = (key, value) =>
    edit(() =>
      setStages((list) =>
        list.map((s, i) =>
          i === active
            ? {
                ...s,
                params: {
                  ...s.params,
                  [key]: value,
                },
              }
            : s,
        ),
      ),
    );
  async function review() {
    setBusy(true);
    setError('');
    try {
      const snapshot = request();
      const result = await api('/preview', snapshot);
      setPreview({
        ...result,
        request: JSON.stringify(snapshot),
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function launch() {
    setBusy(true);
    setError('');
    try {
      const job = await api('/jobs', {
        ...request(),
        endpoint_key: endpointKey,
      });
      setEndpointKey('');
      onLaunched(job);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function prepare(preparationKind) {
    setPreparing(preparationKind);
    setError('');
    try {
      const job = await api('/jobs', {
        kind: preparationKind,
        name:
          preparationKind === 'image_build'
            ? t('Build container')
            : `${t('Pre-download model')} · ${preparationModel.checkpoint}`,
        image,
        model: preparationModel,
        model_hub: preparationHub,
        resources: { timeout_seconds: resources.timeout_seconds },
      });
      onLaunched(job);
    } catch (e) {
      setError(e.message);
    } finally {
      setPreparing('');
    }
  }
  function exportFlow() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(request(), null, 2)], {
        type: 'application/json',
      }),
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = 'arenoflow-workflow.json';
    a.click();
    URL.revokeObjectURL(url);
    notify(t('Workflow exported. Credentials are not included.'));
  }
  return (
    <>
      <PageHeader
        eyebrow={kind === 'training' ? t('WORKFLOW BUILDER') : t('DEPLOYMENTS')}
        title={kind === 'training' ? t('Configure training') : t('Deploy a model checkpoint')}
        action={
          <Button onClick={exportFlow}>
            <Download size={16} />
            {t('Export configuration')}
          </Button>
        }
      >
        {t(
          'Parameters are initialized from algorithm presets and can be edited before submission.',
        )}
      </PageHeader>
      <div className="builder-layout">
        <div className="builder-main">
          <section className="panel">
            <div className="panel-heading">
              <span className="section-number">01</span>
              <h2>{t('Model configuration')}</h2>
              <Badge>{t('Repository catalog')}</Badge>
            </div>
            <div className="field-grid">
              <label className="field">
                <span>{kind === 'training' ? t('Workflow name') : t('Endpoint name')}</span>
                <input value={name} onChange={(e) => edit(() => setName(e.target.value))} />
              </label>
              <label className="field">
                <span>{t('Model family / adapter')}</span>
                <select
                  value={model.adapter}
                  onChange={(e) =>
                    edit(() =>
                      setModel({
                        adapter: e.target.value,
                        checkpoint:
                          catalog.models.find((m) => m.id === e.target.value)?.checkpoint || '',
                      }),
                    )
                  }
                >
                  {catalog.models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field full">
                <span>{t('Model checkpoint')}</span>
                <input
                  list="checkpoint-suggestions"
                  placeholder={t('Hugging Face model ID')}
                  value={model.checkpoint}
                  onChange={(e) =>
                    edit(() =>
                      setModel({
                        ...model,
                        checkpoint: e.target.value,
                      }),
                    )
                  }
                />
                <datalist id="checkpoint-suggestions">
                  {catalog.checkpoints.map((s) => (
                    <option key={s} value={s} />
                  ))}
                </datalist>
                <small>
                  {t(
                    'Filled automatically for the selected model. You can choose another Hugging Face checkpoint; AReno validates its architecture at launch.',
                  )}
                </small>
              </label>
            </div>
          </section>
          {kind === 'training' ? (
            <section className="panel">
              <div className="panel-heading">
                <span className="section-number">02</span>
                <h2>{t('Training stages')}</h2>
                <span className="muted">
                  {stages.length} {t('stage')}
                  {stages.length > 1 ? t('s') : ''}
                </span>
              </div>
              <div className="stage-rail">
                {stages.map((s, i) => (
                  <div className="stage-wrap" key={i}>
                    <button
                      className={`stage-node ${active === i ? 'active' : ''}`}
                      onClick={() => setActive(i)}
                    >
                      <span>
                        {t('STAGE')} {String(i + 1).padStart(2, '0')}
                      </span>
                      <b>
                        <Layers size={16} />
                        {s.algo.toUpperCase()}
                      </b>
                      <small>{i ? t('Previous checkpoint') : t('Base checkpoint')}</small>
                    </button>
                    {i < stages.length - 1 && <ArrowRight size={18} className="stage-arrow" />}
                  </div>
                ))}
                <button
                  className="add-stage"
                  aria-label={t('Add training stage')}
                  disabled={stages.length >= 8}
                  onClick={() =>
                    edit(() => {
                      setStages([
                        ...stages,
                        {
                          algo: 'gspo',
                          params: {
                            ...catalog.presets.gspo,
                          },
                        },
                      ]);
                      setActive(stages.length);
                    })
                  }
                >
                  <Plus size={20} />
                </button>
              </div>
              <div className="stage-settings">
                <div className="stage-title">
                  <h3>
                    {t('Stage')} {active + 1} {t('settings')}
                  </h3>
                  <div className="button-group">
                    <Button
                      disabled={!active}
                      aria-label={t('Move stage earlier')}
                      onClick={() =>
                        edit(() => {
                          const next = [...stages];
                          [next[active - 1], next[active]] = [next[active], next[active - 1]];
                          setStages(next);
                          setActive(active - 1);
                        })
                      }
                    >
                      <ArrowUp size={15} />
                    </Button>
                    <Button
                      disabled={active === stages.length - 1}
                      aria-label={t('Move stage later')}
                      onClick={() =>
                        edit(() => {
                          const next = [...stages];
                          [next[active], next[active + 1]] = [next[active + 1], next[active]];
                          setStages(next);
                          setActive(active + 1);
                        })
                      }
                    >
                      <ArrowDown size={15} />
                    </Button>
                    <Button
                      disabled={stages.length === 1}
                      aria-label={t('Remove stage')}
                      onClick={() =>
                        edit(() => {
                          setStages(stages.filter((_, i) => i !== active));
                          setActive(Math.max(0, active - 1));
                        })
                      }
                    >
                      <Trash2 size={15} />
                    </Button>
                  </div>
                </div>
                <div className="field-grid">
                  <label className="field">
                    <span>{t('Algorithm preset')}</span>
                    <select
                      value={stage.algo}
                      onChange={(e) =>
                        edit(() =>
                          setStages(
                            stages.map((s, i) =>
                              i === active
                                ? {
                                    algo: e.target.value,
                                    params: {
                                      ...catalog.presets[e.target.value],
                                      dataset_path: s.params.dataset_path,
                                    },
                                    dataset_id: s.dataset_id,
                                  }
                                : s,
                            ),
                          ),
                        )
                      }
                    >
                      {catalog.algorithms.map((a) => (
                        <option key={a.id} value={a.id}>
                          {a.id.toUpperCase()}
                        </option>
                      ))}
                    </select>
                    <small>{t('Changing the algorithm applies its recommended preset.')}</small>
                  </label>
                  {[
                    'lr',
                    'max_steps',
                    'batch_size',
                    'mini_bs',
                    ...(catalog.algorithms.find((a) => a.id === stage.algo)?.rollout
                      ? ['n_samples']
                      : []),
                  ].map((key) => {
                    const field = catalog.train.find((f) => f.name === key);
                    return (
                      field && (
                        <Field
                          key={key}
                          field={field}
                          value={stage.params[key]}
                          idPrefix={`common-${active}-`}
                          onChange={(v) => setParam(key, v)}
                        />
                      )
                    );
                  })}
                </div>
                {library.error && <Notice error>{library.error}</Notice>}
                <div className="field-grid subsection">
                  <label className="field full">
                    <span>{t('Dataset')}</span>
                    <select
                      value={stage.dataset_id || ''}
                      onChange={(e) =>
                        edit(() =>
                          setStages((list) =>
                            list.map((s, i) =>
                              i === active
                                ? {
                                    ...s,
                                    dataset_id: e.target.value,
                                    params: {
                                      ...s.params,
                                      dataset_path: null,
                                      dataset_loader_fn: null,
                                    },
                                  }
                                : s,
                            ),
                          ),
                        )
                      }
                    >
                      <option value="">{t('Choose a saved dataset')}</option>
                      {library.datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name}
                        </option>
                      ))}
                    </select>
                    <small>
                      <a href="#datasets">{t('Manage datasets ↗')}</a>
                    </small>
                  </label>
                  {[
                    [
                      'dataset_loader_id',
                      'dataset_loader',
                      'Dataset loader script',
                      'AReno default loader',
                    ],
                    ...(catalog.algorithms.find((a) => a.id === stage.algo)?.rollout
                      ? [
                          [
                            'reward_function_id',
                            'reward',
                            'Reward script',
                            'Repository math verifier',
                          ],
                          ['agentic_function_id', 'agentic', 'Agent script', 'Standard rollout'],
                        ]
                      : []),
                  ].map(([key, kind, label, fallback]) => (
                    <label className="field" key={key}>
                      <span>{t(label)}</span>
                      <select
                        value={stage[key] || ''}
                        onChange={(e) =>
                          edit(() =>
                            setStages((list) =>
                              list.map((s, i) =>
                                i === active
                                  ? {
                                      ...s,
                                      [key]: e.target.value,
                                    }
                                  : s,
                              ),
                            ),
                          )
                        }
                      >
                        <option value="">{t(fallback)}</option>
                        {library.functions
                          .filter((f) => f.kind === kind)
                          .map((f) => (
                            <option key={f.id} value={f.id}>
                              {f.name}
                            </option>
                          ))}
                      </select>
                      <small>
                        <a href="#functions">{t('Manage scripts ↗')}</a>
                      </small>
                    </label>
                  ))}
                </div>
                <div className="subsection">
                  <h3>
                    {stage.algo.toUpperCase()} {t('training parameters')}{' '}
                    <span>
                      {stageSchema.length} {t('applicable controls')}
                    </span>
                  </h3>
                  <Parameters
                    schema={stageSchema}
                    excluded={[
                      'algo',
                      'dataset_path',
                      'dataset_loader_fn',
                      'reward_fn_path',
                      'agent_fn',
                    ]}
                    values={stage.params}
                    idPrefix={`advanced-${active}-`}
                    onChange={setParam}
                  />
                </div>
              </div>
            </section>
          ) : (
            <section className="panel">
              <div className="panel-heading">
                <span className="section-number">02</span>
                <h2>{t('Serving configuration')}</h2>
              </div>
              <label className="field">
                <span>{t('Endpoint API key')}</span>
                <input
                  type="password"
                  autoComplete="new-password"
                  value={endpointKey}
                  onChange={(e) => setEndpointKey(e.target.value)}
                  placeholder={t('Choose at least 24 characters')}
                />
                <small>
                  {t(
                    'Keep your key. It is sent to a Modal Secret and is not saved in local job records.',
                  )}
                </small>
              </label>
              <Parameters
                schema={catalog.serve}
                values={serve}
                onChange={(key, value) =>
                  edit(() =>
                    setServe({
                      ...serve,
                      [key]: value,
                    }),
                  )
                }
                idPrefix="serve-"
              />
              <Notice>
                {t(
                  "The protected gateway exposes the endpoint. AReno's host and port are managed internally.",
                )}
              </Notice>
            </section>
          )}
          <section className="panel">
            <div className="panel-heading">
              <span className="section-number">03</span>
              <h2>{t('Compute & runtime')}</h2>
            </div>
            <div className="field-grid three">
              <label className="field">
                <span>{t('GPU')}</span>
                <select
                  value={resources.gpu}
                  onChange={(e) =>
                    edit(() =>
                      setResources({
                        ...resources,
                        gpu: e.target.value,
                      }),
                    )
                  }
                >
                  {gpu_types.map((g) => (
                    <option key={g}>{g}</option>
                  ))}
                </select>
              </label>
              {[
                ['count', 'GPU count', 1, 8],
                ['cpu', 'CPU cores', 1, 64],
                ['memory_gib', 'Memory · GiB', 4, 512],
                ['timeout_seconds', 'Maximum lifetime · s', 1, 86400],
              ].map(([key, label, min, max]) => (
                <label className="field" key={key}>
                  <span>{t(label)}</span>
                  <input
                    type="number"
                    min={min}
                    max={max}
                    value={resources[key]}
                    onChange={(e) =>
                      edit(() =>
                        setResources({
                          ...resources,
                          [key]: Number(e.target.value),
                        }),
                      )
                    }
                  />
                </label>
              ))}
              <label className="field full">
                <span>{t('AReno container image')}</span>
                <input
                  value={image}
                  onChange={(e) =>
                    edit(() => {
                      setAutoImage(false);
                      setImage(e.target.value);
                    })
                  }
                />
                <small>
                  {autoImage
                    ? imageInfo
                      ? t('Latest published release: {p0}. Checked every 60 seconds.', {
                          p0: imageInfo.tag,
                        })
                      : t('Checking published tags in GHCR…')
                    : t('Pinned manually. Automatic updates are paused.')}{' '}
                  {t('The selected tag is resolved to an immutable digest at launch.')}
                </small>
              </label>
              <label className="field full image-follow">
                <span>
                  <input
                    type="checkbox"
                    checked={autoImage}
                    onChange={(e) => edit(() => setAutoImage(e.target.checked))}
                  />{' '}
                  {t('Follow latest published tag')}
                </span>
                <small>
                  {t(
                    'Uses the highest stable version from GHCR. A new tag requires reviewing commands again.',
                  )}
                </small>
              </label>
              <div className="field full preparation-actions">
                <h3>{t('Prepare runtime')}</h3>
                <label className="field">
                  <span>{t('Model source · shared with runtime')}</span>
                  <select
                    value={preparationHub}
                    onChange={(e) => setPreparationHub(e.target.value)}
                  >
                    <option value="hf">Hugging Face</option>
                    <option value="modelscope">ModelScope</option>
                  </select>
                </label>
                <small>
                  {t(
                    'Pre-download uses the first training stage or deployment model configuration. Changing this source also updates that configuration.',
                  )}
                </small>
                <div className="preparation-buttons">
                  <Button
                    busy={preparing === 'image_build'}
                    disabled={!!preparing || busy || !bootstrap.connected || !image.trim()}
                    onClick={() => prepare('image_build')}
                  >
                    <Box size={16} />
                    {t('Build container')}
                  </Button>
                  <Button
                    busy={preparing === 'model_download'}
                    disabled={
                      !!preparing ||
                      busy ||
                      !bootstrap.connected ||
                      !image.trim() ||
                      !preparationModel.checkpoint.trim()
                    }
                    onClick={() => prepare('model_download')}
                  >
                    <Download size={16} />
                    {t('Pre-download model')}
                  </Button>
                </div>
                <small>
                  {t(
                    'Downloads original weights on CPU into the shared Modal Volume. Hugging Face and ModelScope have separate caches. Modal usage charges apply.',
                  )}
                </small>
                {!bootstrap.connected && (
                  <small>{t('Connect Modal in settings to prepare the runtime.')}</small>
                )}
              </div>
              {autoImage && imageError && (
                <Notice error>
                  {t('Image discovery failed:')}
                  {imageError}
                  {t('. Retrying automatically; you can specify a tag manually.')}
                </Notice>
              )}
            </div>
          </section>
        </div>
        <aside className="review-panel">
          <div className="review-heading">
            <Rocket size={20} />
            <h3>{t('Workflow summary')}</h3>
          </div>
          <dl>
            <dt>{t('Execution')}</dt>
            <dd>{t('Your Modal workspace')}</dd>
            <dt>{t('Training stack')}</dt>
            <dd>{t('AReno native')}</dd>
            <dt>{t('GPU reservation')}</dt>
            <dd>
              {resources.count} × {resources.gpu}
            </dd>
            <dt>{t('Maximum lifetime')}</dt>
            <dd>{resources.timeout_seconds} s</dd>
            <dt>{t('Billing')}</dt>
            <dd>{t('Actual Modal usage')}</dd>
          </dl>
          <div className="review-flow">
            <Box size={16} />
            <span>{model.checkpoint || t('Choose a model')}</span>
            {kind === 'training' &&
              stages.map((s, i) => (
                <div key={i}>
                  <span className="vertical-line" />
                  <Layers size={16} />
                  <b>{s.algo.toUpperCase()}</b>
                </div>
              ))}
          </div>
          <p className="muted">
            {t(
              'Launching reserves cloud compute in your account. Reported costs refresh automatically; Modal may report usage with a delay.',
            )}
          </p>
          <RunEstimate
            resources={resources}
            hours={estimateHours}
            onHoursChange={setEstimateHours}
          />
          {error && <Notice error>{error}</Notice>}
          <Button className="wide" busy={busy} disabled={autoImage && !imageInfo} onClick={review}>
            <Code2 size={16} />
            {t('Review commands')}
          </Button>
          <Button
            className="primary wide"
            disabled={!reviewed || !bootstrap.connected}
            busy={busy}
            onClick={launch}
          >
            {kind === 'training' ? t('Launch training flow') : t('Deploy endpoint')}
            <ArrowUpRightIcon />
          </Button>
          {!bootstrap.connected && (
            <a href="#settings" className="connect-hint">
              {t('Connect Modal to launch →')}
            </a>
          )}
          {reviewed && (
            <div className="command-preview">
              <span>
                <Check size={14} />
                {t('Configuration validated')}
              </span>
              {preview.commands.map((c, i) => (
                <pre key={i}>{c}</pre>
              ))}
            </div>
          )}
        </aside>
      </div>
    </>
  );
}
function ArrowUpRightIcon() {
  return <ArrowRight size={16} />;
}
