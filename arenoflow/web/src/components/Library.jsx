import ScriptGenerator from './ScriptGenerator';
import { t } from '../i18n';
import { lazy, Suspense, useEffect, useState } from 'react';
import { Code2, Database, LoaderCircle, Plus, Save, Trash2, UploadCloud } from 'lucide-react';
import { api } from '../api';
import { Button, Empty, Notice, PageHeader } from './UI';
import Upload, { uploadFile } from './Upload';
const PythonEditor = lazy(() => import('./PythonEditor'));
export const functionKinds = {
  dataset_loader: 'Dataset loader script',
  reward: 'Reward script',
  agentic: 'Agent script',
};
const templates = {
  dataset_loader:
    'def load_training_dataset(dataset_path, *, default_loader, **kwargs):\n    records = default_loader(dataset_path)\n    # Normalize each record for your training task here.\n    return records\n',
  reward:
    'def reward_fn(record):\n    # Example: exact-match reward. Adapt to your task.\n    return float(record.completion.strip() == str(record.answer).strip())\n',
  agentic:
    'async def run_agent(ctx, batch):\n    # Implement your task-specific agent rollout here.\n    raise NotImplementedError("Implement your agent rollout before training")\n',
};
const contracts = {
  dataset_loader:
    'load_training_dataset(dataset_path, *, default_loader, load_dataset, load_from_disk) → training records. Use **kwargs for unused helpers.',
  reward:
    'reward_fn(record) → numeric score for one completion. The record includes completion and task fields such as answer.',
  agentic:
    'run_agent(ctx, batch) → agent rollout results. Async functions are supported; implement the AReno agent rollout contract for your task.',
};
export function useLibrary() {
  const [datasets, setDatasets] = useState([]),
    [functions, setFunctions] = useState([]),
    [error, setError] = useState('');
  async function refresh() {
    try {
      const [data, fns] = await Promise.all([api('/datasets'), api('/functions')]);
      setDatasets(data);
      setFunctions(fns);
      setError('');
    } catch (e) {
      setError(e.message);
    }
  }
  useEffect(() => {
    refresh();
  }, []);
  return {
    datasets,
    functions,
    error,
    refresh,
  };
}
export default function Library({ type, notify }) {
  const isFunction = type === 'functions';
  const library = useLibrary();
  const records = isFunction ? library.functions : library.datasets;
  const [draft, setDraft] = useState(null),
    [error, setError] = useState(''),
    [busy, setBusy] = useState(false);
  const [query, setQuery] = useState(''),
    [deleting, setDeleting] = useState(false);
  useEffect(() => {
    if (isFunction) return;
    const timer = setInterval(library.refresh, 2000);
    return () => clearInterval(timer);
  }, [isFunction]);
  const savedDataset =
    !isFunction && draft?.id ? library.datasets.find((d) => d.id === draft.id) : null;
  function create() {
    setDraft(
      isFunction
        ? {
            name: '',
            kind: 'dataset_loader',
            source: templates.dataset_loader,
          }
        : {
            name: '',
            source_type: 'upload',
            source: '',
            model_hub: 'hf',
            modalities: ['text'],
            media: [],
          },
    );
    setError('');
    setDeleting(false);
  }
  function edit(key, value) {
    setDraft((d) => ({
      ...d,
      [key]: value,
    }));
    setError('');
  }
  async function save(e) {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      setDraft(await api(`/${type}`, isFunction ? draft : { ...draft, loader_id: '' }));
      await library.refresh();
      notify(isFunction ? 'Script saved and syntax checked.' : 'Dataset saved.');
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    setBusy(true);
    setError('');
    try {
      await api(`/${type}/${draft.id}/delete`, {});
      setDraft(null);
      setDeleting(false);
      await library.refresh();
      notify(t('Removed from the library. Existing run artifacts are retained.'));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader
        eyebrow={isFunction ? t('SCRIPT LIBRARY') : t('DATA LIBRARY')}
        title={isFunction ? t('Script Manager') : t('Dataset Manager')}
        action={
          <Button className="primary" onClick={create}>
            <Plus size={16} />
            {isFunction ? t('New script') : t('New dataset')}
          </Button>
        }
      >
        {isFunction
          ? t(
              'Complete Python scripts for data loading, rewards and agentic rollouts. Scripts may include imports, helper functions and classes.',
            )
          : t('Manage dataset sources, media attachments and local sample caches.')}
      </PageHeader>
      {library.error && <Notice error>{library.error}</Notice>}
      {isFunction && (
        <ScriptGenerator
          datasets={library.datasets}
          onSaved={async (scripts) => {
            await library.refresh();
            setDraft(scripts[0]);
            notify(t('Scripts saved.'));
          }}
        />
      )}
      <div className="library-layout">
        <section className="panel library-list">
          <label className="field">
            <span>
              {t('Search')} {isFunction ? t('scripts') : t('datasets')}
            </span>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={t('Search by name…')}
            />
          </label>
          {records
            .filter((r) => r.name.toLowerCase().includes(query.toLowerCase()))
            .map((record) => (
              <button
                key={record.id}
                className={`library-item ${draft?.id === record.id ? 'selected' : ''}`}
                onClick={() => {
                  setDraft(record);
                  setError('');
                  setDeleting(false);
                }}
              >
                {isFunction ? <Code2 size={18} /> : <Database size={18} />}
                <span>
                  <b>{record.name}</b>
                  {!isFunction && (
                    <small>
                      {t(
                        record.sample_status === 'ready'
                          ? 'Cached'
                          : record.sample_status === 'downloading'
                            ? 'Downloading'
                            : record.sample_status === 'failed'
                              ? 'Download failed'
                              : 'Not cached',
                      )}
                    </small>
                  )}
                  <small>
                    {isFunction
                      ? t(functionKinds[record.kind])
                      : record.source_type === 'upload'
                        ? record.source_name
                        : record.source}
                  </small>
                </span>
              </button>
            ))}
          {!records.length && (
            <p className="muted">
              {t('Your saved')} {isFunction ? t('scripts') : t('datasets')} {t('will appear here.')}
            </p>
          )}
        </section>
        {!draft ? (
          <Empty
            icon={isFunction ? Code2 : Database}
            title={isFunction ? t('Python scripts') : t('Dataset configuration')}
            action={
              <Button onClick={create}>
                {t('Create')} {isFunction ? t('a script') : t('a dataset')}
              </Button>
            }
          >
            {isFunction
              ? t(
                  'Write or import a Python script, then select it by name in datasets and training workflows.',
                )
              : t(
                  'Add a repository or upload a dataset. Select the loader script when configuring a training stage.',
                )}
          </Empty>
        ) : (
          <form className="panel library-editor" onSubmit={save}>
            <div className="panel-heading">
              <h2>
                {draft.id ? t('Edit') : t('New')} {isFunction ? t('script') : t('dataset')}
              </h2>
            </div>
            <label className="field">
              <span>{t('Name')}</span>
              <input
                required
                maxLength={120}
                value={draft.name}
                onChange={(e) => edit('name', e.target.value)}
                placeholder={isFunction ? t('Alpaca normalizer') : t('Instruction training set')}
              />
            </label>
            {isFunction ? (
              <>
                <label className="field">
                  <span>{t('Script type')}</span>
                  <select
                    disabled={!!draft.id}
                    value={draft.kind}
                    onChange={(e) =>
                      setDraft({
                        ...draft,
                        kind: e.target.value,
                        source: templates[e.target.value],
                      })
                    }
                  >
                    {Object.entries(functionKinds).map(([value, label]) => (
                      <option key={value} value={value}>
                        {t(label)}
                      </option>
                    ))}
                  </select>
                </label>
                <Notice>{t(contracts[draft.kind])}</Notice>
                {draft.algorithm && (
                  <p className="muted">
                    {t('Generation context')}: {draft.algorithm.toUpperCase()} ·{' '}
                    {library.datasets.find((d) => d.id === draft.dataset_id)?.name ||
                      t('Dataset unavailable')}
                  </p>
                )}
                <div className="upload-dataset">
                  <label className="button">
                    <UploadCloud size={16} />
                    {t('Import Python')}
                    <input
                      type="file"
                      accept=".py"
                      aria-label={t('Import Python')}
                      onChange={async (e) => {
                        const file = e.target.files[0];
                        e.target.value = '';
                        if (!file) return;
                        if (file.size > 256 * 1024) {
                          setError(t('Import Python code up to 256 KiB.'));
                          return;
                        }
                        try {
                          edit('source', await file.text());
                        } catch {
                          setError(t('Could not read Python file.'));
                        }
                      }}
                    />
                  </label>
                </div>
                <PythonEditor value={draft.source} onChange={(source) => edit('source', source)} />
                <p className="muted">
                  {t(
                    'Syntax and entrypoint signatures are checked when saving. Code runs inside your training sandbox; dependencies must be available in the AReno image.',
                  )}
                </p>
              </>
            ) : (
              <>
                {savedDataset && (
                  <div className="notice" role="status">
                    {savedDataset.sample_status === 'downloading' && (
                      <LoaderCircle size={16} className="spin" aria-hidden="true" />
                    )}
                    {t(
                      savedDataset.sample_status === 'ready'
                        ? 'Cached'
                        : savedDataset.sample_status === 'downloading'
                          ? 'Downloading'
                          : savedDataset.sample_status === 'failed'
                            ? 'Download failed'
                            : 'Not cached',
                    )}
                    {savedDataset.sample_status !== 'downloading' && (
                      <Button
                        type="button"
                        onClick={async () => {
                          try {
                            await api('/scripts/sample', { dataset_id: draft.id, retry: true });
                            await library.refresh();
                          } catch (e) {
                            setError(e.message);
                          }
                        }}
                      >
                        {t('Download sample again')}
                      </Button>
                    )}
                  </div>
                )}
                <fieldset className="modality-picker">
                  <legend>{t('Data modalities')}</legend>
                  {['text', 'image', 'audio', 'video'].map((mode) => (
                    <label key={t(mode)}>
                      <input
                        type="checkbox"
                        checked={(draft.modalities || ['text']).includes(mode)}
                        onChange={(e) =>
                          edit(
                            'modalities',
                            e.target.checked
                              ? [...(draft.modalities || ['text']), mode]
                              : (draft.modalities || ['text']).filter((m) => m !== mode),
                          )
                        }
                      />
                      {t(mode)}
                    </label>
                  ))}
                </fieldset>
                <div
                  className="dataset-source-picker"
                  role="group"
                  aria-label={t('Dataset source')}
                >
                  {[
                    ['upload', 'Upload from computer'],
                    ['repository', 'Dataset repository'],
                  ].map(([value, label]) => (
                    <Button
                      key={value}
                      type="button"
                      aria-pressed={draft.source_type === value}
                      disabled={busy}
                      onClick={() => {
                        if (draft.source_type !== value)
                          setDraft({
                            ...draft,
                            source_type: value,
                            source: '',
                            source_name: '',
                            media: [],
                          });
                      }}
                    >
                      {value === 'upload' ? <UploadCloud size={16} /> : <Database size={16} />}
                      {t(label)}
                    </Button>
                  ))}
                </div>
                {draft.source_type === 'repository' ? (
                  <div className="field-grid">
                    <label className="field">
                      <span>{t('Repository ID')}</span>
                      <input
                        required
                        value={draft.source}
                        onChange={(e) => edit('source', e.target.value)}
                      />
                    </label>
                    <label className="field">
                      <span>{t('Dataset hub')}</span>
                      <select
                        value={draft.model_hub}
                        onChange={(e) => edit('model_hub', e.target.value)}
                      >
                        <option value="modelscope">{t('ModelScope')}</option>
                        <option value="hf">{t('Hugging Face')}</option>
                      </select>
                    </label>
                  </div>
                ) : (
                  <>
                    <Upload
                      key={draft.id || 'new'}
                      onBusyChange={setBusy}
                      onUploaded={(source, file) =>
                        setDraft((d) => ({
                          ...d,
                          source,
                          source_name: file.name,
                        }))
                      }
                    />
                    {draft.source && (
                      <p className="muted">
                        {t('Selected:')} {draft.source_name}
                      </p>
                    )}
                  </>
                )}
                {draft.source_type === 'upload' && (
                  <section className="media-attachments">
                    <h3>{t('Media attachments')}</h3>
                    <p className="muted">
                      {t(
                        'Reference filenames in your JSON, JSONL, CSV or TSV samples. Image, audio and video references are connected automatically, including nested messages.',
                      )}
                    </p>
                    <div className="upload-dataset">
                      <label className="button">
                        <UploadCloud size={16} />
                        {busy ? t('Uploading…') : t('Add images, audio or video')}
                        <input
                          type="file"
                          multiple
                          disabled={busy}
                          accept=".jpg,.jpeg,.png,.webp,.gif,.bmp,.wav,.mp3,.flac,.ogg,.m4a,.mp4,.webm,.mov,.mkv"
                          aria-label={t('Upload media attachments')}
                          onChange={async (e) => {
                            const files = [...e.target.files];
                            e.target.value = '';
                            setBusy(true);
                            setError('');
                            try {
                              for (const file of files) {
                                const media = await uploadFile(file);
                                setDraft((d) => ({
                                  ...d,
                                  media: [
                                    ...(d.media || []).filter((m) => m.name !== media.name),
                                    media,
                                  ],
                                }));
                              }
                            } catch (e) {
                              setError(e.message);
                            } finally {
                              setBusy(false);
                            }
                          }}
                        />
                      </label>
                      <small>
                        {t(
                          'Up to 16 MiB per file and 128 attachments. Use a repository for larger datasets.',
                        )}
                      </small>
                    </div>
                    {(draft.media || []).map((media, i) => (
                      <div className="media-item" key={media.name}>
                        <span>
                          {media.name}
                          <small>
                            {(media.bytes / 1024 / 1024).toFixed(2)} {t('MiB')}
                          </small>
                        </span>
                        <Button
                          type="button"
                          aria-label={t('Remove {p0}', {
                            p0: media.name,
                          })}
                          disabled={busy}
                          onClick={() =>
                            edit(
                              'media',
                              draft.media.filter((_, j) => i !== j),
                            )
                          }
                        >
                          <Trash2 size={14} />
                        </Button>
                      </div>
                    ))}
                  </section>
                )}
              </>
            )}
            {error && <Notice error>{error}</Notice>}
            <div className="library-actions">
              <Button type="submit" className="primary" busy={busy}>
                <Save size={16} />
                {t('Save')} {isFunction ? t('script') : t('dataset')}
              </Button>
              {draft.id && (
                <Button
                  type="button"
                  className="danger"
                  disabled={busy}
                  onClick={() => setDeleting(true)}
                >
                  <Trash2 size={15} />
                  {t('Delete')}
                </Button>
              )}
            </div>
            {deleting && (
              <Notice>
                {t('Remove “')}
                {draft.name}
                {t('” from the library? Existing run artifacts are retained.')}
                <div className="button-group">
                  <Button type="button" className="danger" busy={busy} onClick={remove}>
                    {t('Remove now')}
                  </Button>
                  <Button type="button" onClick={() => setDeleting(false)}>
                    {t('Cancel deletion')}
                  </Button>
                </div>
              </Notice>
            )}
          </form>
        )}
      </div>
    </>
  );
}
