import { lazy, Suspense, useEffect, useState } from 'react';
import { api } from '../api';
import { t } from '../i18n';
import { Button, Notice } from './UI';

const PythonEditor = lazy(() => import('./PythonEditor'));
const scriptTypes = {
  dataset_loader: 'Dataset loader script',
  reward: 'Reward script',
  agentic: 'Agent script',
};

export default function ScriptGenerator({ datasets, onSaved }) {
  const [kinds, setKinds] = useState(['dataset_loader']);
  const [saving, setSaving] = useState(false);
  const [dataset, setDataset] = useState('');
  const [algorithm, setAlgorithm] = useState('');
  const [algorithms, setAlgorithms] = useState([]);
  const [sample, setSample] = useState('');
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    api('/bootstrap')
      .then((data) => setAlgorithms(data.catalog.algorithms))
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    setSample('');
    setResult(null);
    setError('');
    if (!dataset) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api('/scripts/sample', { dataset_id: dataset })
      .then((data) => {
        if (!cancelled) setSample(data.sample);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset]);
  const choices = algorithms;
  const supportsRollout = algorithms.find((a) => a.id === algorithm)?.rollout;
  const promptExample = [
    algorithm === 'dpo'
      ? t(
          'Example: The dataset contains prompt, chosen and rejected fields. Preserve both preferred and rejected responses for DPO training.',
        )
      : t(
          'Example: The dataset contains question and answer fields. Use question as the user message and answer as the expected response.',
        ),
    kinds.includes('dataset_loader') &&
      t(
        'Loader: normalize the sample fields into AReno training records and skip records with missing required fields.',
      ),
    kinds.includes('reward') &&
      t(
        'Reward: compare the generated answer with the reference answer after trimming whitespace; return 1 for a match and 0 otherwise.',
      ),
    kinds.includes('agentic') &&
      t(
        'Agent: implement a single-turn rollout using the normalized question and preserve the trajectory metadata required by AReno.',
      ),
    t('Adjust these requirements to match your dataset sample and task.'),
  ]
    .filter(Boolean)
    .join('\n');
  async function generate() {
    setBusy(true);
    setError('');
    setResult(null);
    try {
      setResult(
        await api('/scripts/generate', { kinds, dataset_id: dataset, algorithm, sample, prompt }),
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="panel">
      <summary>{t('Generate scripts with LLM')}</summary>
      <p className="muted">
        {t(
          'The selected sample and prompt are sent to your configured LLM provider. Media files are not sent. Generated code is not executed.',
        )}{' '}
        <a href="#settings">{t('LLM connection')}</a>
      </p>
      <fieldset disabled={busy || saving} style={{ border: 0, padding: 0 }}>
        <div className="field-grid">
          <label className="field">
            <span>{t('Dataset')}</span>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              <option value="">{t('Choose a saved dataset')}</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>{t('Algorithm')}</span>
            <select
              value={algorithm}
              onChange={(e) => {
                setAlgorithm(e.target.value);
                if (!algorithms.find((a) => a.id === e.target.value)?.rollout)
                  setKinds(['dataset_loader']);
                setResult(null);
              }}
            >
              <option value="">{t('Select algorithm')}</option>
              {choices.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.id.toUpperCase()}
                </option>
              ))}
            </select>
          </label>
          <fieldset className="modality-picker script-picker">
            <legend>{t('Scripts to generate')}</legend>
            {Object.entries(scriptTypes).map(([kind, label]) => (
              <label key={kind}>
                <input
                  type="checkbox"
                  checked={kinds.includes(kind)}
                  disabled={kind !== 'dataset_loader' && !supportsRollout}
                  onChange={(e) => {
                    setKinds((old) =>
                      e.target.checked ? [...old, kind] : old.filter((k) => k !== kind),
                    );
                    setResult(null);
                  }}
                />
                {t(label)}
              </label>
            ))}
          </fieldset>
          <label className="field full">
            <span>{t('Dataset sample')}</span>
            <textarea
              rows={7}
              maxLength={12000}
              disabled={loading}
              value={sample}
              onChange={(e) => {
                setSample(e.target.value);
                setResult(null);
              }}
            />
            <small>
              {t(
                'Inspect or paste representative records. Repository, Parquet, Arrow and large JSON samples must be supplied manually.',
              )}
            </small>
          </label>
          <label className="field full">
            <span>{t('Script requirements')}</span>
            <textarea
              placeholder={promptExample}
              rows={4}
              maxLength={12000}
              value={prompt}
              onChange={(e) => {
                setPrompt(e.target.value);
                setResult(null);
              }}
            />
          </label>
        </div>
        <Button
          type="button"
          busy={busy}
          disabled={
            saving ||
            !kinds.length ||
            loading ||
            !dataset ||
            !choices.some((a) => a.id === algorithm) ||
            !sample.trim() ||
            !prompt.trim()
          }
          onClick={generate}
        >
          {t('Generate selected scripts')}
        </Button>
      </fieldset>
      {error && <Notice error>{error}</Notice>}
      {result && (
        <section>
          <p>{t('Review and edit each script. Save the complete set when ready.')}</p>
          {result.scripts.map((script, index) => (
            <section className="panel" key={script.kind}>
              <h3>{t(scriptTypes[script.kind])}</h3>
              <label className="field">
                <span>{t('Name')}</span>
                <input
                  value={script.name}
                  maxLength={120}
                  disabled={saving}
                  onChange={(e) =>
                    setResult((old) => ({
                      ...old,
                      scripts: old.scripts.map((s, i) =>
                        i === index ? { ...s, name: e.target.value } : s,
                      ),
                    }))
                  }
                />
              </label>
              <Suspense fallback={<Notice>{t('Loading editor…')}</Notice>}>
                <PythonEditor
                  value={script.source}
                  onChange={(source) => {
                    if (!saving)
                      setResult((old) => ({
                        ...old,
                        scripts: old.scripts.map((s, i) => (i === index ? { ...s, source } : s)),
                      }));
                  }}
                />
              </Suspense>
            </section>
          ))}
          <Button
            type="button"
            busy={saving}
            disabled={busy || result.scripts.some((s) => !s.name.trim())}
            onClick={async () => {
              setSaving(true);
              setError('');
              try {
                const saved = await api('/scripts/batch', result);
                setResult(null);
                await onSaved(saved.scripts);
              } catch (e) {
                setError(e.message);
              } finally {
                setSaving(false);
              }
            }}
          >
            {t('Save all scripts')}
          </Button>
        </section>
      )}
    </details>
  );
}
