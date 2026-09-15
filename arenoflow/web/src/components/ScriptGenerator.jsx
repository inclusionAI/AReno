import { useEffect, useState } from 'react';
import { api } from '../api';
import { t } from '../i18n';
import { Button, Notice } from './UI';

export default function ScriptGenerator({ kind, datasets, onApply }) {
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
    if (!dataset) return;
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
  const choices = algorithms.filter((a) => kind === 'dataset_loader' || a.rollout);
  async function generate() {
    setBusy(true);
    setError('');
    setResult(null);
    try {
      setResult(
        await api('/scripts/generate', { kind, dataset_id: dataset, algorithm, sample, prompt }),
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="panel">
      <summary>{t('Generate script with LLM')}</summary>
      <p className="muted">
        {t(
          'The selected sample and prompt are sent to your configured LLM provider. Media files are not sent. Generated code is not executed.',
        )}{' '}
        <a href="#settings">{t('LLM connection')}</a>
      </p>
      <fieldset disabled={busy} style={{ border: 0, padding: 0 }}>
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
            loading ||
            !dataset ||
            !choices.some((a) => a.id === algorithm) ||
            !sample.trim() ||
            !prompt.trim()
          }
          onClick={generate}
        >
          {t('Generate script')}
        </Button>
      </fieldset>
      {error && <Notice error>{error}</Notice>}
      {result && (
        <>
          <p>{t('Review the generated script before applying it to the editor.')}</p>
          <pre className="code-block" style={{ maxHeight: 360, overflow: 'auto' }}>
            {result.source}
          </pre>
          <Button
            type="button"
            onClick={() => {
              onApply(result);
              setResult(null);
            }}
          >
            {t('Use generated script')}
          </Button>
        </>
      )}
    </details>
  );
}
