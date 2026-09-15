import { useEffect, useState } from 'react';
import { api } from '../api';
import { t } from '../i18n';
import { Button, Notice } from './UI';

export default function LLMSettings() {
  const [config, setConfig] = useState({ base_url: '', model: '', has_api_key: false });
  const [key, setKey] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    api('/llm')
      .then(setConfig)
      .catch((e) => setError(e.message));
  }, []);
  async function save(e) {
    e.preventDefault();
    setBusy(true);
    setError('');
    setSaved(false);
    try {
      setConfig(
        await api('/llm', {
          base_url: config.base_url,
          model: config.model,
          ...(key ? { api_key: key } : {}),
        }),
      );
      setKey('');
      setSaved(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="panel">
      <h2>{t('LLM connection')}</h2>
      <p className="muted">
        {t(
          'Configure an OpenAI-compatible Chat Completions API for script generation. Include the API version in the base URL when required.',
        )}
      </p>
      <form onSubmit={save} className="field-grid">
        <label className="field full">
          <span>Base URL</span>
          <input
            required
            type="url"
            value={config.base_url}
            placeholder="https://api.example.com/v1"
            onChange={(e) => {
              setConfig({ ...config, base_url: e.target.value });
              setSaved(false);
            }}
          />
        </label>
        <label className="field">
          <span>{t('Model')}</span>
          <input
            required
            value={config.model}
            onChange={(e) => {
              setConfig({ ...config, model: e.target.value });
              setSaved(false);
            }}
          />
        </label>
        <label className="field">
          <span>API Key</span>
          <input
            type="password"
            autoComplete="new-password"
            value={key}
            placeholder={
              config.has_api_key
                ? t('Key configured; leave blank to retain')
                : t('Optional for local providers')
            }
            onChange={(e) => {
              setKey(e.target.value);
              setSaved(false);
            }}
          />
        </label>
        <Button type="submit" busy={busy}>
          {t('Save LLM settings')}
        </Button>
        <Button
          type="button"
          busy={busy}
          onClick={async () => {
            setBusy(true);
            setError('');
            try {
              setConfig(await api('/llm', { ...config, api_key: '' }));
              setKey('');
            } catch (e) {
              setError(e.message);
            } finally {
              setBusy(false);
            }
          }}
        >
          {t('Clear API key')}
        </Button>
      </form>
      <p className="muted">
        {t(
          'LLM settings stay in server memory and must be configured again after restart. Saving settings does not make a provider request.',
        )}
      </p>
      {error && <Notice error>{error}</Notice>}
      {saved && <Notice>{t('LLM settings saved.')}</Notice>}
    </section>
  );
}
