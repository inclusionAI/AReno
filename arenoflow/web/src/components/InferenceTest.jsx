import { useEffect, useRef, useState } from 'react';
import { Send } from 'lucide-react';
import { api } from '../api';
import { t } from '../i18n';
import { Button, Notice } from './UI';

export default function InferenceTest({ job }) {
  const [key, setKey] = useState('');
  const [prompt, setPrompt] = useState('');
  const [system, setSystem] = useState('');
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const pending = useRef(null);
  useEffect(() => () => pending.current?.abort(), []);
  const ready = job.status === 'ready' && !!job.endpoint;
  async function submit(event) {
    event.preventDefault();
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true);
    setError('');
    setResult(null);
    try {
      const data = await api(
        `/jobs/${job.id}/inference`,
        {
          api_key: key,
          prompt,
          system,
          max_tokens: maxTokens,
          temperature,
        },
        controller.signal,
      );
      if (!controller.signal.aborted) setResult(data);
    } catch (e) {
      if (!controller.signal.aborted) setError(e.message);
    } finally {
      if (!controller.signal.aborted) setBusy(false);
    }
  }
  const choice = result?.response?.choices?.[0];
  const message = choice?.message;
  return (
    <section className="panel inference-test">
      <h2>{t('Inference test')}</h2>
      <p className="muted">
        {t(
          'Send a request to this deployment. Prompts, responses and the endpoint key are not saved.',
        )}
      </p>
      {!ready && (
        <Notice>{t('Inference testing is available when the deployment is ready.')}</Notice>
      )}
      <form onSubmit={submit}>
        <div className="field-grid">
          <label className="field full">
            <span>{t('Endpoint API key')}</span>
            <input
              type="password"
              autoComplete="off"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              required
              disabled={busy}
            />
          </label>
          <label className="field full">
            <span>{t('System prompt (optional)')}</span>
            <textarea
              rows={2}
              maxLength={16000}
              value={system}
              onChange={(e) => setSystem(e.target.value)}
              disabled={busy}
              placeholder={t('You are a helpful assistant.')}
            />
          </label>
          <label className="field full">
            <span>{t('Test prompt')}</span>
            <textarea
              rows={5}
              maxLength={32000}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={busy}
              required
              placeholder={t('Explain why the sky appears blue in two sentences.')}
            />
          </label>
          <label className="field">
            <span>{t('Maximum output tokens')}</span>
            <input
              type="number"
              min={1}
              max={32768}
              step={1}
              value={maxTokens}
              onChange={(e) => setMaxTokens(e.target.value === '' ? '' : Number(e.target.value))}
              disabled={busy}
              required
            />
          </label>
          <label className="field">
            <span>{t('Temperature')}</span>
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(e.target.value === '' ? '' : Number(e.target.value))}
              disabled={busy}
              required
            />
          </label>
        </div>
        <div className="inference-actions">
          <Button
            type="submit"
            className="primary"
            busy={busy}
            disabled={!ready || !key.trim() || !prompt.trim()}
          >
            <Send size={16} />
            {t(busy ? 'Generating response…' : 'Send test request')}
          </Button>
        </div>
      </form>
      {error && <Notice error>{error}</Notice>}
      {result && (
        <div className="inference-result" aria-live="polite">
          <h3>{t('Model response')}</h3>
          {message?.reasoning_content && (
            <details>
              <summary>{t('Reasoning')}</summary>
              <pre>{message.reasoning_content}</pre>
            </details>
          )}
          <pre>
            {typeof message?.content === 'string' && message.content
              ? message.content
              : t('No text content returned.')}
          </pre>
          <p className="muted">{t('Response time: {p0} s', { p0: result.elapsed_seconds })}</p>
          {result.response.usage && (
            <p className="muted">
              {t('Tokens · input: {p0} · output: {p1} · total: {p2}', {
                p0: result.response.usage.prompt_tokens ?? '—',
                p1: result.response.usage.completion_tokens ?? '—',
                p2: result.response.usage.total_tokens ?? '—',
              })}
            </p>
          )}
          <p className="muted">{t('Finish reason: {p0}', { p0: choice?.finish_reason || '—' })}</p>
          <details>
            <summary>{t('Raw response')}</summary>
            <pre>{JSON.stringify(result.response, null, 2)}</pre>
          </details>
        </div>
      )}
    </section>
  );
}
