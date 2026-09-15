import { t } from '../i18n';
import { useEffect, useState } from 'react';
import { api, money } from '../api';
import { Notice } from './UI';
export function RunEstimate({ resources, hours, onHoursChange }) {
  const [runs, setRuns] = useState(1),
    [result, setResult] = useState(null),
    [error, setError] = useState('');
  const payload = JSON.stringify({
    resources,
    hours,
    run_count: runs,
  });
  useEffect(() => {
    const abort = new AbortController();
    setResult(null);
    setError('');
    const timer = setTimeout(async () => {
      try {
        setResult(await api('/estimate', JSON.parse(payload), abort.signal));
      } catch (e) {
        if (!abort.signal.aborted) setError(e.message);
      }
    }, 400);
    return () => {
      clearTimeout(timer);
      abort.abort();
    };
  }, [payload]);
  return (
    <section className="estimate-card">
      <h3>{t('Estimated compute cost')}</h3>
      <label className="field">
        <span>{t('Expected runtime · hours')}</span>
        <input
          type="number"
          min={1 / 3600}
          max={resources.timeout_seconds / 3600}
          step="any"
          value={hours}
          onChange={(e) => onHoursChange(e.target.value === '' ? '' : Number(e.target.value))}
        />
        <small>{t('Entire workflow, including all training stages.')}</small>
      </label>
      <label className="field">
        <span>{t('Runs to compare')}</span>
        <input
          type="number"
          min={1}
          max={10000}
          step={1}
          value={runs}
          onChange={(e) => setRuns(e.target.value === '' ? '' : Number(e.target.value))}
        />
        <small>{t('Planning only; launching still submits one workflow.')}</small>
      </label>
      {error ? (
        <Notice error>{error}</Notice>
      ) : (
        <>
          <div className="estimate-value">
            <span>{t('One run')}</span>
            <strong>{money(result?.planned_cost)}</strong>
          </div>
          <div className="estimate-value">
            <span>
              {runs || '—'} {t('runs total')}
            </span>
            <strong>{money(result?.total_cost)}</strong>
          </div>
          <small>
            {result
              ? t('{p0}/hour · {p1} at maximum lifetime', {
                  p0: money(result.hourly_cost),
                  p1: money(result.lifetime_cost),
                })
              : t('Fetching current Modal rates…')}
          </small>
        </>
      )}
      <p>
        {t('Estimate from')}{' '}
        <a href="https://modal.com/pricing" target="_blank" rel="noreferrer">
          {t('live Modal Sandbox list prices ↗')}
        </a>
        {t(
          '. CPU/memory bursts, storage, transfer, credits and negotiated rates can change your actual bill.',
        )}
      </p>
    </section>
  );
}
export function RunEstimates({ compact = false }) {
  const [data, setData] = useState(null),
    [error, setError] = useState('');
  useEffect(() => {
    const abort = new AbortController();
    let timer;
    async function poll() {
      try {
        const result = await api('/estimates', undefined, abort.signal);
        if (!abort.signal.aborted) {
          setData(result);
          setError('');
        }
      } catch (e) {
        if (!abort.signal.aborted) setError(e.message);
      }
      if (!abort.signal.aborted) timer = setTimeout(poll, 10000);
    }
    poll();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, []);
  const known = data && (!data.run_count || data.priced_count);
  return (
    <section className="panel run-estimates">
      <div className="panel-heading">
        <h2>{t('All training runs · estimates')}</h2>
        <span className="muted">
          {data?.run_count ?? '—'} {t('runs')}
        </span>
      </div>
      {error && <Notice error>{error}</Notice>}
      <div className="estimate-totals">
        <div>
          <span>{t('Planned compute total')}</span>
          <strong>{money(known ? data.planned_total : null)}</strong>
        </div>
        <div>
          <span>{t('Estimated compute elapsed')}</span>
          <strong>{money(known ? data.accrued_total : null)}</strong>
        </div>
      </div>
      <p className="muted">
        {t(
          "Local training runs only. Planned costs use expected duration; elapsed costs use recorded runtime and quoted list rates. These are estimates, separate from Modal's reported workspace charges.",
        )}
      </p>
      {!!data?.errors?.length && (
        <Notice error>
          {t('Prices unavailable for')}
          {data.errors.length} {t('runs. Totals include only')} {data.priced_count}{' '}
          {t('priced runs.')}
        </Notice>
      )}
      {!compact && !!data?.rows?.length && (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>{t('Run')}</th>
                <th>{t('Expected hours')}</th>
                <th>{t('Planned estimate')}</th>
                <th>{t('Elapsed estimate')}</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={row.id}>
                  <td>
                    <a href={`#run/${row.id}`}>{row.name}</a>
                  </td>
                  <td>{row.hours}</td>
                  <td>{money(row.planned_cost)}</td>
                  <td>{money(row.accrued_cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
