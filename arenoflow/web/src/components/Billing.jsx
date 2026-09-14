import { t } from '../i18n';
import { CreditCard, RefreshCw, Timer } from 'lucide-react';
import { money, time } from '../api';
import { RunEstimates } from './Estimates';
import { Button, Empty, External, Notice, PageHeader } from './UI';
export default function Billing({ data, error, loading, connected, refresh }) {
  const summary = data?.summary;
  const breakdown = Object.entries(summary?.metered_cost_breakdown || {});
  const total = Number(summary?.metered_cost || 0);
  return (
    <>
      <PageHeader
        eyebrow={t('USAGE & BILLING')}
        title={t('Know where your compute goes.')}
        action={
          <Button busy={loading} onClick={refresh}>
            <RefreshCw size={16} />
            {t('Refresh now')}
          </Button>
        }
      >
        {t('Actual Modal charges and separately labeled compute estimates.')}
      </PageHeader>
      <RunEstimates />
      {!connected ? (
        <Empty
          icon={CreditCard}
          title={t('Connect your workspace to see costs')}
          action={
            <a className="button primary" href="#settings">
              {t('Connect Modal')}
            </a>
          }
        >
          {t('Your billing data stays between this local workspace and Modal.')}
        </Empty>
      ) : (
        <>
          <div className="billing-status">
            <span>
              <i className="live-dot" />
              {t('Auto-refresh · every 15 seconds')}
            </span>
            <span>
              {t('Last response:')} {time(data?.fetched_at)}
            </span>
          </div>
          {error && <Notice error>{error}</Notice>}
          {data?.errors?.summary && (
            <Notice error>
              {t('Billing summary unavailable:')} {data.errors.summary}
            </Notice>
          )}
          <div className="billing-numbers">
            <div>
              <span>{t('Reported metered usage')}</span>
              <strong>{money(summary?.metered_cost)}</strong>
              <small>
                {data?.cycle || t('This month')} {t('· entire Modal workspace')}
              </small>
            </div>
            <div>
              <span>{t('Reported billed amount')}</span>
              <strong>{money(summary?.billed_cost)}</strong>
              <small>{t('After adjustments reported by Modal')}</small>
            </div>
            <div>
              <span>{t('Latest usage')}</span>
              <strong className="pending-cost">
                <Timer size={25} />
                {t('Pending')}
              </strong>
              <small>{t('Unreported usage is not counted as zero')}</small>
            </div>
          </div>
          <Notice>
            {t(
              'Modal billing has a collection delay, typically minutes. Hourly reports exclude the current incomplete hour. These are workspace-wide API totals, not an allocation to an individual ARenoflow run.',
            )}
          </Notice>
          <div className="billing-grid">
            <section className="panel">
              <div className="panel-heading">
                <h2>{t('Resource breakdown')}</h2>
                <span className="muted">{t('USD · metered')}</span>
              </div>
              {breakdown.length ? (
                <div className="cost-bars">
                  {breakdown.map(([key, value]) => (
                    <div key={key}>
                      <div>
                        <span>{key.replaceAll('_', ' ')}</span>
                        <b>{money(value)}</b>
                      </div>
                      <div className="bar-track">
                        <div
                          style={{
                            width: `${total ? Math.max(1, (Number(value) / total) * 100) : 0}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="muted">{t('No resource breakdown has been reported.')}</p>
              )}
            </section>
            <section className="panel">
              <div className="panel-heading">
                <h2>{t('Billing adjustments')}</h2>
              </div>
              {Object.entries(summary?.adjustments || {}).map(([key, value]) => (
                <div className="billing-row" key={key}>
                  <span>{key.replaceAll('_', ' ')}</span>
                  <b>{money(value)}</b>
                </div>
              ))}
              {!Object.keys(summary?.adjustments || {}).length && (
                <p className="muted">{t('No adjustments reported.')}</p>
              )}
              <External href="https://modal.com/settings">{t('Open Modal billing')}</External>
            </section>
          </div>
          <section className="panel">
            <div className="panel-heading">
              <h2>{t('Reported usage over time')}</h2>
              <span className="muted">{t('Completed hourly intervals · UTC')}</span>
            </div>
            {data?.errors?.report && (
              <Notice error>
                {t('Detailed report unavailable:')}
                {data.errors.report}
                {t('. Modal restricts granular billing reports by plan and permissions.')}
              </Notice>
            )}
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>{t('Interval')}</th>
                    <th>{t('Modal object')}</th>
                    <th>{t('Environment')}</th>
                    <th>{t('Cost')}</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.report
                    ?.slice(-200)
                    .reverse()
                    .map((row, i) => (
                      <tr key={`${row.object_id}-${row.interval_start}-${i}`}>
                        <td>{time(row.interval_start)}</td>
                        <td>
                          <b>{row.description || row.object_id}</b>
                          <small>{row.object_id}</small>
                        </td>
                        <td>{row.environment_name}</td>
                        <td>{money(row.cost)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
            {data?.report?.length === 0 && (
              <p className="muted">{t('No completed intervals returned yet.')}</p>
            )}
          </section>
        </>
      )}
    </>
  );
}
