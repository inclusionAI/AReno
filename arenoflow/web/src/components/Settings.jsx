import { t } from '../i18n';
import { useState } from 'react';
import { Check, KeyRound, ShieldCheck } from 'lucide-react';
import { api } from '../api';
import { Badge, Button, External, Notice, PageHeader } from './UI';
export default function Settings({ bootstrap, onConnected }) {
  const [tokenId, setTokenId] = useState(''),
    [tokenSecret, setTokenSecret] = useState('');
  const [busy, setBusy] = useState(false),
    [error, setError] = useState('');
  async function connect(useEnv = false) {
    setBusy(true);
    setError('');
    try {
      await api(
        '/connect',
        useEnv
          ? {}
          : {
              token_id: tokenId,
              token_secret: tokenSecret,
            },
      );
      setTokenSecret('');
      setTokenId('');
      onConnected();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader eyebrow={t('WORKSPACE SETTINGS')} title={t('Your cloud. Your credentials.')}>
        {t('ARenoflow runs locally. Training and serving run in your own Modal workspace.')}
      </PageHeader>
      <div className="settings-layout">
        <section className="panel">
          <div className="panel-heading">
            <KeyRound size={20} />
            <h2>{t('Connect Modal')}</h2>
            {bootstrap.connected && <Badge status="ready">{t('Connected')}</Badge>}
          </div>
          <p className="muted">
            {t(
              'Use the Token ID and Token Secret from your Modal account. An API token is this pair; no third credential is needed.',
            )}
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              connect();
            }}
          >
            <label className="field">
              <span>{t('Token ID')}</span>
              <input
                autoComplete="off"
                placeholder={t('ak-…')}
                value={tokenId}
                onChange={(e) => setTokenId(e.target.value)}
                required
              />
            </label>
            <label className="field">
              <span>{t('Token Secret')}</span>
              <input
                type="password"
                autoComplete="off"
                placeholder={t('as-…')}
                value={tokenSecret}
                onChange={(e) => setTokenSecret(e.target.value)}
                required
              />
            </label>
            {error && <Notice error>{error}</Notice>}
            <Button type="submit" className="primary" busy={busy}>
              <ShieldCheck size={16} />
              {t('Verify & connect')}
            </Button>
            {bootstrap.environment_credentials && (
              <Button type="button" busy={busy} onClick={() => connect(true)}>
                {t('Use environment credentials')}
              </Button>
            )}
          </form>
          <p className="muted">
            {t(
              'Browser-entered credentials stay in server memory. They are never saved in browser storage, job records, or exported configurations. Reconnect after a server restart.',
            )}
          </p>
          <External href="https://modal.com/settings">{t('Manage Modal API tokens')}</External>
        </section>
        <aside className="panel connection-notes">
          <span className="eyebrow">{t('ONE SMALL LOCAL SERVICE')}</span>
          <h2>{t('A clear boundary.')}</h2>
          {[
            'Local React workspace and job history',
            'Modal GPU execution and persistent artifacts',
            'Repository-driven models and parameters',
            'Actual billing from the Modal API',
          ].map((s) => (
            <p key={s}>
              <Check size={16} />
              {s}
            </p>
          ))}
          <Notice>
            {t(
              'Keep this service bound to localhost. Multi-user authentication and public hosting are outside this version.',
            )}
          </Notice>
          <small className="mono">
            {t('AReno')} {bootstrap.catalog.revision.slice(0, 7)}
          </small>
        </aside>
      </div>
    </>
  );
}
