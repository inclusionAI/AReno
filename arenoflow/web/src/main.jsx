import LanguageSwitcher from './components/LanguageSwitcher';
import { t, useLanguage } from './i18n';
import React, { useCallback, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  ArrowUpRight,
  Box,
  Code2,
  Database,
  CreditCard,
  GitBranch,
  Radio,
  Settings as SettingsIcon,
} from 'lucide-react';
import { api, money, setSession } from './api';
import { Brand, Notice, Toast } from './components/UI';
import Landing from './components/Landing';
import Workflow from './components/Workflow';
import Library from './components/Library';
import Billing from './components/Billing';
import Settings from './components/Settings';
import { JobDetail, JobList } from './components/Jobs';
import './styles.css';
function App() {
  useLanguage();
  const [route, setRoute] = useState(location.hash.slice(1) || 'home');
  const [bootstrap, setBootstrap] = useState(null),
    [fatal, setFatal] = useState(''),
    [toast, setToast] = useState('');
  const [jobs, setJobs] = useState([]),
    [draft, setDraft] = useState(null);
  const [trainingDraft, setTrainingDraft] = useState(null);
  const [billing, setBilling] = useState(null),
    [billingError, setBillingError] = useState(''),
    [billingLoading, setBillingLoading] = useState(false);
  useEffect(() => {
    const listener = () => {
      const next = location.hash.slice(1) || 'home';
      setRoute(next);
      if (['workflow', 'models'].includes(next)) {
        requestAnimationFrame(() =>
          document.getElementById(next)?.scrollIntoView({
            behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
          }),
        );
      } else window.scrollTo(0, 0);
    };
    window.addEventListener('hashchange', listener);
    return () => window.removeEventListener('hashchange', listener);
  }, []);
  const load = useCallback(async () => {
    try {
      const data = await api('/bootstrap');
      setSession(data.csrf);
      setBootstrap(data);
    } catch (e) {
      setFatal(e.message);
    }
  }, []);
  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    if (!bootstrap) return;
    let cancelled = false,
      timer;
    async function poll() {
      try {
        const rows = await api('/jobs');
        if (!cancelled) setJobs(rows);
      } catch (e) {
        if (!cancelled) setToast(e.message);
      }
      if (!cancelled) timer = setTimeout(poll, 5000);
    }
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [!!bootstrap]);
  const refreshBilling = useCallback(async () => {
    if (!bootstrap?.connected) return;
    setBillingLoading(true);
    try {
      setBilling(await api('/billing'));
      setBillingError('');
    } catch (e) {
      setBillingError(e.message);
    } finally {
      setBillingLoading(false);
    }
  }, [bootstrap?.connected]);
  useEffect(() => {
    if (!bootstrap?.connected) return;
    refreshBilling();
    const id = setInterval(refreshBilling, 15000);
    return () => clearInterval(id);
  }, [refreshBilling, bootstrap?.connected]);
  const launched = (job) => {
    setJobs((old) => [job, ...old]);
    location.hash = `run/${job.id}`;
    setToast(t('Submitted to your Modal workspace.'));
  };
  const deploy = (job) => {
    const stage = job.manifest.stages.at(-1)?.params || {};
    const serve = job.adapter_only
      ? {
          lora_adapter_path: job.checkpoint,
          lora_rank: stage.lora_rank,
          lora_alpha: stage.lora_alpha,
          lora_target_modules: stage.lora_target_modules,
        }
      : {};
    setDraft({
      model: {
        ...job.manifest.model,
        checkpoint: job.adapter_only ? job.manifest.model.checkpoint : job.checkpoint,
      },
      serve,
      name: `${job.name} endpoint`,
    });
    location.hash = 'deploy';
  };
  if (!bootstrap)
    return (
      <div className="initial-load">
        <Brand />
        <Notice error={!!fatal}>{fatal || t('Reading your AReno workspace…')}</Notice>
        {fatal && (
          <button className="button" onClick={load}>
            {t('Retry connection')}
          </button>
        )}
      </div>
    );
  if (route === 'home' || route === 'workflow' || route === 'models')
    return <Landing catalog={bootstrap.catalog} />;
  const navigation = [
    ['workspace', GitBranch, 'Create a flow'],
    ['runs', Activity, 'Training runs'],
    ['datasets', Database, 'Dataset Manager'],
    ['functions', Code2, 'Script Manager'],
    ['deployments', Radio, 'Deployments'],
    ['billing', CreditCard, 'Usage & billing'],
    ['settings', SettingsIcon, 'Settings'],
  ];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Brand compact />
        <span className="sidebar-label">{t('YOUR WORKSPACE')}</span>
        <nav>
          {navigation.map(([href, Icon, label]) => (
            <a
              key={href}
              href={`#${href}`}
              className={
                route === href || (href === 'deployments' && route === 'deploy') ? 'active' : ''
              }
            >
              <Icon size={18} />
              {t(label)}
              {href === 'runs' &&
                jobs.some(
                  (j) => j.kind === 'training' && ['running', 'starting'].includes(j.status),
                ) && <i className="live-dot" />}
            </a>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="connection-status">
            <i className={bootstrap.connected ? 'live-dot' : 'offline-dot'} />
            {t('Modal')} {bootstrap.connected ? t('connected') : t('not connected')}
          </div>
          <a href="https://github.com/inclusionAI/AReno" target="_blank" rel="noreferrer">
            {t('Open-source on GitHub')}
            <ArrowUpRight size={14} />
          </a>
          <small>
            {t('AReno')} {bootstrap.catalog.revision.slice(0, 7)}
          </small>
        </div>
      </aside>
      <div className="workspace">
        <div className="workspace-topbar">
          <LanguageSwitcher />
          <span>
            <Box size={15} /> {t('Local workspace')} <span className="slash">/</span>
            {t('AReno')}
          </span>
          <a href="#billing" className="live-billing">
            <i className={billingError ? 'offline-dot' : 'live-dot'} />
            {t('Modal reported usage')}
            <b>{money(billing?.summary?.metered_cost)}</b>
          </a>
        </div>
        <main>
          {route === 'workspace' && (
            <Workflow
              key="training"
              initial={trainingDraft}
              onDraft={setTrainingDraft}
              bootstrap={bootstrap}
              onLaunched={launched}
              notify={setToast}
            />
          )}
          {route === 'deploy' && (
            <Workflow
              key="deployment"
              bootstrap={bootstrap}
              kind="deployment"
              initial={draft}
              onDraft={setDraft}
              onLaunched={launched}
              notify={setToast}
            />
          )}
          {route === 'runs' && <JobList jobs={jobs} />}
          {['datasets', 'functions'].includes(route) && (
            <Library key={route} type={route} notify={setToast} />
          )}
          {route === 'deployments' && <JobList jobs={jobs} deployments />}
          {route.startsWith('run/') && (
            <JobDetail
              id={route.split('/')[1]}
              onDeploy={deploy}
              notify={setToast}
              connected={bootstrap.connected}
            />
          )}
          {route === 'settings' && (
            <Settings
              bootstrap={bootstrap}
              onConnected={() => {
                load();
                setToast(t('Modal workspace connected.'));
              }}
            />
          )}
          {route === 'billing' && (
            <Billing
              connected={bootstrap.connected}
              data={billing}
              error={billingError}
              loading={billingLoading}
              refresh={refreshBilling}
            />
          )}
          {![
            'workspace',
            'deploy',
            'runs',
            'deployments',
            'settings',
            'billing',
            'datasets',
            'functions',
          ].includes(route) &&
            !route.startsWith('run/') && (
              <Notice>
                {t('Page not found.')}
                <a href="#workspace">{t('Return to workspace')}</a>
              </Notice>
            )}
        </main>
        <footer className="workspace-footer">
          <span>{t('AReno · Local control plane')}</span>
          <span>
            {bootstrap.catalog.models.length} {t('adapters ·')} {bootstrap.catalog.train.length}
            {t('training parameters')}
          </span>
        </footer>
      </div>
      <Toast message={toast} onClose={() => setToast('')} />
    </div>
  );
}
class ErrorBoundary extends React.Component {
  state = {
    error: null,
  };
  static getDerivedStateFromError(error) {
    return {
      error,
    };
  }
  render() {
    return this.state.error ? (
      <div className="initial-load">
        <Brand />
        <Notice error>
          {t('Something went wrong:')} {this.state.error.message}
        </Notice>
        <button className="button" onClick={() => location.reload()}>
          {t('Reload workspace')}
        </button>
      </div>
    ) : (
      this.props.children
    );
  }
}
createRoot(document.getElementById('root')).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>,
);
