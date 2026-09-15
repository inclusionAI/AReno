import LanguageSwitcher from './LanguageSwitcher';
import { t } from '../i18n';
import { useEffect, useRef } from 'react';
import {
  ArrowRight,
  ArrowUpRight,
  GitBranch,
  Activity,
  Box,
  Check,
  Cpu,
  Layers,
  Radio,
  Terminal,
} from 'lucide-react';
import { Brand, Button, Badge, External } from './UI';
import { Chart } from './Chart';
const illustrative = Array.from(
  {
    length: 35,
  },
  (_, i) => ({
    step: i * 3,
    value: 1.9 * Math.exp(-i / 12) + 0.22 + Math.sin(i * 1.8) * 0.04,
  }),
);
export default function Landing({ catalog }) {
  const root = useRef();
  useEffect(() => {
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let frame;
    const scroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() =>
        root.current?.style.setProperty(
          '--scroll',
          reduced.matches ? 0 : Math.min(window.scrollY, 1600),
        ),
      );
    };
    window.addEventListener('scroll', scroll, {
      passive: true,
    });
    reduced.addEventListener('change', scroll);
    scroll();
    return () => {
      window.removeEventListener('scroll', scroll);
      reduced.removeEventListener('change', scroll);
      cancelAnimationFrame(frame);
    };
  }, []);
  return (
    <div className="landing" ref={root}>
      <nav className="home-nav">
        <LanguageSwitcher />
        <Brand />
        <div className="home-links">
          <a href="#workflow">{t('Workflow')}</a>
          <a href="#models">{t('Models')}</a>
          <External href="https://github.com/inclusionAI/AReno">{t('GitHub')}</External>
        </div>
        <a className="button dark" href="#workspace">
          {t('Open workspace')}
          <ArrowUpRight size={15} />
        </a>
      </nav>
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">
            <i className="live-dot" />
            {t('AReno training and deployment')}
          </span>
          <h1>
            {t('Model training')}
            <br />
            <em>{t('Deployment')}</em>
          </h1>
          <p>
            {t('Configure training workflows and deploy model checkpoints.')}
            <br className="desktop" />
            {t('Training and inference run on Modal using AReno containers.')}
          </p>
          <div className="hero-actions">
            <a href="#workspace" className="button primary">
              {t('Create training workflow')}
              <ArrowRight size={17} />
            </a>
            <a href="#workflow" className="text-button">
              {t('View workflow overview')}
              <span>↓</span>
            </a>
          </div>
          <div className="hero-footnote">
            <Terminal size={14} /> {t('Run locally')} <span>·</span> {t('Train in the cloud')}{' '}
            <span>·</span>
            {t('Local configuration')}
          </div>
        </div>
        <div
          className="hero-scene"
          aria-label={t('Illustration of a training workflow, not a live run')}
        >
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="scene-label">
            <span>{t('Workflow example')}</span>
            <span>01 — 03</span>
          </div>
          <div className="float-card flow-card">
            <div className="card-kicker">
              <GitBranch size={15} /> {t('math-reasoning-flow')}{' '}
              <span className="demo-label">{t('Illustration')}</span>
            </div>
            <div className="mini-flow">
              <div>
                <Box />
                <b>{t('Base model')}</b>
                <small>{t('Initial checkpoint')}</small>
              </div>
              <ArrowRight size={14} />
              <div className="selected">
                <Layers />
                <b>{t('SFT')}</b>
                <small>{t('Supervised fine-tuning')}</small>
              </div>
              <ArrowRight size={14} />
              <div>
                <Activity />
                <b>{t('GSPO')}</b>
                <small>{t('Reinforcement learning')}</small>
              </div>
            </div>
            <div className="card-footer">
              <span>
                <i className="live-dot" />
                {t('Sequential training stages')}
              </span>
              <span>{t('Modal GPU ↗')}</span>
            </div>
          </div>
          <div className="float-card metric-card">
            <div className="card-kicker">
              {t('TRAINING LOSS')}
              <Activity size={14} />
            </div>
            <div className="metric-number">
              0.34 <span>{t('illustrative curve')}</span>
            </div>
            <Chart points={illustrative} label={t('Illustrative training loss')} demo />
          </div>
          <div className="float-card endpoint-card">
            <div className="endpoint-icon">
              <Radio size={22} />
            </div>
            <div>
              <span className="eyebrow">{t('Inference API')}</span>
              <b>{t('Model endpoint')}</b>
              <code>/v1/chat/completions</code>
            </div>
            <ArrowUpRight size={18} />
          </div>
          <div className="scene-coordinate">{t('CHECKPOINT → SFT → GSPO')}</div>
        </div>
      </section>
      <div className="capability-strip">
        <span>{t('Supported algorithms')}</span>
        {['SFT', 'DPO', 'GRPO', 'GSPO', 'PPO'].map((s) => (
          <span key={s}>{s}</span>
        ))}
        <span>
          {t('OpenAI-compatible serving')}
          <ArrowUpRight size={14} />
        </span>
      </div>
      <section id="workflow" className="story-section">
        <div className="story-intro">
          <span className="eyebrow">{t('Workflow orchestration')}</span>
          <h2>
            {t('Multi-stage training')}
            <br />
            {t('Checkpoint transfer')}
          </h2>
          <p>
            {t(
              'Configure sequential training stages. Each stage uses the checkpoint produced by the previous stage.',
            )}
          </p>
          <a href="#workspace" className="text-button">
            {t('Create a workflow')}
            <ArrowRight size={16} />
          </a>
        </div>
        <div className="story-steps">
          {[
            [
              '01',
              'Select a model and configure parameters',
              'Model adapters are read from the AReno registry. Algorithm presets provide initial values; applicable training parameters can be edited.',
              Box,
            ],
            [
              '02',
              'Monitor training metrics and logs',
              'Run SFT and reinforcement learning stages sequentially. View reported training metrics and sandbox logs.',
              Activity,
            ],
            [
              '03',
              'Deploy a model checkpoint',
              'Deploy a checkpoint as an authenticated API endpoint. Inspect endpoint status and workspace billing reported by Modal.',
              Radio,
            ],
          ].map(([number, title, description, Icon]) => (
            <article key={number}>
              <span className="step-number">{number}</span>
              <div>
                <Icon size={21} />
                <h3>{t(title)}</h3>
                <p>{t(description)}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
      <section className="model-section" id="models">
        <div>
          <span className="eyebrow">{t('Model registry')}</span>
          <h2>{t('Supported models')}</h2>
          <p>{t('Available model adapters are read from the current AReno checkout.')}</p>
        </div>
        <div className="model-tags">
          {[...new Set(catalog.models.map((m) => m.family))].map((f) => (
            <span key={f}>
              <Box size={16} />
              {f.replaceAll('_', '.')}
            </span>
          ))}
        </div>
        <div className="source-note">
          <GitBranch size={14} /> {catalog.revision.slice(0, 7)} <span>·</span>{' '}
          {catalog.models.length} {t('registered adapters')} <span>·</span> {catalog.train.length}
          {t('training controls')}
        </div>
      </section>
      <section className="closing">
        <span className="eyebrow">{t('Local UI · Modal compute')}</span>
        <h2>
          {t('Configure a')}
          <br />
          <em>{t('Training workflow')}</em>
        </h2>
        <a href="#workspace" className="button primary">
          {t('Open AReno')}
          <ArrowRight size={17} />
        </a>
      </section>
      <footer>
        <Brand compact />
        <span>{t('AReno training and inference interface')}</span>
        <External href="https://github.com/inclusionAI/AReno">{t('Source on GitHub')}</External>
      </footer>
    </div>
  );
}
