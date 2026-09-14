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
            {t('OPEN SOURCE. YOUR WORKSPACE. YOUR MODELS.')}
          </span>
          <h1>
            {t('From model')}
            <br />
            {t('to')} <em>{t('momentum.')}</em>
          </h1>
          <p>
            {t('A clear path from your first fine-tune to a live endpoint.')}
            <br className="desktop" />
            {t('Built on AReno. Powered by your Modal workspace.')}
          </p>
          <div className="hero-actions">
            <a href="#workspace" className="button primary">
              {t('Build your first flow')}
              <ArrowRight size={17} />
            </a>
            <a href="#workflow" className="text-button">
              {t('Explore the workflow')}
              <span>↓</span>
            </a>
          </div>
          <div className="hero-footnote">
            <Terminal size={14} /> {t('Run locally')} <span>·</span> {t('Train in the cloud')}{' '}
            <span>·</span>
            {t('Keep control')}
          </div>
        </div>
        <div
          className="hero-scene"
          aria-label={t('Illustration of a training workflow, not a live run')}
        >
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="scene-label">
            <span>{t('YOUR NEXT MODEL')}</span>
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
                <small>{t('Start with AReno')}</small>
              </div>
              <ArrowRight size={14} />
              <div className="selected">
                <Layers />
                <b>{t('SFT')}</b>
                <small>{t('Teach the task')}</small>
              </div>
              <ArrowRight size={14} />
              <div>
                <Activity />
                <b>{t('GSPO')}</b>
                <small>{t('Refine reasoning')}</small>
              </div>
            </div>
            <div className="card-footer">
              <span>
                <i className="live-dot" />
                {t('A connected training journey')}
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
              <span className="eyebrow">{t('NEXT STOP')}</span>
              <b>{t('Your model, deployed.')}</b>
              <code>/v1/chat/completions</code>
            </div>
            <ArrowUpRight size={18} />
          </div>
          <div className="scene-coordinate">{t('WEIGHTS → KNOWLEDGE → POSSIBILITY')}</div>
        </div>
      </section>
      <div className="capability-strip">
        <span>{t('One native training stack')}</span>
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
          <span className="eyebrow">{t('LESS GLUE CODE. MORE PROGRESS.')}</span>
          <h2>
            {t('Every step.')}
            <br />
            {t('In the same flow.')}
          </h2>
          <p>
            {t(
              'Choose a model, teach it your task, and put it to work. The checkpoint connects the stages. You stay in control.',
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
              'Start with a good foundation.',
              'Models discovered from the AReno registry. Practical presets, with the complete training surface one click away.',
              Box,
            ],
            [
              '02',
              'Make the learning visible.',
              'Compose SFT and reinforcement learning stages. Follow actual training metrics and logs as the work happens.',
              Activity,
            ],
            [
              '03',
              'Put your model to work.',
              'Launch a protected API from a checkpoint. See endpoint status and pull reported costs directly from Modal.',
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
          <span className="eyebrow">{t('IN SYNC WITH THE SOURCE')}</span>
          <h2>{t('Your toolkit keeps growing.')}</h2>
          <p>
            {t('The model catalog comes from this AReno checkout. No separate list to maintain.')}
          </p>
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
        <span className="eyebrow">{t('LOCAL CONTROL. CLOUD COMPUTE.')}</span>
        <h2>
          {t('Make your next')}
          <br />
          <em>{t('model yours.')}</em>
        </h2>
        <a href="#workspace" className="button primary">
          {t('Open AReno')}
          <ArrowRight size={17} />
        </a>
      </section>
      <footer>
        <Brand compact />
        <span>{t('Built with AReno. Open by design.')}</span>
        <External href="https://github.com/inclusionAI/AReno">{t('Source on GitHub')}</External>
      </footer>
    </div>
  );
}
