import { t } from '../i18n';
import { useId } from 'react';
export function Chart({ points = [], label = 'Metric', demo = false }) {
  const id = useId().replaceAll(':', '');
  const data = points.filter((p) => Number.isFinite(p.value)).slice(-200);
  if (!data.length)
    return (
      <div className="chart-empty">
        {t('Waiting for')} {label.toLowerCase()} {t('samples')}
      </div>
    );
  const values = data.map((p) => p.value),
    low = Math.min(...values),
    high = Math.max(...values);
  const span = high - low || Math.abs(high) * 0.1 || 1;
  const path = data
    .map(
      (p, i) =>
        `${i ? 'L' : 'M'}${36 + (i / Math.max(1, data.length - 1)) * 480},${150 - ((p.value - low) / span) * 110}`,
    )
    .join(' ');
  return (
    <div className="chart-wrap">
      <svg
        viewBox="0 0 540 185"
        role="img"
        aria-label={t('{p0}, {p1} samples, latest {p2}', {
          p0: label,
          p1: data.length,
          p2: values.at(-1),
        })}
      >
        <defs>
          <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="currentColor" stopOpacity=".14" />
            <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[40, 95, 150].map((y) => (
          <line key={y} x1="36" x2="516" y1={y} y2={y} className="gridline" />
        ))}
        <path d={`${path} L516,150 L36,150 Z`} fill={`url(#${id})`} />
        <path d={path} fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinejoin="round" />
        <text x="36" y="178">
          {demo ? t('Step 0') : data[0].step}
        </text>
        <text x="478" y="178">
          {demo ? '100' : data.at(-1).step}
        </text>
        <text x="4" y="42">
          {high.toPrecision(2)}
        </text>
        <text x="4" y="153">
          {low.toPrecision(2)}
        </text>
      </svg>
    </div>
  );
}
