import { getLocale, t } from './i18n';
let csrf = '';
export function setSession(value) {
  csrf = value;
}
export async function api(path, body, signal) {
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers:
      body === undefined ? {} : { 'Content-Type': 'application/json', 'X-Arenoflow-CSRF': csrf },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
export const money = (value) =>
  value == null
    ? '—'
    : new Intl.NumberFormat(getLocale(), {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 4,
      }).format(Number(value));
export const time = (value) =>
  value
    ? new Date(typeof value === 'number' ? value * 1000 : value).toLocaleString(getLocale())
    : '—';
export function duration(start, end) {
  if (!start) return t('Not started');
  const seconds = Math.max(0, Math.floor((end || Date.now() / 1000) - start));
  if (getLocale() === 'zh-CN')
    return `${Math.floor(seconds / 3600)} 小时 ${Math.floor((seconds % 3600) / 60)} 分 ${seconds % 60} 秒`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m ${seconds % 60}s`;
}
