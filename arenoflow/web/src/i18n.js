import { useSyncExternalStore } from 'react';
import zh from './locales/zh.js';

const storageKey = 'arenoflow.language';
const listeners = new Set();
function initialLanguage() {
  try {
    const saved = globalThis.localStorage?.getItem(storageKey);
    if (['en', 'zh'].includes(saved)) return saved;
  } catch {
    /* Private browsing may disable storage. */
  }
  return globalThis.navigator?.language?.startsWith('zh') ? 'zh' : 'en';
}
let language = initialLanguage();
export const getLanguage = () => language;
export const getLocale = () => (language === 'zh' ? 'zh-CN' : 'en-US');
export function t(key, values = {}) {
  if (typeof key !== 'string') return key;
  const message = language === 'zh' ? (zh[key] ?? key) : key;
  return message.replace(/\{(\w+)\}/g, (match, name) =>
    Object.hasOwn(values, name) ? String(values[name]) : match,
  );
}
function updateDocument() {
  if (!globalThis.document) return;
  document.documentElement.lang = getLocale();
  document.title = language === 'zh' ? t('Page title') : 'AReno — From model to momentum';
}
export function setLanguage(next) {
  if (!['en', 'zh'].includes(next) || next === language) return;
  language = next;
  try {
    globalThis.localStorage?.setItem(storageKey, next);
  } catch {
    /* Session-only fallback. */
  }
  updateDocument();
  listeners.forEach((listener) => listener());
}
function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
export const useLanguage = () => useSyncExternalStore(subscribe, getLanguage, getLanguage);
updateDocument();
