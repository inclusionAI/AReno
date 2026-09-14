import { test } from 'node:test';
import assert from 'node:assert/strict';
import { getLocale, setLanguage, t } from './i18n.js';
import zh from './locales/zh.js';

test('switches UI strings while preserving unknown identifiers and interpolation values', () => {
  setLanguage('zh');
  assert.equal(getLocale(), 'zh-CN');
  assert.equal(t('Dataset Manager'), '数据集管理');
  assert.equal(t('Qwen/Qwen3-0.6B'), 'Qwen/Qwen3-0.6B');
  assert.equal(t('Remove {p0}', { p0: 'my dataset {p1}' }), '移除 my dataset {p1}');
  setLanguage('en');
  assert.equal(t('Dataset Manager'), 'Dataset Manager');
  setLanguage('invalid');
  assert.equal(getLocale(), 'en-US');
});

test('all translations retain their interpolation placeholders', () => {
  const placeholders = (value) => [...value.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
  for (const [source, translation] of Object.entries(zh)) {
    assert.deepEqual(placeholders(translation), placeholders(source), source);
  }
});
