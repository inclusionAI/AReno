import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeConfigSections, formatConfigValue } from './config-display.js';

test('Modal stages display separate parameter sections without losing false or zero', () => {
  const sections = normalizeConfigSections({ kind: 'training', model: { checkpoint: 'org/model' }, stages: [
    { algo: 'sft', params: { dataset_loader_fn: 'loader.py', adam_4bit: true, weight_decay: 0, offload: false } },
    { algo: 'grpo', params: { n_samples: 4 } },
  ] });
  assert.deepEqual(sections.map(s => s.title), ['Launch', 'Model', 'Stage 1 · SFT', 'Stage 2 · GRPO']);
  const params = Object.fromEntries(sections[2].items.map(({ key, value }) => [key, value]));
  assert.equal(params.dataset_loader_fn, 'loader.py');
  assert.equal(params.offload, false);
  assert.equal(params.weight_decay, 0);
  assert.ok(!sections.flatMap(s => s.items).some(item => item.key === 'stages'));
});

test('nested arrays render values instead of implicit object strings', () => {
  const text = formatConfigValue([{ algo: 'sft', params: { lr: 0.001 } }, { algo: 'grpo' }]);
  assert.ok(text.includes('"lr":0.001'));
  assert.ok(!text.includes('[object Object]'));
  assert.equal(formatConfigValue([1, 2]), '1 · 2');
});

test('ordinary and preformatted local configurations retain their sections', () => {
  assert.deepEqual(normalizeConfigSections({ algo: 'sft', unused: null }), [{ title: 'Launch', items: [{ key: 'algo', value: 'sft' }] }]);
  assert.deepEqual(normalizeConfigSections({ sections: [{ title: 'Train', items: [{ key: 'steps', value: 0 }] }] }), [{ title: 'Train', items: [{ key: 'steps', value: 0 }] }]);
});
