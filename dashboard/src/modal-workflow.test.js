import test from 'node:test';
import assert from 'node:assert/strict';
import { modalWorkflow } from './modal-workflow.js';
const settings = { adapter: 'qwen3', gpu: 'H100', count: 2, cpu: 4, memory_gib: 32, duration_hours: 2 };
const catalog = { train: [{ name: 'ckpt' }, { name: 'world_size' }, { name: 'attn_backend' }, { name: 'adam_4bit', type: 'bool' }, { name: 'save_path' }, { name: 'metrics_log_dir' }, { name: 'gspo_clip_eps', algorithms: ['gspo'] }], serve: [{ name: 'model_path' }, { name: 'world_size' }, { name: 'attn_backend' }] };
test('shared train configuration maps to Modal without changing the form state', () => {
 const config = { algo: 'sft', ckpt: 'org/model', world_size: 2, attn_backend: 'flash', adam_4bit: true, save_path: 'outputs/train', metrics_dir: 'outputs/metrics', gspo_clip_eps: 0.2 };
 const request = modalWorkflow('train', config, catalog, settings);
 assert.equal(request.stages[0].params.world_size, 2);
 assert.equal(request.stages[0].params.adam_4bit, true);
 assert.equal(request.stages[0].params.attn_backend, 'flash');
 assert.equal(request.stages[0].params.save_path, '/artifacts/outputs/train');
 assert.equal(request.stages[0].params.gspo_clip_eps, undefined);
 assert.equal(config.save_path, 'outputs/train');
 assert.equal(request.resources.timeout_seconds, 7200);
});
test('shared serve form produces a deployment with the same checkpoint', () => {
 const request = modalWorkflow('serve', { model_path: 'org/model', world_size: 2, attn_backend: 'flash' }, catalog, settings);
 assert.equal(request.kind, 'deployment');
 assert.equal(request.serve.model_path, 'org/model');
 assert.equal(request.model.checkpoint, 'org/model');
});
