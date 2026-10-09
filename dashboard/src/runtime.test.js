import test from "node:test";
import assert from "node:assert/strict";
import { runtimeSummary, runtimeFacts } from "./runtime.js";

const mlx = {
  metal: { available: true, error: null },
  dependencies: {
    mlx: { version: "0.32.3" },
    mlx_lm: { version: "0.32.0" },
    mlx_vlm: { version: "0.7.6" },
  },
};

test("MLX runtime displays Metal without requiring Torch or CUDA fields", () => {
  assert.deepEqual(runtimeSummary(mlx), {
    label: "MLX / Metal", value: "0.32.3 / Metal", detail: "Metal runtime available", tone: "ok",
  });
  const facts = Object.fromEntries(runtimeFacts(mlx));
  assert.equal(facts["MLX-LM"], "0.32.0");
  assert.equal(facts["MLX-VLM"], "0.7.6");
  assert.equal(facts["Metal available"], true);
  assert.ok(!JSON.stringify(facts).match(/CUDA|PyTorch|NVIDIA|NVCC/));
});

test("Metal failures remain visible, including false availability", () => {
  const report = { metal: { available: false, error: "Metal probe failed" } };
  assert.equal(runtimeSummary(report).tone, "warn");
  assert.equal(runtimeSummary(report).detail, "Metal probe failed");
  assert.equal(Object.fromEntries(runtimeFacts(report))["Metal available"], false);
  assert.equal(Object.fromEntries(runtimeFacts(report))["Metal error"], "Metal probe failed");
});

test("CUDA runtime versions and failure states are preserved", () => {
  const report = { torch: { version: "2.6.0", cuda_build: "12.4", cuda_available: true } };
  assert.deepEqual(runtimeSummary(report), {
    label: "PyTorch / CUDA", value: "2.6.0 / 12.4", detail: "Compatible runtime detected", tone: "ok",
  });
  assert.equal(Object.fromEntries(runtimeFacts(report))["CUDA build"], "12.4");
  report.torch.cuda_available = false;
  assert.equal(runtimeSummary(report).tone, "warn");
  assert.equal(runtimeSummary(report).detail, "CUDA runtime unavailable");
  assert.equal(Object.fromEntries(runtimeFacts(report))["CUDA available"], false);
});
