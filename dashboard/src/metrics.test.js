import test from "node:test";
import assert from "node:assert/strict";
import { sampleMetricPoints } from "./metrics.js";

test("large chart retains endpoints and spikes with bounded rendering", () => {
  const points = Array.from({ length: 250000 }, (_, step) => ({ step, value: Math.sin(step) }));
  points[12345].value = 1000;
  points[23456].value = -1000;
  const sampled = sampleMetricPoints(points);
  assert.ok(sampled.length <= 1200);
  assert.equal(sampled[0], points[0]);
  assert.equal(sampled.at(-1), points.at(-1));
  assert.ok(sampled.includes(points[12345]));
  assert.ok(sampled.includes(points[23456]));
  assert.ok(sampled.every((point, index) => index === 0 || point.step > sampled[index - 1].step));
  assert.equal(points.length, 250000);
});

test("small and flat series remain visible", () => {
  const small = [{ step: 0, value: 1 }];
  assert.equal(sampleMetricPoints(small), small);
  const flat = Array.from({ length: 100000 }, (_, step) => ({ step, value: 1 }));
  const sampled = sampleMetricPoints(flat);
  assert.ok(sampled.length > 1 && sampled.length <= 1200);
  assert.equal(sampled.at(-1).step, 99999);
});
