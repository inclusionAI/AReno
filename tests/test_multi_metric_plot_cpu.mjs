// CPU tests for the multi-metric plotting pure helpers (issue #265).
//
// These exercise the dependency-free functions embedded in
// dashboard/src/main.jsx (downsampleLttb, normalizeSeries, assignAxes,
// buildMultiMetricPlot) WITHOUT a DOM or React. They are plain node asserts.
//
// Run:  node tests/test_multi_metric_plot_cpu.mjs
//
// Covers: thousands-of-points downsampling, differently scaled series,
// normalization toggles, deterministic output, and a boundary/failure path.

import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const SRC = new URL("../dashboard/src/main.jsx", import.meta.url);
const src = readFileSync(SRC, "utf8");

// Extract the pure-helper block: smoothTensorboard, buildMetricPlot,
// compactNumber, and all multi-metric helpers down to TimePerfView. The
// extracted module is written to the OS temp dir (never the repo) so the test
// does not add build artifacts to the tree.
const start = src.indexOf("function smoothTensorboard");
const end = src.indexOf("function TimePerfView");
const block = src.slice(start, end);
const tmp = join(tmpdir(), `_areno_multi_metric_helpers_${process.pid}.mjs`);
writeFileSync(tmp, `${block}\nexport { downsampleLttb, normalizeSeries, assignAxes, buildMultiMetricPlot, metricColor, METRIC_PALETTE };\n`);

const { downsampleLttb, normalizeSeries, assignAxes, buildMultiMetricPlot, metricColor, METRIC_PALETTE } = await import(tmp);

const mk = (vals) => vals.map((v, i) => ({ step: i, value: v }));
let passed = 0;
function check(name, fn) {
  fn();
  passed += 1;
  console.log(`ok ${name}`);
}

// --- downsampleLttb ---
check("lttb caps to target length", () => {
  const pts = mk(Array.from({ length: 2000 }, (_, i) => Math.sin(i / 50)));
  const out = downsampleLttb(pts, 480);
  assert.equal(out.length, 480, "should downsample to exactly target");
});

check("lttb preserves first and last points", () => {
  const pts = mk(Array.from({ length: 1000 }, (_, i) => i));
  const out = downsampleLttb(pts, 100);
  assert.equal(out[0], pts[0]);
  assert.equal(out[out.length - 1], pts[pts.length - 1]);
});

check("lttb does not mutate input", () => {
  const pts = mk([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const snapshot = pts.map((p) => ({ ...p }));
  downsampleLttb(pts, 5);
  assert.deepEqual(pts, snapshot);
});

check("lttb passthrough when already small", () => {
  const pts = mk([1, 2, 3]);
  assert.equal(downsampleLttb(pts, 480), pts, "returns same ref when n <= target");
});

check("lttb passthrough for invalid target", () => {
  const pts = mk([1, 2, 3, 4, 5]);
  assert.equal(downsampleLttb(pts, 1), pts);
  assert.equal(downsampleLttb(pts, 0), pts);
});

// --- normalizeSeries ---
check("normalize scales to [0,1]", () => {
  const n = normalizeSeries(mk([0, 5, 10]));
  assert.equal(n[0].value, 0);
  assert.equal(n[2].value, 1);
});

check("normalize preserves rawValue and does not mutate input", () => {
  const pts = mk([2, 4, 6]);
  const n = normalizeSeries(pts);
  assert.equal(n[0].rawValue, 2);
  assert.equal(n[2].rawValue, 6);
  assert.equal(pts[0].value, 2, "input untouched");
});

check("normalize handles constant series without divide-by-zero", () => {
  const n = normalizeSeries(mk([3, 3, 3]));
  assert.ok(Number.isFinite(n[0].value));
});

// --- assignAxes (differently scaled series) ---
check("different scales split across two axes", () => {
  const series = { a: mk([1, 2, 3]), b: mk([0.0001, 0.0002, 0.0003]) };
  const axes = assignAxes(series, ["a", "b"]);
  assert.equal(axes.get("a"), 0, "first series -> left");
  assert.equal(axes.get("b"), 1, "tiny-scale series -> right");
});

check("same-scale series share the left axis", () => {
  const series = { a: mk([10, 20, 30]), b: mk([12, 18, 27]) };
  const axes = assignAxes(series, ["a", "b"]);
  assert.equal(axes.get("a"), 0);
  assert.equal(axes.get("b"), 0);
});

check("single series uses left axis only", () => {
  const axes = assignAxes({ a: mk([1, 2, 3]) }, ["a"]);
  assert.equal(axes.size, 1);
  assert.equal(axes.get("a"), 0);
});

// --- buildMultiMetricPlot (deterministic, view-only) ---
check("plot output is deterministic across repeated calls", () => {
  const series = { "train/loss": mk([2, 1.5, 1, 0.7]), "train/reward": mk([-1, 0, 2, 4]) };
  const axes = assignAxes(series, ["train/loss", "train/reward"]);
  const a = buildMultiMetricPlot(series, ["train/loss", "train/reward"], { normalize: false, axes, smooth: 0.6 });
  const b = buildMultiMetricPlot(series, ["train/loss", "train/reward"], { normalize: false, axes, smooth: 0.6 });
  assert.deepEqual(a, b);
});

check("plot reports raw min/max labels even when normalized", () => {
  // View-only normalization: labels must reflect the un-modified stored values.
  const series = { "train/loss": mk([2, 4, 6]) };
  const axes = assignAxes(series, ["train/loss"]);
  const [plot] = buildMultiMetricPlot(series, ["train/loss"], { normalize: true, axes, smooth: 0 });
  assert.equal(plot.minLabel, "2");
  assert.equal(plot.maxLabel, "6");
});

check("plot produces rawPoly and smoothPoly strings", () => {
  const series = { "train/loss": mk([1, 2, 3, 4]) };
  const axes = assignAxes(series, ["train/loss"]);
  const [plot] = buildMultiMetricPlot(series, ["train/loss"], { normalize: false, axes, smooth: 0.5 });
  assert.equal(typeof plot.rawPoly, "string");
  assert.equal(typeof plot.smoothPoly, "string");
  assert.ok(plot.rawPoly.includes(","));
  assert.ok(plot.smoothPoly.includes(","));
});

// --- boundary / failure path ---
check("empty order yields empty plot list", () => {
  const axes = assignAxes({}, []);
  const plots = buildMultiMetricPlot({}, [], { normalize: false, axes, smooth: 0.6 });
  assert.equal(plots.length, 0);
});

check("palette is colorblind-safe sized and cyclic", () => {
  assert.ok(METRIC_PALETTE.length >= 8);
  assert.equal(metricColor(0), metricColor(METRIC_PALETTE.length), "wraps around");
});

console.log(`\n${passed} multi-metric plot checks passed`);