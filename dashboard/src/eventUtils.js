/**
 * Pure utility functions for training-event overlay on metric charts.
 *
 * This file contains NO JSX and uses ES module exports so it can be loaded
 * via `node --input-type=module` for smoke testing without a build step.
 */

// --- Event kind metadata -------------------------------------------------

export const EVENT_KIND_META = {
  non_finite: { label: "Non-finite", color: "#ef4444", shape: "cross" },
  constant_reward: { label: "Constant reward", color: "#f59e0b", shape: "diamond" },
  invalid_batch: { label: "Invalid batch", color: "#8b5cf6", shape: "square" },
  oom: { label: "OOM", color: "#dc2626", shape: "triangle" },
};

export const ALL_EVENT_KINDS = Object.keys(EVENT_KIND_META);

// --- Pure utility functions ----------------------------------------------

/**
 * Map an event's step to an x-coordinate using the same projection as
 * buildMetricPlot in main.jsx.
 *
 * @param {object} event     - Event object with a `step` field.
 * @param {object} plotRange - { stepMin, stepSpan, width } from the active chart.
 * @returns {number} x position in chart pixels.
 */
export function eventStepToX(event, plotRange) {
  if (!event || !plotRange) return 0;
  const step = Number(event.step || 0);
  const { stepMin = 0, stepSpan = 1, width = 700 } = plotRange;
  return ((step - stepMin) / Math.max(stepSpan, 1)) * width + 10;
}

/**
 * Filter events by enabled kinds.
 *
 * @param {Array}  events       - List of event objects.
 * @param {object} enabledTypes - { [kind]: boolean } map.
 * @returns {Array} Filtered events.
 */
export function filterEvents(events, enabledTypes) {
  if (!Array.isArray(events)) return [];
  return events.filter((e) => {
    const kind = e && e.kind;
    return kind && enabledTypes[kind] !== false;
  });
}

/**
 * Extract a bounded window of log lines around a given index for OOM
 * click-through.
 *
 * @param {Array}  logs   - Array of log line strings.
 * @param {number} index  - Centre line index.
 * @param {number} radius - Lines before/after (default 20).
 * @returns {Array} Sliced log lines with relative offsets.
 */
export function extractLogWindow(logs, index, radius = 20) {
  if (!Array.isArray(logs) || index < 0 || index >= logs.length) return [];
  const start = Math.max(0, index - radius);
  const end = Math.min(logs.length, index + radius + 1);
  const result = [];
  for (let i = start; i < end; i++) {
    result.push({ lineIndex: i, offset: i - index, text: logs[i] });
  }
  return result;
}

/**
 * Extract metric values around a given step for metric-context click-through.
 *
 * @param {Array}  points - Metric series points [{ step, value }, ...].
 * @param {number} step   - Centre step.
 * @param {number} radius - Steps before/after (default 3).
 * @returns {Array} Nearby points sorted by step.
 */
export function extractMetricContext(points, step, radius = 3) {
  if (!Array.isArray(points)) return [];
  return points
    .filter((p) => Math.abs(Number(p.step || 0) - step) <= radius)
    .sort((a, b) => Number(a.step) - Number(b.step));
}
