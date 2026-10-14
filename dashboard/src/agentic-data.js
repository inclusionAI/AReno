export const valueText = (value) =>
  typeof value === "string" ? value : JSON.stringify(value, null, 2);
export const numeric = (value) =>
  typeof value === "number" && Number.isFinite(value);
export function sampleTrace(events) {
  const turns = new Map(),
    tools = new Map();
  let grader, mask;
  const masks = new Map();
  for (const e of events) {
    const d = e.data || {};
    if (e.kind === "turn" || e.kind === "turn_snapshot") {
      // Exact live snapshots take precedence over retrospective snapshots.
      const prev = turns.get(d.turn);
      if (!prev || e.kind === "turn") turns.set(d.turn, { ...d, time: e.time });
    }
    if (e.kind === "tool") {
      const prev = tools.get(d.tool_id) || {};
      tools.set(d.tool_id, {
        ...prev,
        ...d,
        arguments: d.arguments ?? prev.arguments,
        start: prev.start ?? e.time,
        end: d.status === "running" ? null : e.time,
      });
    }
    if (e.kind === "grader") grader = d;
    if (e.kind === "loss_mask") {
      mask = d;
      masks.set(d.turn ?? 0, d);
    }
  }
  const ordered = [...turns.values()].sort((a, b) => a.turn - b.turn);
  const recordedResults = new Map();
  for (const turn of ordered) {
    for (const message of turn.messages || []) {
      if (message.role === "tool" && message.tool_call_id)
        recordedResults.set(message.tool_call_id, message);
    }
  }
  for (const turn of ordered) {
    for (const call of turn.response?.tool_calls || []) {
      if (!call.id || tools.has(call.id)) continue;
      const result = recordedResults.get(call.id);
      tools.set(call.id, {
        tool_id: call.id,
        name: call.function?.name || "Tool",
        arguments: call.function?.arguments,
        result: result?.content,
        status: result
          ? result.is_error
            ? "failed"
            : "completed"
          : "recorded",
        start: null,
        end: null,
      });
    }
  }
  const seenResults = new Set(),
    assigned = new Set();
  for (const turn of ordered) {
    turn.toolResults = (turn.messages || []).filter((message) => {
      if (message.role !== "tool") return false;
      const key = message.tool_call_id || JSON.stringify(message);
      if (seenResults.has(key)) return false;
      seenResults.add(key);
      return true;
    });
    turn.tools = (turn.response?.tool_calls || [])
      .map((call) => {
        const tool = tools.get(call.id);
        if (tool) assigned.add(call.id);
        return tool;
      })
      .filter(Boolean);
  }
  return {
    turns: ordered,
    tools: [...tools.values()],
    unassignedTools: [...tools.values()].filter(
      (tool) => !assigned.has(tool.tool_id),
    ),
    grader,
    mask,
    masks: [...masks.values()],
  };
}
export function rewardHistogram(samples) {
  const bins = Array.from({ length: 10 }, (_, i) => ({
    label: `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`,
    value: 0,
  }));
  for (const s of samples)
    if (numeric(s.reward) && s.reward >= 0 && s.reward <= 1)
      bins[Math.min(9, Math.floor(s.reward * 10))].value++;
  return bins;
}
export function stageSegments(events) {
  const rows = [];
  for (let i = 0; i < events.length - 1; i++) {
    const e = events[i],
      next = events[i + 1];
    const seconds = next.time - e.time;
    if (
      seconds < 0 ||
      !e.data.stage?.endsWith("_start") ||
      next.step !== e.step
    )
      continue;
    rows.push({
      step: e.step,
      stage: e.data.stage.replace("_start", ""),
      seconds,
      time: e.time,
    });
  }
  return rows;
}
export function messageText(message) {
  if (!message) return "";
  if (typeof message.content === "string") return message.content;
  return (message.content || [])
    .map((c) => c.text || (c.type === "image" ? "[Image]" : `[${c.type}]`))
    .join("\n");
}

export function isAgenticJob(job, samples = []) {
  const configured = (value) =>
    value && !["none", "null", "false"].includes(String(value).toLowerCase());
  return (
    [job.launch?.agent_fn, job.config?.agent_fn, job.config?.agent_run_fn].some(
      configured,
    ) ||
    (job.command || []).includes("--agent-fn") ||
    samples.some((sample) => sample.harness || sample.mode === "agentic")
  );
}

export function rewardVariation(learning) {
  if (!learning || !numeric(learning.varying)) return null;
  const eligible = ["zero", "one", "varying", "flat_other"].reduce(
    (sum, key) => sum + (learning[key] || 0),
    0,
  );
  return eligible ? learning.varying / eligible : null;
}

export function runtimeNotices(job, recorded = []) {
  const notices = recorded.map((event) => ({
    id: event.id,
    step: event.step,
    time: event.time,
    message:
      event.data.message ||
      event.data.stage?.replaceAll("_", " ") ||
      event.data.status,
    source: "Trainer event",
  }));
  for (const line of job.logs || []) {
    if (
      /\b(?:WARNING|ERROR)\b|^(?:Traceback|torch\..*Error|RuntimeError|ValueError)/.test(
        line,
      ) &&
      !line.includes("rollout_completion=")
    )
      notices.push({ id: `log:${line}`, message: line, source: "Job log" });
  }
  if (["failed", "stopped"].includes(job.status))
    notices.push({
      id: `job:${job.status}`,
      message: `Job ${job.status} at ${job.stage || "unreported stage"}`,
      source: "Job status",
    });
  return [...new Map(notices.map((notice) => [notice.id, notice])).values()]
    .slice(-30)
    .reverse();
}

export function sampleSourceLabel(sample) {
  const record = sample.source || {};
  const source = record.source || record;
  if (typeof source === "string") return source;
  return (
    source.dataset ||
    source.dataset_id ||
    source.repo ||
    record.repo ||
    "Unlabeled source"
  );
}

export function assistantText(turn) {
  const text =
    messageText(turn.response) ||
    (turn.messages?.at(-1)?.role === "assistant"
      ? messageText(turn.messages.at(-1))
      : "");
  if (!turn.tools?.length) return text;
  return text
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "")
    .replace(/<\|(?:role_end|im_end|eot_id)\|>/g, "")
    .trim();
}

export function timeWindowSeries(series, seconds, end) {
  if (!numeric(end))
    end = series.reduce(
      (latest, row) =>
        row.points.reduce(
          (time, point) =>
            numeric(point.step) ? Math.max(time, point.step) : time,
          latest,
        ),
      -Infinity,
    );
  return series.map((row) => ({
    ...row,
    points: row.points.filter(
      (point) =>
        numeric(point.step) && point.step >= end - seconds && point.step <= end,
    ),
  }));
}
