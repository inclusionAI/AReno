import test from "node:test";
import assert from "node:assert/strict";
import { sampleTrace } from "./agentic-data.js";
import {
  exportAtif,
  filterMessages,
  snapshotGroups,
  toolInventory,
  trajectoryMessages,
  trajectoryStats,
  trajectorySteps,
} from "./trajectory-data.js";

function fixture() {
  const system = { role: "system", content: "System instruction" },
    user = { role: "user", content: "Fix loader" };
  const response = {
    role: "assistant",
    content: "Inspect",
    reasoning_content: "Hidden reasoning needle",
    tool_calls: [
      {
        id: "call",
        function: { name: "read", arguments: '{"path":"loader.py"}' },
      },
    ],
  };
  return [
    { kind: "sample", time: 0, data: { status: "running" } },
    {
      kind: "turn",
      time: 2,
      data: {
        turn: 0,
        messages: [system, user],
        response,
        seconds: 2,
        input_tokens: 100,
        output_tokens: 12,
      },
    },
    {
      kind: "tool",
      time: 2,
      data: {
        tool_id: "call",
        name: "read",
        arguments: { path: "loader.py" },
        status: "running",
      },
    },
    {
      kind: "tool",
      time: 3,
      data: { tool_id: "call", status: "failed", result: "Permission denied" },
    },
    {
      kind: "turn",
      time: 5,
      data: {
        turn: 1,
        messages: [
          system,
          user,
          response,
          { role: "tool", tool_call_id: "call", content: "Permission denied" },
        ],
        response: { role: "assistant", content: "Report failure" },
        seconds: 2,
        input_tokens: 200,
        output_tokens: 8,
      },
    },
    { kind: "reward", time: 7, data: { status: "completed", reward: 0 } },
  ];
}
test("repeated full histories form one transcript with linked action/observation steps", () => {
  const events = fixture(),
    trace = sampleTrace(events),
    messages = trajectoryMessages(trace, { prompt: "duplicate summary" });
  assert.deepEqual(
    messages.map((message) => message.role),
    ["system", "user", "assistant", "tool", "assistant"],
  );
  assert.equal(messages[3].owner, messages[2].id);
  assert.equal(trajectorySteps(messages).length, 4);
  assert.equal(
    filterMessages(trajectorySteps(messages), "hidden reasoning needle").length,
    1,
  );
  assert.equal(
    filterMessages(
      trajectorySteps(messages),
      "permission denied",
      "assistant",
      "failed",
    ).length,
    1,
  );
  assert.equal(filterMessages(messages, "", "tool").length, 1);
});
test("identical assistant responses and repeated user messages remain distinct actual turns", () => {
  const user = { role: "user", content: "Again" },
    response = { role: "assistant", content: "Same" };
  const trace = sampleTrace([
    { kind: "turn", data: { turn: 0, messages: [user], response } },
    {
      kind: "turn",
      data: { turn: 1, messages: [user, response, user], response },
    },
  ]);
  assert.deepEqual(
    trajectoryMessages(trace).map((message) => message.role),
    ["user", "assistant", "user", "assistant"],
  );
});
test("statistics use timestamp zero, cumulative tokens and complete timing without estimates", () => {
  const events = fixture(),
    trace = sampleTrace(events),
    stats = trajectoryStats(trace, events);
  assert.deepEqual(stats, {
    elapsed: 7,
    llm: 4,
    tools: 1,
    other: 2,
    inputTokens: 300,
    outputTokens: 20,
    rounds: 2,
    toolRounds: 1,
    toolCalls: 1,
  });
  trace.turns[0].seconds = undefined;
  assert.equal(trajectoryStats(trace, events).other, null);
  assert.equal(trajectoryStats(trace, events).llm, null);
  trace.turns[0].seconds = 10;
  assert.equal(trajectoryStats(trace, events).other, null);
});
test("ATIF action references resolve within their own step and raw malformed arguments survive export", () => {
  const events = fixture(),
    trace = sampleTrace(events),
    messages = trajectoryMessages(trace),
    stats = trajectoryStats(trace, events);
  trace.tools[0].arguments = "broken-json";
  const atif = exportAtif(messages, trace, { id: "sample", reward: 0 }, stats);
  assert.equal(atif.schema_version, "ATIF-v1.7");
  assert.deepEqual(
    atif.steps.map((step) => step.source),
    ["system", "user", "agent", "agent"],
  );
  const action = atif.steps[2];
  assert.equal(
    action.observation.results[0].source_call_id,
    action.tool_calls[0].tool_call_id,
  );
  assert.equal(action.tool_calls[0].extra.raw_arguments, "broken-json");
  assert.equal(action.metrics.prompt_tokens, 100);
  assert.equal(atif.extra.reward, 0);
  assert.equal(atif.final_metrics.total_steps, atif.steps.length);
});
test("snapshot filename search preserves step/task/sample identities and zero rewards", () => {
  const samples = [
    {
      id: "a",
      step: 2,
      prompt_idx: 0,
      sample_idx: 0,
      source: { instance_id: "task-a" },
      reward: 0,
    },
    {
      id: "b",
      step: 2,
      prompt_idx: 0,
      sample_idx: 1,
      source: { instance_id: "task-a" },
    },
    {
      id: "c",
      step: 3,
      prompt_idx: 0,
      sample_idx: 0,
      source: { instance_id: "task-b" },
    },
  ];
  const groups = snapshotGroups(samples, "trajectory_1.json");
  assert.equal(groups.length, 1);
  assert.equal(groups[0].tasks[0].samples[0].id, "b");
  assert.equal(snapshotGroups(samples)[0].tasks[0].samples[0].reward, 0);
  assert.equal(snapshotGroups(samples, "task-b")[0].step, 3);
});
test("tool inventory uses captured definitions and labels schema-free historical calls", () => {
  const events = fixture(),
    trace = sampleTrace(events);
  assert.equal(toolInventory(events, trace)[0].schema_recorded, false);
  const definition = {
    type: "function",
    function: {
      name: "read",
      parameters: { type: "object", properties: { path: { type: "string" } } },
    },
  };
  events.push({ kind: "turn", data: { tool_definitions: [definition] } });
  assert.deepEqual(toolInventory(events, trace), [definition]);
});
