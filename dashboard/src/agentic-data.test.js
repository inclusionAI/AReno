import test from "node:test";
import assert from "node:assert/strict";
import {
  sampleTrace,
  rewardHistogram,
  stageSegments,
  matchingTurns,
  prettyArguments,
  turnLabel,
} from "./agentic-data.js";
test("live exact turns prevail over delayed snapshots and tool durations merge", () => {
  const events = [
    {
      kind: "turn",
      time: 1,
      data: { turn: 0, output_tokens: 0, response: { content: "exact" } },
    },
    {
      kind: "turn_snapshot",
      time: 3,
      data: { turn: 0, response: { content: "snapshot" } },
    },
    {
      kind: "tool",
      time: 1,
      data: {
        tool_id: "x",
        name: "edit",
        arguments: { path: "a.py" },
        status: "running",
      },
    },
    {
      kind: "tool",
      time: 4,
      data: { tool_id: "x", status: "failed", result: "failure" },
    },
  ];
  const data = sampleTrace(events);
  assert.equal(data.turns.length, 1);
  assert.equal(data.turns[0].response.content, "exact");
  assert.equal(data.tools[0].end - data.tools[0].start, 3);
  assert.equal(data.tools[0].arguments.path, "a.py");
});
test("zero and perfect reward are counted while missing and unbounded rewards are excluded", () => {
  const bins = rewardHistogram([
    { reward: 0 },
    { reward: 1 },
    { reward: null },
    { reward: -2 },
  ]);
  assert.equal(bins[0].value, 1);
  assert.equal(bins[9].value, 1);
});
test("phase timing never bridges two trainer steps", () => {
  assert.deepEqual(
    stageSegments([
      { step: 0, time: 1, data: { stage: "train_start" } },
      { step: 1, time: 4, data: { stage: "rollout_start" } },
    ]),
    [],
  );
  assert.equal(
    stageSegments([
      { step: 0, time: 1, data: { stage: "rollout_start" } },
      { step: 0, time: 4, data: { stage: "rollout_end" } },
    ])[0].seconds,
    3,
  );
});

test("tool outputs stay with their call and repeated conversation history is deduplicated", () => {
  const events = [
    {
      kind: "tool",
      time: 2,
      data: {
        tool_id: "call",
        name: "read",
        status: "completed",
        result: "file",
      },
    },
    {
      kind: "turn",
      data: {
        turn: 0,
        response: { tool_calls: [{ id: "call" }] },
        messages: [],
      },
    },
    {
      kind: "turn",
      data: {
        turn: 1,
        response: { content: "next" },
        messages: [{ role: "tool", tool_call_id: "call", content: "file" }],
      },
    },
    {
      kind: "turn",
      data: {
        turn: 2,
        response: { content: "done" },
        messages: [{ role: "tool", tool_call_id: "call", content: "file" }],
      },
    },
  ];
  const trace = sampleTrace(events);
  assert.equal(trace.turns[0].tools[0].tool_id, "call");
  assert.equal(trace.unassignedTools.length, 0);
  assert.equal(trace.turns[1].toolResults.length, 1);
  assert.equal(trace.turns[2].toolResults.length, 0);
});

test("ordinary training and harness training share views without misclassifying ordinary samples", async () => {
  const { isAgenticJob } = await import("./agentic-data.js");
  assert.equal(isAgenticJob({ config: { algo: "gspo" } }), false);
  assert.equal(
    isAgenticJob({ config: { agent_fn: null }, command: [] }, [
      { mode: "rollout" },
    ]),
    false,
  );
  assert.equal(isAgenticJob({ config: { agent_fn: "None" } }), false);
  assert.equal(isAgenticJob({ launch: { agent_fn: "pi/run_agent.py" } }), true);
  assert.equal(isAgenticJob({}, [{ harness: "pi" }]), true);
  assert.equal(isAgenticJob({}, [{ mode: "agentic" }]), true);
});

test("generic trajectories render recorded tool calls without requiring harness changes", () => {
  const events = [
    {
      kind: "turn_snapshot",
      data: {
        turn: 0,
        response: {
          tool_calls: [
            {
              id: "edit-1",
              function: {
                name: "edit",
                arguments: '{"path":"a.py","newText":"fixed"}',
              },
            },
          ],
        },
        messages: [],
      },
    },
    {
      kind: "turn_snapshot",
      data: {
        turn: 1,
        response: { content: "done" },
        messages: [
          { role: "tool", tool_call_id: "edit-1", content: "Updated a.py" },
        ],
      },
    },
  ];
  const trace = sampleTrace(events);
  assert.equal(trace.tools.length, 1);
  assert.equal(trace.turns[0].tools[0].name, "edit");
  assert.equal(trace.tools[0].result, "Updated a.py");
  assert.equal(trace.tools[0].status, "completed");
  assert.equal(trace.tools[0].start, null);
  assert.equal(trace.tools[0].end, null);
  assert.equal(trace.unassignedTools.length, 0);
});

test("learning rate excludes groups with insufficient scored attempts and does not invent missing data", async () => {
  const { rewardVariation } = await import("./agentic-data.js");
  assert.equal(
    rewardVariation({
      zero: 2,
      one: 1,
      varying: 1,
      flat_other: 0,
      insufficient: 10,
    }),
    0.25,
  );
  assert.equal(rewardVariation({ insufficient: 3, varying: 0 }), null);
  assert.equal(rewardVariation({ varying: null }), null);
});

test("runtime notices deduplicate actual events and logs without treating completion text as incidents", async () => {
  const { runtimeNotices } = await import("./agentic-data.js");
  const result = runtimeNotices(
    {
      status: "stopped",
      stage: "train",
      logs: [
        "INFO normal",
        "WARNING slow",
        "WARNING slow",
        'INFO rollout_completion={"content":"ERROR"}',
      ],
    },
    [{ id: "event", step: 2, time: 10, data: { stage: "train_skip" } }],
  );
  assert.equal(result.length, 3);
  assert.ok(result.some((notice) => notice.message === "WARNING slow"));
  assert.ok(result.some((notice) => notice.message === "Job stopped at train"));
});

test("tool cards suppress duplicate tool markup while retaining assistant prose", async () => {
  const { assistantText } = await import("./agentic-data.js");
  const content =
    'Checking the catalog.\n<tool_call>search_catalog <arg_key>categories</arg_key><arg_value>["jacket"]</arg_value></tool_call><|role_end|>';
  assert.equal(
    assistantText({
      response: { content },
      tools: [{ name: "search_catalog" }],
    }),
    "Checking the catalog.",
  );
  assert.equal(assistantText({ response: { content }, tools: [] }), content);
  assert.equal(
    assistantText({
      response: {
        content: "<tool_call>search_catalog</tool_call><|role_end|>",
      },
      tools: [{}],
    }),
    "",
  );
});

test("time windows follow the latest recorded timestamp and align memory series", async () => {
  const { timeWindowSeries } = await import("./agentic-data.js");
  const series = [
    {
      name: "host",
      points: [
        { step: 100, value: 1 },
        { step: 200, value: 2 },
        { step: 500, value: 3 },
      ],
    },
    {
      name: "cuda",
      points: [
        { step: 150, value: 4 },
        { step: 250, value: 5 },
      ],
    },
  ];
  assert.deepEqual(
    timeWindowSeries(series, 300).map((row) => row.points.map((p) => p.step)),
    [[200, 500], [250]],
  );
  assert.deepEqual(
    timeWindowSeries(series, 60).map((row) => row.points.map((p) => p.step)),
    [[500], []],
  );
  assert.deepEqual(
    timeWindowSeries(series, 100, 250).map((row) =>
      row.points.map((p) => p.step),
    ),
    [[200], [150, 250]],
  );
  assert.deepEqual(timeWindowSeries([], 60), []);
  assert.equal(series[0].points.length, 3);
});

test("trajectory search includes collapsed reasoning, tool arguments and output across all turns", () => {
  const turns = Array.from({ length: 65 }, (_, turn) => ({
    turn,
    response: { content: `Answer ${turn}` },
    tools: [],
  }));
  turns[0].response.reasoning_content = "Inspect the MISSING fixture";
  turns[1].messages = [
    { role: "assistant", content: "Inspect the MISSING fixture" },
  ];
  turns[31].tools = [
    {
      name: "read",
      arguments: '{"path":"src/loader.py"}',
      result: "Permission denied",
      status: "failed",
    },
  ];
  turns[64].tools = [
    { name: "test", result: "65 passed", status: "completed" },
  ];
  assert.deepEqual(
    matchingTurns(turns, " missing ").map((t) => t.turn),
    [0],
  );
  assert.deepEqual(
    matchingTurns(turns, "loader.py").map((t) => t.turn),
    [31],
  );
  assert.deepEqual(
    matchingTurns(turns, "PERMISSION DENIED", "failed").map((t) => t.turn),
    [31],
  );
  assert.deepEqual(
    matchingTurns(turns, "", "tools").map((t) => t.turn),
    [31, 64],
  );
  assert.equal(matchingTurns(turns, "65 passed", "failed").length, 0);
  assert.equal(matchingTurns(turns, "").length, 65);
  assert.equal(turns.length, 65);
});

test("partial live tool records are enriched from captured calls and results without inventing timing", () => {
  const trace = sampleTrace([
    { kind: "tool", time: 12, data: { tool_id: "call", status: "failed" } },
    {
      kind: "turn",
      data: {
        turn: 0,
        response: {
          tool_calls: [
            {
              id: "call",
              function: { name: "shell", arguments: '{"command":"pytest"}' },
            },
          ],
        },
      },
    },
    {
      kind: "turn",
      data: {
        turn: 1,
        messages: [
          {
            role: "tool",
            tool_call_id: "call",
            content: "command failed",
            is_error: true,
          },
        ],
      },
    },
  ]);
  assert.equal(trace.turns[0].tools[0].name, "shell");
  assert.equal(trace.tools[0].arguments, '{"command":"pytest"}');
  assert.equal(trace.tools[0].result, "command failed");
  assert.equal(trace.tools[0].status, "failed");
  assert.equal(trace.tools[0].start, null);
  assert.equal(trace.tools[0].end, 12);
  assert.equal(turnLabel(trace.turns[0]), "shell");
});

test("argument display formats valid JSON and preserves malformed and plain text verbatim", () => {
  assert.equal(
    prettyArguments('{"path":"a.py","lines":0}'),
    '{\n  "path": "a.py",\n  "lines": 0\n}',
  );
  assert.equal(prettyArguments('{"path":'), '{"path":');
  assert.equal(prettyArguments("ls -la"), "ls -la");
  assert.equal(prettyArguments({ enabled: false }), '{\n  "enabled": false\n}');
});

test("tool timing requires an observed start and finish, including timestamp zero", async () => {
  const { toolSeconds } = await import("./agentic-data.js");
  assert.equal(toolSeconds({ start: null, end: 100 }), null);
  assert.equal(toolSeconds({ start: 1, end: null }), null);
  assert.equal(toolSeconds({ start: 0, end: 2 }), 2);
  assert.equal(toolSeconds({ start: 4, end: 3 }), null);
});
