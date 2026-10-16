import {
  assistantText,
  messageText,
  numeric,
  toolSeconds,
  valueText,
} from "./agentic-data.js";

const messageKey = (message) =>
  message.role === "tool" && message.tool_call_id
    ? `tool:${message.tool_call_id}`
    : JSON.stringify({
        role: message.role,
        content: message.content ?? "",
        reasoning: message.reasoning_content ?? "",
        calls: (message.tool_calls || []).map((call) => [
          call.id,
          call.function?.name,
          call.function?.arguments,
        ]),
      });

// Build one transcript from repeated full histories without losing repeated turns.
export function trajectoryMessages(trace, sample) {
  const messages = [],
    counts = new Map(),
    attached = new Set();
  const toolMap = new Map(trace.tools.map((tool) => [tool.tool_id, tool]));
  const append = (message, metadata = {}) => {
    const key = messageKey(message);
    counts.set(key, (counts.get(key) || 0) + 1);
    const item = {
      ...metadata,
      id: `message-${messages.length}`,
      index: messages.length + 1,
      role: message.role || "assistant",
      message,
    };
    messages.push(item);
    return item;
  };
  if (
    !trace.turns.some((turn) =>
      turn.messages?.some((message) => message.role === "user"),
    ) &&
    sample?.prompt
  )
    append({ role: "user", content: sample.prompt }, { media: true });
  for (const turn of trace.turns) {
    const occurrences = new Map();
    for (const message of turn.messages || []) {
      const key = messageKey(message),
        occurrence = (occurrences.get(key) || 0) + 1;
      occurrences.set(key, occurrence);
      if (occurrence <= (counts.get(key) || 0)) continue;
      append(message, { tool: toolMap.get(message.tool_call_id) });
    }
    if (!turn.response) continue;
    const step = append(
      { ...turn.response, role: "assistant" },
      {
        turn: turn.turn,
        time: turn.time,
        seconds: turn.seconds,
        inputTokens: turn.input_tokens,
        outputTokens: turn.output_tokens,
        tools: turn.tools,
        text: assistantText(turn),
      },
    );
    for (const tool of turn.tools) {
      attached.add(tool.tool_id);
      if (tool.result == null) continue;
      append(
        {
          role: "tool",
          tool_call_id: tool.tool_id,
          name: tool.name,
          content: valueText(tool.result),
        },
        {
          tool,
          owner: step.id,
          turn: turn.turn,
          time: tool.end,
        },
      );
    }
  }
  for (const tool of trace.unassignedTools) {
    if (
      attached.has(tool.tool_id) ||
      messages.some((message) => message.message.tool_call_id === tool.tool_id)
    )
      continue;
    append(
      {
        role: "tool",
        tool_call_id: tool.tool_id,
        name: tool.name,
        content: valueText(tool.result),
      },
      { tool, time: tool.end },
    );
  }
  return messages;
}

export function trajectorySteps(messages) {
  return messages
    .filter((message) => !message.owner)
    .map((message, index) => ({ ...message, step: index + 1 }));
}

export function filterMessages(
  messages,
  query = "",
  role = "all",
  kind = "all",
) {
  const needle = query.trim().toLocaleLowerCase();
  return messages.filter((message) => {
    if (role !== "all" && message.role !== role) return false;
    const tools = message.tools || (message.tool ? [message.tool] : []);
    if (kind === "tools" && !tools.length) return false;
    if (kind === "failed" && !tools.some((tool) => tool.status === "failed"))
      return false;
    return (
      !needle ||
      valueText({ message: message.message, tools })
        .toLocaleLowerCase()
        .includes(needle)
    );
  });
}

export function messagePreview(step) {
  return (
    step.tools?.map((tool) => tool.name).join(" · ") ||
    step.tool?.name ||
    messageText(step.message).replace(/\s+/g, " ").trim() ||
    "空消息"
  );
}

export function trajectoryStats(trace, events) {
  const sum = (rows, get) =>
    rows.length && rows.every((row) => numeric(get(row)))
      ? rows.reduce((total, row) => total + get(row), 0)
      : null;
  const start = events.find(
    (event) => event.kind === "sample" && numeric(event.time),
  )?.time;
  const finish = events
    .filter(
      (event) =>
        ["sample", "reward"].includes(event.kind) &&
        ["completed", "failed", "stopped", "cancelled"].includes(
          event.data?.status,
        ) &&
        numeric(event.time),
    )
    .at(-1)?.time;
  const elapsed =
    numeric(start) && numeric(finish) && finish >= start
      ? finish - start
      : null;
  const llm = sum(trace.turns, (turn) => turn.seconds);
  const tools = trace.tools.length ? sum(trace.tools, toolSeconds) : 0;
  const other =
    numeric(elapsed) && numeric(llm) && numeric(tools) && llm + tools <= elapsed
      ? elapsed - llm - tools
      : null;
  return {
    elapsed,
    llm,
    tools,
    other,
    inputTokens: sum(trace.turns, (turn) => turn.input_tokens),
    outputTokens: sum(trace.turns, (turn) => turn.output_tokens),
    rounds: trace.turns.length,
    toolRounds: trace.turns.filter((turn) => turn.tools.length).length,
    toolCalls: trace.tools.length,
  };
}

export const snapshotName = (sample) =>
  `trajectory_${sample.sample_idx ?? 0}.json`;
export const snapshotTask = (sample) =>
  sample.source?.task_id ||
  sample.source?.instance_id ||
  sample.capture?.source_record?.instance_id ||
  `Task ${(sample.prompt_idx ?? 0) + 1}`;

export function snapshotGroups(samples, query = "") {
  const groups = new Map(),
    needle = query.trim().toLocaleLowerCase();
  for (const sample of samples) {
    if (
      needle &&
      !`${snapshotName(sample)} ${snapshotTask(sample)} Step ${sample.step} ${sample.status}`
        .toLocaleLowerCase()
        .includes(needle)
    )
      continue;
    if (!groups.has(sample.step)) groups.set(sample.step, new Map());
    const tasks = groups.get(sample.step),
      key = sample.prompt_idx;
    if (!tasks.has(key))
      tasks.set(key, { key, name: snapshotTask(sample), samples: [] });
    tasks.get(key).samples.push(sample);
  }
  return [...groups].map(([step, tasks]) => ({
    step,
    tasks: [...tasks.values()],
  }));
}

// ATIF v1.7 groups each agent action and its observations in one step.
// https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
export function exportAtif(messages, trace, sample, stats) {
  const steps = trajectorySteps(messages).map((item, index) => {
    const step = {
      step_id: index + 1,
      source:
        item.role === "assistant"
          ? "agent"
          : item.role === "system"
            ? "system"
            : "user",
      message: messageText(item.message),
      extra: { areno_message: item.message, areno_turn: item.turn },
    };
    if (numeric(item.time))
      step.timestamp = new Date(item.time * 1000).toISOString();
    if (item.role === "assistant") {
      if (item.message.reasoning_content)
        step.reasoning_content = item.message.reasoning_content;
      step.metrics = {
        ...(numeric(item.inputTokens)
          ? { prompt_tokens: item.inputTokens }
          : {}),
        ...(numeric(item.outputTokens)
          ? { completion_tokens: item.outputTokens }
          : {}),
        ...(numeric(item.seconds)
          ? { extra: { duration_seconds: item.seconds } }
          : {}),
      };
      if (item.tools?.length) {
        step.tool_calls = item.tools.map((tool) => {
          let argumentsValue = tool.arguments,
            malformed = false;
          if (typeof argumentsValue === "string") {
            try {
              argumentsValue = JSON.parse(argumentsValue);
            } catch {
              malformed = true;
            }
          }
          if (
            !argumentsValue ||
            typeof argumentsValue !== "object" ||
            Array.isArray(argumentsValue)
          )
            malformed = true;
          return {
            tool_call_id: tool.tool_id,
            function_name: tool.name,
            arguments: malformed ? {} : argumentsValue,
            extra: {
              status: tool.status,
              ...(malformed
                ? { raw_arguments: tool.arguments, arguments_parseable: false }
                : {}),
            },
          };
        });
        const results = item.tools
          .filter((tool) => tool.result != null)
          .map((tool) => ({
            source_call_id: tool.tool_id,
            content: valueText(tool.result),
            extra: { status: tool.status, duration_seconds: toolSeconds(tool) },
          }));
        if (results.length) step.observation = { results };
      }
    } else if (item.role === "tool") {
      // ATIF has no tool source: retain orphan feedback without an invented call.
      step.observation = {
        results: [
          {
            content: messageText(item.message),
            extra: { original_tool_call_id: item.message.tool_call_id },
          },
        ],
      };
    }
    return step;
  });
  return {
    schema_version: "ATIF-v1.7",
    session_id: sample?.id || "areno-capture",
    agent: { name: sample?.harness || "AReno", version: "unknown" },
    steps,
    final_metrics: {
      total_steps: steps.length,
      total_prompt_tokens: stats.inputTokens,
      total_completion_tokens: stats.outputTokens,
      extra: {
        elapsed_seconds: stats.elapsed,
        llm_seconds: stats.llm,
        tool_seconds: stats.tools,
        other_seconds: stats.other,
      },
    },
    notes:
      "Converted from captured AReno OpenAI-style messages. Original messages are retained in step.extra. Orphan tool feedback uses source user. Unparseable tool arguments are retained in tool_call.extra.raw_arguments. Agent version was not recorded.",
    extra: {
      sample_status: sample?.status,
      reward: sample?.reward,
      finish_reason: sample?.reason,
      loss_masks: trace.masks,
    },
  };
}

export function toolInventory(events, trace) {
  const definitions = new Map();
  for (const event of events) {
    for (const definition of event.data?.tool_definitions || []) {
      if (definition.function?.name)
        definitions.set(definition.function.name, definition);
    }
  }
  const names = new Set([
    ...definitions.keys(),
    ...trace.tools.map((tool) => tool.name),
  ]);
  return [...names].map(
    (name) =>
      definitions.get(name) || {
        name,
        schema_recorded: false,
        captured_calls: trace.tools.filter((tool) => tool.name === name),
      },
  );
}
