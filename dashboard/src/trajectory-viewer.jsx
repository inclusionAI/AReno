import React, { useEffect, useId, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  Box,
  Brain,
  Braces,
  ChevronDown,
  Clock,
  Download,
  FileJson,
  LayoutList,
  List,
  MessageSquare,
  Search,
  Terminal,
} from "lucide-react";
import {
  messageText,
  numeric,
  prettyArguments,
  sampleTrace,
  toolSeconds,
  valueText,
} from "./agentic-data";
import {
  exportAtif,
  toolInventory,
  filterMessages,
  messagePreview,
  snapshotName,
  trajectoryMessages,
  trajectoryStats,
  trajectorySteps,
} from "./trajectory-data";
import "./trajectory-viewer.css";
import { CopyAction, TrajectoryPopup } from "./trajectory-json";

const fmt = (value) =>
  numeric(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: 1 })
    : "未记录";
const tokens = (value) =>
  numeric(value)
    ? value >= 1000
      ? `${(value / 1000).toFixed(1)}K`
      : fmt(value)
    : "未记录";
const duration = (value) =>
  !numeric(value)
    ? "未记录"
    : value >= 60
      ? `${Math.floor(value / 60)}m ${(value % 60).toFixed(2)}s`
      : `${value.toFixed(2)}s`;
const roles = ["all", "user", "assistant", "tool", "system"];

function saveJson(value, name) {
  const url = URL.createObjectURL(
    new Blob([valueText(value)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function Body({ text, markdown }) {
  if (!markdown)
    return <pre className="tv-content">{text || "（空消息）"}</pre>;
  return (
    <div className="tv-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ children, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
          img: ({ alt }) => <span>[Image: {alt || "see captured media"}]</span>,
        }}
      >
        {text || "（空消息）"}
      </ReactMarkdown>
    </div>
  );
}
function Block({
  title,
  icon: Icon = MessageSquare,
  tone = "message",
  text,
  markdown = false,
  children,
}) {
  return (
    <details className={`tv-block tv-block-${tone}`} open>
      <summary>
        <Icon size={18} />
        <strong>{title}</strong>
        <CopyAction text={text} />
        <ChevronDown className="tv-block-chevron" size={16} />
      </summary>
      <div className="tv-block-body">
        {children || <Body text={text} markdown={markdown} />}
      </div>
    </details>
  );
}
function Role({ role }) {
  return (
    <span className={`tv-role tv-role-${role}`}>{role.toUpperCase()}</span>
  );
}
function TrainingMask({ mask }) {
  if (!mask) return null;
  return (
    <details className="tv-training" open>
      <summary>
        <Activity size={15} />
        训练信号 · Turn {(mask.turn ?? 0) + 1}
        <ChevronDown size={14} />
      </summary>
      <p>
        <b>{fmt(mask.trained_tokens)}</b> / {fmt(mask.input_tokens)} tokens 参与
        loss
      </p>
      <div className="av-mask">
        {mask.spans?.map(([start, end, enabled]) => (
          <i
            key={start}
            style={{
              flex: end - start,
              background: enabled ? "var(--tv-blue)" : "var(--tv-line)",
            }}
            title={`Tokens ${start}–${end}: ${enabled ? "trained" : "masked"}`}
          />
        ))}
      </div>
    </details>
  );
}
function StepBody({ step, grouped, markdown, masks, renderMedia, sample }) {
  const text =
    markdown && step.role === "assistant"
      ? (step.text ?? messageText(step.message))
      : messageText(step.message);
  const mask = masks.find((value) => value.turn === step.turn);
  return (
    <>
      {step.message.reasoning_content && (
        <Block
          title="Thinking（思考过程）"
          icon={Brain}
          tone="thinking"
          text={step.message.reasoning_content}
          markdown={markdown}
        />
      )}
      {step.role !== "tool" && (
        <Block
          title="Message Content（消息正文）"
          text={text}
          markdown={markdown}
        >
          <>
            {step.media &&
              renderMedia?.(sample.capture || { source_record: sample.source })}
            <Body text={text} markdown={markdown} />
          </>
        </Block>
      )}
      {step.tools?.map((tool) => (
        <React.Fragment key={tool.tool_id}>
          <Block
            title={`Action · ${tool.name}`}
            icon={Terminal}
            tone="action"
            text={prettyArguments(tool.arguments) ?? "未记录参数"}
          />
          {grouped && (
            <Block
              title={`Observation · ${tool.name} · ${tool.status}`}
              icon={Box}
              tone={tool.status === "failed" ? "failed" : "observation"}
              text={
                tool.result == null
                  ? "尚未捕获工具返回"
                  : prettyArguments(tool.result)
              }
            />
          )}
        </React.Fragment>
      ))}
      {step.role === "tool" && (
        <Block
          title={`Observation · ${step.tool?.name || step.message.name || "Tool"}${step.tool?.status ? ` · ${step.tool.status}` : ""}`}
          icon={Box}
          tone={step.tool?.status === "failed" ? "failed" : "observation"}
          text={prettyArguments(step.message.content) ?? "尚未捕获工具返回"}
        />
      )}
      {step.role === "assistant" && <TrainingMask mask={mask} />}
    </>
  );
}
function Stats({ stats, messages, sample }) {
  const parts = [
    { name: "LLM", value: stats.llm, color: "var(--tv-blue)" },
    { name: "TOOL", value: stats.tools, color: "var(--tv-green)" },
    { name: "OTHER", value: stats.other, color: "var(--tv-orange)" },
  ];
  return (
    <section className="tv-stats" aria-label="轨迹统计">
      <header>
        <Activity size={20} />
        <strong>轨迹统计</strong>
        <span className="tv-completion">{sample?.status || "captured"}</span>
        <small>共 {messages.length} 条消息</small>
      </header>
      <div className="tv-stat-grid">
        <article>
          <label>总耗时</label>
          <strong>{duration(stats.elapsed)}</strong>
          <small>
            LLM {duration(stats.llm)} · Tool {duration(stats.tools)} · Other{" "}
            {duration(stats.other)}
          </small>
        </article>
        <article>
          <label>交互轮次</label>
          <strong>{stats.rounds}</strong>
          <small>
            LLM {stats.rounds} 次 · Tool {stats.toolRounds} 轮 /{" "}
            {stats.toolCalls} 次
          </small>
        </article>
        <article>
          <label>TOKEN 用量</label>
          <strong className="tv-token-total">
            <span>{tokens(stats.inputTokens)}</span>
            <i>/</i>
            <span>{tokens(stats.outputTokens)}</span>
          </strong>
          <small>
            Prompt {tokens(stats.inputTokens)} · Completion{" "}
            {tokens(stats.outputTokens)}（各轮累计）
          </small>
        </article>
      </div>
      {numeric(stats.other) && stats.elapsed > 0 ? (
        <>
          <div className="tv-duration-bar" aria-label="耗时占比">
            {parts.map((part) => (
              <i
                key={part.name}
                style={{ flex: part.value, background: part.color }}
                title={`${part.name} ${duration(part.value)}`}
              />
            ))}
          </div>
          <div className="tv-duration-legend">
            {parts.map((part) => (
              <span key={part.name}>
                <i style={{ background: part.color }} />
                {part.name} {((100 * part.value) / stats.elapsed).toFixed(1)}%
              </span>
            ))}
          </div>
        </>
      ) : (
        <p className="tv-timing-note">
          {numeric(stats.elapsed) && numeric(stats.llm) && numeric(stats.tools)
            ? "已记录的 LLM 与 Tool 耗时存在重叠，不推算 Other 占比。"
            : "部分时间戳或耗时未记录，暂不计算完整耗时占比。"}
        </p>
      )}
    </section>
  );
}

export function TrajectoryViewer({ events, sample, renderMedia, toolFocus }) {
  const trace = useMemo(() => sampleTrace(events), [events]);
  const messages = useMemo(
    () => trajectoryMessages(trace, sample),
    [trace, sample?.id, sample?.prompt],
  );
  const steps = useMemo(() => trajectorySteps(messages), [messages]);
  const stats = useMemo(() => trajectoryStats(trace, events), [trace, events]);
  const atif = useMemo(
    () => exportAtif(messages, trace, sample, stats),
    [messages, trace, sample, stats],
  );
  const [mode, setMode] = useState("steps"),
    [query, setQuery] = useState(""),
    [role, setRole] = useState("all"),
    [kind, setKind] = useState("all");
  const [selectedId, setSelectedId] = useState(null),
    [markdown, setMarkdown] = useState(false),
    [panel, setPanel] = useState(null);
  const [page, setPage] = useState(0);
  const root = useRef(null),
    content = useRef(null),
    pendingScroll = useRef(false);
  const instance = useId();
  const stepList =
    role === "tool"
      ? messages
          .filter((message) => message.role === "tool")
          .map((message) => ({ ...message, step: message.index }))
      : steps;
  const candidates = mode === "steps" ? stepList : messages;
  const filtered = useMemo(
    () => filterMessages(candidates, query, role, kind),
    [candidates, query, role, kind],
  );
  const selection = messages.find((message) => message.id === selectedId);
  const selected =
    filtered.find(
      (message) =>
        message.id === selectedId ||
        (mode === "steps" && message.id === selection?.owner),
    ) || filtered[0];
  const pageSize = filtered.length > 200 ? 50 : Math.max(1, filtered.length);
  const visible = filtered.slice(
    Math.min(page, Math.max(0, Math.ceil(filtered.length / pageSize) - 1)) *
      pageSize,
    (page + 1) * pageSize,
  );
  const tools = useMemo(() => toolInventory(events, trace), [events, trace]);
  useEffect(() => {
    setSelectedId(null);
    setQuery("");
    setRole("all");
    setKind("all");
    setPanel(null);
    setPage(0);
  }, [sample?.id]);
  useEffect(() => {
    setPage(0);
  }, [query, role, kind]);
  useEffect(() => {
    if (!toolFocus) return;
    const owner =
      messages.find((message) =>
        message.tools?.some((tool) => tool.tool_id === toolFocus.id),
      ) || messages.find((message) => message.tool?.tool_id === toolFocus.id);
    if (!owner) return;
    setSelectedId(owner.id);
    setMode("steps");
    setQuery("");
    setRole("all");
    setKind("all");
    pendingScroll.current = true;
  }, [toolFocus]);
  useEffect(() => {
    if (!content.current) return;
    if (mode === "steps") content.current.scrollTop = 0;
    else if (pendingScroll.current) {
      const node = [
        ...content.current.querySelectorAll("[data-message-id]"),
      ].find((el) => el.dataset.messageId === selected?.id);
      if (node)
        content.current.scrollTop +=
          node.getBoundingClientRect().top -
          content.current.getBoundingClientRect().top -
          16;
    }
    pendingScroll.current = false;
  }, [selected?.id, mode, page]);
  const select = (message) => {
    setSelectedId(message.id);
    pendingScroll.current = true;
    if (mode === "timeline")
      setPage(
        Math.floor(
          filtered.findIndex((item) => item.id === message.id) / pageSize,
        ),
      );
  };
  const switchMode = (value) => {
    setMode(value);
    pendingScroll.current = true;
    setPage(0);
  };
  const expand = (open) => {
    for (const node of root.current.querySelectorAll(
      "details.tv-block, details.tv-training",
    ))
      node.open = open;
  };
  const empty = (
    <p className="tv-empty">
      {events.length
        ? "没有匹配的步骤，请调整搜索或角色筛选。"
        : sample
          ? "等待捕获轨迹…"
          : "选择一条轨迹开始查看"}
    </p>
  );
  return (
    <section className="tv-root" ref={root} aria-label="轨迹查看器">
      <header className="tv-header">
        <span className="tv-protocol">
          <i />
          Protocol: OPENAI
        </span>
        <span className="tv-count">
          总步数：<b>{steps.length}</b>
        </span>
        <span className="tv-count">
          工具调用：<b className="tv-orange">{trace.tools.length}</b>
        </span>
        <div className="tv-modes" role="group" aria-label="轨迹视图">
          <button
            className={mode === "steps" ? "active" : ""}
            aria-pressed={mode === "steps"}
            onClick={() => switchMode("steps")}
          >
            <LayoutList size={17} />
            分步详情
          </button>
          <button
            className={mode === "timeline" ? "active" : ""}
            aria-pressed={mode === "timeline"}
            onClick={() => switchMode("timeline")}
          >
            <List size={17} />
            时间线全览
          </button>
        </div>
        <button
          className="tv-action"
          aria-expanded={panel === "tools"}
          onClick={() => setPanel(panel === "tools" ? null : "tools")}
        >
          <Box size={17} />
          Tools 清单 ({tools.length})
        </button>
        <button
          className="tv-action"
          aria-expanded={panel === "source"}
          onClick={() => setPanel(panel === "source" ? null : "source")}
        >
          <Braces size={17} />
          ATIF 源码
        </button>
        <button
          className="tv-action tv-primary"
          onClick={() =>
            saveJson(atif, `${sample?.id || "trajectory"}-atif.json`)
          }
        >
          <Download size={17} />
          导出 ATIF
        </button>
      </header>
      <div className="tv-toolbar">
        <label className="tv-search">
          <Search size={17} />
          <input
            aria-label="搜索轨迹"
            placeholder="搜索 Step 内容、Thinking、Action 参数或 Observation 返回…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <div className="tv-roles" role="group" aria-label="角色筛选">
          {roles.map((value) => (
            <button
              key={value}
              className={role === value ? "active" : ""}
              aria-pressed={role === value}
              onClick={() => setRole(value)}
            >
              {value === "all" ? "全部角色" : value.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      <div className="tv-utilities">
        <span>
          {snapshotName(sample || {})}
          {sample ? ` · ${sample.status}` : ""}
          {numeric(sample?.reward) ? ` · Reward ${fmt(sample.reward)}` : ""}
          {sample?.reason ? ` · ${sample.reason}` : ""}
        </span>
        <select
          aria-label="筛选工具状态"
          value={kind}
          onChange={(event) => setKind(event.target.value)}
        >
          <option value="all">全部步骤</option>
          <option value="tools">含工具调用</option>
          <option value="failed">失败工具</option>
        </select>
        <button
          className="tv-text-action"
          aria-pressed={markdown}
          onClick={() => setMarkdown(!markdown)}
        >
          Markdown
        </button>
        <button className="tv-text-action" onClick={() => expand(true)}>
          全部展开
        </button>
        <button className="tv-text-action" onClick={() => expand(false)}>
          全部收起
        </button>
      </div>
      {panel && (
        <TrajectoryPopup
          key={panel}
          kind={panel}
          tools={tools}
          atif={atif}
          events={events}
          onClose={() => setPanel(null)}
        />
      )}
      {mode === "steps" ? (
        <div className="tv-step-layout">
          <nav className="tv-step-list" aria-label="轨迹步骤">
            {filtered.map((step) => (
              <button
                key={step.id}
                className={selected?.id === step.id ? "selected" : ""}
                aria-current={selected?.id === step.id ? "step" : undefined}
                onClick={() => select(step)}
              >
                <div>
                  <span className="tv-step-badge">Step #{step.step}</span>
                  <Role role={step.role} />
                  <i className={`tv-role-dot tv-role-dot-${step.role}`} />
                </div>
                {step.tools?.length || step.role === "tool" ? (
                  <p className="tv-step-tool">
                    <Terminal size={14} />
                    {messagePreview(step)}
                  </p>
                ) : (
                  <p className="tv-step-preview">
                    {messagePreview(step).slice(0, 100)}
                  </p>
                )}
                <footer>
                  <span>{step.tools?.length || 0} Actions</span>
                  <i>·</i>
                  <span>
                    {step.tools?.filter((tool) => tool.result != null).length ||
                      (step.role === "tool" ? 1 : 0)}{" "}
                    Obs
                  </span>
                  {step.tools?.some((tool) => tool.status === "failed") && (
                    <b>Failed</b>
                  )}
                </footer>
              </button>
            ))}
            {!filtered.length && empty}
          </nav>
          <div className="tv-step-content" ref={content}>
            {selected ? (
              <>
                <div className="tv-step-heading">
                  <span className="tv-step-badge">Step #{selected.step}</span>
                  <Role role={selected.role} />
                  {numeric(selected.time) && (
                    <time>
                      <Clock size={15} />
                      {new Date(selected.time * 1000).toLocaleTimeString()}
                    </time>
                  )}
                  <CopyAction
                    text={valueText(selected)}
                    label="复制完整 Step JSON"
                  />
                </div>
                <StepBody
                  key={selected.id}
                  step={selected}
                  grouped
                  markdown={markdown}
                  masks={trace.masks}
                  renderMedia={renderMedia}
                  sample={sample}
                />
              </>
            ) : (
              empty
            )}
          </div>
        </div>
      ) : (
        <div className="tv-timeline" ref={content}>
          <nav className="tv-number-strip" aria-label="时间线消息">
            {filtered.map((message) => (
              <button
                key={message.id}
                className={`${message.role === "assistant" ? "agent" : "context"} ${selected?.id === message.id ? "selected" : ""}`}
                title={`#${message.index} ${message.role.toUpperCase()} · ${messagePreview(message).slice(0, 80)}`}
                aria-label={`定位消息 ${message.index}`}
                aria-current={selected?.id === message.id ? "step" : undefined}
                onClick={() => select(message)}
              >
                {message.index}
              </button>
            ))}
          </nav>
          <div className="tv-timeline-inner">
            <div className="tv-snapshot-heading">
              <FileJson size={17} />
              <strong>SNAPSHOT</strong>
              <span>{snapshotName(sample || {})}</span>
              <CopyAction text={valueText(atif)} label="复制轨迹 JSON" />
            </div>
            <Stats stats={stats} messages={messages} sample={sample} />
            {filtered.length > pageSize && (
              <div className="tv-paging">
                <button disabled={!page} onClick={() => setPage(page - 1)}>
                  较早消息
                </button>
                <span>
                  {page * pageSize + 1}–
                  {Math.min((page + 1) * pageSize, filtered.length)} /{" "}
                  {filtered.length}
                </span>
                <button
                  disabled={(page + 1) * pageSize >= filtered.length}
                  onClick={() => setPage(page + 1)}
                >
                  更多消息
                </button>
              </div>
            )}
            {visible.map((message) => (
              <article
                className={`tv-timeline-message ${selected?.id === message.id ? "selected" : ""}`}
                key={message.id}
                id={`${instance}-${message.id}`}
                data-message-id={message.id}
              >
                <div className="tv-step-heading">
                  <span className="tv-step-badge">#{message.index}</span>
                  <Role role={message.role} />
                  <small>Index: {message.index - 1}</small>
                  {numeric(message.time) && (
                    <time>
                      <Clock size={14} />
                      {new Date(message.time * 1000).toLocaleTimeString()}
                    </time>
                  )}
                  <CopyAction
                    text={valueText(message)}
                    label="复制完整 Step JSON"
                  />
                </div>
                <StepBody
                  step={message}
                  markdown={markdown}
                  masks={trace.masks}
                  renderMedia={renderMedia}
                  sample={sample}
                />
              </article>
            ))}
            {!filtered.length && empty}
          </div>
        </div>
      )}
    </section>
  );
}
