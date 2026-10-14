import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ArrowDown,
  Check,
  ChevronDown,
  Clock,
  Code,
  Download,
  MessageSquare,
  Search,
  Terminal,
  Wrench,
  X,
} from "lucide-react";
import {
  sampleTrace,
  rewardHistogram,
  isAgenticJob,
  messageText,
  valueText,
  numeric,
  rewardVariation,
  runtimeNotices,
  sampleSourceLabel,
  assistantText,
  timeWindowSeries,
} from "./agentic-data";
import { sampleMetricPoints } from "./metrics";
import "./agentic.css";
const fmt = (n) =>
  numeric(n)
    ? n.toLocaleString(undefined, { maximumFractionDigits: 3 })
    : "Not recorded";
const seconds = (n) => (numeric(n) ? `${n.toFixed(1)}s` : "Not recorded");
const gib = (n) =>
  numeric(n) ? `${(n / 2 ** 30).toFixed(1)} GiB` : "Not recorded";
const clock = (n) =>
  numeric(n)
    ? new Date(n * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "—";
const colors = ["var(--deck-cyan)", "#a78bfa", "#e8b86d", "#6bba99", "#e77e88"];
function Panel({ title, note, children, wide = false }) {
  return (
    <section className={`panel av-panel ${wide ? "av-wide" : ""}`}>
      <header>
        <h3>{title}</h3>
        {note && <small>{note}</small>}
      </header>
      {children}
    </section>
  );
}
function TimeWindow({ label, value, onChange }) {
  return (
    <label className="av-time-window">
      <span>Time window</span>
      <select
        aria-label={`${label} time window`}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        <option value={60}>Last 1 min</option>
        <option value={300}>Last 5 min</option>
        <option value={900}>Last 15 min</option>
      </select>
    </label>
  );
}
function Empty({ children = "Not recorded for this job" }) {
  return <div className="av-empty">{children}</div>;
}
function Spark({ series, onSelect, format = fmt, xFormat = fmt }) {
  const available = series.filter((s) =>
    s.points?.some((p) => numeric(p.value)),
  );
  const [hover, setHover] = useState(null);
  if (!available.length) return <Empty />;
  const pts = available
    .flatMap((s) => s.points)
    .filter((p) => numeric(p.value));
  const bounds = pts.reduce(
    (b, p) => ({
      maxX: Math.max(b.maxX, p.step),
      minX: Math.min(b.minX, p.step),
      maxY: Math.max(b.maxY, p.value),
      minY: Math.min(b.minY, p.value),
    }),
    { maxX: -Infinity, minX: Infinity, maxY: -Infinity, minY: 0 },
  );
  const { maxX, minX, maxY, minY } = bounds;
  const x = (n) => 36 + ((n - minX) / Math.max(1, maxX - minX)) * 584,
    y = (n) => 140 - ((n - minY) / Math.max(1e-10, maxY - minY)) * 115;
  return (
    <>
      <svg
        viewBox="0 0 640 166"
        className="av-chart"
        role="img"
        aria-label={available.map((s) => s.name).join(", ")}
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1="36"
              x2="620"
              y1={25 + t * 115}
              y2={25 + t * 115}
              className="av-grid"
            />
            <text x="32" y={29 + t * 115} textAnchor="end">
              {format(maxY - t * (maxY - minY))}
            </text>
          </g>
        ))}
        {available.map((s, i) => (
          <g key={s.name}>
            <path
              fill="none"
              stroke={colors[i % colors.length]}
              strokeWidth="2"
              d={sampleMetricPoints(
                s.points.filter((p) => numeric(p.value)),
                400,
              )
                .map((p, j) => `${j ? "L" : "M"}${x(p.step)},${y(p.value)}`)
                .join(" ")}
            />
            {sampleMetricPoints(
              s.points.filter((p) => numeric(p.value)),
              80,
            ).map((p, j) => (
              <circle
                key={j}
                cx={x(p.step)}
                cy={y(p.value)}
                r="5"
                fill="transparent"
                tabIndex="0"
                aria-label={`${s.name} step ${p.step}: ${p.value}`}
                onMouseEnter={() => setHover({ name: s.name, ...p })}
                onFocus={() => setHover({ name: s.name, ...p })}
                onClick={() => onSelect?.(p.step)}
                onKeyDown={(e) => e.key === "Enter" && onSelect?.(p.step)}
              />
            ))}
          </g>
        ))}
        <text x="36" y="162">
          {xFormat(minX)}
        </text>
        <text x="620" y="162" textAnchor="end">
          {xFormat(maxX)}
        </text>
      </svg>
      <div className="av-legend">
        {available.map((s, i) => (
          <span key={s.name}>
            <i style={{ background: colors[i % colors.length] }} />
            {s.name}
          </span>
        ))}
      </div>
      <div className="av-tooltip">
        {hover
          ? `${hover.name} · ${xFormat(hover.step)} · ${format(hover.value)}`
          : onSelect
            ? "Select a point to inspect its step"
            : "Hover a point for its recorded value"}
      </div>
    </>
  );
}
function DownloadButton({ data, name }) {
  return (
    <button
      className="av-icon"
      title="Download full output"
      aria-label="Download full output"
      onClick={() => {
        const url = URL.createObjectURL(
          new Blob([valueText(data)], { type: "text/plain" }),
        );
        const a = document.createElement("a");
        a.href = url;
        a.download = name;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }}
    >
      <Download size={14} />
    </button>
  );
}
function Tool({ tool, onJump }) {
  return (
    <details
      className={`av-tool av-${tool.status}`}
      id={`av-tool-${tool.tool_id}`}
    >
      <summary>
        <Wrench size={14} />
        <strong>{tool.name}</strong>
        <span>{tool.status}</span>
        <small>{seconds(tool.end ? tool.end - tool.start : null)}</small>
        <ChevronDown size={14} />
      </summary>
      <div className="av-toolbody">
        <label>Arguments</label>
        <pre>{valueText(tool.arguments)}</pre>
        {tool.result != null && (
          <>
            <label>
              Output{" "}
              <DownloadButton
                data={tool.result}
                name={`${tool.name}-output.txt`}
              />
            </label>
            <pre>{valueText(tool.result)}</pre>
          </>
        )}
      </div>
    </details>
  );
}
function Conversation({ events, sample, agentic, renderMedia }) {
  const trace = useMemo(() => sampleTrace(events), [events]);
  const ref = useRef(null);
  const follow = useRef(true);
  const [unread, setUnread] = useState(false);
  const [windowEnd, setWindowEnd] = useState(null);
  const [signal, setSignal] = useState(false);
  const [maskTurn, setMaskTurn] = useState("");
  const positions = useRef(new Map());
  useEffect(() => {
    const el = ref.current;
    if (el) {
      el.scrollTop = positions.current.get(sample?.id) ?? 0;
      follow.current = !positions.current.has(sample?.id);
    }
    setWindowEnd(null);
    setUnread(false);
  }, [sample?.id]);
  useEffect(() => {
    if (follow.current && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    } else setUnread(true);
  }, [events.length]);
  const selectedMask =
    trace.masks.find((m) => String(m.turn) === maskTurn) || trace.mask;
  const end = windowEnd ?? trace.turns.length;
  const visible = trace.turns.slice(Math.max(0, end - 30), end);
  return (
    <div className="av-conversation">
      <div className="av-conversation-head">
        <MessageSquare size={17} />
        <strong>
          {sample
            ? `${agentic ? "Task" : "Prompt"} ${sample.prompt_idx + 1} · Sample ${sample.sample_idx + 1}`
            : "Select a sample"}
        </strong>
        <span className={`av-status av-${sample?.status}`}>
          {sample?.status}
        </span>
        {trace.masks.length > 0 && (
          <button
            className={`av-chip ${signal ? "active" : ""}`}
            onClick={() => setSignal(!signal)}
          >
            Training signal
          </button>
        )}
        <DownloadButton data={events} name="sample-events.json" />
      </div>
      <div
        ref={ref}
        className="av-messages"
        onScroll={(e) => {
          const el = e.currentTarget;
          follow.current =
            el.scrollHeight - el.scrollTop - el.clientHeight < 64;
          if (sample) positions.current.set(sample.id, el.scrollTop);
          if (follow.current) setUnread(false);
        }}
      >
        {end > 30 && (
          <button
            className="secondaryButton"
            onClick={() => setWindowEnd(Math.max(30, end - 30))}
          >
            Earlier turns ({end - 30})
          </button>
        )}
        {windowEnd !== null && (
          <button
            className="secondaryButton"
            onClick={() => setWindowEnd(null)}
          >
            Latest turns
          </button>
        )}
        {sample?.prompt && (
          <article className="av-message av-user">
            <label>{agentic ? "User task" : "Prompt"}</label>
            {renderMedia?.(
              sample.capture || {
                source_record: sample.source,
                prompt_messages: trace.turns[0]?.messages,
              },
            )}
            <div>{sample.prompt}</div>
          </article>
        )}
        {visible.map((t) => (
          <article className="av-turn" key={t.turn} id={`av-turn-${t.turn}`}>
            <div className="av-turnline">
              <span>{agentic ? `TURN ${t.turn + 1}` : "OUTPUT"}</span>
              <span>
                {fmt(t.input_tokens)} in · {fmt(t.output_tokens)} out ·{" "}
                {seconds(t.seconds)}
              </span>
            </div>
            {(t.toolResults || [])
              .filter(
                (m) =>
                  !trace.tools.some((tool) => tool.tool_id === m.tool_call_id),
              )
              .map((m, i) => (
                <details className="av-tool" key={i}>
                  <summary>
                    <Terminal size={14} />
                    Tool result · {m.tool_call_id}
                  </summary>
                  <pre>{messageText(m)}</pre>
                </details>
              ))}
            <div className="av-message av-assistant">
              <label>Assistant</label>
              {t.response?.reasoning_content && (
                <details>
                  <summary>Thinking</summary>
                  <pre>{t.response.reasoning_content}</pre>
                </details>
              )}
              {assistantText(t) && <div>{assistantText(t)}</div>}
              {(t.response?.tool_calls || [])
                .filter((c) => !t.tools.some((tool) => tool.tool_id === c.id))
                .map((c) => (
                  <details className="av-tool" key={c.id}>
                    <summary>
                      <Code size={14} />
                      {c.function?.name}
                    </summary>
                    <pre>{valueText(c.function?.arguments)}</pre>
                  </details>
                ))}
            </div>
            {t.tools.map((tool) => (
              <Tool key={tool.tool_id} tool={tool} />
            ))}
          </article>
        ))}
        {trace.unassignedTools.map((t) => (
          <Tool key={t.tool_id} tool={t} />
        ))}
        {signal && (
          <section className="av-signal">
            <strong>Exact training mask</strong>
            {trace.masks.length > 1 && (
              <label>
                Model turn{" "}
                <select
                  aria-label="Training mask turn"
                  value={maskTurn || String(trace.mask?.turn ?? 0)}
                  onChange={(e) => setMaskTurn(e.target.value)}
                >
                  {trace.masks.map((m) => (
                    <option key={m.turn} value={m.turn}>
                      Turn {(m.turn ?? 0) + 1}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {selectedMask ? (
              <>
                <p>
                  {fmt(selectedMask.trained_tokens)} /{" "}
                  {fmt(selectedMask.input_tokens)} tokens participate in loss
                </p>
                <div className="av-mask">
                  {selectedMask.spans?.map(([a, b, enabled]) => (
                    <i
                      key={a}
                      style={{
                        flex: b - a,
                        background: enabled
                          ? "var(--deck-cyan)"
                          : "var(--deck-line)",
                      }}
                      title={`Tokens ${a}–${b}: ${enabled ? "trained" : "masked"}`}
                    />
                  ))}
                </div>
              </>
            ) : (
              <Empty />
            )}
          </section>
        )}
        {!events.length && (
          <Empty>
            {sample
              ? "Waiting for captured output…"
              : "Choose a sample in the matrix"}
          </Empty>
        )}
      </div>
      {unread && (
        <button
          className="av-new"
          onClick={() => {
            follow.current = true;
            setWindowEnd(null);
            ref.current.scrollTop = ref.current.scrollHeight;
            setUnread(false);
          }}
        >
          <ArrowDown size={14} />
          New events
        </button>
      )}
    </div>
  );
}
function useTrace(jobId, sampleId, api, base) {
  const [events, setEvents] = useState([]);
  const [state, setState] = useState("connecting");
  useEffect(() => {
    setEvents([]);
    if (!sampleId) return;
    let active = true,
      stream;
    let cursor = 0;
    const append = (batch) => {
      if (!active) return;
      setEvents((old) => {
        const known = new Set(old.map((e) => e.id));
        return [...old, ...batch.filter((e) => !known.has(e.id))];
      });
    };
    (async () => {
      try {
        while (active) {
          const data = await api(
            `/api/jobs/${jobId}/agent-events?sample=${encodeURIComponent(sampleId)}&after=${cursor}`,
          );
          append(data.events);
          if (data.events?.length) cursor = data.events.at(-1).cursor;
          if (data.events?.length < 250) break;
        }
        if (!active) return;
        stream = new EventSource(
          `${base}api/jobs/${jobId}/agent-events?stream=1&sample=${encodeURIComponent(sampleId)}&after=${cursor}`,
        );
        stream.onopen = () => setState("live");
        stream.onerror = () => setState("reconnecting");
        stream.onmessage = (e) => {
          try {
            append([JSON.parse(e.data)]);
          } catch {
            setState("reconnecting");
          }
        };
      } catch {
        if (active) setState("unavailable");
      }
    })();
    return () => {
      active = false;
      stream?.close();
    };
  }, [jobId, sampleId]);
  return { events, state };
}
function Tests({ grader }) {
  if (!grader) return <Empty>Waiting for grader results</Empty>;
  const r = grader.result || {};
  const text = r.verify_log || "";
  const tests =
    grader.tests ||
    text
      .split("\n")
      .filter((l) => /^(FAILED|PASSED|ERROR)\s/.test(l))
      .map((l) => ({
        name: l.replace(/^(FAILED|PASSED|ERROR)\s+/, ""),
        status: l.startsWith("PASSED") ? "passed" : "failed",
      }));
  return (
    <>
      <div className="av-testsummary">
        <span className={`av-status av-${grader.status}`}>{grader.status}</span>
        <span>Exit {r.verify_returncode ?? "—"}</span>
        <span>{r.verify_timed_out ? "Timed out" : "Verifier"}</span>
      </div>
      {tests.length ? (
        <div className="av-tests">
          {tests.map((t, i) => (
            <div key={i}>
              <span className={`av-status av-${t.status}`}>
                {t.status === "passed" ? <Check size={14} /> : <X size={14} />}
              </span>
              <code>{t.name}</code>
              <small>{t.group || "Group not recorded"}</small>
            </div>
          ))}
        </div>
      ) : (
        <p className="av-muted">
          Individual test names and FAIL_TO_PASS / PASS_TO_PASS groups were not
          recorded.
        </p>
      )}
      <details>
        <summary>
          Grader output <DownloadButton data={text} name="grader-output.txt" />
        </summary>
        <pre>{text || "Not recorded"}</pre>
      </details>
    </>
  );
}
function ToolGraph({ tools, onJump }) {
  if (!tools.length) return <Empty />;
  const visible = tools.slice(-24);
  return (
    <>
      <svg
        className="av-toolgraph"
        viewBox={`0 0 620 ${Math.ceil(visible.length / 4) * 76 + 12}`}
        role="img"
        aria-label="Tool execution order"
      >
        {visible.map((t, i) => {
          const x = 10 + (i % 4) * 154,
            y = 12 + Math.floor(i / 4) * 76;
          return (
            <g
              key={t.tool_id}
              tabIndex="0"
              role="button"
              aria-label={`Jump to ${t.name}`}
              onClick={() => onJump(t)}
              onKeyDown={(e) => e.key === "Enter" && onJump(t)}
            >
              {i % 4 !== 3 && i < visible.length - 1 && (
                <path d={`M${x + 140} ${y + 25}h14`} className="av-link" />
              )}
              <rect
                x={x}
                y={y}
                width="140"
                height="50"
                rx="8"
                className={`av-node av-${t.status}`}
              />
              <text x={x + 12} y={y + 21}>
                {i + 1}. {t.name}
              </text>
              <text x={x + 12} y={y + 38} className="av-muted">
                {seconds(t.end ? t.end - t.start : null)}
              </text>
            </g>
          );
        })}
      </svg>
      <small className="av-muted">
        Observed execution order · last 24 calls
      </small>
    </>
  );
}
function Changes({ tools }) {
  const edits = tools.filter((t) => ["edit", "write"].includes(t.name));
  if (!edits.length) return <Empty>No file edits recorded</Empty>;
  return (
    <div className="av-changes">
      {edits.map((t) => {
        let a = t.arguments || {};
        if (typeof a === "string") {
          try {
            a = JSON.parse(a);
          } catch {
            a = {};
          }
        }
        const changes = a.edits || [a];
        return (
          <details key={t.tool_id}>
            <summary>
              <Code size={14} />
              {a.path || a.file_path || t.name}
            </summary>
            {changes.map((c, i) => (
              <div className="av-diff" key={i}>
                {(c.oldText || c.old_string || "").split("\n").map((l, j) => (
                  <div className="av-deletion" key={`a${j}`}>
                    − {l}
                  </div>
                ))}
                {(c.newText || c.new_string || c.content || "")
                  .split("\n")
                  .map((l, j) => (
                    <div className="av-addition" key={`b${j}`}>
                      + {l}
                    </div>
                  ))}
              </div>
            ))}
          </details>
        );
      })}
    </div>
  );
}
function LearningSignal({ overview, onSelect }) {
  const learning = overview.learning;
  if (!learning || !Object.values(learning).some(numeric))
    return <Empty>Reward-group statistics were not recorded.</Empty>;
  const groups = [
    ["varying", "Reward variation"],
    ["zero", "All zero"],
    ["one", "All perfect"],
    ["flat_other", "Other equal rewards"],
    ["insufficient", "Fewer than two scored attempts"],
  ];
  const count = groups.reduce((n, [key]) => n + (learning[key] || 0), 0);
  return (
    <>
      <div className="av-signal-summary">
        <div>
          <small>Groups with reward variation</small>
          <strong>
            {numeric(rewardVariation(learning))
              ? `${(rewardVariation(learning) * 100).toFixed(1)}%`
              : "Not recorded"}
          </strong>
        </div>
        <div>
          <small>Scored prompt groups</small>
          <strong>{fmt(count)}</strong>
        </div>
      </div>
      <div className="av-group-signal">
        {groups.map(([key, label], i) => (
          <div key={key}>
            <span>
              <i style={{ background: colors[i] }} />
              {label}
            </span>
            <strong>{fmt(learning[key])}</strong>
            <div>
              <i
                style={{
                  width: `${count ? (100 * (learning[key] || 0)) / count : 0}%`,
                  background: colors[i],
                }}
              />
            </div>
          </div>
        ))}
      </div>
      <Spark
        series={[
          {
            name: "Groups with reward variation",
            points: (overview.batches || []).map((batch) => ({
              step: batch.step,
              value: rewardVariation(batch.learning),
            })),
          },
        ]}
        format={(value) => `${(value * 100).toFixed(1)}%`}
        onSelect={onSelect}
      />
      <p className="av-muted">
        Reward variation gives group-relative updates a relative signal. This
        does not imply a nonzero total gradient when other loss terms are
        present.
      </p>
    </>
  );
}

function SamplingFunnel({ funnel }) {
  if (!funnel || !Object.values(funnel).some(numeric))
    return <Empty>Sampling counts were not recorded.</Empty>;
  const rows = [
    ["requested", "Requested"],
    ["returned", "Returned"],
    ["scored", "Scored"],
    ["accepted", "Accepted for training"],
    ["trained", "Completed training"],
  ];
  return (
    <div className="av-funnel">
      {rows.map(([key, label], i) => (
        <div key={key}>
          <span className="av-funnel-number">{i + 1}</span>
          <div>
            <label>{label}</label>
            <div>
              <i
                style={{
                  width: `${numeric(funnel[key]) && funnel.requested ? Math.min(100, (100 * funnel[key]) / funnel.requested) : 0}%`,
                }}
              />
            </div>
          </div>
          <strong>{fmt(funnel[key])}</strong>
        </div>
      ))}
      <p className="av-muted">
        Returned includes invalid attempts. Accepted counts episodes; a
        multi-turn episode can produce many training rows. Completed training
        updates after the trainer reports completion.
      </p>
    </div>
  );
}

function SourceMix({ sources, onSearch }) {
  if (!sources?.length)
    return <Empty>Dataset/source breakdown was not recorded.</Empty>;
  const total = sources.reduce((n, source) => n + source.samples, 0);
  return (
    <>
      <div className="av-source-strip" aria-label="Accepted samples by source">
        {sources.map((source, i) => (
          <i
            key={source.label}
            style={{
              flex: source.samples,
              background: colors[i % colors.length],
            }}
            title={`${source.label}: ${source.samples} accepted samples`}
          />
        ))}
      </div>
      <div className="av-source-table">
        <table>
          <thead>
            <tr>
              <th>Dataset / source</th>
              <th>Accepted</th>
              <th>Mean reward</th>
              <th>Train rows</th>
              <th>Context tokens</th>
            </tr>
          </thead>
          <tbody>
            {sources.map((source, i) => (
              <tr key={source.label}>
                <td>
                  <button
                    onClick={() => onSearch(source.label)}
                    title={source.label}
                  >
                    <i style={{ background: colors[i % colors.length] }} />
                    {source.label}
                  </button>
                </td>
                <td>
                  {fmt(source.samples)}
                  <small>
                    {((100 * source.samples) / Math.max(1, total)).toFixed(1)}%
                  </small>
                </td>
                <td>{fmt(source.reward)}</td>
                <td>{fmt(source.train_rows)}</td>
                <td>{fmt(source.train_tokens)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="av-muted">
        Counts cover accepted training data. Context tokens include repeated
        conversation history, not only newly generated output.
      </p>
    </>
  );
}

function RunNotices({ notices, onSelect }) {
  return notices.length ? (
    <div className="av-notices">
      {notices.map((notice) => (
        <article key={notice.id}>
          <div>
            <span>{notice.source}</span>
            {numeric(notice.step) && (
              <button onClick={() => onSelect(notice.step)}>
                Step {notice.step}
              </button>
            )}
            {numeric(notice.time) && <time>{clock(notice.time)}</time>}
          </div>
          <p>{notice.message}</p>
        </article>
      ))}
    </div>
  ) : (
    <Empty>No runtime incidents recorded.</Empty>
  );
}

function FloatingTimeline({ jobId, api, step, onSelect, onClear }) {
  const [query, setQuery] = useState("");
  const [timeline, setTimeline] = useState({ rows: [], total: 0 });
  const [hovered, setHovered] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true,
      pending = false;
    const load = async () => {
      if (pending) return;
      pending = true;
      try {
        const result = await api(
          `/api/jobs/${jobId}/timeline${query.trim() ? `?step=${encodeURIComponent(query.trim())}` : ""}`,
        );
        if (active) {
          setTimeline(result);
          setError("");
        }
      } catch (e) {
        if (active) setError(e.message);
      } finally {
        pending = false;
      }
    };
    load();
    const timer = setInterval(load, 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [jobId, query]);
  const rows = timeline.rows;

  const [expanded, setExpanded] = useState(
    () => window.matchMedia("(min-width: 1280px)").matches,
  );
  useEffect(() => {
    const media = window.matchMedia("(min-width: 1280px)");
    const update = (e) => setExpanded(e.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  const phases = [
    ...new Set(rows.flatMap((row) => row.segments.map((part) => part.name))),
  ];
  const color = (name) =>
    colors[
      Math.max(
        0,
        ["rollout", "reward", "advantage", "train", "save", "other"].indexOf(
          name,
        ),
      ) % colors.length
    ];
  const inspect = (event, row) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setHovered({
      ...row,
      left: Math.max(12, rect.left - 252),
      top: Math.max(12, Math.min(rect.top, window.innerHeight - 280)),
    });
  };
  const timeRow = (row) => {
    const total = row.segments.reduce((n, part) => n + part.seconds, 0);
    const average = row.step === "avg";
    const active = average ? step === "" : String(row.step) === step;
    return (
      <button
        key={row.step}
        className={`av-time-row ${average ? "av-average " : ""}${active ? "selected" : ""}`}
        aria-pressed={active}
        aria-label={average ? "AVG · All steps" : `Inspect step ${row.step}`}
        aria-describedby={
          hovered?.step === row.step ? "av-timeline-tooltip" : undefined
        }
        onClick={() => {
          if (average) {
            onClear();
            setQuery("");
          } else onSelect(row.step);
          setHovered(null);
        }}
        onMouseEnter={(e) => inspect(e, row)}
        onMouseLeave={() => setHovered(null)}
        onFocus={(e) => inspect(e, row)}
        onBlur={() => setHovered(null)}
      >
        <span>
          {average ? "AVG" : `Step ${row.step}`}
          <small>{seconds(total)}</small>
        </span>
        <div>
          {row.segments.map((part, i) => (
            <i
              key={i}
              style={{ flex: part.seconds, background: color(part.name) }}
            />
          ))}
        </div>
      </button>
    );
  };
  return createPortal(
    <aside
      className={`av-floating-timeline ${expanded ? "expanded" : ""}`}
      aria-label="Step timeline"
    >
      <button
        className="av-timeline-toggle"
        aria-expanded={expanded}
        aria-controls="av-timeline-body"
        onClick={() => {
          setExpanded(!expanded);
          setHovered(null);
        }}
      >
        <Clock size={15} />
        <strong>Step timeline</strong>
        <ChevronDown size={15} />
      </button>
      {expanded && (
        <div id="av-timeline-body" className="av-timeline-body">
          <label className="av-timeline-search">
            <Search size={14} />
            <input
              aria-label="Search timeline step"
              type="search"
              inputMode="numeric"
              placeholder="Search step…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          {error && (
            <p className="av-muted" role="alert">
              Timeline update unavailable. {error}
            </p>
          )}

          <div className="av-timeline-filter">
            <span>{step === "" ? "All steps" : `Step ${step}`}</span>
            <button
              onClick={() => {
                onClear();
                setQuery("");
              }}
              disabled={step === "" && !query}
            >
              Clear filter
            </button>
          </div>
          {timeline.average?.count > 0 && timeRow(timeline.average)}
          <div className="av-timelines">
            {rows.length ? (
              rows.map(timeRow)
            ) : (
              <Empty>
                {query.trim()
                  ? "No recorded timing for this step."
                  : "No phase timings recorded yet."}
              </Empty>
            )}
          </div>
          {rows.length > 0 && (
            <div className="av-legend">
              {phases.map((name) => (
                <span key={name}>
                  <i style={{ background: color(name) }} />
                  {name.replaceAll("_", " ")}
                </span>
              ))}
            </div>
          )}
          <small className="av-timeline-hint">
            {rows.length} of {timeline.total} steps ·{" "}
            {query.trim() ? "search result" : "uniformly sampled"}. Select a
            step to inspect its samples.
          </small>
        </div>
      )}
      {hovered && (
        <div
          id="av-timeline-tooltip"
          role="tooltip"
          className="av-timeline-detail"
          style={{ left: hovered.left, top: hovered.top }}
        >
          <strong>
            {hovered.step === "avg"
              ? `AVG · ${hovered.count} recorded steps`
              : `Step ${hovered.step}`}
          </strong>
          <div className="av-timeline-total">
            <span>Total</span>
            <b>
              {seconds(
                hovered.segments.reduce((n, part) => n + part.seconds, 0),
              )}
            </b>
          </div>
          {hovered.segments.map((part, i) => (
            <div key={i}>
              <span>
                <i style={{ background: color(part.name) }} />
                {part.name.replaceAll("_", " ")}
              </span>
              <b>{seconds(part.seconds)}</b>
            </div>
          ))}
        </div>
      )}
    </aside>,
    document.body,
  );
}

export function AgenticJobDetail({ job, api, base, renderMedia }) {
  const [overview, setOverview] = useState({
    samples: [],
    stages: [],
    resources: [],
    total: 0,
  });
  const [selected, setSelected] = useState(null);
  const [compare, setCompare] = useState(null);
  const [step, setStep] = useState("");
  const [offset, setOffset] = useState(0);
  const [query, setQuery] = useState("");
  const [metrics, setMetrics] = useState([]);
  const [error, setError] = useState("");
  const [memoryWindow, setMemoryWindow] = useState(300);
  const [mstWindow, setMstWindow] = useState(60);
  useEffect(() => {
    let active = true,
      pending = false;
    const load = async () => {
      if (pending) return;
      pending = true;
      try {
        const d = await api(
          `/api/jobs/${job.id}/agentic?step_limit=10&offset=${offset}${step !== "" ? `&step=${step}` : ""}`,
        );
        if (active) {
          setOverview(d);
          setError("");
          setSelected((old) =>
            d.samples.some((s) => s.id === old)
              ? old
              : d.samples[0]?.id || null,
          );
        }
      } catch (e) {
        if (active) setError(e.message);
      } finally {
        pending = false;
      }
    };
    load();
    const timer = setInterval(load, 3000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [job.id, step, offset]);
  useEffect(() => {
    let active = true,
      pending = false;
    const load = async () => {
      if (pending) return;
      pending = true;
      try {
        const d = await api(`/api/jobs/${job.id}/metrics`);
        const wanted = d.metrics.filter((m) =>
          /reward|loss|grad_norm|logp(?:rob)?_abs_diff|memory|mst/i.test(
            m.name,
          ),
        );
        const results = await Promise.all(
          wanted.map(async (m) => ({
            name: m.name,
            points: (
              await api(
                `/api/jobs/${job.id}/metric?name=${encodeURIComponent(m.name)}&limit=2000`,
              )
            ).points,
          })),
        );
        if (active) setMetrics(results);
      } catch {
      } finally {
        pending = false;
      }
    };
    load();
    const timer = setInterval(load, 10000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [job.id]);
  const trace = useTrace(job.id, selected, api, base),
    comparison = useTrace(job.id, compare, api, base);
  const samples = overview.samples || [],
    current = samples.find((s) => s.id === selected),
    other = samples.find((s) => s.id === compare);
  const data = useMemo(() => sampleTrace(trace.events), [trace.events]);
  const filtered = samples.filter((s) =>
    `${s.source?.task_id || s.prompt || ""} ${sampleSourceLabel(s)} ${s.status}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const tasks = [...new Set(filtered.map((s) => `${s.step}:${s.prompt_idx}`))];
  const hist = overview.stats?.histogram
    ? overview.stats.histogram.map((value, i) => ({
        label: `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`,
        value,
      }))
    : rewardHistogram(samples);
  const scored = samples.filter((s) => numeric(s.reward));
  const history = overview.history || [];
  const failure =
    overview.stats?.failures ||
    samples.reduce((acc, s) => {
      if (s.reason) acc[s.reason] = (acc[s.reason] || 0) + 1;
      else if (s.status === "failed")
        acc.execution_error = (acc.execution_error || 0) + 1;
      return acc;
    }, {});
  const agentic = isAgenticJob(job, samples);
  const hasRewards =
    (overview.stats?.scored || 0) > 0 ||
    metrics.some((m) => /reward|accuracy/.test(m.name));
  const latestMetric = (pattern) =>
    metrics.find((m) => pattern.test(m.name))?.points?.at(-1)?.value;
  const chooseStep = (n) => {
    setStep(String(n));
    setOffset(0);
  };
  const resourceSeries = (keys) =>
    timeWindowSeries(
      keys.map((key) => ({
        name: key,
        points: (overview.resources || [])
          .filter((e) => numeric(e.data[key]))
          .map((e) => ({ step: e.time, value: e.data[key] })),
      })),
      memoryWindow,
      overview.resources?.at(-1)?.time,
    );
  const jump = (t) => {
    const el = document.getElementById(`av-tool-${t.tool_id}`);
    if (el) {
      el.open = true;
      el.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  };
  return (
    <section className="av-root" aria-label="Training observability">
      <div className="av-heading">
        <div>
          <h2>Training workspace</h2>
          <p>
            {agentic
              ? "Inspect execution, compare attempts, and follow each update."
              : "Inspect samples and follow loss, gradients, memory, and each update."}
          </p>
        </div>
        <span className="av-live">
          <i />
          {["succeeded", "exited", "stopped", "failed"].includes(job.status)
            ? "Recorded history"
            : selected
              ? trace.state
              : "Live metrics"}
        </span>
      </div>
      <div className="av-kpis">
        {[
          ["Samples", overview.matched_total ?? overview.total],
          ...((overview.stats?.scored || scored.length) > 0
            ? [
                [
                  "Mean reward",
                  overview.stats?.mean_reward ??
                    (scored.length
                      ? scored.reduce((n, s) => n + s.reward, 0) / scored.length
                      : null),
                ],
                [
                  "Perfect reward",
                  overview.stats?.scored
                    ? `${overview.stats.perfect}/${overview.stats.scored}`
                    : scored.length
                      ? `${scored.filter((s) => s.reward === 1).length}/${scored.length}`
                      : "—",
                ],
              ]
            : hasRewards
              ? [
                  ["Latest reward", latestMetric(/rewards?_mean/)],
                  ["Latest loss", latestMetric(/loss/)],
                ]
              : [
                  ["Latest loss", latestMetric(/loss/)],
                  ["Grad norm", latestMetric(/grad_norm/)],
                ]),
          ["Current stage", job.stage],
          ["Step", job.step],
        ].map(([label, value]) => (
          <div key={label}>
            <small>{label}</small>
            <strong>
              {typeof value === "number"
                ? fmt(value)
                : (value ?? "Not recorded")}
            </strong>
          </div>
        ))}
      </div>
      {error && (
        <div role="alert" className="av-error">
          Live update unavailable: {error}. Retaining the last snapshot.
        </div>
      )}
      <div className="av-controls">
        <label>
          <Search size={15} />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              agentic ? "Search task or status" : "Search prompt or status"
            }
          />
        </label>
        <label>
          Step{" "}
          <input
            aria-label="Filter step"
            type="number"
            min="0"
            value={step}
            onChange={(e) => {
              setStep(e.target.value);
              setOffset(0);
            }}
            placeholder="All"
          />
        </label>
        <button
          className="secondaryButton"
          disabled={step === ""}
          onClick={() => {
            setStep("");
            setOffset(0);
          }}
        >
          All steps
        </button>
        <span>
          {overview.total ? offset + 1 : 0}–
          {Math.min(offset + 100, overview.total)} / {overview.total}
        </span>
        <button
          className="av-icon"
          disabled={!offset}
          onClick={() => setOffset(Math.max(0, offset - 100))}
        >
          ←
        </button>
        <button
          className="av-icon"
          disabled={offset + 100 >= overview.total}
          onClick={() => setOffset(offset + 100)}
        >
          →
        </button>
      </div>
      <div className="av-workspace">
        <aside className="av-samplelist">
          <div className="av-listheader">
            <strong>Sample matrix</strong>
            <small>Click to inspect · compare via +</small>
          </div>
          {tasks.length ? (
            tasks.map((key) => {
              const group = filtered.filter(
                (s) => `${s.step}:${s.prompt_idx}` === key,
              );
              return (
                <div className="av-task" key={key}>
                  <div title={group[0].prompt}>
                    <strong>
                      {group[0].source?.task_id ||
                        `${agentic ? "Task" : "Prompt"} ${group[0].prompt_idx + 1}`}
                    </strong>
                    <small>Step {group[0].step}</small>
                  </div>
                  <div className="av-samplecells">
                    {group.map((s) => (
                      <div
                        key={s.id}
                        className={`av-cell av-${s.status} ${selected === s.id ? "selected" : ""}`}
                      >
                        <button
                          onClick={() => setSelected(s.id)}
                          title={s.reason || s.status}
                        >
                          S{s.sample_idx + 1}
                          <strong>
                            {numeric(s.reward)
                              ? s.reward.toFixed(2)
                              : s.status === "completed"
                                ? "done"
                                : s.status || "queued"}
                          </strong>
                        </button>
                        <button
                          aria-label={`Compare sample ${s.sample_idx + 1}`}
                          title="Compare this sample"
                          onClick={() =>
                            setCompare(compare === s.id ? null : s.id)
                          }
                        >
                          +
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })
          ) : (
            <Empty>
              {agentic
                ? "No attempts recorded yet."
                : "No samples recorded for this job. Training metrics and progress remain available."}
            </Empty>
          )}
        </aside>
        <div className={`av-chatcolumns ${compare ? "av-comparing" : ""}`}>
          <Conversation
            events={trace.events}
            sample={current}
            agentic={agentic}
            renderMedia={renderMedia}
          />
          {compare && (
            <div>
              <button
                className="av-closecompare"
                onClick={() => setCompare(null)}
              >
                <X size={14} />
                Close comparison
              </button>
              <Conversation
                events={comparison.events}
                sample={other}
                agentic={agentic}
                renderMedia={renderMedia}
              />
            </div>
          )}
        </div>
        <aside className="av-inspector">
          <h3>Attempt details</h3>
          <dl>
            {[
              ...(agentic
                ? [
                    ["Harness", current?.harness],
                    [
                      "Turns",
                      `${data.turns.length}${current?.max_turns ? ` / ${current.max_turns}` : ""}`,
                    ],
                  ]
                : []),
              ...(hasRewards ? [["Reward", fmt(current?.reward)]] : []),
              ["Source", current?.source?.issue_url],
              ["Finish reason", current?.reason || current?.status],
              ...(agentic ? [["Tools", data.tools.length]] : []),
              [
                "Model output tokens",
                data.turns.length &&
                data.turns.every((t) => numeric(t.output_tokens))
                  ? data.turns.reduce((n, t) => n + t.output_tokens, 0)
                  : null,
              ],
            ].map(([k, v]) => (
              <React.Fragment key={k}>
                <dt>{k}</dt>
                <dd>{v ?? "Not recorded"}</dd>
              </React.Fragment>
            ))}
          </dl>
          {hasRewards && (
            <p className="av-muted">
              Reward measures task outcome. Execution errors are tracked
              separately.
            </p>
          )}
        </aside>
      </div>
      <div className="av-analytics">
        <Panel
          title="Run notices"
          note="Reported trainer events and job output"
          wide
        >
          <RunNotices
            notices={runtimeNotices(job, overview.notices)}
            onSelect={chooseStep}
          />
        </Panel>
        {(hasRewards || agentic) && (
          <>
            <Panel
              title="Learning signal"
              note="Actual scored groups · separate from success rate"
            >
              <LearningSignal overview={overview} onSelect={chooseStep} />
            </Panel>
            <Panel
              title="Sampling funnel"
              note="Episode counts · current step filter"
            >
              <SamplingFunnel funnel={overview.funnel} />
            </Panel>
            <Panel
              title="Dataset mix & outcomes"
              note="Recorded source labels and accepted training volume"
              wide
            >
              <SourceMix sources={overview.sources} onSearch={setQuery} />
            </Panel>
          </>
        )}

        {hasRewards && (
          <Panel
            title={
              (overview.stats?.scored || scored.length) > 0
                ? "Reward distribution"
                : "Reward history"
            }
            note="Recorded rewards · sample scores in [0, 1]"
          >
            {(overview.stats?.scored || scored.length) > 0 && (
              <div className="av-histogram">
                {hist.map((b) => (
                  <div key={b.label} title={`${b.label}: ${b.value}`}>
                    <strong>{b.value}</strong>
                    <i
                      style={{
                        height: `${Math.max(2, (b.value / Math.max(1, ...hist.map((x) => x.value))) * 100)}px`,
                      }}
                    />
                    <small>{b.label.split("–")[0]}</small>
                  </div>
                ))}
              </div>
            )}
            <Spark
              series={metrics.filter((m) =>
                /rewards?_mean|accuracy|success_rate/.test(m.name),
              )}
              onSelect={chooseStep}
            />
          </Panel>
        )}
        {hasRewards && (agentic || scored.length > 0) && (
          <Panel
            title="Sample reward scatter"
            note="Every recorded score on this page"
          >
            <svg
              className="av-chart"
              viewBox="0 0 640 170"
              role="img"
              aria-label="Sample rewards by step"
            >
              {[0, 0.5, 1].map((v) => (
                <g key={v}>
                  <line
                    x1="30"
                    x2="620"
                    y1={140 - v * 115}
                    y2={140 - v * 115}
                    className="av-grid"
                  />
                  <text x="10" y={144 - v * 115}>
                    {v}
                  </text>
                </g>
              ))}
              {scored.map((s, i) => (
                <circle
                  key={s.id}
                  cx={
                    45 +
                    ((s.step - Math.min(...scored.map((x) => x.step))) /
                      Math.max(
                        1,
                        Math.max(...scored.map((x) => x.step)) -
                          Math.min(...scored.map((x) => x.step)),
                      )) *
                      535 +
                    (s.sample_idx % 4) * 5
                  }
                  cy={140 - s.reward * 115}
                  r="4"
                  fill={colors[s.prompt_idx % colors.length]}
                  tabIndex="0"
                  role="button"
                  aria-label={`Step ${s.step} sample ${s.sample_idx + 1} reward ${s.reward}`}
                  onClick={() => setSelected(s.id)}
                  onKeyDown={(e) => e.key === "Enter" && setSelected(s.id)}
                >
                  <title>
                    {s.source?.task_id || `Task ${s.prompt_idx + 1}`} · Step{" "}
                    {s.step} · {s.reward}
                  </title>
                </circle>
              ))}
            </svg>
            {!scored.length && <Empty />}
          </Panel>
        )}
        {agentic && (
          <Panel title="Tests & grading" note="Selected sample">
            <Tests grader={data.grader} />
          </Panel>
        )}
        {(agentic || Object.keys(failure).length > 0) && (
          <Panel
            title="Failure reasons"
            note="Execution outcomes, separate from reward"
          >
            {Object.keys(failure).length ? (
              <div className="av-bars">
                {Object.entries(failure).map(([name, count]) => (
                  <div key={name}>
                    <span>{name.replaceAll("_", " ")}</span>
                    <i
                      style={{
                        width: `${(count / Math.max(...Object.values(failure))) * 100}%`,
                      }}
                    />
                    <strong>{count}</strong>
                  </div>
                ))}
              </div>
            ) : (
              <Empty>No failure reasons recorded on this page</Empty>
            )}
          </Panel>
        )}
        <Panel
          title="Context & token growth"
          note="Exact token counts · selected sample"
        >
          <Spark
            series={
              data.turns.some(
                (t) => numeric(t.input_tokens) || numeric(t.output_tokens),
              )
                ? ["input_tokens", "output_tokens"].map((k) => ({
                    name: k.replace("_", " "),
                    points: data.turns.map((t) => ({
                      step: t.turn + 1,
                      value: t[k],
                    })),
                  }))
                : metrics.filter((m) => /prompt_len|response_len/.test(m.name))
            }
          />
          <div className="av-budget">
            <span>Context limit</span>
            <strong>
              {fmt(job.launch?.max_context_len ?? job.config?.max_context_len)}
            </strong>
          </div>
        </Panel>
        {agentic && (
          <Panel
            title="Tool & generation timing"
            note="Select a tool to jump to its output"
          >
            {data.tools.length ? (
              <div className="av-tool-times">
                {data.tools.map((t) => (
                  <button key={t.tool_id} onClick={() => jump(t)}>
                    <span>{t.name}</span>
                    <i
                      style={{
                        width: `${Math.max(2, (100 * (t.end ? t.end - t.start : 0)) / Math.max(1, ...data.tools.map((x) => (x.end ? x.end - x.start : 0))))}%`,
                      }}
                    />
                    <small>{seconds(t.end ? t.end - t.start : null)}</small>
                  </button>
                ))}
              </div>
            ) : (
              <Empty />
            )}
            <Spark
              series={[
                {
                  name: "Model generation seconds",
                  points: data.turns
                    .filter((t) => numeric(t.seconds))
                    .map((t) => ({ step: t.turn + 1, value: t.seconds })),
                },
              ]}
            />
          </Panel>
        )}
        <Panel title="Training loss" note="Native trainer metrics">
          <Spark
            series={metrics.filter((m) => /loss/.test(m.name))}
            onSelect={chooseStep}
          />
        </Panel>
        <Panel
          title="Gradient & policy agreement"
          note="Native grad norm and logprob absolute difference"
        >
          <Spark
            series={metrics.filter((m) => /grad_norm/.test(m.name))}
            onSelect={chooseStep}
          />
          <Spark
            series={metrics.filter((m) => /logp(?:rob)?_abs_diff/.test(m.name))}
            onSelect={chooseStep}
          />
        </Panel>
        <Panel
          title="Memory residency"
          note="Unified host memory and CUDA allocations overlap; do not add"
        >
          <TimeWindow
            label="Memory"
            value={memoryWindow}
            onChange={setMemoryWindow}
          />
          <Spark
            series={resourceSeries([
              "host_used",
              "host_available",
              "cuda_allocated",
              "cuda_reserved",
            ])}
            format={gib}
            xFormat={clock}
          />
          <Spark
            series={resourceSeries(["swap_used"])}
            format={gib}
            xFormat={clock}
          />
        </Panel>
        <Panel title="MST" note="Recorded scheduler and training progress">
          <TimeWindow label="MST" value={mstWindow} onChange={setMstWindow} />
          <Spark
            series={timeWindowSeries(
              ["chunk_size"].map((key) => ({
                name: "MST chunk tokens",
                points: (overview.progress || [])
                  .filter((e) => numeric(e.data[key]))
                  .map((e) => ({ step: e.time, value: e.data[key] })),
              })),
              mstWindow,
            )}
            xFormat={clock}
          />

          <p className="av-muted">
            Scheduler progress appears when emitted by this runtime.
          </p>
        </Panel>
        {agentic && (
          <Panel
            title="Tool execution map"
            note="Actual calls, their order and outcomes"
          >
            <ToolGraph tools={data.tools} onJump={jump} />
          </Panel>
        )}
        {agentic && (
          <Panel
            title="Files & code changes"
            note="Edits captured by the harness"
          >
            <Changes tools={data.tools} />
          </Panel>
        )}
        {agentic && (
          <Panel
            title="Task learning history"
            note="Mean reward per task and step · latest 1,000 observations"
            wide
          >
            <div className="av-heatmap">
              {[...new Set(history.map((row) => row.task))].map((task) => {
                const rows = history
                  .filter((row) => row.task === task)
                  .sort((a, b) => a.step - b.step);
                return (
                  <div key={task}>
                    <strong title={task}>{task}</strong>
                    {rows.map((row) => (
                      <button
                        key={row.step}
                        className="av-heat"
                        title={`Step ${row.step} · ${row.samples} samples · Mean reward ${fmt(row.reward)}`}
                        style={{
                          background: numeric(row.reward)
                            ? `color-mix(in srgb, var(--deck-cyan) ${Math.max(8, row.reward * 85)}%, var(--deck-panel-2))`
                            : undefined,
                        }}
                        onClick={() => chooseStep(row.step)}
                      >
                        {numeric(row.reward) ? row.reward.toFixed(1) : "—"}
                      </button>
                    ))}
                  </div>
                );
              })}
            </div>
            {!history.length && <Empty />}
          </Panel>
        )}
      </div>
      <FloatingTimeline
        jobId={job.id}
        api={api}
        step={step}
        onSelect={chooseStep}
        onClear={() => {
          setStep("");
          setOffset(0);
        }}
      />
    </section>
  );
}
