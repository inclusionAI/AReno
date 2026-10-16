import React, { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Box, Braces, Check, ChevronDown, Copy, Search, X } from "lucide-react";
import { valueText } from "./agentic-data";

export function CopyAction({ text, label = "复制正文" }) {
  const [state, setState] = useState("");
  useEffect(() => {
    if (!state) return;
    const timer = setTimeout(() => setState(""), 1800);
    return () => clearTimeout(timer);
  }, [state]);
  return (
    <button
      className="tv-action"
      title={label}
      aria-label={label}
      onClick={async (event) => {
        event.preventDefault();
        event.stopPropagation();
        try {
          await navigator.clipboard.writeText(text ?? "");
          setState("已复制");
        } catch {
          setState("复制不可用，请选择文本");
        }
      }}
    >
      {state === "已复制" ? <Check size={15} /> : <Copy size={15} />}
      <span role="status">{state || label}</span>
    </button>
  );
}
function JsonNode({ name, value, depth = 0, command, query = "" }) {
  const [open, setOpen] = useState(depth < 2);
  useEffect(() => {
    if (command) setOpen(command.open);
  }, [command]);
  const keyMatches =
    name != null && String(name).toLocaleLowerCase().includes(query);
  if (
    query &&
    !keyMatches &&
    !JSON.stringify(value)?.toLocaleLowerCase().includes(query)
  )
    return null;
  const label =
    name == null ? null : (
      <>
        <span className="tv-json-key">{JSON.stringify(String(name))}</span>
        <span>: </span>
      </>
    );
  if (value === null || typeof value !== "object")
    return (
      <div className="tv-json-leaf">
        {label}
        <span className={`tv-json-${value === null ? "null" : typeof value}`}>
          {JSON.stringify(value) ?? "null"}
        </span>
      </div>
    );
  const array = Array.isArray(value),
    entries = Object.entries(value),
    opening = array ? "[" : "{",
    closing = array ? "]" : "}";
  return (
    <details
      className="tv-json-node"
      open={query ? true : open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <ChevronDown size={14} />
        {label}
        <span>{opening}</span>
        <small>{entries.length} items</small>
      </summary>
      {(open || query) && (
        <div className="tv-json-children">
          {entries.map(([key, child]) => (
            <JsonNode
              key={key}
              name={array ? null : key}
              value={child}
              depth={depth + 1}
              command={command}
              query={keyMatches ? "" : query}
            />
          ))}
          <div>{closing}</div>
        </div>
      )}
    </details>
  );
}
export function JsonPreview({ data }) {
  const root = useRef(null);
  const [query, setQuery] = useState(""),
    [text, setText] = useState(false),
    [command, setCommand] = useState(null);
  const full = valueText(data);
  const matches =
    !query.trim() ||
    full.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase());
  return (
    <section className="tv-json-preview" ref={root} aria-label="JSON Preview">
      <header>
        <span className="tv-json-title">
          <Braces size={17} />
          JSON PREVIEW
        </span>
        <div className="tv-json-expand">
          <button onClick={() => setCommand({ open: true })}>展开全部</button>
          <button onClick={() => setCommand({ open: false })}>折叠全部</button>
        </div>
        <CopyAction text={full} label="复制完整 JSON" />
        <label className="tv-search">
          <Search size={16} />
          <input
            aria-label="搜索 JSON Key 或 Value"
            placeholder="搜索 Key 或 Value…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <div className="tv-modes" role="group" aria-label="JSON 显示方式">
          <button
            aria-pressed={!text}
            className={!text ? "active" : ""}
            onClick={() => setText(false)}
          >
            结构树
          </button>
          <button
            aria-pressed={text}
            className={text ? "active" : ""}
            onClick={() => setText(true)}
          >
            纯文本
          </button>
        </div>
      </header>
      <div className="tv-json-body">
        {!matches ? (
          <p className="tv-empty">没有匹配的 Key 或 Value</p>
        ) : text ? (
          <pre>{full}</pre>
        ) : (
          <JsonNode
            value={data}
            command={command}
            query={query.trim().toLocaleLowerCase()}
          />
        )}
      </div>
    </section>
  );
}
export function TrajectoryPopup({ kind, tools, atif, events, onClose }) {
  const ref = useRef(null),
    title = useId();
  const [raw, setRaw] = useState(false);
  useEffect(() => {
    const previous = document.activeElement,
      overflow = document.body.style.overflow;
    ref.current.showModal();
    document.body.style.overflow = "hidden";
    return () => {
      ref.current?.close();
      document.body.style.overflow = overflow;
      previous?.focus();
    };
  }, []);
  const data = kind === "tools" ? tools : raw ? events : atif;
  return createPortal(
    <dialog
      className="tv-json-modal"
      ref={ref}
      aria-labelledby={title}
      onCancel={onClose}
      onClose={onClose}
      onClick={(event) => {
        if (event.target !== ref.current) return;
        const bounds = ref.current.getBoundingClientRect();
        if (
          event.clientX < bounds.left ||
          event.clientX > bounds.right ||
          event.clientY < bounds.top ||
          event.clientY > bounds.bottom
        )
          onClose();
      }}
    >
      <header>
        {kind === "tools" ? <Box size={22} /> : <Braces size={22} />}
        <h2 id={title}>
          {kind === "tools"
            ? `工具清单 (${tools.length})`
            : raw
              ? "原始 AReno 事件"
              : "ATIF 轨迹结构"}
        </h2>
        {kind === "source" && (
          <button className="tv-text-action" onClick={() => setRaw(!raw)}>
            {raw ? "查看 ATIF" : "原始事件"}
          </button>
        )}
        <CopyAction
          text={valueText(data)}
          label={kind === "tools" ? "复制 Tools" : "复制全部 ATIF JSON"}
        />
        <button className="tv-action" aria-label="关闭弹窗" onClick={onClose}>
          <X size={19} />
        </button>
      </header>
      <div className="tv-popup-content">
        <JsonPreview key={raw ? "raw" : kind} data={data} />
      </div>
    </dialog>,
    document.body,
  );
}
