import React, { useMemo, useRef, useState } from "react";
import {
  ChevronDown,
  FileJson,
  Folder,
  FolderOpen,
  Layers,
  Plus,
  Search,
} from "lucide-react";
import { numeric } from "./agentic-data";
import { snapshotGroups, snapshotName } from "./trajectory-data";

export function SnapshotTree({
  samples,
  total,
  selected,
  compare,
  onSelect,
  onCompare,
  width = 260,
  onWidthChange,
  collapsed = false,
  onToggle,
}) {
  const root = useRef(null),
    drag = useRef(null);
  const [query, setQuery] = useState("");
  const groups = useMemo(
    () => snapshotGroups(samples, query),
    [samples, query],
  );
  return (
    <aside
      className={`tv-snapshots ${collapsed ? "collapsed" : ""}`}
      aria-label="Snapshots"
      ref={root}
    >
      <header>
        <Layers size={18} />
        <strong>SNAPSHOTS</strong>
        <span>
          {samples.length === total ? total : `${samples.length} / ${total}`}
        </span>
        <button
          className="tv-tree-toggle"
          aria-label={collapsed ? "展开 SNAPSHOTS" : "收起 SNAPSHOTS"}
          aria-expanded={!collapsed}
          title={collapsed ? "展开 SNAPSHOTS" : "收起 SNAPSHOTS"}
          onClick={onToggle}
        >
          {collapsed ? "›" : "‹"}
        </button>
      </header>
      {!collapsed && (
        <>
          <label className="tv-search">
            <Search size={16} />
            <input
              aria-label="Search snapshots"
              placeholder="Search by filename or task…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <div className="tv-tree-controls">
            <button
              onClick={() => {
                for (const node of root.current.querySelectorAll("details"))
                  node.open = true;
              }}
            >
              展开目录
            </button>
            <button
              onClick={() => {
                for (const node of root.current.querySelectorAll("details"))
                  node.open = false;
              }}
            >
              合上目录
            </button>
          </div>
          <div className="tv-tree">
            {groups.map((group) => (
              <details key={group.step} open>
                <summary>
                  <ChevronDown size={14} />
                  <FolderOpen className="tv-folder" size={17} />
                  <strong>STEP {group.step}</strong>
                  <small>
                    {group.tasks.reduce(
                      (n, task) => n + task.samples.length,
                      0,
                    )}
                  </small>
                </summary>
                <div className="tv-tree-children">
                  {group.tasks.map((task) => (
                    <details key={task.key} open>
                      <summary title={task.name}>
                        <ChevronDown size={14} />
                        <Folder className="tv-folder" size={17} />
                        <strong>{task.name}</strong>
                        <small>{task.samples.length}</small>
                      </summary>
                      <div className="tv-tree-children">
                        {task.samples.map((sample) => (
                          <div
                            key={sample.id}
                            className={`tv-snapshot ${selected === sample.id ? "selected" : ""}`}
                          >
                            <button
                              onClick={() => onSelect(sample.id)}
                              aria-current={
                                selected === sample.id ? "true" : undefined
                              }
                              title={`${snapshotName(sample)} · ${sample.status}${numeric(sample.reward) ? ` · reward ${sample.reward}` : ""}`}
                            >
                              <FileJson size={16} />
                              <span>{snapshotName(sample)}</span>
                              <small>
                                {numeric(sample.reward)
                                  ? sample.reward.toFixed(2)
                                  : sample.status}
                              </small>
                            </button>
                            <button
                              className={compare === sample.id ? "active" : ""}
                              aria-label={`Compare sample ${sample.sample_idx + 1}`}
                              aria-pressed={compare === sample.id}
                              title="比较此轨迹"
                              onClick={() =>
                                onCompare(
                                  compare === sample.id ? null : sample.id,
                                )
                              }
                            >
                              <Plus size={15} />
                            </button>
                          </div>
                        ))}
                      </div>
                    </details>
                  ))}
                </div>
              </details>
            ))}
            {!groups.length && (
              <p className="tv-empty">
                {samples.length ? "没有匹配的轨迹" : "等待捕获轨迹…"}
              </p>
            )}
          </div>
          <div
            className="tv-tree-resizer"
            role="separator"
            tabIndex="0"
            aria-label="调整 SNAPSHOTS 宽度"
            aria-orientation="vertical"
            aria-valuemin={180}
            aria-valuemax={520}
            aria-valuenow={width}
            onPointerDown={(event) => {
              if (event.button !== 0) return;
              event.preventDefault();
              drag.current = { id: event.pointerId, x: event.clientX, width };
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
            onPointerMove={(event) => {
              if (drag.current?.id !== event.pointerId) return;
              const max = Math.max(
                180,
                Math.min(520, root.current.parentElement.clientWidth - 360),
              );
              onWidthChange?.(
                Math.min(
                  max,
                  Math.max(
                    180,
                    drag.current.width + event.clientX - drag.current.x,
                  ),
                ),
              );
            }}
            onPointerUp={(event) => {
              if (drag.current?.id === event.pointerId) {
                drag.current = null;
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
            onPointerCancel={() => {
              drag.current = null;
            }}
            onKeyDown={(event) => {
              const changes = {
                ArrowLeft: width - 16,
                ArrowRight: width + 16,
                Home: 180,
                End: 520,
              };
              if (changes[event.key] == null) return;
              event.preventDefault();
              onWidthChange?.(Math.min(520, Math.max(180, changes[event.key])));
            }}
          />
        </>
      )}
    </aside>
  );
}
