"""Incremental, disk-backed indexes for complete agentic job histories."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


def rollout_events(sample, identifier):
    """Display existing rollout captures in the same workspace as harness traces."""
    prompt = sample.get("prompt") or sample.get("decoded_prompt") or ""
    output = next(
        (sample[key] for key in ("completion", "rendered_completion", "final_answer") if sample.get(key)),
        "",
    )
    data = dict(
        prompt=prompt,
        status="completed",
        mode=sample.get("kind", "rollout"),
        source=sample.get("source_record") or {},
        capture=sample,
    )
    if sample.get("reward") is not None:
        data["reward"] = sample["reward"]
    common = dict(
        step=int(sample.get("step", 0)),
        prompt_idx=int(sample.get("prompt_idx", 0)),
        sample_idx=int(sample.get("sample_idx", 0)),
        time=sample.get("time"),
    )
    return [
        dict(common, id=identifier + ":sample", kind="sample", data=data),
        dict(
            common,
            id=identifier + ":turn",
            kind="turn_snapshot",
            data=dict(
                turn=0,
                messages=sample.get("prompt_messages") or [],
                response=dict(content=output, tool_calls=sample.get("tool_calls") or []),
                input_tokens=sample.get("input_token_count"),
                output_tokens=sample.get("output_token_count"),
            ),
        ),
    ]


def timeline_overview(native, stages, *, step=None, limit=10):
    """One timeline per step, uniformly sampled without discarding searchable history."""
    rows = {}
    for event, following in zip(stages, stages[1:]):
        stage = event.get("data", {}).get("stage", "")
        elapsed = following["time"] - event["time"]
        if not stage.endswith("_start") or event["step"] != following["step"] or elapsed <= 0:
            continue
        row = rows.setdefault(event["step"], dict(step=event["step"], segments=[]))
        row["segments"].append(dict(name=stage.removesuffix("_start"), seconds=elapsed))
    for row in native:
        segments = [segment for segment in row.get("segments", []) if segment.get("seconds", 0) > 0]
        if segments:
            rows[row["step"]] = dict(step=row["step"], segments=segments)
    averages = {}
    for row in rows.values():
        for segment in row["segments"]:
            averages[segment["name"]] = averages.get(segment["name"], 0) + segment["seconds"]
    average = dict(
        step="avg",
        segments=[dict(name=name, seconds=seconds / len(rows)) for name, seconds in averages.items()],
        count=len(rows),
    )
    ordered = [rows[number] for number in sorted(rows) if step is None or number == step]
    total = len(ordered)
    limit = min(10, max(2, limit))
    if total > limit:
        ordered = [ordered[round(i * (total - 1) / (limit - 1))] for i in range(limit)]
    return dict(rows=ordered, total=total, average=average)


class EventIndex:
    def __init__(self, directory: Path, pid=None, *, rollout_files=()):
        self.directory = directory
        self.pid = pid
        self.rollout_files = rollout_files
        self.lock = threading.Lock()
        self.offsets = {}
        self.db = sqlite3.connect(directory / f".agent-index.{pid or 'all'}.sqlite", check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, id TEXT UNIQUE, step INTEGER, sample TEXT, kind TEXT, time REAL, body TEXT)"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS offsets (path TEXT PRIMARY KEY, inode INTEGER, position INTEGER)")
        self.offsets = {
            Path(path): (inode, position)
            for path, inode, position in self.db.execute("SELECT path,inode,position FROM offsets")
        }
        self.db.execute("CREATE TABLE IF NOT EXISTS summaries (sample TEXT PRIMARY KEY, step INTEGER, body TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS batches (step INTEGER PRIMARY KEY, body TEXT)")
        self.db.execute("CREATE INDEX IF NOT EXISTS summaries_step ON summaries(step)")
        self.db.execute("CREATE INDEX IF NOT EXISTS sample_events ON events(sample,seq)")
        self.db.execute("CREATE INDEX IF NOT EXISTS step_events ON events(step,seq)")

    def refresh(self):
        files = sorted(self.directory.glob("agent_events.*.jsonl"))
        if self.pid is not None:
            prefix = f"agent_events.{self.pid}"
            files = [p for p in files if p.name == prefix + ".jsonl" or p.name.startswith(prefix + ".")]
        for path in [*files, *self.rollout_files]:
            stat = path.stat()
            inode, offset = self.offsets.get(path, (stat.st_ino, 0))
            if inode != stat.st_ino or offset > stat.st_size:
                offset = 0
            with path.open("rb") as stream:
                stream.seek(offset)
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    offset = stream.tell()
                    try:
                        raw = json.loads(line)
                        if path in self.rollout_files:
                            if self.pid is not None and raw.get("pid") not in (None, self.pid):
                                continue
                            key = f"{raw.get('step', 0)}:{raw.get('prompt_idx', 0)}:{raw.get('sample_idx', 0)}"
                            # Structured harness samples include richer, live histories.
                            if self.db.execute("SELECT 1 FROM summaries WHERE sample=?", (key,)).fetchone():
                                continue
                            events = rollout_events(raw, f"rollout:{path.name}:{stat.st_ino}:{offset}")
                        else:
                            events = [raw]
                        for e in events:
                            self._insert(e)
                    except (ValueError, KeyError, TypeError):
                        continue
            self.offsets[path] = (stat.st_ino, offset)
            self.db.execute("INSERT OR REPLACE INTO offsets VALUES(?,?,?)", (str(path), stat.st_ino, offset))
        self.db.commit()

    def _insert(self, event):
        e = event
        sample = None if e.get("prompt_idx") is None else f"{e['step']}:{e['prompt_idx']}:{e['sample_idx']}"
        inserted = self.db.execute(
            "INSERT OR IGNORE INTO events(id,step,sample,kind,time,body) VALUES(?,?,?,?,?,?)",
            (e["id"], e["step"], sample, e["kind"], e.get("time"), json.dumps(e)),
        )
        if inserted.rowcount and (
            e["kind"] == "batch" or (e["kind"] == "stage" and "trained_samples" in e.get("data", {}))
        ):
            existing = self.db.execute("SELECT body FROM batches WHERE step=?", (e["step"],)).fetchone()
            batch = json.loads(existing[0]) if existing else {}
            if e["kind"] == "batch":
                batch.update(e["data"])
                batch.update(step=e["step"], time=e["time"])
            else:
                batch["trained"] = e["data"]["trained_samples"]
            self.db.execute("INSERT OR REPLACE INTO batches VALUES(?,?)", (e["step"], json.dumps(batch)))
        if not inserted.rowcount or not sample or e["kind"] not in {"sample", "reward"}:
            return
        row = self.db.execute("SELECT body FROM summaries WHERE sample=?", (sample,)).fetchone()
        summary = (
            json.loads(row[0])
            if row
            else dict(id=sample, step=e["step"], prompt_idx=e["prompt_idx"], sample_idx=e["sample_idx"])
        )
        summary.update(e.get("data", {}))
        summary["time"] = e.get("time")
        summary.pop("result", None)
        self.db.execute("INSERT OR REPLACE INTO summaries VALUES(?,?,?)", (sample, e["step"], json.dumps(summary)))

    def read(self, *, after=0, sample=None, limit=250):
        with self.lock:
            self.refresh()
            sql = "SELECT seq,body FROM events WHERE seq>?"
            args = [max(0, after)]
            if sample is not None:
                sql += " AND sample=?"
                args.append(sample)
            sql += " ORDER BY seq LIMIT ?"
            args.append(min(max(limit, 1), 1000))
            rows = self.db.execute(sql, args).fetchall()
            return [dict(json.loads(body), cursor=seq) for seq, body in rows]

    def stages(self):
        with self.lock:
            self.refresh()
            return [
                json.loads(body)
                for (body,) in self.db.execute("SELECT body FROM events WHERE kind='stage' ORDER BY seq")
            ]

    def overview(self, *, step=None, offset=0, limit=100, step_limit=None):
        with self.lock:
            self.refresh()
            where, args = (" AND step=?", [step]) if step is not None else ("", [])
            filter_sql = " WHERE step=?" if step is not None else ""
            total = self.db.execute("SELECT COUNT(*) FROM summaries" + filter_sql, args).fetchone()[0]
            matched_total = total
            sample_filter, sample_args = filter_sql, args
            if step is None and step_limit is not None:
                numbers = [row[0] for row in self.db.execute("SELECT DISTINCT step FROM summaries ORDER BY step")]
                count = min(max(step_limit, 1), 10, len(numbers))
                chosen = (
                    [numbers[round(i * (len(numbers) - 1) / (count - 1))] for i in range(count)]
                    if count > 1
                    else numbers[:count]
                )
                if chosen:
                    sample_filter = " WHERE step IN (" + ",".join("?" for _ in chosen) + ")"
                    sample_args = chosen
                total = self.db.execute("SELECT COUNT(*) FROM summaries" + sample_filter, sample_args).fetchone()[0]
            samples = [
                json.loads(body)
                for (body,) in self.db.execute(
                    "SELECT body FROM summaries" + sample_filter + " ORDER BY step DESC,sample LIMIT ? OFFSET ?",
                    [*sample_args, min(max(limit, 1), 500), max(offset, 0)],
                )
            ]
            scored_count, mean_reward, perfect = self.db.execute(
                "SELECT COUNT(json_extract(body,'$.reward')), AVG(json_extract(body,'$.reward')), "
                "COALESCE(SUM(json_extract(body,'$.reward')=1),0) FROM summaries" + filter_sql,
                args,
            ).fetchone()
            histogram = [0] * 10
            for bucket, count in self.db.execute(
                "SELECT MIN(9, CAST(json_extract(body,'$.reward')*10 AS INTEGER)), COUNT(*) FROM summaries "
                "WHERE json_type(body,'$.reward') IN ('integer','real') AND json_extract(body,'$.reward') BETWEEN 0 AND 1"
                + where
                + " GROUP BY 1",
                args,
            ):
                histogram[bucket] = count
            failures = dict(
                self.db.execute(
                    "SELECT COALESCE(json_extract(body,'$.reason'),'execution_error'), COUNT(*) FROM summaries "
                    "WHERE json_extract(body,'$.status') IN ('failed','filtered')" + where + " GROUP BY 1",
                    args,
                )
            )
            history = [
                dict(task=task, step=number, reward=reward, samples=count, id=identifier)
                for task, number, reward, count, identifier in self.db.execute(
                    "SELECT COALESCE(json_extract(body,'$.source.task_id'),CAST(json_extract(body,'$.prompt_idx') AS TEXT)), "
                    "step, AVG(json_extract(body,'$.reward')), COUNT(*), MIN(sample) FROM summaries "
                    "GROUP BY 1,step ORDER BY step DESC LIMIT 1000"
                )
            ]
            stats = dict(
                scored=scored_count, mean_reward=mean_reward, perfect=perfect, histogram=histogram, failures=failures
            )
            stages = [
                dict(json.loads(b), cursor=seq)
                for seq, b in self.db.execute(
                    "SELECT seq,body FROM events WHERE kind='stage' ORDER BY seq DESC LIMIT 500"
                ).fetchall()
            ][::-1]
            resources = [
                json.loads(b)
                for (b,) in self.db.execute(
                    "SELECT body FROM events WHERE kind='resource' ORDER BY seq DESC LIMIT 1000"
                ).fetchall()
            ][::-1]
            progress = [
                json.loads(b)
                for (b,) in self.db.execute(
                    "SELECT body FROM events WHERE kind='progress' ORDER BY seq DESC LIMIT 1000"
                ).fetchall()
            ][::-1]
            aggregate_keys = ["requested", "returned", "scored", "accepted", "trained"]
            sums = self.db.execute(
                "SELECT "
                + ",".join(f"SUM(json_extract(body,'$.{key}'))" for key in aggregate_keys)
                + " FROM batches"
                + filter_sql,
                args,
            ).fetchone()
            funnel = dict(zip(aggregate_keys, sums))
            learning_keys = ["zero", "one", "varying", "flat_other", "insufficient"]
            sums = self.db.execute(
                "SELECT "
                + ",".join(f"SUM(json_extract(body,'$.learning.{key}'))" for key in learning_keys)
                + " FROM batches"
                + filter_sql,
                args,
            ).fetchone()
            learning = dict(zip(learning_keys, sums))
            sources = [
                dict(label=label, samples=count, reward=reward, train_rows=rows, train_tokens=tokens)
                for label, count, reward, rows, tokens in self.db.execute(
                    "SELECT json_extract(source.value,'$.label'), SUM(json_extract(source.value,'$.samples')), "
                    "SUM(json_extract(source.value,'$.reward_sum')) / (1.0 * SUM(json_extract(source.value,'$.samples'))), "
                    "SUM(json_extract(source.value,'$.train_rows')), SUM(json_extract(source.value,'$.train_tokens')) "
                    "FROM batches JOIN json_each(batches.body,'$.sources') AS source"
                    + (" WHERE batches.step=?" if step is not None else "")
                    + " GROUP BY 1 ORDER BY 2 DESC",
                    args,
                )
            ]
            batches = [
                json.loads(body)
                for (body,) in self.db.execute(
                    "SELECT body FROM batches" + filter_sql + " ORDER BY step DESC LIMIT 1000", args
                )
            ][::-1]
            notices = [
                json.loads(body)
                for (body,) in self.db.execute(
                    "SELECT body FROM events WHERE (kind='notice' OR (kind='stage' AND "
                    "(json_extract(body,'$.data.status') IN ('failed','stopped') OR json_extract(body,'$.data.stage') LIKE '%skip%')))"
                    + where
                    + " ORDER BY seq DESC LIMIT 30",
                    args,
                )
            ]
            cursor = self.db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
            return dict(
                samples=samples,
                funnel=funnel,
                learning=learning,
                sources=sources,
                batches=batches,
                notices=notices,
                stats=stats,
                history=history,
                total=total,
                matched_total=matched_total,
                offset=offset,
                stages=stages,
                resources=resources,
                progress=progress,
                cursor=cursor,
            )
