"""Small SQLite store. Jobs survive browser and local-server restarts."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = directory
        self.lock = threading.RLock()
        self.db = sqlite3.connect(directory / "areno.dashboard.flow.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS functions (id TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT, body TEXT);
          CREATE INDEX IF NOT EXISTS events_job ON events(job, id);
          CREATE TABLE IF NOT EXISTS metric_events (id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT, body TEXT);
          CREATE UNIQUE INDEX IF NOT EXISTS metric_events_unique ON metric_events(job, body);
          CREATE INDEX IF NOT EXISTS metric_events_job ON metric_events(job, id);
          INSERT OR IGNORE INTO metric_events(job, body)
            SELECT job, body FROM events WHERE json_extract(body, '$.type') = 'metric';
        """)
        self.db.commit()

    def save_plan(self, identifier, request, expires, status="proposed"):
        safe = {key: value for key, value in request.items() if key != "endpoint_key"}
        body = json.dumps({"request": safe, "expires": expires, "status": status})
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO plans VALUES (?, ?)", (identifier, body))

    def get_plan(self, identifier):
        with self.lock:
            row = self.db.execute("SELECT body FROM plans WHERE id=?", (identifier,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, record):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?)", (record["id"], json.dumps(record)))

    def get(self, job_id):
        with self.lock:
            row = self.db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError("Job not found")
        return json.loads(row[0])

    def update(self, job_id, **changes):
        with self.lock:
            record = self.get(job_id)
            record.update(changes, updated_at=time.time())
            self.put(record)
            return record

    def jobs(self):
        with self.lock:
            result = [json.loads(row[0]) for row in self.db.execute("SELECT body FROM jobs")]
        return sorted(result, key=lambda r: r["created_at"], reverse=True)

    def event(self, job_id, event):
        with self.lock, self.db:
            body = json.dumps(event, sort_keys=True)
            if event.get("type") == "metric":
                self.db.execute("INSERT OR IGNORE INTO metric_events(job, body) VALUES (?, ?)", (job_id, body))
            # Modal replays stdout on reattachment. Preserve original remote timestamps
            # and ignore repeated structured events so charts do not double-count steps.
            if (
                event.get("type") != "log"
                and self.db.execute("SELECT 1 FROM events WHERE job=? AND body=? LIMIT 1", (job_id, body)).fetchone()
            ):
                return
            self.db.execute("INSERT INTO events(job, body) VALUES (?, ?)", (job_id, body))
            # Keep a bounded tail per run; complete stdout remains available in Modal.
            self.db.execute(
                "DELETE FROM events WHERE job=? AND id NOT IN "
                "(SELECT id FROM events WHERE job=? ORDER BY id DESC LIMIT 20000)",
                (job_id, job_id),
            )

    def events(self, job_id, after=0):
        with self.lock:
            rows = self.db.execute(
                "SELECT id, body FROM events WHERE job=? AND id>? ORDER BY id LIMIT 2000", (job_id, after)
            ).fetchall()
        return [dict(json.loads(body), cursor=cursor) for cursor, body in rows]

    def metrics(self, job_id):
        """Durable metric history, independent of the bounded stdout event tail."""
        with self.lock:
            rows = self.db.execute("SELECT body FROM metric_events WHERE job=? ORDER BY id", (job_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def latest_metrics(self, job_id):
        with self.lock:
            rows = self.db.execute(
                "SELECT body FROM metric_events WHERE id IN (SELECT MAX(id) FROM metric_events "
                "WHERE job=? GROUP BY json_extract(body, '$.tag'), json_extract(body, '$.index'))",
                (job_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def datasets(self):
        with self.lock:
            rows = self.db.execute("SELECT body FROM datasets").fetchall()
        return sorted((json.loads(row[0]) for row in rows), key=lambda d: d["updated_at"], reverse=True)

    def save_dataset(self, record):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO datasets VALUES (?, ?)", (record["id"], json.dumps(record)))
        return record

    def delete_dataset(self, identifier):
        with self.lock, self.db:
            if not self.db.execute("DELETE FROM datasets WHERE id=?", (identifier,)).rowcount:
                raise KeyError("Dataset not found")
        return {"deleted": identifier}

    def functions(self):
        with self.lock:
            rows = self.db.execute("SELECT body FROM functions").fetchall()
        return sorted((json.loads(row[0]) for row in rows), key=lambda f: f["updated_at"], reverse=True)

    def save_function(self, record):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO functions VALUES (?, ?)", (record["id"], json.dumps(record)))
        return record

    def save_functions(self, records):
        with self.lock, self.db:
            self.db.executemany("INSERT INTO functions VALUES (?, ?)", [(r["id"], json.dumps(r)) for r in records])
        return records

    def delete_function(self, identifier):
        with self.lock, self.db:
            if any(d.get("loader_id") == identifier for d in self.datasets()):
                raise ValueError("This function is assigned to a dataset. Change its loader before deleting it.")
            if not self.db.execute("DELETE FROM functions WHERE id=?", (identifier,)).rowcount:
                raise KeyError("Function not found")
        return {"deleted": identifier}
