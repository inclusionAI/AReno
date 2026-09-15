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
        self.db = sqlite3.connect(directory / "arenoflow.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS functions (id TEXT PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT, body TEXT);
          CREATE INDEX IF NOT EXISTS events_job ON events(job, id);
        """)
        self.db.commit()

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
