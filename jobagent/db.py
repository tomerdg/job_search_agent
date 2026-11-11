"""DuckDB store: dedup by job_id, incremental load, history for analytics.

Tables (the raw layer that the dbt project in analytics/ builds on):
  jobs              one row per job, current state (first/last seen, open/closed, score, sent)
  runs              one row per agent run
  job_observations  append-only: which jobs were live in which run (open-duration history)
  source_runs       append-only: per run and source, did the fetch work and how many jobs came back
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from jobagent.models import Job

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       VARCHAR PRIMARY KEY,
    source       VARCHAR NOT NULL,
    company      VARCHAR NOT NULL,
    sector       VARCHAR,
    external_id  VARCHAR,
    title        VARCHAR NOT NULL,
    location     VARCHAR,
    department   VARCHAR,
    url          VARCHAR,
    description  VARCHAR,
    posted_at    VARCHAR,
    first_seen   TIMESTAMP NOT NULL,
    last_seen    TIMESTAMP NOT NULL,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    closed_at    TIMESTAMP,          -- set when a job disappears from a source that was fetched fine
    prefilter    BOOLEAN,            -- passed the keyword + location filter
    score        INTEGER,            -- 0-100 from the model, NULL = not scored
    analysis     VARCHAR,            -- JSON: summary, strengths, gaps
    notified_at  TIMESTAMP
);
CREATE TABLE IF NOT EXISTS runs (
    run_id       VARCHAR PRIMARY KEY,
    started_at   TIMESTAMP NOT NULL,
    finished_at  TIMESTAMP,
    n_fetched    INTEGER,
    n_new        INTEGER
);
CREATE TABLE IF NOT EXISTS job_observations (
    run_id       VARCHAR NOT NULL,
    job_id       VARCHAR NOT NULL,
    observed_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, job_id)
);
CREATE TABLE IF NOT EXISTS source_runs (
    run_id       VARCHAR NOT NULL,
    company      VARCHAR NOT NULL,
    source_type  VARCHAR NOT NULL,
    ok           BOOLEAN NOT NULL,
    n_jobs       INTEGER NOT NULL,
    duration_s   DOUBLE,
    error        VARCHAR,
    PRIMARY KEY (run_id, company, source_type)
);
-- Added later; ADD COLUMN IF NOT EXISTS migrates databases created before.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS prompt_version VARCHAR;  -- prompt that produced score
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS model VARCHAR;           -- model that produced score
"""

JOB_COLS = ["job_id", "source", "company", "sector", "external_id", "title", "location",
            "department", "url", "description", "posted_at"]


def now() -> datetime:
    """UTC timestamp without tz info, stored as DuckDB TIMESTAMP."""
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def is_sqlite_file(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    with open(p, "rb") as f:
        return f.read(16) == b"SQLite format 3\x00"


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.conn = duckdb.connect(self.path)
        self.conn.execute(SCHEMA)

    # ---- helpers ----
    def _all(self, sql: str, params: list | tuple = ()) -> list[dict]:
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _one(self, sql: str, params: list | tuple = ()) -> dict | None:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        """Checkpoint and close, so the file is complete (no WAL) for the cache and for dbt."""
        try:
            self.conn.execute("CHECKPOINT")
        finally:
            self.conn.close()

    # ---- migration ----
    def import_sqlite(self, sqlite_path: str | Path) -> int:
        """One-time copy of the old SQLite jobs table. Skipped if this DB already has jobs."""
        if self.conn.execute("SELECT count(*) FROM jobs").fetchone()[0]:
            return 0
        src = sqlite3.connect(str(sqlite_path))
        src.row_factory = sqlite3.Row
        rows = src.execute("SELECT * FROM jobs").fetchall()
        src.close()

        def ts(v):
            return datetime.fromisoformat(v).astimezone(timezone.utc).replace(tzinfo=None) if v else None

        data = [
            tuple(r[c] for c in JOB_COLS) + (
                ts(r["first_seen"]), ts(r["last_seen"]), bool(r["is_active"]),
                None if r["prefilter"] is None else bool(r["prefilter"]),
                r["score"], r["analysis"], ts(r["notified_at"]),
            )
            for r in rows
        ]
        if data:
            self.conn.executemany(
                f"INSERT INTO jobs ({', '.join(JOB_COLS)}, first_seen, last_seen, is_active, prefilter,"
                " score, analysis, notified_at) VALUES (" + ", ".join(["?"] * (len(JOB_COLS) + 7)) + ")",
                data,
            )
        log.info("Imported %d jobs from SQLite %s", len(data), sqlite_path)
        return len(data)

    # ---- runs and history ----
    def start_run(self, ts: datetime | None = None) -> str:
        ts = ts or now()
        run_id = ts.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.conn.execute("INSERT INTO runs (run_id, started_at) VALUES (?, ?)", (run_id, ts))
        return run_id

    def finish_run(self, run_id: str, n_fetched: int, n_new: int, ts: datetime | None = None) -> None:
        self.conn.execute("UPDATE runs SET finished_at=?, n_fetched=?, n_new=? WHERE run_id=?",
                          (ts or now(), n_fetched, n_new, run_id))

    def record_sources(self, run_id: str, stats: list[dict]) -> None:
        if not stats:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO source_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(run_id, s["company"], s["source_type"], bool(s["ok"]), int(s["n_jobs"]),
              s.get("duration_s"), (s.get("error") or "")[:300] or None) for s in stats],
        )

    def record_observations(self, run_id: str, job_ids: list[str], ts: datetime | None = None) -> None:
        ts = ts or now()
        ids = list(dict.fromkeys(job_ids))
        if ids:
            self.conn.executemany("INSERT OR IGNORE INTO job_observations VALUES (?, ?, ?)",
                                  [(run_id, i, ts) for i in ids])

    # ---- jobs ----
    def upsert(self, jobs: list[Job], ts: datetime | None = None) -> list[str]:
        """Insert unseen jobs, refresh last_seen on known ones. Returns ids of NEW jobs."""
        ts = ts or now()
        incoming: dict[str, Job] = {}
        for j in jobs:
            incoming.setdefault(j.job_id, j)
        if not incoming:
            return []
        self.conn.execute("BEGIN")
        try:
            self.conn.execute(
                "CREATE OR REPLACE TEMP TABLE incoming (seq INTEGER, "
                + ", ".join(f"{c} VARCHAR" for c in JOB_COLS) + ")"
            )
            self.conn.executemany(
                "INSERT INTO incoming VALUES (" + ", ".join(["?"] * (len(JOB_COLS) + 1)) + ")",
                [(i, *(getattr(j, c) for c in JOB_COLS))
                 for i, j in enumerate(incoming.values())],
            )
            new_ids = [r[0] for r in self.conn.execute(
                "SELECT i.job_id FROM incoming i ANTI JOIN jobs USING (job_id) ORDER BY i.seq").fetchall()]
            self.conn.execute(
                "UPDATE jobs SET last_seen=?, is_active=TRUE, closed_at=NULL, title=i.title,"
                " location=i.location, description=i.description FROM incoming i WHERE jobs.job_id=i.job_id",
                (ts,),
            )
            self.conn.execute(
                f"INSERT INTO jobs ({', '.join(JOB_COLS)}, first_seen, last_seen) "
                f"SELECT {', '.join('i.' + c for c in JOB_COLS)}, ?, ? FROM incoming i "
                "ANTI JOIN jobs USING (job_id) ORDER BY i.seq",
                (ts, ts),
            )
            self.conn.execute("DROP TABLE incoming")
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return new_ids

    def mark_inactive(self, companies_fetched: set[str], active_ids: set[str],
                      ts: datetime | None = None) -> None:
        """Jobs from a company we fetched successfully but that are gone now are closed."""
        ts = ts or now()
        for company in companies_fetched:
            rows = self.conn.execute(
                "SELECT job_id FROM jobs WHERE company=? AND is_active", (company,)
            ).fetchall()
            gone = [r[0] for r in rows if r[0] not in active_ids]
            if gone:
                self.conn.executemany("UPDATE jobs SET is_active=FALSE, closed_at=? WHERE job_id=?",
                                      [(ts, g) for g in gone])

    def set_prefilter(self, job_id: str, passed: bool) -> None:
        self.conn.execute("UPDATE jobs SET prefilter=? WHERE job_id=?", (bool(passed), job_id))

    def unscored_active(self) -> list[dict]:
        return self._all("SELECT * FROM jobs WHERE score IS NULL AND is_active ORDER BY first_seen, job_id")

    def get(self, job_id: str) -> dict | None:
        return self._one("SELECT * FROM jobs WHERE job_id=?", (job_id,))

    def pending_scoring(self, limit: int = 40) -> list[dict]:
        return self._all(
            "SELECT * FROM jobs WHERE prefilter AND score IS NULL AND is_active "
            "ORDER BY first_seen DESC, job_id LIMIT ?", (limit,)
        )

    def save_score(self, job_id: str, score: int, analysis: dict,
                   prompt_version: str | None = None, model: str | None = None) -> None:
        self.conn.execute(
            "UPDATE jobs SET score=?, analysis=?, prompt_version=?, model=? WHERE job_id=?",
            (int(score), json.dumps(analysis, ensure_ascii=False), prompt_version, model, job_id),
        )

    def top_unsent(self, n: int = 3, min_score: int = 50) -> list[dict]:
        return self._all(
            "SELECT * FROM jobs WHERE score >= ? AND notified_at IS NULL AND is_active "
            "ORDER BY score DESC, first_seen DESC, job_id LIMIT ?", (min_score, n)
        )

    def resolve_ids(self, prefixes: list[str]) -> list[str]:
        """Expand short ids (as shown in the digest) to full job ids."""
        out = []
        for p in prefixes:
            rows = self.conn.execute("SELECT job_id FROM jobs WHERE starts_with(job_id, ?)",
                                     (p.strip(),)).fetchall()
            out.extend(r[0] for r in rows)
        return out

    def mark_notified(self, job_ids: list[str]) -> None:
        ts = now()
        if job_ids:
            self.conn.executemany("UPDATE jobs SET notified_at=? WHERE job_id=?", [(ts, i) for i in job_ids])


def open_store(path: str | Path) -> Store:
    """Open the DuckDB store. An old SQLite jobs.db is imported once into jobs.duckdb next to it."""
    path = Path(path)
    legacy = None
    if is_sqlite_file(path):  # JOBS_DB still points at the old SQLite file
        legacy, path = path, path.with_suffix(".duckdb")
        log.warning("%s is SQLite; using %s instead", legacy, path)
    elif not path.exists() and is_sqlite_file(path.with_name("jobs.db")):
        legacy = path.with_name("jobs.db")
    store = Store(path)
    if legacy is not None:
        t0 = time.time()
        n = store.import_sqlite(legacy)
        if n:
            log.info("Migrated %d jobs from SQLite in %.1fs", n, time.time() - t0)
    return store
