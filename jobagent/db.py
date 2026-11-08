"""SQLite store: dedup by job_id, incremental load, remembers scores and what was sent."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from jobagent.models import Job

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    company      TEXT NOT NULL,
    sector       TEXT,
    external_id  TEXT,
    title        TEXT NOT NULL,
    location     TEXT,
    department   TEXT,
    url          TEXT,
    description  TEXT,
    posted_at    TEXT,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    is_active    INTEGER NOT NULL DEFAULT 1,
    prefilter    INTEGER,            -- 1 passed keyword+location filter, 0 rejected
    score        INTEGER,            -- 0-100 from the model, NULL = not scored
    analysis     TEXT,               -- JSON: summary, strengths, gaps
    notified_at  TEXT
);
CREATE INDEX IF NOT EXISTS ix_jobs_first_seen ON jobs(first_seen);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def upsert(self, jobs: list[Job]) -> list[str]:
        """Insert unseen jobs, refresh last_seen on known ones. Returns ids of NEW jobs."""
        ts = now()
        new_ids = []
        seen_ids = set()
        for j in jobs:
            if j.job_id in seen_ids:
                continue
            seen_ids.add(j.job_id)
            cur = self.conn.execute(
                "UPDATE jobs SET last_seen=?, is_active=1, title=?, location=?, description=? "
                "WHERE job_id=?",
                (ts, j.title, j.location, j.description, j.job_id),
            )
            if cur.rowcount == 0:
                self.conn.execute(
                    "INSERT INTO jobs (job_id, source, company, sector, external_id, title, location,"
                    " department, url, description, posted_at, first_seen, last_seen)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (j.job_id, j.source, j.company, j.sector, j.external_id, j.title, j.location,
                     j.department, j.url, j.description, j.posted_at, ts, ts),
                )
                new_ids.append(j.job_id)
        self.conn.commit()
        return new_ids

    def mark_inactive(self, companies_fetched: set[str], active_ids: set[str]) -> None:
        """Jobs from a company we fetched successfully but that are gone now are closed."""
        for company in companies_fetched:
            rows = self.conn.execute(
                "SELECT job_id FROM jobs WHERE company=? AND is_active=1", (company,)
            ).fetchall()
            gone = [r["job_id"] for r in rows if r["job_id"] not in active_ids]
            self.conn.executemany("UPDATE jobs SET is_active=0 WHERE job_id=?", [(g,) for g in gone])
        self.conn.commit()

    def set_prefilter(self, job_id: str, passed: bool) -> None:
        self.conn.execute("UPDATE jobs SET prefilter=? WHERE job_id=?", (int(passed), job_id))
        self.conn.commit()

    def unscored_active(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM jobs WHERE score IS NULL AND is_active=1").fetchall()

    def get(self, job_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()

    def pending_scoring(self, limit: int = 40) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM jobs WHERE prefilter=1 AND score IS NULL AND is_active=1 "
            "ORDER BY first_seen DESC LIMIT ?", (limit,)
        ).fetchall()

    def save_score(self, job_id: str, score: int, analysis: dict) -> None:
        self.conn.execute(
            "UPDATE jobs SET score=?, analysis=? WHERE job_id=?",
            (int(score), json.dumps(analysis, ensure_ascii=False), job_id),
        )
        self.conn.commit()

    def top_unsent(self, n: int = 3, min_score: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM jobs WHERE score >= ? AND notified_at IS NULL AND is_active=1 "
            "ORDER BY score DESC, first_seen DESC LIMIT ?", (min_score, n)
        ).fetchall()

    def mark_notified(self, job_ids: list[str]) -> None:
        ts = now()
        self.conn.executemany("UPDATE jobs SET notified_at=? WHERE job_id=?", [(ts, i) for i in job_ids])
        self.conn.commit()
