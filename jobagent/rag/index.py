"""Incremental index of JD chunks and their embeddings in jobs.duckdb.

Every run re-chunks the eligible jobs (cheap, no model) and diffs the result against job_chunks by chunk_id,
a hash of job_id and the embedded text. So:
  - an unchanged JD costs nothing: no rows written, nothing embedded;
  - a changed JD loses the chunks that changed and gets the new ones; unchanged chunks keep their vectors;
  - only chunks without a vector for the current model are embedded (a new model re-embeds everything).
Closed jobs keep their chunks, because the history is the point; retrieval filters on jobs.is_active.
Only counts are logged, never chunk text.
"""
from __future__ import annotations

import hashlib
import logging
import time

import pandas as pd

from jobagent.db import Store, now
from jobagent.rag.chunking import CHUNKING_VERSION, chunk_jd, content_hash, embed_text
from jobagent.rag.embed import Embedder

log = logging.getLogger(__name__)

EMBED_BATCH = 256  # chunks per embed call and per commit, so a crash keeps what was embedded


def chunk_id(job_id: str, chash: str) -> str:
    return hashlib.sha1(f"{job_id}\x1f{chash}".encode()).hexdigest()[:16]


def eligible_jobs(store: Store) -> pd.DataFrame:
    """Active jobs worth indexing: passed the pre-filter, or have a data-ish title (role_family, the same
    rules as the relevance model and dbt's int_job_roles), so the index is not limited to today's profile."""
    from jobagent.ml.features import role_family
    df = store.conn.execute(
        "SELECT job_id, company, title, description, prefilter FROM jobs WHERE is_active ORDER BY job_id").df()
    if df.empty:
        return df
    keep = df["prefilter"].fillna(False).astype(bool) | (role_family(df["title"]) != "other")
    return df[keep.to_numpy()].reset_index(drop=True)


def _chunk_rows(jobs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows, dropped = [], 0
    for job_id, company, title, description in jobs[["job_id", "company", "title", "description"]].itertuples(
            index=False):
        res = chunk_jd(description, company, title)
        dropped += res.dropped
        for ch in res.chunks:
            h = content_hash(embed_text(company, title, ch.section, ch.text))
            rows.append((chunk_id(job_id, h), job_id, ch.section, ch.seq, ch.text, h))
    df = pd.DataFrame(rows, columns=["chunk_id", "job_id", "section", "seq", "text", "content_hash"])
    return df.drop_duplicates("chunk_id"), dropped


def embed_missing(store: Store, embedder: Embedder, batch: int = EMBED_BATCH, ts=None) -> int:
    """Embed every chunk that has no vector for embedder.model_id. Returns how many were embedded."""
    con = store.conn
    todo = con.execute(
        "SELECT c.chunk_id, j.company, j.title, c.section, c.text FROM job_chunks c JOIN jobs j USING (job_id) "
        "WHERE c.chunk_id NOT IN (SELECT chunk_id FROM chunk_embeddings WHERE model_id = ?) ORDER BY c.chunk_id",
        [embedder.model_id]).df()
    done = 0
    for start in range(0, len(todo), batch):
        part = todo.iloc[start:start + batch]
        vecs = embedder.embed([embed_text(*r) for r in
                               part[["company", "title", "section", "text"]].itertuples(index=False)])
        if vecs.shape != (len(part), embedder.dim):
            raise ValueError(f"embedder returned {vecs.shape}, expected {(len(part), embedder.dim)}")
        emb = pd.DataFrame({"chunk_id": part["chunk_id"].to_numpy(), "emb": list(vecs)})  # noqa: F841 (SQL)
        con.execute(f"INSERT INTO chunk_embeddings SELECT chunk_id, ?, emb::FLOAT[{embedder.dim}], ? FROM emb",
                    [embedder.model_id, ts or now()])
        done += len(part)
    return done


def index_jobs(store: Store, embedder: Embedder, run_id: str | None = None, ts=None) -> dict:
    """Chunk the eligible jobs, apply the diff to job_chunks, embed what is missing, record the counts.
    ts: the run time (default now); the demo passes its synthetic clock."""
    t0, ts, con = time.time(), ts or now(), store.conn
    jobs = eligible_jobs(store)
    new, dropped = _chunk_rows(jobs) if not jobs.empty else (pd.DataFrame(
        columns=["chunk_id", "job_id", "section", "seq", "text", "content_hash"]), 0)
    ids = pd.DataFrame({"job_id": jobs["job_id"] if not jobs.empty else pd.Series([], dtype=str)})  # noqa: F841

    con.execute("BEGIN")
    try:
        old = con.execute("SELECT job_id, chunk_id FROM job_chunks WHERE job_id IN (SELECT job_id FROM ids)").df()
        removed = old[~old["chunk_id"].isin(new["chunk_id"])]
        added = new[~new["chunk_id"].isin(old["chunk_id"])]
        jobs_changed = len(set(old["job_id"]) & (set(removed["job_id"]) | set(added["job_id"])))
        if not removed.empty:
            con.execute("DELETE FROM chunk_embeddings WHERE chunk_id IN (SELECT chunk_id FROM removed)")
            con.execute("DELETE FROM job_chunks WHERE chunk_id IN (SELECT chunk_id FROM removed)")
        if not added.empty:
            con.execute("INSERT INTO job_chunks SELECT chunk_id, job_id, section, seq, text, content_hash, ? "
                        "FROM added", [ts])
        if not new.empty:  # an unchanged chunk can move when a paragraph above it changes
            con.execute("UPDATE job_chunks SET seq = n.seq FROM new n "
                        "WHERE job_chunks.chunk_id = n.chunk_id AND job_chunks.seq <> n.seq")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    embedded = embed_missing(store, embedder, ts=ts)
    stats = {"jobs_indexed": len(jobs), "jobs_changed": jobs_changed, "chunks_total": len(new),
             "chunks_added": len(added), "chunks_removed": len(removed), "chunks_dropped": dropped,
             "chunks_embedded": embedded, "seconds": round(time.time() - t0, 2)}
    con.execute("INSERT INTO rag_index_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [ts, run_id, embedder.model_id, CHUNKING_VERSION, *stats.values()])
    log.info("RAG index: %(jobs_indexed)d jobs, %(chunks_added)d chunks added, %(chunks_removed)d replaced, "
             "%(chunks_dropped)d dropped as boilerplate, %(chunks_embedded)d embedded in %(seconds).1fs", stats)
    return stats


def index_step(store: Store, run_id: str | None = None) -> dict | None:
    """The daily run's RAG step: skipped when no embedder is available, and never fails the run."""
    from jobagent.rag.embed import get_embedder
    try:
        embedder = get_embedder()
        return index_jobs(store, embedder, run_id) if embedder else None
    except Exception as e:  # noqa: BLE001 - the index is an extra; scoring and the digest go on without it
        log.warning("RAG index failed (%s); continuing without it", type(e).__name__)
        return None
