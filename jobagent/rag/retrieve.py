"""Search over the JD chunks: vector (DuckDB array_cosine_distance), BM25 (scipy sparse) and hybrid (RRF).

    search(store, embedder, "dbt and Snowflake", k=10, mode="hybrid", active_only=True)
    similar_jobs(store, embedder, job_id, k=5)

Why BM25 here and not DuckDB's fts extension: fts is downloaded at runtime (tests and CI must run without
network), its default tokenizer keeps only a-z (it drops Hebrew), and its index is a full snapshot that
has to be rebuilt by hand after every insert. A sparse term matrix over ~thousands of chunks is a few
milliseconds per query, is built from the same table, and uses a tokenizer that knows Hebrew prefixes.

Skill names (dbt, ClickHouse, Airflow) are exact tokens that embeddings blur, and paraphrases ("first data
hire") are what BM25 misses, so hybrid fuses both rankings with reciprocal rank fusion:
score = sum over modes of 1 / (RRF_K + rank).
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse

from jobagent.db import Store
from jobagent.rag.chunking import embed_text
from jobagent.rag.embed import Embedder

MODES = ("vector", "bm25", "hybrid")
RRF_K = 60
POOL = 200  # chunks per mode that enter the fusion

_TOKEN = re.compile(r"[a-z0-9א-ת][a-z0-9א-ת+#]*")
_HEBREW_PREFIX = set("ובהלמשכ")
STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or our the to we with you your will this that who "
    "של עם את על או גם כל זה זו כמו אנו אנחנו".split())


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; '+' and '#' stay (c++, c#). A Hebrew word with a one-letter prefix (ו, ה, ב, ל, מ,
    ש, כ) also yields the word without it, so 'בפייתון' matches 'פייתון'."""
    out = []
    for tok in _TOKEN.findall((text or "").lower()):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        out.append(tok)
        if len(tok) >= 4 and tok[0] in _HEBREW_PREFIX and "א" <= tok[1] <= "ת":
            out.append(tok[1:])
    return out


class BM25:
    """Okapi BM25 over a list of documents, as a scipy sparse term-frequency matrix."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        vocab: dict[str, int] = {}
        rows, cols = [], []
        for i, doc in enumerate(docs):
            for tok in tokenize(doc):
                rows.append(i)
                cols.append(vocab.setdefault(tok, len(vocab)))
        self.vocab = vocab
        tf = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                               shape=(len(docs), max(len(vocab), 1)))
        tf.sum_duplicates()
        self.tf = tf.tocsc()
        n = len(docs)
        df = np.diff(self.tf.indptr)  # documents per term
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        self.norm = k1 * (1 - b + b * dl / (dl.mean() if n and dl.mean() else 1))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.tf.shape[0], dtype=np.float64)
        for t in {self.vocab[t] for t in tokenize(query) if t in self.vocab}:
            lo, hi = self.tf.indptr[t], self.tf.indptr[t + 1]
            docs, f = self.tf.indices[lo:hi], self.tf.data[lo:hi]
            out[docs] += self.idf[t] * f * (self.k1 + 1) / (f + self.norm[docs])
        return out


_BM25_CACHE: dict[tuple, tuple[BM25, dict[str, int]]] = {}


def _bm25_index(store: Store) -> tuple[BM25, dict[str, int]]:
    """One BM25 index over all chunks (company | title | section + text), rebuilt when the chunks change."""
    key = (store.path, *store.conn.execute(
        "SELECT count(*), max(created_at), bit_xor(hash(chunk_id)) FROM job_chunks").fetchone())
    if key not in _BM25_CACHE:
        rows = store.conn.execute(
            "SELECT c.chunk_id, j.company, j.title, c.section, c.text FROM job_chunks c JOIN jobs j USING (job_id) "
            "ORDER BY c.chunk_id").fetchall()
        _BM25_CACHE.clear()
        _BM25_CACHE[key] = (BM25([embed_text(*r[1:]) for r in rows]), {r[0]: i for i, r in enumerate(rows)})
    return _BM25_CACHE[key]


def _ranks(values: pd.Series) -> pd.Series:
    """1 = best; only the top POOL positive values get a rank, the rest NaN."""
    v = values.where(values > 0)
    r = v.rank(ascending=False, method="first")
    return r.where(r <= POOL)


def _since(since) -> datetime | None:
    if since is None or isinstance(since, datetime):
        return since
    if isinstance(since, str) and (m := re.fullmatch(r"(\d+)d", since.strip())):
        return datetime.now() - pd.Timedelta(days=int(m.group(1)))
    return pd.Timestamp(since).to_pydatetime()


def search(store: Store, embedder: Embedder | None, query: str, k: int = 10, mode: str = "hybrid",
           active_only: bool = False, since=None, section: str | None = None, company: str | None = None,
           per_job: int = 2, job_ids: list[str] | None = None) -> list[dict]:
    """Top k chunks for a query. Each hit: chunk_id, job_id, company, title, section, text, score,
    first_seen, is_active. At most per_job chunks per job, so one long JD cannot fill the results.
    since: a datetime, a date string or 'Nd' (first seen in the last N days)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode != "bm25" and embedder is None:
        raise ValueError(f"mode {mode!r} needs an embedder")
    where, params = ["TRUE"], []
    if active_only:
        where.append("j.is_active")
    if (since_ts := _since(since)) is not None:
        where.append("j.first_seen >= ?")
        params.append(since_ts)
    if section:
        where.append("c.section = ?")
        params.append(section)
    if company:
        where.append("lower(j.company) = lower(?)")
        params.append(company)
    if job_ids is not None:
        where.append("list_contains(?, c.job_id)")
        params.append(list(job_ids))

    if mode == "bm25":
        sim_sql, sim_params, join = "NULL::DOUBLE AS sim", [], ""
    else:
        qv = embedder.embed([query])[0].astype(float).tolist()
        sim_sql = f"1 - array_cosine_distance(e.embedding, ?::FLOAT[{embedder.dim}]) AS sim"
        join = "LEFT JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id AND e.model_id = ?"
        sim_params = [qv, embedder.model_id]  # in SQL order: the SELECT, then the JOIN
    df = store.conn.execute(
        f"SELECT c.chunk_id, c.job_id, j.company, j.title, c.section, c.text, j.first_seen, j.is_active, {sim_sql} "
        f"FROM job_chunks c JOIN jobs j USING (job_id) {join} WHERE {' AND '.join(where)}",
        sim_params + params).df()
    if df.empty:
        return []

    fused = pd.Series(0.0, index=df.index)
    if mode in ("vector", "hybrid"):
        r = _ranks(df["sim"].astype(float).fillna(-1) + 1)  # cosine in [-1, 1] -> positive
        fused += (1 / (RRF_K + r)).fillna(0) if mode == "hybrid" else df["sim"].astype(float).where(r.notna(), 0)
    if mode in ("bm25", "hybrid"):
        bm, pos = _bm25_index(store)
        all_scores = bm.scores(query)
        s = pd.Series(all_scores[df["chunk_id"].map(pos).to_numpy()], index=df.index)
        fused += (1 / (RRF_K + _ranks(s))).fillna(0) if mode == "hybrid" else s
    df["score"] = fused
    df = df[df["score"] > 0].sort_values(["score", "chunk_id"], ascending=[False, True])
    df = df[df.groupby("job_id").cumcount() < per_job].head(k)
    cols = ["chunk_id", "job_id", "company", "title", "section", "text", "score", "first_seen", "is_active"]
    return df[cols].to_dict("records")


def rank_jobs(hits: list[dict]) -> list[str]:
    """Job ids in the order they first appear in the hits."""
    return list(dict.fromkeys(h["job_id"] for h in hits))


def job_vectors(store: Store, embedder: Embedder) -> tuple[list[str], np.ndarray]:
    """One unit vector per job: the mean of its requirements + responsibilities chunks (all of its chunks
    when it has neither). Returns (job_ids, matrix)."""
    df = store.conn.execute(
        "SELECT c.job_id, c.section IN ('requirements', 'responsibilities') AS core, e.embedding "
        "FROM job_chunks c JOIN chunk_embeddings e USING (chunk_id) WHERE e.model_id = ? ORDER BY c.job_id, c.seq",
        [embedder.model_id]).df()
    if df.empty:
        return [], np.zeros((0, embedder.dim), dtype=np.float32)
    df["has_core"] = df.groupby("job_id")["core"].transform("any")
    df = df[df["core"] | ~df["has_core"]]
    ids, vecs = [], []
    for job_id, g in df.groupby("job_id", sort=True):
        v = np.stack(g["embedding"].to_numpy()).mean(axis=0)
        ids.append(job_id)
        vecs.append(v / (np.linalg.norm(v) or 1))
    return ids, np.asarray(vecs, dtype=np.float32)


def similar_jobs(store: Store, embedder: Embedder, job_id: str, k: int = 5, labeled_only: bool = True,
                 exclude_ids: set[str] | frozenset = frozenset(),
                 exclude_feedback_sources: tuple[str, ...] = ()) -> list[dict]:
    """The k jobs most similar to job_id, each with title, company, the LLM score and the owner's decision.
    Excludes the job itself, its duplicates (same company + title), exclude_ids, and every job whose
    feedback came from one of exclude_feedback_sources (the eval's leakage rule). The owner's note is
    never read."""
    ids, mat = job_vectors(store, embedder)
    if job_id not in ids:
        return []
    sims = mat @ mat[ids.index(job_id)]
    meta = store.conn.execute(
        "SELECT j.job_id, j.company, j.title, j.score, j.first_seen, j.is_active, f.decision, f.source AS fb_source "
        "FROM jobs j LEFT JOIN job_feedback f USING (job_id) WHERE list_contains(?, j.job_id)", [ids]).df()
    meta = meta.set_index("job_id").loc[ids].reset_index()
    meta["similarity"] = sims.astype(float)
    target = meta.loc[meta["job_id"] == job_id].iloc[0]
    key = meta["company"].str.lower().str.strip() + "\x1f" + meta["title"].str.lower().str.strip()
    target_key = key[meta["job_id"] == job_id].iloc[0]
    keep = (meta["job_id"] != target["job_id"]) & (key != target_key) & ~meta["job_id"].isin(set(exclude_ids))
    if exclude_feedback_sources:
        keep &= ~meta["fb_source"].isin(list(exclude_feedback_sources))
    if labeled_only:
        keep &= meta["decision"].notna()
    out = meta[keep].assign(_key=key[keep]).sort_values(["similarity", "job_id"], ascending=[False, True])
    out = out.drop_duplicates("_key").head(k)
    cols = ["job_id", "company", "title", "score", "decision", "similarity", "first_seen", "is_active"]
    rows = out[cols].astype(object).where(out[cols].notna(), None).to_dict("records")
    for r in rows:
        r["score"] = None if r["score"] is None else int(r["score"])
    return rows


class CvIndex:
    """The CV's chunks with their vectors and a BM25 index, in memory only (never stored or logged).
    search() fuses vector and BM25 rankings with RRF, like search() over JDs."""

    def __init__(self, cv_text: str, embedder: Embedder):
        from jobagent.rag.chunking import chunk_cv, cv_embed_text
        self.embedder = embedder
        self.chunks = chunk_cv(cv_text)
        texts = [cv_embed_text(c.section, c.text) for c in self.chunks]
        self.vectors = embedder.embed(texts) if texts else np.zeros((0, embedder.dim), dtype=np.float32)
        self.bm25 = BM25(texts) if texts else None

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.chunks:
            return []
        sims = self.vectors @ self.embedder.embed([query])[0]
        fused = pd.Series(0.0, index=range(len(self.chunks)))
        fused += (1 / (RRF_K + _ranks(pd.Series(sims + 1)))).fillna(0)
        fused += (1 / (RRF_K + _ranks(pd.Series(self.bm25.scores(query))))).fillna(0)
        best = fused[fused > 0].sort_values(ascending=False, kind="stable").head(k)
        return [{"section": self.chunks[i].section, "text": self.chunks[i].text} for i in best.index]
