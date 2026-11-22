"""Retrieval evaluation: does search() bring back the jobs a query is about?

    python -m eval.retrieval --modes vector bm25 hybrid          # corpus: the eval set's JDs
    python -m eval.retrieval --db data/jobs.duckdb               # corpus: every active job in a jobs DB

queries.jsonl holds generic queries (skills, Hebrew phrasings, role types, a few paraphrases), each with a
*rule* that says which jobs are relevant, not a list of job ids:

    {"qid": "skill-dbt-snowflake", "kind": "skill", "query": "dbt and Snowflake",
     "relevant": {"skills": ["dbt", "snowflake"]}}

    skills  every named skill (jobagent.ml.features.SKILLS) appears in the JD
    role    the title's role_family (jobagent.ml.features.ROLE_RULES)
    jd      a regex over the JD (case-insensitive)

Rules make the file public (no ids from the personal eval set, nothing about preferences), let it run on
any corpus, and are easy to review. The cost: skill rules are keyword matches, which favours BM25, so the
kinds are reported separately; "paraphrase" queries (no shared keyword) are where vectors have to earn it.
Ranking is per job (one chunk per job), so recall@k counts jobs. Only numbers are logged.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

from eval import metrics as M
from jobagent.db import Store
from jobagent.models import Job

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_PATH = Path(__file__).parent / "queries.jsonl"
KS = (5, 10)


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for q in rows:
        missing = {"qid", "kind", "query", "relevant"} - set(q)
        if missing:
            raise ValueError(f"query {q.get('qid')}: missing {sorted(missing)}")
        if not set(q["relevant"]) <= {"skills", "role", "jd"}:
            raise ValueError(f"query {q['qid']}: unknown rule {sorted(set(q['relevant']) - {'skills', 'role', 'jd'})}")
    return rows


def queries_hash(path: Path = QUERIES_PATH) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def relevant_jobs(rule: dict, jobs: pd.DataFrame) -> set[str]:
    """job_ids in `jobs` (columns job_id, title, description) that satisfy every part of the rule."""
    from jobagent.ml.features import SKILLS, role_family
    desc = jobs["description"].fillna("")
    keep = pd.Series(True, index=jobs.index)
    for skill in rule.get("skills", []):
        keep &= desc.str.contains(SKILLS[skill], regex=True, flags=re.I)
    if "role" in rule:
        keep &= role_family(jobs["title"]) == rule["role"]
    if "jd" in rule:
        keep &= desc.str.contains(rule["jd"], regex=True, flags=re.I)
    return set(jobs.loc[keep, "job_id"])


def eval_set_store(path: Path) -> Store:
    """A fresh DuckDB with the eval set's JDs (decrypted in memory in CI) as active jobs."""
    from eval import dataset as ds
    store = Store(path)
    store.upsert([Job(source="eval", company=c["company"] or "unknown", external_id=c["case_id"], title=c["title"],
                      url=c["url"], location=c["location"], department=c["department"],
                      description=c["description"]) for c in ds.load_jobs()])
    return store


def evaluate(store: Store, embedder, queries: list[dict], modes: tuple[str, ...], k: int = max(KS)) -> dict:
    """{mode: {"all": summary, "<kind>": summary, "per_query": {qid: (recall@k, rr)}}} over the active jobs."""
    from jobagent.rag.retrieve import rank_jobs, search
    jobs = store.conn.execute("SELECT job_id, title, description FROM jobs WHERE is_active").df()
    gold = {q["qid"]: relevant_jobs(q["relevant"], jobs) for q in queries}
    out = {}
    for mode in modes:
        results = {q["qid"]: (rank_jobs(search(store, embedder, q["query"], k=k, mode=mode, active_only=True,
                                               per_job=1)), gold[q["qid"]]) for q in queries}
        summary = {"all": M.retrieval_summary(list(results.values()), KS)}
        for kind in sorted({q["kind"] for q in queries}):
            summary[kind] = M.retrieval_summary([results[q["qid"]] for q in queries if q["kind"] == kind], KS)
        summary["per_query"] = {qid: {"n_relevant": len(g), "recall_at_10": M.recall_at_k(r, g, 10),
                                      "rr": M.reciprocal_rank(r, g)} for qid, (r, g) in results.items()}
        out[mode] = summary
    return out


def index_corpus(store: Store, embedder) -> dict:
    from jobagent.rag.index import index_jobs
    return index_jobs(store, embedder, run_id="retrieval-eval", scope="all")


def temp_copy(db: Path) -> tuple[tempfile.TemporaryDirectory, Store]:
    """A writable copy of a jobs DB, so the eval can index it without touching the original."""
    import shutil
    tmp = tempfile.TemporaryDirectory()
    dst = Path(tmp.name) / "jobs.duckdb"
    shutil.copy(db, dst)
    return tmp, Store(dst)
