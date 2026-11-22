from datetime import datetime

import numpy as np
import pytest

from eval import metrics as M
from eval.retrieval import evaluate, load_queries, relevant_jobs
from jobagent.db import Store
from jobagent.models import Job
from jobagent.rag.embed import FakeEmbedder
from jobagent.rag.index import index_jobs
from jobagent.rag.retrieve import BM25, rank_jobs, search, similar_jobs, tokenize

JDS = {
    "de1": ("Acme", "Data Engineer", "Requirements:\nBuild pipelines with dbt and Snowflake. Airflow is a must."),
    "de2": ("Beta", "Data Engineer", "Requirements:\nKafka streaming, Spark and Scala for real time events."),
    "an1": ("Gamma", "Data Analyst", "Requirements:\nSQL, Tableau dashboards and product analytics for growth."),
    "he1": ("דלתא", "מהנדס/ת נתונים", "דרישות:\nניסיון בפייתון ו-Spark, עבודה עם Airflow ותהליכי ETL."),
    "dup": ("Acme", "Data Engineer", "Requirements:\nBuild pipelines with dbt and Snowflake. Airflow nice."),
}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    for ext, (company, title, desc) in JDS.items():
        s.upsert([Job(source="greenhouse", company=company, external_id=ext, title=title, url=f"u/{ext}",
                      description=desc)], ts=datetime(2026, 11, 1 + len(ext) % 3))
    index_jobs(s, FakeEmbedder(), scope="all")
    yield s
    s.close()


def jid(s, ext):
    return s.conn.execute("SELECT job_id FROM jobs WHERE external_id=?", [ext]).fetchone()[0]


def test_tokenize_keeps_skills_and_strips_hebrew_prefixes():
    assert "c++" in tokenize("C++ and C#") and "c#" in tokenize("C++ and C#")
    toks = tokenize("ניסיון בפייתון")
    assert "בפייתון" in toks and "פייתון" in toks
    assert "and" not in tokenize("dbt and Snowflake")


def test_bm25_prefers_documents_with_the_rare_term():
    bm = BM25(["dbt snowflake", "snowflake snowflake", "python airflow"])
    s = bm.scores("dbt")
    assert s[0] > 0 and s[1] == 0 and s[2] == 0
    assert np.argmax(bm.scores("airflow python")) == 2


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_search_modes_find_the_skill(store, mode):
    hits = search(store, FakeEmbedder(), "Kafka Scala", k=3, mode=mode)
    assert hits and hits[0]["job_id"] == jid(store, "de2")
    assert set(hits[0]) == {"chunk_id", "job_id", "company", "title", "section", "text", "score",
                            "first_seen", "is_active"}


def test_search_filters(store):
    e = FakeEmbedder()
    assert {h["company"] for h in search(store, e, "Airflow", company="acme")} == {"Acme"}
    assert all(h["section"] == "requirements" for h in search(store, e, "Airflow", section="requirements"))
    store.mark_inactive({"Beta"}, set())
    assert jid(store, "de2") not in rank_jobs(search(store, e, "Kafka", active_only=True))
    assert jid(store, "de2") in rank_jobs(search(store, e, "Kafka"))
    assert search(store, e, "Kafka", since=datetime(2030, 1, 1)) == []


def test_search_caps_chunks_per_job(store):
    hits = search(store, FakeEmbedder(), "Airflow pipelines", k=10, per_job=1)
    assert len(rank_jobs(hits)) == len(hits)


def test_search_needs_an_embedder_except_bm25(store):
    assert search(store, None, "dbt", mode="bm25")
    with pytest.raises(ValueError):
        search(store, None, "dbt", mode="hybrid")


def test_similar_jobs_excludes_self_duplicates_and_eval_feedback(store):
    e = FakeEmbedder()
    de1 = jid(store, "de1")
    store.record_feedback([jid(store, "de2")], "applied", note="private note", source="cli")
    store.record_feedback([jid(store, "an1")], "skipped", source="eval")
    store.record_feedback([jid(store, "dup")], "applied", source="cli")
    got = similar_jobs(store, e, de1, k=5)
    ids = [r["job_id"] for r in got]
    assert de1 not in ids and jid(store, "dup") not in ids  # itself and its duplicate (same company + title)
    assert set(ids) == {jid(store, "de2"), jid(store, "an1")}  # labeled only
    assert all("note" not in r for r in got) and {r["decision"] for r in got} == {"applied", "skipped"}
    leak_free = similar_jobs(store, e, de1, k=5, exclude_feedback_sources=("eval",))
    assert [r["job_id"] for r in leak_free] == [jid(store, "de2")]
    assert len(similar_jobs(store, e, de1, k=5, labeled_only=False)) == 3
    assert similar_jobs(store, e, "nope") == []


def test_retrieval_metrics():
    assert M.recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 2) == 0.5
    assert M.recall_at_k(["a", "b"], {"a"}, 10) == 1.0
    assert np.isnan(M.recall_at_k(["a"], set(), 5))
    assert M.reciprocal_rank(["x", "a"], {"a"}) == 0.5 and M.reciprocal_rank(["x"], {"a"}) == 0.0
    assert M.ndcg_at_k(["a", "b"], {"a"}, 10) == 1.0
    assert M.ndcg_at_k(["b", "a"], {"a"}, 10) == pytest.approx(1 / np.log2(3))
    s = M.retrieval_summary([(["a"], {"a"}), (["b"], {"a"}), (["c"], set())])
    assert s["n_queries"] == 2 and s["mrr"] == 0.5 and s["recall_at_10"] == 0.5


def test_retrieval_eval_end_to_end(store):
    queries = [{"qid": "q1", "kind": "skill", "query": "Kafka", "relevant": {"skills": ["kafka"]}},
               {"qid": "q2", "kind": "role", "query": "data analyst", "relevant": {"role": "analytics"}}]
    out = evaluate(store, FakeEmbedder(), queries, ("bm25", "hybrid"))
    assert out["bm25"]["all"]["recall_at_10"] == 1.0 and out["hybrid"]["skill"]["n_queries"] == 1


def test_committed_queries_are_valid():
    import pandas as pd
    queries = load_queries()
    assert 25 <= len(queries) <= 40 and len({q["qid"] for q in queries}) == len(queries)
    jobs = pd.DataFrame([{"job_id": "x", "title": "Data Engineer", "description": "dbt"}])
    for q in queries:
        relevant_jobs(q["relevant"], jobs)  # every rule names a known skill / role and a valid regex
