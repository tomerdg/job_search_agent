from pathlib import Path

import pytest

from jobagent.db import Store
from jobagent.models import Job
from jobagent.rag.embed import FakeEmbedder
from jobagent.rag.index import index_jobs, index_step

FIX = Path(__file__).parent / "fixtures" / "rag"


class CountingEmbedder(FakeEmbedder):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return super().embed(texts)


def _job(ext: str, title: str, description: str, company: str = "Acme Data") -> Job:
    return Job(source="greenhouse", company=company, external_id=ext, title=title,
               url=f"https://example.com/{ext}", location="Tel Aviv", description=description)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    s.upsert([
        _job("1", "Data Engineer", (FIX / "jd_en.txt").read_text(encoding="utf-8")),
        _job("2", "מהנדס/ת נתונים", (FIX / "jd_he.txt").read_text(encoding="utf-8"), company="חברה"),
        _job("3", "Account Executive", "Sell our product.\nRequirements:\nFive years of B2B sales in SaaS."),
    ])
    yield s
    s.close()


def count(s: Store, table: str) -> int:
    return s.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_first_run_indexes_data_jobs_only(store):
    e = CountingEmbedder()
    stats = index_jobs(store, e, run_id="r1")
    assert stats["jobs_indexed"] == 2  # the sales job is not data-ish and did not pass a pre-filter
    assert stats["chunks_added"] == count(store, "job_chunks") == e.calls == stats["chunks_embedded"]
    assert count(store, "chunk_embeddings") == stats["chunks_total"]
    assert stats["chunks_dropped"] == 2
    assert count(store, "rag_index_runs") == 1
    dims = store.conn.execute("SELECT DISTINCT len(embedding) FROM chunk_embeddings").fetchall()
    assert dims == [(384,)]


def test_prefilter_pass_makes_any_title_eligible(store):
    sales = store.conn.execute("SELECT job_id FROM jobs WHERE title='Account Executive'").fetchone()[0]
    store.set_prefilter(sales, True)
    assert index_jobs(store, FakeEmbedder())["jobs_indexed"] == 3


def test_second_run_embeds_nothing(store):
    index_jobs(store, FakeEmbedder())
    e = CountingEmbedder()
    stats = index_jobs(store, e)
    assert e.calls == 0 and stats["chunks_added"] == stats["chunks_removed"] == stats["jobs_changed"] == 0


def test_changed_jd_reembeds_only_its_changed_chunks(store):
    index_jobs(store, FakeEmbedder())
    before = dict(store.conn.execute("SELECT chunk_id, job_id FROM job_chunks").fetchall())
    text = (FIX / "jd_en.txt").read_text(encoding="utf-8").replace("ClickHouse, Spark, Terraform.",
                                                                   "ClickHouse, Spark, Terraform, Flink.")
    store.upsert([_job("1", "Data Engineer", text)])
    e = CountingEmbedder()
    stats = index_jobs(store, e)
    assert stats["jobs_changed"] == 1 and stats["chunks_removed"] == 1 and stats["chunks_added"] == 1
    assert e.calls == 1
    after = dict(store.conn.execute("SELECT chunk_id, job_id FROM job_chunks").fetchall())
    assert len(set(before) - set(after)) == 1
    assert count(store, "chunk_embeddings") == count(store, "job_chunks")  # the old vector went with it
    assert store.conn.execute("SELECT count(*) FROM job_chunks WHERE text LIKE '%Flink%'").fetchone()[0] == 1


def test_closed_jobs_keep_their_chunks(store):
    index_jobs(store, FakeEmbedder())
    n = count(store, "job_chunks")
    store.mark_inactive({"Acme Data"}, set())
    stats = index_jobs(store, FakeEmbedder())
    assert count(store, "job_chunks") == n and stats["jobs_indexed"] == 1 and stats["chunks_removed"] == 0


def test_new_model_embeds_everything_again(store):
    index_jobs(store, FakeEmbedder())
    e = CountingEmbedder(model_id="fake-bow-v2")
    index_jobs(store, e)
    assert e.calls == count(store, "job_chunks")
    assert store.conn.execute("SELECT count(DISTINCT model_id) FROM chunk_embeddings").fetchone()[0] == 2


def test_index_step_never_fails_the_run(store, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model download failed")
    monkeypatch.setenv("RAG_EMBEDDER", "fake")
    monkeypatch.setattr("jobagent.rag.index.index_jobs", boom)
    assert index_step(store) is None


def test_index_step_skips_without_fastembed(store, monkeypatch):
    monkeypatch.setattr("jobagent.rag.embed.get_embedder", lambda: None)
    assert index_step(store) is None and count(store, "job_chunks") == 0
