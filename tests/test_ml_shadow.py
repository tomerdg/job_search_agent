"""Relevance model in shadow mode: scores and reorders, never filters, falls back quietly. Synthetic data."""
import builtins
from types import SimpleNamespace as NS

import pytest

from jobagent.db import Store
from jobagent.demo import DEMO_CV, DEMO_PROFILE, build
from jobagent.ml.shadow import shadow_score
from jobagent.models import Job


@pytest.fixture
def demo_store(tmp_path):
    db = tmp_path / "demo.duckdb"
    build(db, days=60)
    s = Store(db)
    s.conn.execute("UPDATE jobs SET ml_score=NULL, ml_model_version=NULL, ml_scored_at=NULL")
    yield s
    s.conn.close()


def _unscored_ids(s):
    return {r["job_id"] for r in s.unscored_active()}


def test_scores_every_active_unscored_job(demo_store):
    s = demo_store
    out = shadow_score(s, DEMO_CV, DEMO_PROFILE)
    assert out and out["n_scored"] == len(_unscored_ids(s)) > 0 and out["version"].startswith("logreg-")
    rows = s.conn.execute("SELECT job_id, ml_score, ml_model_version, score, is_active FROM jobs").fetchall()
    scored = {r[0] for r in rows if r[1] is not None}
    assert scored == _unscored_ids(s)                                   # LLM-scored and closed jobs untouched
    assert all(0 <= r[1] <= 1 and r[2] == out["version"] for r in rows if r[1] is not None)


def test_never_filters_only_reorders(demo_store):
    s = demo_store
    before = {r["job_id"] for r in s.pending_scoring(limit=10_000)}
    shadow_score(s, DEMO_CV, DEMO_PROFILE)
    queue = s.pending_scoring(limit=10_000)
    assert {r["job_id"] for r in queue} == before                       # same jobs reach the LLM
    ml = [r["ml_score"] for r in queue]
    assert ml == sorted(ml, reverse=True)                               # best guesses first


def test_pending_scoring_puts_unscored_ml_last(tmp_path):
    s = Store(tmp_path / "j.duckdb")
    jobs = [Job(source="adam", company="Acme", external_id=str(i), title="Data Engineer", url=f"u{i}") for i in range(3)]
    s.upsert(jobs)
    for j in jobs:
        s.set_prefilter(j.job_id, True)
    s.save_ml_scores({jobs[0].job_id: 0.2, jobs[2].job_id: 0.9}, "v")
    assert [r["job_id"] for r in s.pending_scoring()] == [jobs[2].job_id, jobs[0].job_id, jobs[1].job_id]


def test_falls_back_on_too_few_labels(demo_store, caplog):
    s = demo_store
    s.conn.execute("DELETE FROM job_feedback")
    s.conn.execute("UPDATE jobs SET score = NULL")                     # no labels except weak negatives
    assert shadow_score(s, DEMO_CV, DEMO_PROFILE) is None
    assert "not trained this run" in caplog.text
    assert s.conn.execute("SELECT count(ml_score) FROM jobs").fetchone()[0] == 0


def test_falls_back_without_ml_packages(demo_store, monkeypatch, caplog):
    real = builtins.__import__

    def no_sklearn(name, *a, **kw):
        if name.startswith("sklearn") or name.startswith("jobagent.ml.train"):
            raise ImportError(name=name.split(".")[0])
        return real(name, *a, **kw)

    import sys
    monkeypatch.delitem(sys.modules, "jobagent.ml.train", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_sklearn)
    assert shadow_score(demo_store, DEMO_CV, DEMO_PROFILE) is None


def test_unexpected_error_does_not_break_the_run(demo_store, monkeypatch, caplog):
    import jobagent.ml.train as T
    monkeypatch.setattr(T, "load_jobs", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    assert shadow_score(demo_store, DEMO_CV, DEMO_PROFILE) is None
    assert "RuntimeError" in caplog.text


def test_main_run_calls_shadow_and_respects_no_ml(demo_store, monkeypatch):
    import jobagent.main as main
    monkeypatch.setattr(main, "fetch_all", lambda sources, stats=None: [])
    monkeypatch.setattr(main, "cv_text", lambda profile: DEMO_CV)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = []
    import jobagent.ml.shadow as shadow
    monkeypatch.setattr(shadow, "shadow_score", lambda *a, **k: calls.append(1))
    args = NS(mark_seen=None, no_llm=True, no_notify=True, mark_printed=False, no_ml=False)
    main.run(demo_store, args, [], DEMO_PROFILE)
    main.run(demo_store, NS(**{**vars(args), "no_ml": True}), [], DEMO_PROFILE)
    assert calls == [1]
