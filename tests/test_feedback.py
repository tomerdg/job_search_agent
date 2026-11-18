"""Feedback (labels): store, CLI flags, export/import sync, eval bootstrap. Synthetic data only."""
import shutil
from datetime import datetime
from types import SimpleNamespace as NS

import duckdb
import pytest

from jobagent import feedback
from jobagent.db import Store
from jobagent.main import main, run
from jobagent.models import Job


def _job(eid, url=None):
    return Job(source="adam", company="Acme", external_id=eid, title="Data Engineer",
               url=url or f"https://example.com/{eid}", location="Tel Aviv")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    s.upsert([_job("1"), _job("2"), _job("3")])
    yield s
    s.conn.close()


def test_record_feedback_upserts_and_marks_sent(store):
    a, b = _job("1").job_id, _job("2").job_id
    for jid in (a, b):
        store.set_prefilter(jid, True)
        store.save_score(jid, 90, {})
    assert store.record_feedback([a, a], "applied", note="private") == 1
    assert store.record_feedback([a], "skipped") == 1       # latest decision wins
    fb = {r["job_id"]: r for r in store.feedback()}
    assert fb[a]["decision"] == "skipped" and fb[a]["source"] == "cli" and fb[a]["note"] is None
    assert [r["job_id"] for r in store.top_unsent()] == [b]  # decided job is not sent again


def test_bad_decision_rejected(store):
    with pytest.raises(ValueError):
        store.record_feedback([_job("1").job_id], "maybe")
    with pytest.raises(duckdb.ConstraintException):  # the table enforces it too
        store.conn.execute("INSERT INTO job_feedback VALUES ('x', 'maybe', now(), 'cli', NULL)")


def test_old_db_gets_feedback_table(tmp_path):
    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE jobs (job_id VARCHAR PRIMARY KEY)")
    con.close()
    s = Store(path)
    assert s.feedback() == []


def test_cli_flags(tmp_path, monkeypatch, capsys):
    db = tmp_path / "jobs.duckdb"
    s = Store(db)
    s.upsert([_job("1"), _job("2")])
    s.close()
    monkeypatch.delenv("PROFILE_YAML", raising=False)
    base = ["--db", str(db), "--profile", str(tmp_path / "none.yaml")]
    a, b = _job("1").job_id, _job("2").job_id

    main(base + ["--applied", a[:8], "--note", "n"])
    main(base + ["--irrelevant", b])
    main(base + ["--skipped", "zzzzzzzz", "0123456789abcdef"])  # unknown short id skipped, full id kept
    out = capsys.readouterr().out
    assert "Recorded 1 job(s) as applied" in out and "Recorded 1 job(s) as skipped" in out

    s = Store(db)
    fb = {r["job_id"]: r for r in s.feedback()}
    s.close()
    assert fb[a]["decision"] == "applied" and fb[a]["note"] == "n"
    assert fb[b]["decision"] == "irrelevant"
    assert fb["0123456789abcdef"]["decision"] == "skipped"

    with pytest.raises(SystemExit):
        main(base + ["--applied", a, "--skipped", b])
    with pytest.raises(SystemExit):
        main(base + ["--note", "x"])


def test_mark_seen_unchanged(store):
    a = _job("1").job_id
    run(store, NS(mark_seen=[a[:8]]), [], {})
    assert store.get(a)["notified_at"] is not None
    assert store.feedback() == []


def test_export_import_round_trip(tmp_path, store, monkeypatch):
    monkeypatch.delenv("DB_PASSPHRASE", raising=False)
    a, b = _job("1").job_id, _job("2").job_id
    store.record_feedback([a], "applied", note="secret note", ts=datetime(2026, 11, 1, 9))
    store.record_feedback([b], "skipped", ts=datetime(2026, 11, 2, 9))
    csv_path = tmp_path / "feedback.csv"
    assert feedback.export(store, csv_path, enc_path=None) == 2
    text = csv_path.read_text()
    assert text.splitlines()[0] == "job_id,decision,decided_at,source"
    assert "secret note" not in text and "Acme" not in text and "Data Engineer" not in text

    other = Store(tmp_path / "ci.duckdb")
    other.upsert([_job("1"), _job("2")])
    other.record_feedback([b], "applied", ts=datetime(2026, 11, 5, 9))  # newer than the file: kept
    assert feedback.import_file(other, csv_path) == 1
    fb = {r["job_id"]: r for r in other.feedback()}
    assert fb[a]["decision"] == "applied" and fb[a]["decided_at"] == datetime(2026, 11, 1, 9)
    assert fb[b]["decision"] == "applied"
    assert other.get(a)["notified_at"] is not None
    assert feedback.import_file(other, csv_path) == 0  # idempotent
    other.close()


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")
def test_encrypted_sync(tmp_path, store, monkeypatch):
    monkeypatch.setenv("DB_PASSPHRASE", "test-only-passphrase")
    store.record_feedback([_job("1").job_id], "applied")
    csv_path, enc = tmp_path / "feedback.csv", tmp_path / "feedback.csv.enc"
    feedback.export(store, csv_path, enc)
    first = enc.read_bytes()
    assert b"applied" not in first
    feedback.export(store, csv_path, enc)
    assert enc.read_bytes() == first  # unchanged content is not re-encrypted

    other = Store(tmp_path / "ci.duckdb")
    monkeypatch.delenv("FEEDBACK_CSV", raising=False)
    monkeypatch.setenv("FEEDBACK_CSV_ENC", str(enc))
    feedback.sync_at_start(other)
    assert [r["decision"] for r in other.feedback()] == ["applied"]
    other.close()


def test_sync_at_start_never_fails_the_run(tmp_path, store, monkeypatch):
    bad = tmp_path / "feedback.csv"
    bad.write_text("job_id,decision,decided_at,source\nx,nope,2026-01-01 00:00:00,cli\n")
    monkeypatch.setenv("FEEDBACK_CSV", str(bad))
    feedback.sync_at_start(store)
    assert store.feedback() == []


def test_import_eval_counts_only(store, capsys):
    a, b = _job("1").job_id, _job("2").job_id
    store.record_feedback([b], "skipped")  # an existing decision is never overwritten
    jobs = [{"case_id": "c1", "url": _job("1").url}, {"case_id": "c2", "url": _job("2").url},
            {"case_id": "c3", "url": "https://example.com/unknown"}, {"case_id": "c4", "url": _job("3").url}]
    labels = {"c1": {"label": "maybe"}, "c2": {"label": "no_fit"}, "c3": {"label": "fit"}, "c4": {"label": ""}}
    c = feedback.import_eval(store, jobs, labels)
    assert c == {"cases": 4, "matched": 2, "unlabeled": 1, "recorded": 1}
    fb = {r["job_id"]: r for r in store.feedback()}
    assert fb[a]["decision"] == "applied" and fb[a]["source"] == "eval"
    assert fb[b]["decision"] == "skipped" and fb[b]["source"] == "cli"


def test_match_cases_shared_url_needs_title():
    from jobagent.feedback import match_cases
    db = [("j1", "https://example.com/careers", "Data Engineer"),
          ("j2", "https://example.com/careers", "Sales Rep"),       # same careers-page url for every job
          ("j3", "https://example.com/jobs/3", "BI Developer")]
    cases = [{"case_id": "a", "url": "https://example.com/careers", "title": "data  engineer"},
             {"case_id": "b", "url": "https://example.com/careers", "title": "Product Manager"},
             {"case_id": "c", "url": "https://example.com/jobs/3", "title": "anything"}]
    assert match_cases(cases, db) == {"a": ["j1"], "b": [], "c": ["j3"]}
