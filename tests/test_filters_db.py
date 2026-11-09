from jobagent.db import Store
from jobagent.filters import PreFilter
from jobagent.models import Job

# Test-only profile, so the tests do not depend on anyone's real preferences.
PROFILE = {
    "keywords": ["data engineer", "BI", "מפתח/ת BI"],
    "locations_allowed": ["תל אביב", "רמת גן"],
    "locations_blocked": ["לוד"],
}


def test_prefilter_role_and_location():
    pf = PreFilter(PROFILE)
    assert pf("Data Engineer", "", "תל אביב")
    assert pf("מפתח/ת BI", "", "גוש דן | בית הראל רמת גן")
    assert not pf("Data Engineer", "", "לוד")            # blocked city
    assert not pf("Data Engineer", "", "חיפה")           # outside the area
    assert pf("Data Engineer", "", "")                   # unknown location kept
    assert not pf("נציג/ת שירות לקוחות", "מענה טלפוני", "רמת גן")
    assert not pf("Biology lab tech", "", "תל אביב")     # 'BI' must be a whole word


def _job(eid, title="Data Engineer"):
    return Job(source="adam", company="Acme", external_id=eid, title=title, url="u", location="רמת גן")


def test_store_dedup_and_incremental(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    assert len(s.upsert([_job("1"), _job("2"), _job("2")])) == 2
    assert s.upsert([_job("1"), _job("3")]) == [_job("3").job_id]
    s.mark_inactive({"Acme"}, {_job("3").job_id})
    assert s.get(_job("1").job_id)["is_active"] == 0
    s.set_prefilter(_job("3").job_id, True)
    s.save_score(_job("3").job_id, 90, {"summary": "x", "strengths": [], "gaps": []})
    top = s.top_unsent()
    assert [r["job_id"] for r in top] == [_job("3").job_id]
    s.mark_notified([r["job_id"] for r in top])
    assert s.top_unsent() == []


def test_unscored_active(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    s.upsert([_job("1"), _job("2")])
    s.save_score(_job("1").job_id, 70, {})
    assert [r["job_id"] for r in s.unscored_active()] == [_job("2").job_id]


def test_load_dotenv(tmp_path, monkeypatch):
    from jobagent.main import load_dotenv
    monkeypatch.delenv("JA_TEST_KEY", raising=False)
    monkeypatch.setenv("JA_TEST_KEEP", "real")
    (tmp_path / ".env").write_text('# c\nJA_TEST_KEY="abc"\nexport JA_TEST_KEEP=fromfile\n', encoding="utf-8")
    load_dotenv(tmp_path / ".env")
    import os
    assert os.environ["JA_TEST_KEY"] == "abc"
    assert os.environ["JA_TEST_KEEP"] == "real"


def test_resolve_ids_and_mark_seen(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    s.upsert([_job("1")])
    jid = _job("1").job_id
    s.save_score(jid, 90, {})
    assert s.resolve_ids([jid[:8]]) == [jid]
    s.mark_notified(s.resolve_ids([jid[:8]]))
    assert s.top_unsent() == []


def test_history_tables(tmp_path):
    from datetime import datetime
    s = Store(tmp_path / "jobs.duckdb")
    t1, t2 = datetime(2026, 1, 1, 8), datetime(2026, 1, 3, 8)
    r1 = s.start_run(ts=t1)
    s.upsert([_job("1"), _job("2")], ts=t1)
    s.record_observations(r1, [_job("1").job_id, _job("2").job_id, _job("2").job_id], ts=t1)
    s.record_sources(r1, [{"company": "Acme", "source_type": "adam", "ok": True, "n_jobs": 2}])
    r2 = s.start_run(ts=t2)
    s.upsert([_job("2")], ts=t2)
    s.mark_inactive({"Acme"}, {_job("2").job_id}, ts=t2)
    s.record_observations(r2, [_job("2").job_id], ts=t2)
    s.finish_run(r2, 1, 0, ts=t2)
    closed = s.get(_job("1").job_id)
    assert closed["is_active"] is False and closed["closed_at"] == t2
    assert s.get(_job("2").job_id)["first_seen"] == t1
    assert s.get(_job("2").job_id)["last_seen"] == t2
    assert s.conn.execute("SELECT count(*) FROM job_observations").fetchone()[0] == 3
    assert s.conn.execute("SELECT count(*) FROM source_runs").fetchone()[0] == 1
    # A closed job that comes back is open again.
    s.upsert([_job("1")], ts=t2)
    assert s.get(_job("1").job_id)["closed_at"] is None


def test_migrates_old_sqlite_db(tmp_path):
    import sqlite3
    from jobagent.db import open_store
    old = sqlite3.connect(tmp_path / "jobs.db")
    old.executescript("""
        CREATE TABLE jobs (job_id TEXT PRIMARY KEY, source TEXT, company TEXT, sector TEXT,
          external_id TEXT, title TEXT, location TEXT, department TEXT, url TEXT, description TEXT,
          posted_at TEXT, first_seen TEXT, last_seen TEXT, is_active INTEGER, prefilter INTEGER,
          score INTEGER, analysis TEXT, notified_at TEXT);
        INSERT INTO jobs VALUES ('abc123', 'adam', 'Acme', 'retail', '1', 'Data Engineer', 'רמת גן',
          '', 'u', 'd', '', '2026-01-01T08:00:00+00:00', '2026-01-05T08:00:00+00:00', 1, 1, 90, '{}',
          '2026-01-02T08:00:00+00:00');
    """)
    old.commit()
    old.close()
    s = open_store(tmp_path / "jobs.duckdb")
    row = s.get("abc123")
    assert row["score"] == 90 and row["is_active"] is True and row["notified_at"] is not None
    s.close()
    # Second open does not import again, and pointing JOBS_DB at the SQLite file also works.
    s = open_store(tmp_path / "jobs.db")
    assert s.path.endswith("jobs.duckdb")
    assert s.conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
