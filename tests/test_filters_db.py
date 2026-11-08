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
    return Job(source="adam", company="Harel", external_id=eid, title=title, url="u", location="רמת גן")


def test_store_dedup_and_incremental(tmp_path):
    s = Store(tmp_path / "jobs.db")
    assert len(s.upsert([_job("1"), _job("2"), _job("2")])) == 2
    assert s.upsert([_job("1"), _job("3")]) == [_job("3").job_id]
    s.mark_inactive({"Harel"}, {_job("3").job_id})
    assert s.get(_job("1").job_id)["is_active"] == 0
    s.set_prefilter(_job("3").job_id, True)
    s.save_score(_job("3").job_id, 90, {"summary": "x", "strengths": [], "gaps": []})
    top = s.top_unsent()
    assert [r["job_id"] for r in top] == [_job("3").job_id]
    s.mark_notified([r["job_id"] for r in top])
    assert s.top_unsent() == []


def test_unscored_active(tmp_path):
    s = Store(tmp_path / "jobs.db")
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
    s = Store(tmp_path / "jobs.db")
    s.upsert([_job("1")])
    jid = _job("1").job_id
    s.save_score(jid, 90, {})
    assert s.resolve_ids([jid[:8]]) == [jid]
    s.mark_notified(s.resolve_ids([jid[:8]]))
    assert s.top_unsent() == []
