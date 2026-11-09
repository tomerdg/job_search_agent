from pathlib import Path

from jobagent.fetchers import adam, comeet, topmatch

FIX = Path(__file__).parent / "fixtures"


def test_adam_parse():
    jobs = adam.parse_page((FIX / "adam_page.html").read_text(encoding="utf-8"), "Acme")
    assert [j.external_id for j in jobs] == ["5712", "5800"]
    j = jobs[0]
    assert j.title == "חתמ/ת פרט"
    assert j.location == "גוש דן | בית הראל רמת גן"
    assert j.department == "צוות 5 - מחוז המשמר"
    assert j.url.startswith("https://career.adamtotal.co.il/Jobs/JobDetails?token=abc")
    assert "חיתום פרט" in j.description and "<" not in j.description


def test_topmatch_parse():
    data = {"positions": [
        {"compPositionID": 199575, "jobTitleText": "Data Engineer", "displayLocation": "בני ברק",
         "description": "<ul><li>Python</li></ul>", "activationDate": "2026-11-03T10:43:50", "isActivePosition": True,
         "fieldDesc": "טכנולוגיה"},
        {"compPositionID": 1, "jobTitleText": "closed", "isActivePosition": False},
    ]}
    jobs = topmatch.parse(data, {"company": "Meitav", "slug": "Meitav"})
    assert len(jobs) == 1
    assert jobs[0].external_id == "199575" and jobs[0].location == "בני ברק"
    assert jobs[0].posted_at == "2026-11-03" and jobs[0].description == "Python"


def test_comeet_parse():
    data = [{"uid": "A1.B2", "name": "BI Developer", "department": "Data",
             "location": {"city": "Tel Aviv", "country": "IL", "is_remote": False},
             "details": [{"name": "Requirements", "value": "<p>SQL</p>"}],
             "url_comeet_hosted_page": "https://www.comeet.com/jobs/x/A1.B2", "time_updated": "2026-11-01T00:00:00Z"}]
    jobs = comeet.parse(data, {"company": "ONE ZERO"})
    assert jobs[0].location == "Tel Aviv, IL"
    assert "SQL" in jobs[0].description


def test_greenhouse_parse():
    from jobagent.fetchers import greenhouse
    data = {"jobs": [{
        "id": 123, "title": " Data Engineer ", "absolute_url": "https://x/jobs/123",
        "location": {"name": "Tel Aviv, Israel"}, "departments": [{"name": "R&D"}],
        "content": "&lt;p&gt;Build &lt;b&gt;pipelines&lt;/b&gt;&lt;/p&gt;",
        "first_published": "2026-11-01T10:00:00Z", "updated_at": "2026-11-05T10:00:00Z",
    }]}
    jobs = greenhouse.parse(data, {"company": "Taboola"})
    assert len(jobs) == 1
    j = jobs[0]
    assert (j.external_id, j.title, j.location, j.department) == ("123", "Data Engineer", "Tel Aviv, Israel", "R&D")
    assert "pipelines" in j.description and "<" not in j.description
    assert j.posted_at == "2026-11-01"


def test_fetch_all_fills_city(monkeypatch):
    from jobagent import fetchers
    from jobagent.models import Job
    monkeypatch.setitem(fetchers.FETCHERS, "fake", lambda src: [
        Job(source="fake", company="eToro", external_id="1", title="t", url="u", location="Israel"),
        Job(source="fake", company="eToro", external_id="2", title="t", url="u", location="Limassol, Cyprus"),
    ])
    jobs = fetchers.fetch_all([{"company": "eToro", "type": "fake", "city": "Bnei Brak, Israel"}])
    assert [j.location for j in jobs] == ["Bnei Brak, Israel", "Limassol, Cyprus"]


def test_fetch_all_records_source_stats(monkeypatch):
    from jobagent import fetchers

    def boom(src):
        raise ValueError("site changed")

    monkeypatch.setitem(fetchers.FETCHERS, "fake", lambda src: [])
    monkeypatch.setitem(fetchers.FETCHERS, "broken", boom)
    stats = []
    fetchers.fetch_all([{"company": "A", "type": "fake"}, {"company": "B", "type": "broken"}], stats=stats)
    assert [(s["company"], s["ok"], s["n_jobs"]) for s in stats] == [("A", True, 0), ("B", False, 0)]
    assert stats[1]["error"] == "ValueError: site changed"
