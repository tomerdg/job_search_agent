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


WORKDAY_SRC = {"company": "Acme", "type": "workday", "url": "https://acme.wd1.myworkdayjobs.com/en-US/External",
               "facets": {"locations": ["loc1"]}}


def test_workday_site_parts():
    from jobagent.fetchers import workday
    assert workday.site_parts(WORKDAY_SRC) == ("https://acme.wd1.myworkdayjobs.com", "acme", "External")
    src = {**WORKDAY_SRC, "url": "https://acme.wd3.myworkdayjobs.com/Careers", "tenant": "acmecorp"}
    assert workday.site_parts(src) == ("https://acme.wd3.myworkdayjobs.com", "acmecorp", "Careers")


def test_workday_parse_with_and_without_detail():
    import json
    from jobagent.fetchers import workday
    postings = workday.parse_list(json.loads((FIX / "workday_jobs.json").read_text()))
    detail = json.loads((FIX / "workday_job.json").read_text())

    j = workday.parse(postings[0], detail, WORKDAY_SRC)
    assert (j.external_id, j.title, j.posted_at) == ("JR0001", "Data Engineer", "2026-11-12")
    assert j.location == "Israel, Petah-Tikva | Israel, Haifa"
    assert "data pipelines" in j.description and "Python & SQL" in j.description and "<" not in j.description
    assert j.url.endswith("/External/job/Israel-Petah-Tikva/Data-Engineer_JR0001")

    bare = workday.parse(postings[1], None, WORKDAY_SRC)
    assert (bare.external_id, bare.title, bare.location, bare.description) == ("JR0002", "BI Analyst", "Israel, Haifa", "")
    assert bare.url == "https://acme.wd1.myworkdayjobs.com/External/job/Israel-Haifa/BI-Analyst_JR0002"


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


def test_workday_fetch_paginates_and_survives_bad_detail(monkeypatch):
    import json
    from jobagent.fetchers import workday
    listing = json.loads((FIX / "workday_jobs.json").read_text())
    detail = json.loads((FIX / "workday_job.json").read_text())
    monkeypatch.setattr(workday, "PAGE_SIZE", 2)
    posts, gets = [], []

    def fake_post(url, json, **kw):
        posts.append((url, dict(json)))
        off = json["offset"]
        # Like Workday, only the first page carries the real total.
        return _Resp({"total": 3 if off == 0 else 0, "jobPostings": listing["jobPostings"][off:off + 2]})

    def fake_get(url, **kw):
        gets.append(url)
        if url.endswith("_JR0002"):
            return _Resp({}, 500)
        return _Resp(detail) if url.endswith("_JR0001") else _Resp({})

    monkeypatch.setattr(workday.session, "post", fake_post)
    monkeypatch.setattr(workday.session, "get", fake_get)
    jobs = workday.fetch(WORKDAY_SRC)

    assert [p[1]["offset"] for p in posts] == [0, 2]
    assert posts[0][0] == "https://acme.wd1.myworkdayjobs.com/wday/cxs/acme/External/jobs"
    assert posts[0][1]["appliedFacets"] == {"locations": ["loc1"]}
    assert gets[0] == "https://acme.wd1.myworkdayjobs.com/wday/cxs/acme/External/job/Israel-Petah-Tikva/Data-Engineer_JR0001"
    assert [j.external_id for j in jobs] == ["JR0001", "JR0002", "JR0003"]
    assert jobs[1].description == ""  # failed detail keeps the posting


def test_workday_fetch_respects_max_jobs(monkeypatch):
    import json
    from jobagent.fetchers import workday
    listing = json.loads((FIX / "workday_jobs.json").read_text())
    monkeypatch.setattr(workday, "PAGE_SIZE", 2)
    monkeypatch.setattr(workday.session, "post",
                        lambda url, json, **kw: _Resp({"total": 3, "jobPostings": listing["jobPostings"][json["offset"]:json["offset"] + 2]}))
    monkeypatch.setattr(workday.session, "get", lambda url, **kw: _Resp({}))
    assert len(workday.fetch({**WORKDAY_SRC, "max_jobs": 2})) == 2
