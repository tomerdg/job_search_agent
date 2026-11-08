from pathlib import Path

from jobagent.fetchers import adam, comeet, topmatch

FIX = Path(__file__).parent / "fixtures"


def test_adam_parse():
    jobs = adam.parse_page((FIX / "adam_page.html").read_text(encoding="utf-8"), "Harel")
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
