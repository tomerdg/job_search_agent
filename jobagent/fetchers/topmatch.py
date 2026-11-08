"""TopMatch (careers.topmatch.co.il): JSON API keyed by an affiliate GUID."""
from __future__ import annotations

from jobagent.models import Job, clean_text

from .http import TIMEOUT, session

BASE = "https://careers.topmatch.co.il"


def parse(data: dict, src: dict) -> list[Job]:
    jobs = []
    for p in data.get("positions") or []:
        if p.get("isActivePosition") is False:
            continue
        pid = str(p.get("compPositionID"))
        jobs.append(Job(
            source="topmatch",
            company=src["company"],
            external_id=pid,
            title=(p.get("jobTitleText") or "").strip(),
            url=f"{BASE}/{src.get('slug', '')}/",  # TopMatch has no stable per-job URL; id is in external_id
            location=p.get("displayLocation") or p.get("location") or "",
            department=p.get("fieldDesc") or "",
            description=clean_text(p.get("description")),
            posted_at=(p.get("activationDate") or "")[:10],
        ))
    return jobs


def fetch(src: dict) -> list[Job]:
    url = f"{BASE}/CandidateAPI/api/position/Search/{src['affiliate']}"
    body = {"KeyWords": "", "CategoryId": [], "countryId": "2", "cityId": []}
    resp = session.post(url, json=body, timeout=TIMEOUT,
                        headers={"Referer": f"{BASE}/{src.get('slug', '')}/"})
    resp.raise_for_status()
    return parse(resp.json(), src)
