"""Greenhouse public job board API (boards-api.greenhouse.io)."""
from __future__ import annotations

import html

from jobagent.models import Job, clean_text

from .http import TIMEOUT, session

BASE = "https://boards-api.greenhouse.io/v1/boards"


def parse(data: dict, src: dict) -> list[Job]:
    jobs = []
    for p in (data or {}).get("jobs") or []:
        depts = p.get("departments") or []
        jobs.append(Job(
            source="greenhouse",
            company=src["company"],
            external_id=str(p.get("id", "")),
            title=(p.get("title") or "").strip(),
            url=p.get("absolute_url") or "",
            location=((p.get("location") or {}).get("name") or "").strip(),
            department=(depts[0].get("name") or "") if depts else "",
            description=clean_text(html.unescape(p.get("content") or "")),
            posted_at=(p.get("first_published") or p.get("updated_at") or "")[:10],
        ))
    return jobs


def fetch(src: dict) -> list[Job]:
    resp = session.get(f"{BASE}/{src['board']}/jobs", params={"content": "true"}, timeout=TIMEOUT)
    resp.raise_for_status()
    return parse(resp.json(), src)
