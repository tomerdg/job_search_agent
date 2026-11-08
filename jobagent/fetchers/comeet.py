"""Comeet public careers API."""
from __future__ import annotations

from jobagent.models import Job, clean_text

from .http import TIMEOUT, session

BASE = "https://www.comeet.co/careers-api/2.0/company"


def _location(p: dict) -> str:
    loc = p.get("location") or {}
    parts = [loc.get("city"), loc.get("name"), loc.get("country")]
    text = ", ".join(x for x in parts if x)
    if loc.get("is_remote"):
        text += " (remote)"
    return text


def parse(data: list, src: dict) -> list[Job]:
    jobs = []
    for p in data or []:
        details = "\n".join(
            f"{d.get('name', '')}:\n{clean_text(d.get('value'))}" for d in p.get("details") or []
        )
        jobs.append(Job(
            source="comeet",
            company=src["company"],
            external_id=p.get("uid") or p.get("position_uid") or p.get("name", ""),
            title=(p.get("name") or "").strip(),
            url=p.get("url_comeet_hosted_page") or p.get("url_active_page") or "",
            location=_location(p),
            department=p.get("department") or "",
            description=details,
            posted_at=(p.get("time_updated") or "")[:10],
        ))
    return jobs


def fetch(src: dict) -> list[Job]:
    url = f"{BASE}/{src['uid']}/positions"
    resp = session.get(url, params={"token": src["token"], "details": "true"}, timeout=TIMEOUT)
    resp.raise_for_status()
    return parse(resp.json(), src)
