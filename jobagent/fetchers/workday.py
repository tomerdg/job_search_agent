"""Workday career sites ({tenant}.wdN.myworkdayjobs.com).

Every Workday site serves the same public JSON API that its careers page calls:
  POST /wday/cxs/{tenant}/{site}/jobs        -> paged list (title, path, req id; no description)
  GET  /wday/cxs/{tenant}/{site}{externalPath} -> one posting with the full description

Config (sources.yaml):
  type: workday
  url: https://intel.wd1.myworkdayjobs.com/External   # the careers page, without query string
  tenant: intel            # optional, defaults to the first label of the host
  facets:                  # optional, the same ids the careers page puts in its URL query
    locations: [<facet id>]
  search_text: ""          # optional free-text search
  max_jobs: 200            # optional cap on postings per run
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from jobagent.models import Job, clean_text

from .http import TIMEOUT, session

log = logging.getLogger(__name__)

PAGE_SIZE = 20  # Workday rejects larger pages
DEFAULT_MAX_JOBS = 200


def site_parts(src: dict) -> tuple[str, str, str]:
    """Return (origin, tenant, site) from the source's careers-page URL."""
    u = urlparse(src["url"])
    path = [p for p in u.path.split("/") if p]
    # Careers URLs sometimes carry a locale first, e.g. /en-US/External.
    if len(path) > 1 and "-" in path[0] and len(path[0]) <= 5:
        path = path[1:]
    if not path:
        raise ValueError(f"Workday url has no site name: {src['url']}")
    tenant = src.get("tenant") or u.hostname.split(".")[0]
    return f"{u.scheme}://{u.netloc}", tenant, path[0]


def _req_id(p: dict) -> str:
    bullets = p.get("bulletFields") or []
    return str(bullets[0]) if bullets else p.get("externalPath", "")


def parse_list(data: dict) -> list[dict]:
    return [p for p in (data or {}).get("jobPostings") or [] if p.get("externalPath")]


def parse(posting: dict, detail: dict | None, src: dict) -> Job:
    """Build a Job from a list entry and (if it could be fetched) its detail payload."""
    origin, _, site = site_parts(src)
    info = (detail or {}).get("jobPostingInfo") or {}
    locations = [info.get("location") or "", *(info.get("additionalLocations") or [])]
    location = " | ".join(x for x in locations if x) or (posting.get("locationsText") or "")
    return Job(
        source="workday",
        company=src["company"],
        external_id=str(info.get("jobReqId") or _req_id(posting)),
        title=(info.get("title") or posting.get("title") or "").strip(),
        url=info.get("externalUrl") or f"{origin}/{site}{posting['externalPath']}",
        location=location,
        description=clean_text(info.get("jobDescription")),
        posted_at=(info.get("startDate") or "")[:10],
        extra={"time_type": info["timeType"]} if info.get("timeType") else {},
    )


def fetch(src: dict) -> list[Job]:
    origin, tenant, site = site_parts(src)
    api = f"{origin}/wday/cxs/{tenant}/{site}"
    cap = int(src.get("max_jobs", DEFAULT_MAX_JOBS))
    body = {"appliedFacets": src.get("facets") or {}, "searchText": src.get("search_text", ""),
            "limit": PAGE_SIZE, "offset": 0}
    headers = {"Accept": "application/json", "Accept-Language": "en-US"}

    postings: list[dict] = []
    total = None
    while True:
        resp = session.post(f"{api}/jobs", json=body, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if total is None:  # Workday only reports the real total on the first page
            total = int(data.get("total") or 0)
        page = parse_list(data)
        postings.extend(page)
        body["offset"] += PAGE_SIZE
        if not page or body["offset"] >= total or len(postings) >= cap:
            break
    if total > cap:
        log.warning("%s: %d postings match, keeping the first %d (raise max_jobs or narrow facets)",
                    src["company"], total, cap)

    jobs = []
    for p in postings[:cap]:
        detail = None
        try:
            r = session.get(f"{api}{p['externalPath']}", headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            detail = r.json()
        except Exception as exc:  # noqa: BLE001 - keep the posting without its description
            log.warning("%s: detail for %s failed: %s", src["company"], p["externalPath"], exc)
        jobs.append(parse(p, detail, src))
    return jobs
