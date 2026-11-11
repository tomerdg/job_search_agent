"""Fetch public job postings by URL and append them to the evaluation draft in eval/out/.

    python scripts/fetch_eval_postings.py urls.txt --label fit --note applied

urls.txt holds one posting per line, `URL` or `company<TAB>URL`. Keep it outside git (e.g. in eval/out/):
a list of jobs you picked is personal. Each fetched posting is appended to eval/out/jobs.jsonl, with a
matching row in eval/out/labels.csv (the given label pre-filled, to review). case_id is a hash of the
URL, so hand-added jobs look like the exported ones. Postings already in the draft are skipped.

Supported: Greenhouse, Ashby, Workable and Hunter HRMS through their public APIs; any other page through its
schema.org JobPosting data, falling back to the page text. LinkedIn is skipped (it blocks automated
access). A posting whose text comes back shorter than --min-chars is reported and not added: it is
usually a closed job or a page that needs JavaScript.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jobagent.agent import JD_MAX_CHARS  # noqa: E402
from jobagent.fetchers.http import TIMEOUT, session  # noqa: E402
from jobagent.models import clean_text  # noqa: E402

LABEL_COLS = ["case_id", "label", "note", "company", "title", "location", "url"]


def _greenhouse(url: str) -> dict | None:
    m = re.search(r"greenhouse\.io/([^/]+)/jobs/(\d+)", url) or re.search(r"[?&]gh_jid=(\d+)", url)
    if not m:
        return None
    if len(m.groups()) == 1:  # company careers page that embeds Greenhouse (?gh_jid=); board unknown
        return None
    board, jid = m.groups()
    for host in ("boards-api.greenhouse.io", "boards-api.eu.greenhouse.io"):
        r = session.get(f"https://{host}/v1/boards/{board}/jobs/{jid}", timeout=TIMEOUT)
        if r.ok:
            d = r.json()
            return {"title": d.get("title", ""), "location": (d.get("location") or {}).get("name", ""),
                    "department": ((d.get("departments") or [{}])[0] or {}).get("name", ""),
                    "description": clean_text(html.unescape(d.get("content") or ""))}
    return None


def _ashby(url: str) -> dict | None:
    m = re.search(r"jobs\.ashbyhq\.com/([^/]+)/([0-9a-f-]{36})", url)
    if not m:
        return None
    org, jid = m.groups()
    r = session.get(f"https://api.ashbyhq.com/posting-api/job-board/{org}", timeout=TIMEOUT)
    if not r.ok:
        return None
    for j in r.json().get("jobs", []):
        if j.get("id") == jid:
            return {"title": j.get("title", ""), "location": j.get("location", ""),
                    "department": j.get("department", ""),
                    "description": clean_text(j.get("descriptionHtml") or j.get("descriptionPlain") or "")}
    return None


def _workable(url: str) -> dict | None:
    m = re.search(r"apply\.workable\.com/([^/]+)/j/([A-Z0-9]+)", url)
    if not m:
        return None
    acct, code = m.groups()
    r = session.get(f"https://apply.workable.com/api/v2/accounts/{acct}/jobs/{code}", timeout=TIMEOUT)
    if not r.ok:
        return None
    d = r.json()
    loc = d.get("location") or {}
    text = " ".join(d.get(k) or "" for k in ("description", "requirements", "benefits"))
    return {"title": d.get("title", ""), "location": ", ".join(v for v in (loc.get("city"), loc.get("country")) if v),
            "department": ", ".join(d.get("department") or []) if isinstance(d.get("department"), list)
            else (d.get("department") or ""),
            "description": clean_text(text)}


def _hunter(url: str) -> dict | None:
    """Hunter HRMS career sites (<tenant>-careersite.hunterhrms.com): a JavaScript page that loads the job
    from the tenant's backend, the same call the page makes."""
    m = re.search(r"//([a-z0-9-]+)-careersite\.hunterhrms\.com/job\?jid=(\d+)", url)
    if not m:
        return None
    tenant, jid = m.groups()
    r = session.post(f"https://niloo-server.herokuapp.com/actions-{tenant}-career",
                     json={"cmd": "get-job", "data": {"jid": jid}}, timeout=TIMEOUT)
    if not r.ok or not isinstance(r.json(), dict):
        return None
    d = r.json()
    text = " ".join(html.unescape(d.get(k) or "") for k in ("description", "requirements", "skills"))
    return {"title": d.get("jobTitle", ""), "location": d.get("locationAddress") or "", "department": "",
            "description": clean_text(text)}


def _jsonld_posting(page: str) -> dict | None:
    for block in re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', page, re.S | re.I):
        try:
            data = json.loads(block.strip())
        except ValueError:
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for it in items:
            if isinstance(it, dict) and it.get("@type") == "JobPosting":
                loc = it.get("jobLocation") or {}
                loc = loc[0] if isinstance(loc, list) and loc else loc
                addr = (loc.get("address") or {}) if isinstance(loc, dict) else {}
                place = addr.get("addressLocality", "") if isinstance(addr, dict) else ""
                return {"title": it.get("title", ""), "location": place, "department": "",
                        "description": clean_text(html.unescape(it.get("description") or ""))}
    return None


def _generic(url: str) -> dict | None:
    r = session.get(url, timeout=TIMEOUT)
    if not r.ok:
        return None
    if r.encoding and r.encoding.lower() == "iso-8859-1":  # requests' default when no charset is sent
        r.encoding = r.apparent_encoding
    page = r.text
    found = _jsonld_posting(page)
    if found and len(found["description"]) >= 1000:
        return found
    title = re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I)
    code_free = re.sub(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
    # Dropping nav/header/footer removes menus, but some sites wrap the posting in a <header>;
    # when that leaves little text, keep them.
    body = clean_text(re.sub(r"<(nav|footer|header)[^>]*>.*?</\1>", " ", code_free, flags=re.S | re.I))
    if len(body) < 1000:
        body = max(body, clean_text(code_free), key=len)
    if found and len(found["description"]) >= len(body):  # some JobPosting data is only a short summary
        return found
    return {"title": (found or {}).get("title") or (clean_text(title.group(1)) if title else ""),
            "location": (found or {}).get("location", ""), "department": "", "description": body}


def fetch_posting(url: str) -> dict | None:
    host = urlparse(url).netloc
    if "linkedin.com" in host:
        return None
    for fn in (_greenhouse, _ashby, _workable, _hunter):
        got = fn(url)
        if got:
            return got
    return _generic(url)


def read_urls(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        company, _, url = line.rpartition("\t")
        out.append((company.strip(), url.strip()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("urls", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "out")
    ap.add_argument("--label", default="", choices=["", "fit", "maybe", "no_fit"])
    ap.add_argument("--note", default="")
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    jobs_path, labels_path = args.out / "jobs.jsonl", args.out / "labels.csv"
    known = set()
    if jobs_path.exists():
        known = {json.loads(l)["case_id"] for l in jobs_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    new_labels = not labels_path.exists()

    added, failed = 0, []
    with open(jobs_path, "a", encoding="utf-8") as jf, open(labels_path, "a", encoding="utf-8", newline="") as lf:
        lw = csv.writer(lf)
        if new_labels:
            lw.writerow(LABEL_COLS)
        for company, url in read_urls(args.urls):
            case_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
            if case_id in known:
                continue
            try:
                got = fetch_posting(url)
            except Exception as exc:  # noqa: BLE001
                failed.append((company or url, type(exc).__name__))
                continue
            if not got or len(got["description"]) < args.min_chars:
                failed.append((company or url, "skipped (LinkedIn)" if "linkedin.com" in url
                               else "no posting text (closed, or page needs JavaScript)"))
                continue
            row = {"case_id": case_id, "company": company or urlparse(url).netloc, "title": got["title"].strip(),
                   "location": got["location"] or "", "department": got["department"] or "", "url": url,
                   "description": got["description"][:JD_MAX_CHARS]}
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            lw.writerow([case_id, args.label, args.note, row["company"], row["title"], row["location"], url])
            known.add(case_id)
            added += 1

    print(f"added {added}, not added {len(failed)}")
    for who, why in failed:
        print(f"  {who}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
