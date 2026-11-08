"""Adam (career.adamtotal.co.il): server-rendered HTML, 25 jobs per page, ?page=N."""
from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup

from jobagent.models import Job, clean_text

from .http import TIMEOUT, session

BASE = "https://career.adamtotal.co.il"
MAX_PAGES = 20


def parse_page(html: str, company: str) -> list[Job]:
    soup = BeautifulSoup(html, "html.parser")
    jobs = []
    for card in soup.select("article.job-card"):
        job_no = card.get("data-job-id", "").strip()
        title = card.get("data-job-title") or ""
        if not title:
            h = card.select_one(".job-title-button")
            title = h.get_text(strip=True) if h else ""
        location = department = ""
        for span in card.select(".job-meta span"):
            icon = span.find("i")
            classes = " ".join(icon.get("class", [])) if icon else ""
            text = span.get_text(" ", strip=True)
            if "map-marker" in classes:
                location = text
            elif "building" in classes:
                department = text
        link = card.select_one("a.job-detail-link")
        desc = card.select_one(".description-text")
        jobs.append(Job(
            source="adam",
            company=company,
            external_id=job_no or title,
            title=title.strip(),
            url=urljoin(BASE, link["href"]) if link else BASE,
            location=location,
            department=department,
            description=clean_text(desc.decode_contents()) if desc else "",
        ))
    return jobs


def fetch(src: dict) -> list[Job]:
    token = src["token"]
    seen, out = set(), []
    for page in range(1, MAX_PAGES + 1):
        resp = session.get(f"{BASE}/Home/Index", params={"page": page, "token": token}, timeout=TIMEOUT)
        resp.raise_for_status()
        batch = [j for j in parse_page(resp.text, src["company"]) if j.external_id not in seen]
        if not batch:
            break
        seen.update(j.external_id for j in batch)
        out.extend(batch)
    return out
