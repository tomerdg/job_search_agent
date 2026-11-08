from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass, field


def clean_text(value: str | None) -> str:
    """Strip HTML tags and collapse whitespace."""
    if not value:
        return ""
    text = re.sub(r"<br\s*/?>|</p>|</li>", "\n", value, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


@dataclass
class Job:
    source: str          # fetcher type, e.g. "adam"
    company: str
    external_id: str     # id inside the source system
    title: str
    url: str
    location: str = ""
    department: str = ""
    description: str = ""
    posted_at: str = ""
    sector: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def job_id(self) -> str:
        raw = f"{self.source}|{self.company}|{self.external_id}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
