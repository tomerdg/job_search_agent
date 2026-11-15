"""Cheap filters that run before any model call."""
from __future__ import annotations

import re


def _compile(terms: list[str]) -> list[re.Pattern]:
    pats = []
    for t in terms:
        if re.fullmatch(r"[A-Za-z0-9 .+\-/]+", t):
            pats.append(re.compile(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", re.I))
        else:  # Hebrew and mixed terms: plain substring
            pats.append(re.compile(re.escape(t), re.I))
    return pats


class PreFilter:
    def __init__(self, profile: dict):
        self.keywords = _compile(profile.get("keywords", []))
        self.allowed = _compile(profile.get("locations_allowed", []))
        self.blocked = _compile(profile.get("locations_blocked", []))

    def matches_role(self, title: str, description: str) -> bool:
        text = f"{title}\n{description}"
        return any(p.search(text) for p in self.keywords)

    def location_ok(self, location: str) -> bool:
        loc = (location or "").strip()
        if not loc:
            return True  # unknown: let the model decide
        # Some boards hyphenate place names ("Petah-Tikva"); match either spelling.
        variants = {loc, loc.replace("-", " ")}
        if any(p.search(v) for p in self.allowed for v in variants):
            return True
        if any(p.search(v) for p in self.blocked for v in variants):
            return False
        return False  # a named place outside the target area

    def __call__(self, title: str, description: str, location: str) -> bool:
        return self.location_ok(location) and self.matches_role(title, description)
