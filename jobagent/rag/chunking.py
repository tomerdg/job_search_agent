"""Split a job description (or the CV) into chunks that keep their section, and drop boilerplate.

A JD is split at its section headings (English and Hebrew), so a chunk is "the requirements" or "the day to
day" and not an arbitrary window. Each chunk is tagged with one of SECTIONS. Text with no headings falls
back to ~800-character windows with 150 characters of overlap, split on sentence boundaries.

Boilerplate (benefits, location and hours, EEO statements, "apply now", #LI- tags) is dropped and counted,
so the index holds what a JD actually asks for. The text that gets embedded is prefixed with
`company | title | section`, so a chunk keeps its context once it is retrieved on its own.

The CV goes through chunk_cv(), in memory only: its chunks are never written to disk or logged.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

CHUNKING_VERSION = "c1"  # bump when the rules change; logged with every index run and retrieval eval
SECTIONS = ("about", "responsibilities", "requirements", "advantage", "other")
CV_SECTIONS = ("summary", "experience", "skills", "projects", "education", "other")

WINDOW, OVERLAP = 800, 150  # fallback windows, in characters
MAX_SECTION = 1200          # a longer section is split into windows too
MIN_CHARS = 20              # shorter leftovers ("משרה מלאה") are dropped


@dataclass(frozen=True)
class Chunk:
    section: str
    seq: int
    text: str


@dataclass
class ChunkResult:
    chunks: list[Chunk] = field(default_factory=list)
    dropped: int = 0  # chunks dropped as boilerplate or too short


# Heading text -> section. First match wins, so the more specific kinds come first
# ("preferred qualifications" is an advantage, not a requirement). Matched against a normalised heading.
_HEADING_RULES: list[tuple[str, str]] = [
    ("benefits", r"perks|benefits|what we offer|we offer|why join|hiring philosophy|how (do|to) (i )?apply"
                 r"|ready to|working style|work environment|היקף משרה|אנחנו על המפה|הטבות|מה אנחנו מציעים"
                 r"|תנאי(ם)? העסקה|מיקום המשרה"),
    ("advantage", r"nice to have|nice-to-have|bonus|advantage|preferred|would also be nice|\bplus\b"
                  r"|יתרון|יתרונות"),
    ("requirements", r"requirement|qualification|what you('ll| will) (bring|need)|you('ll| will) need"
                     r"|looking for|who you are|\byou are\b|\byou have\b|must have|skills|about you"
                     r"|experience|דרישות|מה אנחנו מחפשים|למי זה מתאים|כישורים|ניסיון נדרש|מחפשים"),
    ("responsibilities", r"responsibilit|what you('ll| will) (do|be doing|own|solve|impact)"
                         r"|you('ll| will) (own|solve|impact|do)\b|make an impact|day.to.day|your role"
                         r"|in this role|what will you be doing|היומיום|במסגרת התפקיד|התפקיד כולל"
                         r"|כולל התפקיד|תחומי אחריות|מה תעשו|מה מחכה לך|אחריות"),
    ("about", r"about|description|overview|the role|role summary|the opportunity|this is us|who we are"
              r"|^why |^meet |^join |company|unique|תיאור|מי אנחנו|על החברה|על התפקיד|קצת עלינו"),
]
_HEADING_RX = [(name, re.compile(rx, re.I)) for name, rx in _HEADING_RULES]
_INLINE_HEADING = re.compile(r"^\s*([^:]{2,40}):\s*(\S.*)$")

# A line that matches is removed from its chunk (not counted). EEO, apply-now and LinkedIn tags.
_BOILERPLATE_LINE = re.compile(
    r"^#li-|equal opportunit|\beeo\b|regardless of (race|gender|age|sex)|without regard to"
    r"|reasonable accommodation|apply now|click (here )?to apply|send (us )?(your )?(cv|resume)"
    r"|המשרה (מיועדת|פונה)|לנשים ולגברים|לגברים ולנשים|privacy (notice|policy)|לשליחת (קו\"ח|קורות חיים)",
    re.I,
)
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|\n+")


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def embed_text(company: str | None, title: str | None, section: str, text: str) -> str:
    """The text that is embedded for a JD chunk: its context first, then the chunk."""
    return f"{company or ''} | {title or ''} | {section}\n{text}"


def cv_embed_text(section: str, text: str) -> str:
    return f"CV | {section}\n{text}"


def _norm_heading(line: str) -> str:
    s = line.replace("﻿", "").replace("’", "'").strip().lower()
    return s.strip(" \t*#•·-–—:?!.").strip()


def heading_section(line: str) -> str | None:
    """The section a heading line starts ('benefits' for one to drop), or None if it is not a heading.
    A heading is short, and either ends with ':' '?' '-' or is three words at most and matches a rule."""
    raw = line.replace("﻿", "").strip()
    norm = _norm_heading(raw)
    if not norm or len(norm) > 60 or len(norm.split()) > 8 or raw.startswith(("•", "*", "·")):
        return None
    marked = raw.rstrip().endswith((":", "?", "-", "–"))
    for name, rx in _HEADING_RX:
        if rx.search(norm):
            return name if marked or len(norm.split()) <= 3 else None
    return "other" if raw.endswith("?") else None


def windows(text: str, size: int = WINDOW, overlap: int = OVERLAP) -> list[str]:
    """~size-character windows on sentence boundaries; each starts with the last ~overlap chars of
    sentences from the one before. A single sentence longer than size is cut by characters."""
    sentences: list[str] = []
    for s in (p.strip() for p in _SENTENCE_END.split(text)):
        while len(s) > size:
            sentences.append(s[:size])
            s = s[size - overlap:]
        if s:
            sentences.append(s)
    out, cur = [], []
    for s in sentences:
        if cur and sum(len(c) + 1 for c in cur) + len(s) > size:
            out.append(" ".join(cur))
            keep: list[str] = []
            for c in reversed(cur):
                if sum(len(k) + 1 for k in keep) + len(c) > overlap:
                    break
                keep.insert(0, c)
            cur = keep
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def _clean(lines: list[str]) -> str:
    return "\n".join(" ".join(ln.split()) for ln in lines if ln.strip())


def chunk_jd(description: str | None, company: str | None = None, title: str | None = None) -> ChunkResult:
    """Chunks of one JD, by section. company/title are not part of the chunk text (see embed_text)."""
    result = ChunkResult()
    text = (description or "").replace("\r\n", "\n")
    sections: list[tuple[str, list[str]]] = [("about", [])]  # text before the first heading: an intro
    found_heading = False
    for line in text.split("\n"):
        if _BOILERPLATE_LINE.search(line):
            continue
        name, rest = heading_section(line), None
        if name is None and (m := _INLINE_HEADING.match(line)):  # "Requirements: 3+ years of SQL"
            name, rest = heading_section(m.group(1) + ":"), m.group(2)
        if name is None:
            sections[-1][1].append(line)
            continue
        found_heading = True
        sections.append((name, [rest] if rest else []))

    if not found_heading:  # no structure: plain windows
        body = _clean(text.split("\n"))
        parts = [p for p in windows(body) if len(p) >= MIN_CHARS]
        result.chunks = [Chunk("other", i, p) for i, p in enumerate(parts)]
        result.dropped = int(bool(body) and not parts)
        return result

    seen: set[str] = set()
    for name, lines in sections:
        body = _clean(lines)
        if not body:
            continue
        if name == "benefits" or len(body) < MIN_CHARS:
            result.dropped += 1
            continue
        for part in windows(body) if len(body) > MAX_SECTION else [body]:
            if part in seen:  # the same paragraph twice in one JD
                continue
            seen.add(part)
            result.chunks.append(Chunk(name, len(result.chunks), part))
    return result


# ---- CV (in memory only) ----
_CV_RULES: list[tuple[str, str]] = [
    ("summary", r"summary|profile|about me|תקציר|פרופיל|על עצמי"),
    ("skills", r"skills|technologies|tech stack|כישורים|מיומנויות"),
    ("experience", r"experience|employment|career history|ניסיון"),
    ("projects", r"projects?|פרויקטים|פרויקט"),
    ("education", r"education|training|certifications?|courses|military|השכלה|שירות צבאי|קורסים"),
]
_CV_KEYWORDS = r"SUMMARY|PROFILE|SKILLS|EXPERIENCE|PROJECTS?|EDUCATION|TRAINING|MILITARY SERVICE|CERTIFICATIONS?"
# In text extracted from a PDF one word per line, headings are found as UPPERCASE runs instead of lines.
_CV_UPPER_HEADING = re.compile(rf"(?<![A-Za-z])((?:[A-Z][A-Z/]* ){{0,2}}(?:{_CV_KEYWORDS})(?: & [A-Z]+)?)(?![a-z])")
_DATE_RANGE = re.compile(r"(?:(?:19|20)\d\d)\s*[-–—]\s*(?:(?:19|20)\d\d|present|current|now|today|היום|כיום)",
                         re.I)


def _cv_section(heading: str) -> str:
    norm = _norm_heading(heading)
    return next((name for name, rx in _CV_RULES if re.search(rx, norm, re.I)), "other")


def _split_jobs(text: str) -> list[str]:
    """Split an experience section before each job: the header (company, title) ahead of each date
    range starts a new part. The split goes back to the end of the previous sentence or bullet."""
    cuts = [0]
    for m in _DATE_RANGE.finditer(text):
        lo = cuts[-1]
        nl = text.rfind("\n", lo, m.start())
        if nl >= lo:  # line mode: the header is this line, or the line above when the date starts the line
            if text[nl + 1:m.start()].strip():
                cut = nl + 1
            else:
                prev = text.rfind("\n", lo, nl)
                cut = prev + 1 if prev >= lo else lo
        else:  # collapsed text: back to the end of the previous sentence or bullet
            end = max(text.rfind(". ", lo, m.start()), text.rfind("•", lo, m.start()))
            cut = end + 1 if end >= lo else lo
        if cut - lo >= MIN_CHARS:
            cuts.append(cut)
    return [p.strip(" •\n") for p in (text[a:b] for a, b in zip(cuts, cuts[1:] + [len(text)])) if p.strip()]


def chunk_cv(cv_text: str) -> list[Chunk]:
    """The CV by its headings (summary, each job, skills, projects, education). Never persisted."""
    lines = [ln.strip() for ln in (cv_text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    parts: list[tuple[str, str]] = []
    word_per_line = len(lines) > 40 and sorted(len(ln) for ln in lines)[len(lines) // 2] < 15
    if word_per_line:  # PDF extraction: collapse, then cut at UPPERCASE headings
        flat = " ".join(lines)
        marks = list(_CV_UPPER_HEADING.finditer(flat))
        bounds = [(0, "other")] + [(m.start(), _cv_section(m.group(1))) for m in marks]
        for (a, name), (b, _) in zip(bounds, bounds[1:] + [(len(flat), "")]):
            parts.append((name, flat[a:b]))
    else:
        name, buf = "other", []
        for ln in lines:
            norm = _norm_heading(ln)
            if len(norm.split()) <= 4 and any(re.fullmatch(rf"(\w+ )?({rx})( & \w+)?", norm, re.I)
                                               for _, rx in _CV_RULES):
                parts.append((name, "\n".join(buf)))
                name, buf = _cv_section(ln), []
            else:
                buf.append(ln)
        parts.append((name, "\n".join(buf)))

    chunks: list[Chunk] = []
    for name, body in parts:
        body = body.strip()
        if len(body) < MIN_CHARS:
            continue
        pieces = _split_jobs(body) if name == "experience" else [body]
        for piece in pieces:
            for w in windows(piece) if len(piece) > MAX_SECTION else [piece]:
                chunks.append(Chunk(name, len(chunks), w))
    return chunks
