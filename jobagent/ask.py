"""Ask questions over the collected job descriptions, with citations. Local only.

    python -m jobagent.ask "which companies asked for Kafka in the last 30 days?" --since 30d
    python -m jobagent.ask "what do analyst roles require most often?" --active --k 12

Hybrid search (jobagent/rag/retrieve.py) picks the JD chunks, and one Claude call answers from them as
numbered sources, citing [job_id] for every claim. For an aggregate question that names a skill ("how many",
"which companies"), the counts come from SQL over the chunks and are passed in as facts, so numbers never
come from the model. A cited id that is not one of the retrieved jobs is dropped and flagged.

The answer is printed to the terminal only: never logged in CI, never sent in the digest.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from jobagent.db import Store
from jobagent.rag.retrieve import _since, rank_jobs, search

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("ASK_MODEL") or os.environ.get("CLAUDE_MODEL") or "claude-sonnet-5-5"
AGGREGATE = re.compile(r"\bhow many\b|\bwhich compan|\bmost (often|common)|\bcount\b|\bhow often\b"
                       r"|כמה|אילו חברות|איזה חברות|הכי הרבה|הכי נפוץ", re.I)
CITATION = re.compile(r"\[([0-9a-f]{6,16})\]")

SYSTEM = """You answer questions about job postings that were collected from company careers sites.
Use only the numbered sources below. Every claim must cite the job it comes from as [job_id], using the
job_id shown on the source. If the sources do not answer the question, say so plainly instead of guessing.
If FACTS FROM SQL are given, take every number from them: they count all matching jobs, while the sources
are only a sample. Answer in the language of the question. Be concise."""


@dataclass
class Answer:
    text: str
    sources: list[dict]
    cited: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def skills_in(question: str) -> list[str]:
    """Skills from jobagent.ml.features.SKILLS that the question names."""
    from jobagent.ml.features import SKILLS
    return [name for name, rx in SKILLS.items() if re.search(rx, question, re.I)]


def skill_facts(store: Store, question: str, since=None, active_only: bool = False) -> list[str]:
    """For an aggregate question that names a skill: how many jobs (and which companies) have a chunk
    that matches it, counted in SQL over job_chunks with the same filters as the search."""
    if not AGGREGATE.search(question):
        return []
    from jobagent.ml.features import SKILLS
    facts = []
    for skill in skills_in(question):
        where, params = ["regexp_matches(c.text, ?, 'i')"], [SKILLS[skill]]
        if active_only:
            where.append("j.is_active")
        if (ts := _since(since)) is not None:
            where.append("j.first_seen >= ?")
            params.append(ts)
        rows = store.conn.execute(
            "SELECT j.company, count(DISTINCT j.job_id) AS n FROM job_chunks c JOIN jobs j USING (job_id) "
            f"WHERE {' AND '.join(where)} GROUP BY 1 ORDER BY n DESC, 1", params).fetchall()
        total = sum(n for _, n in rows)
        top = ", ".join(f"{c} ({n})" for c, n in rows[:10]) or "none"
        facts.append(f"{total} job(s) mention {skill} across {len(rows)} company(ies); by company: {top}")
    return facts


def format_sources(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] job_id={h['job_id']} | {h['company']} | {h['title']} | {h['section']} | "
        f"first seen {str(h['first_seen'])[:10]}{'' if h['is_active'] else ' | closed'}\n{h['text']}"
        for i, h in enumerate(hits, 1))


def check_citations(text: str, retrieved_ids: list[str]) -> tuple[str, list[str], list[str]]:
    """Keep citations of retrieved jobs (a unique prefix counts), drop the rest and mark them in the text.
    Returns (text, valid ids, invalid ids)."""
    valid, invalid = [], []

    def fix(m: re.Match) -> str:
        ref = m.group(1)
        matches = [j for j in retrieved_ids if j.startswith(ref)]
        if len(matches) == 1:
            valid.append(matches[0])
            return f"[{matches[0]}]"
        invalid.append(ref)
        return "[citation removed: not a retrieved job]"

    return CITATION.sub(fix, text), list(dict.fromkeys(valid)), invalid


def ask(store: Store, embedder, question: str, k: int = 12, since=None, active_only: bool = False,
        client=None, model: str = MODEL) -> Answer:
    hits = search(store, embedder, question, k=k, mode="hybrid" if embedder else "bm25",
                  since=since, active_only=active_only, per_job=2)
    facts = skill_facts(store, question, since, active_only)
    if not hits and not facts:
        return Answer("No collected job matches this question (with these filters).", [])
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    user = (f"QUESTION: {question}\n\n" + ("FACTS FROM SQL:\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"
                                          if facts else "") + f"SOURCES:\n{format_sources(hits)}")
    resp = client.messages.create(model=model, max_tokens=1500, system=SYSTEM,
                                  messages=[{"role": "user", "content": user}])
    raw = "".join(getattr(b, "text", "") for b in resp.content)
    text, cited, invalid = check_citations(raw, rank_jobs(hits))
    from jobagent.costs import add_usage, empty_usage
    usage = empty_usage()
    add_usage(usage, getattr(resp, "usage", None))
    return Answer(text, hits, cited, invalid, facts, usage)


def main(argv=None) -> None:
    from jobagent.main import load_dotenv
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Ask a question over the collected job descriptions (local only)")
    ap.add_argument("question")
    ap.add_argument("--db", default=os.environ.get("JOBS_DB", str(ROOT / "data" / "jobs.duckdb")))
    ap.add_argument("--since", help="only jobs first seen since: 30d, or a date (2026-10-01)")
    ap.add_argument("--active", action="store_true", help="only jobs that are still open")
    ap.add_argument("--k", type=int, default=12, help="chunks to retrieve")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--sources", action="store_true", help="also print the retrieved sources")
    args = ap.parse_args(argv)

    from jobagent.rag.embed import get_embedder
    embedder = get_embedder()
    store = Store(args.db)
    try:
        if not store.conn.execute("SELECT count(*) FROM job_chunks").fetchone()[0]:
            print("The index is empty. Build it with: python -m jobagent.rag.index --scope all")
            return
        a = ask(store, embedder, args.question, k=args.k, since=args.since, active_only=args.active,
                model=args.model)
    finally:
        store.close()
    print(a.text)
    if a.invalid:
        print(f"\n[!] {len(a.invalid)} citation(s) removed: not among the retrieved jobs: {', '.join(a.invalid)}")
    if a.facts:
        print("\nCounts (SQL):\n" + "\n".join(f"  {f}" for f in a.facts))
    print(f"\n{len(a.cited)} job(s) cited out of {len(rank_jobs(a.sources))} retrieved")
    if args.sources:
        print("\n" + format_sources(a.sources))


if __name__ == "__main__":
    main()
