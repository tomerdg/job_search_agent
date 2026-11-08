"""Claude tool-calling agent that scores new jobs against the CV."""
from __future__ import annotations

import json
import logging
import os


from jobagent.db import Store

log = logging.getLogger(__name__)

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 80
JD_MAX_CHARS = 3500
BATCH = 15  # jobs scored per run; the rest wait for the next day

TOOLS = [
    {
        "name": "list_new_jobs",
        "description": "List new jobs that passed the keyword and location pre-filter and are not scored yet. "
                       "Returns id, company, title, location and department for each.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_job_description",
        "description": "Read the full job description (JD) of one job by id.",
        "input_schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "read_cv",
        "description": "Read the candidate's CV as plain text.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "save_match",
        "description": "Save the match score for one job. Call once for every job returned by list_new_jobs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "score": {"type": "integer", "minimum": 0, "maximum": 100,
                          "description": "0-100 fit of the candidate for this job."},
                "summary": {"type": "string", "description": "One or two sentences in Hebrew: what the role is and why it fits or not."},
                "strengths": {"type": "array", "items": {"type": "string"}, "description": "Up to 3 matching points."},
                "gaps": {"type": "array", "items": {"type": "string"}, "description": "Up to 3 gaps or risks."},
            },
            "required": ["job_id", "score", "summary", "strengths", "gaps"],
        },
    },
]

SYSTEM = """You are a job-matching agent for one candidate in Israel.
Target roles: {roles}. Allowed locations: {allowed}. Excluded locations: {blocked}.

Work like this:
1. Call read_cv once.
2. Call list_new_jobs.
3. For every job, call read_job_description, then save_match. You may call several tools in one turn.
Scoring guide: 85-100 = the role is one of the target roles and the requirements match the CV well;
60-84 = a target role with real gaps, or an adjacent data role (e.g. data analyst with heavy SQL/ETL);
30-59 = data-related but a different job family; 0-29 = not a data role, or the location is clearly outside the area.
Judge the JD itself, not just the title. Write summary, strengths and gaps in Hebrew. Be concise.
When every job has a saved score, reply with the single word DONE."""


class Agent:
    def __init__(self, store: Store, cv_text: str, profile: dict, client=None):
        self.store = store
        self.cv_text = cv_text
        self.profile = profile
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.saved: set[str] = set()

    # ---- tools ----
    def _list_new_jobs(self, _: dict) -> str:
        rows = self.store.pending_scoring(limit=BATCH)
        return json.dumps([
            {"job_id": r["job_id"], "company": r["company"], "title": r["title"],
             "location": r["location"], "department": r["department"]}
            for r in rows
        ], ensure_ascii=False)

    def _read_job_description(self, args: dict) -> str:
        row = self.store.get(args["job_id"])
        if row is None:
            return "error: unknown job_id"
        return (f"Company: {row['company']}\nTitle: {row['title']}\nLocation: {row['location']}\n"
                f"Department: {row['department']}\n\n{(row['description'] or '')[:JD_MAX_CHARS]}")

    def _read_cv(self, _: dict) -> str:
        return self.cv_text

    def _save_match(self, args: dict) -> str:
        if self.store.get(args["job_id"]) is None:
            return "error: unknown job_id"
        score = max(0, min(100, int(args["score"])))
        self.store.save_score(args["job_id"], score, {
            "summary": args.get("summary", ""),
            "strengths": args.get("strengths", [])[:3],
            "gaps": args.get("gaps", [])[:3],
        })
        self.saved.add(args["job_id"])
        return "saved"

    def _dispatch(self, name: str, args: dict) -> str:
        fn = {
            "list_new_jobs": self._list_new_jobs,
            "read_job_description": self._read_job_description,
            "read_cv": self._read_cv,
            "save_match": self._save_match,
        }.get(name)
        if fn is None:
            return f"error: unknown tool {name}"
        try:
            return fn(args or {})
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    # ---- loop ----
    def run(self) -> int:
        if not self.store.pending_scoring(limit=1):
            log.info("Nothing new to score.")
            return 0
        messages = [{"role": "user", "content": "Score today's new jobs."}]
        p = self.profile
        system = SYSTEM.format(roles=", ".join(p.get("target_roles", [])),
                               allowed=", ".join(p.get("locations_allowed", [])) or "any",
                               blocked=", ".join(p.get("locations_blocked", [])) or "none")
        for turn in range(MAX_TURNS):
            resp = self.client.messages.create(
                model=MODEL, max_tokens=4096, system=system, tools=TOOLS, messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                break
            results = []
            for tu in tool_uses:
                out = self._dispatch(tu.name, tu.input)
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
            # Cache the conversation so far: each turn re-sends it, caching keeps that cheap.
            for m in messages:
                if m["role"] == "user" and isinstance(m["content"], list):
                    for block in m["content"]:
                        block.pop("cache_control", None)
            results[-1]["cache_control"] = {"type": "ephemeral"}
            messages.append({"role": "user", "content": results})
        log.info("Agent scored %d jobs in %d turns", len(self.saved), turn + 1)
        return len(self.saved)
