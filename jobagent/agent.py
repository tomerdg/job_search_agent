"""Claude tool-calling agent that scores new jobs against the CV."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from jobagent.costs import add_usage, empty_usage, estimate_cost_usd
from jobagent.db import Store

log = logging.getLogger(__name__)

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v1")
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

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(version: str) -> str:
    """System prompt template jobagent/prompts/{version}.txt (placeholders: roles, allowed, blocked)."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", version) or version.startswith("."):
        raise ValueError(f"bad prompt version: {version!r}")
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file for version {version!r}: {path}")
    return path.read_text(encoding="utf-8").rstrip("\n")


class Agent:
    def __init__(self, store: Store, cv_text: str, profile: dict, client=None,
                 model: str | None = None, prompt_version: str | None = None):
        self.store = store
        self.cv_text = cv_text
        self.profile = profile
        self.model = model or MODEL
        self.prompt_version = prompt_version or PROMPT_VERSION
        self.prompt = load_prompt(self.prompt_version)  # fail early on a bad version
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.saved: set[str] = set()
        self.usage: dict[str, int] = empty_usage()  # summed over every API call of every run()
        self.turns = 0  # API calls made

    @property
    def cost_usd(self) -> float | None:
        return estimate_cost_usd(self.usage, self.model)

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
        }, prompt_version=self.prompt_version, model=self.model)
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
        system = self.prompt.format(roles=", ".join(p.get("target_roles", [])),
                               allowed=", ".join(p.get("locations_allowed", [])) or "any",
                               blocked=", ".join(p.get("locations_blocked", [])) or "none")
        for _ in range(MAX_TURNS):
            resp = self.client.messages.create(
                model=self.model, max_tokens=4096, system=system, tools=TOOLS, messages=messages,
            )
            self.turns += 1
            add_usage(self.usage, getattr(resp, "usage", None))
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
        cost = self.cost_usd
        log.info("Agent scored %d jobs in %d turns (prompt %s, model %s); tokens %s; est. cost %s",
                 len(self.saved), self.turns, self.prompt_version, self.model,
                 self.usage, "unknown" if cost is None else f"${cost:.4f}")
        return len(self.saved)
