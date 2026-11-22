"""Small eval for `python -m jobagent.ask` (local only, real API calls: about 10 questions per run).

    python -m eval.ask --db data/jobs.duckdb

questions.jsonl: each question has a rule (as in eval/retrieval) for the jobs that answer it. Per question:
  hit            the answer cites at least one job the rule matches, or names one of their companies;
                 for a rule that matches nothing, the answer cites nothing (it should say it cannot tell)
  citations      valid = cited ids that were retrieved; invalid ones are dropped by the post-check
Reported: hit_rate and citation_validity, to MLflow experiment "ask". Answers are not logged or saved.
"""
from __future__ import annotations

import json
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def judge(answer, expected_jobs: set[str], expected_companies: set[str]) -> bool:
    """Did the answer find the jobs the question is about (or correctly find none)?"""
    if not expected_jobs:
        return not answer.cited
    text = answer.text.lower()
    return bool(set(answer.cited) & expected_jobs) or any(c.lower() in text for c in expected_companies)


def summarize(rows: list[dict]) -> dict:
    valid = sum(r["valid"] for r in rows)
    invalid = sum(r["invalid"] for r in rows)
    return {"n_questions": len(rows), "hit_rate": sum(r["hit"] for r in rows) / len(rows) if rows else 0.0,
            "citation_validity": valid / (valid + invalid) if valid + invalid else 1.0,
            "citations_per_answer": valid / len(rows) if rows else 0.0}
