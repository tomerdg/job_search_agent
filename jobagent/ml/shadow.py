"""Relevance model in shadow mode, inside the daily run.

Each run: retrain the logistic-regression model from the labels in the DB (no model file is kept), then
score every active job the LLM has not scored yet and store P(fit or maybe) in jobs.ml_score. Nothing is
filtered: the score only orders the LLM queue (Store.pending_scoring), so the best guesses are scored first.
dbt's mart_ml_vs_llm then compares the model with the LLM on the jobs both scored.

Labels: LLM scores (silver), job_feedback (your decisions and the imported eval labels), jobs with no role
keyword (weak negatives), and, when EVAL_JOBS_JSONL and EVAL_LABELS_CSV point at the decrypted eval set
(the workflow mounts it read-only), the eval cases as gold rows with their own text. An eval case that is
also a DB job (job id, url + title, or company + title) is counted once. Too few labels, a failed check, or missing ML
packages: a log line, and the run continues exactly as before.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

FEEDBACK_SOURCES = ("cli", "eval")


def load_eval_set() -> tuple[list[dict], dict[str, dict]]:
    """The eval cases and labels from EVAL_JOBS_JSONL / EVAL_LABELS_CSV (plain files), or empty.
    A small reader of its own: the eval package is not in the Docker image."""
    jobs_path, labels_path = os.environ.get("EVAL_JOBS_JSONL"), os.environ.get("EVAL_LABELS_CSV")
    if not (jobs_path and labels_path and Path(jobs_path).is_file() and Path(labels_path).is_file()):
        return [], {}
    fields = ("case_id", "company", "title", "location", "department", "url", "description")
    cases = []
    for line in Path(jobs_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("case_id") and row.get("title"):
                cases.append({k: row.get(k) or "" for k in fields})
    with open(labels_path, encoding="utf-8", newline="") as f:
        labels = {r["case_id"].strip(): {"label": (r.get("label") or "").strip()}
                  for r in csv.DictReader(f) if (r.get("case_id") or "").strip()}
    return cases, labels


def shadow_score(store, cv_text: str, profile: dict, min_positives: int | None = None) -> dict | None:
    """Retrain and score. Returns counts, or None when it fell back (never raises)."""
    try:
        from jobagent.ml import train as T
        from jobagent.ml import validate as V
    except ImportError as e:  # the image or venv has no ML packages
        log.info("Relevance model skipped: %s not installed", e.name)
        return None
    try:
        v = V.Validation()
        jobs = T.load_jobs(store)
        feedback = store.conn.execute("SELECT job_id, decision, source FROM job_feedback").df()
        cases, labels = load_eval_set()
        rows = T.labeled_rows(jobs, feedback, profile, cases, labels, feedback_sources=FEEDBACK_SOURCES)
        V.check_unique_rows(rows, v)
        X, y = T.frame(rows, cv_text, profile), rows["y"].to_numpy()
        V.check_features(X, v)
        n = min_positives if min_positives is not None else V.MIN_POSITIVES
        V.check_labels(y, v, min_positives=n, min_negatives=n)
        model = T._fit(T.make_logreg(), X, y)
        version = f"logreg-{T.dataset_hash(rows)[:10]}"

        targets = pd.DataFrame(store.unscored_active())
        if targets.empty:
            return {"n_labels": len(rows), "n_positives": int(y.sum()), "n_eval_cases": len(cases), "n_scored": 0,
                    "version": version}
        p = T.predict(model, targets, cv_text, profile, v)
        store.save_ml_scores(dict(zip(targets["job_id"], p)), version)
    except V.ValidationError as e:
        log.warning("Relevance model not trained this run (%s); LLM queue order unchanged", e)
        return None
    except Exception as e:  # noqa: BLE001 - shadow mode must never break the daily run
        log.warning("Relevance model failed (%s: %s); continuing without it", type(e).__name__, e)
        return None
    out = {"n_labels": len(rows), "n_positives": int(y.sum()), "n_eval_cases": len(cases),
           "n_scored": len(targets), "version": version}
    log.info("Relevance model %s: trained on %d labels (%d positive, %d eval cases), scored %d jobs",
             version, out["n_labels"], out["n_positives"], out["n_eval_cases"], out["n_scored"])
    return out
