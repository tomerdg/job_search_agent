"""Relevance model in shadow mode, inside the daily run.

Each run: retrain the logistic-regression model from the labels in the DB (no model file is kept), then
score every active job the LLM has not scored yet and store P(fit or maybe) in jobs.ml_score. Nothing is
filtered: the score only orders the LLM queue (Store.pending_scoring), so the best guesses are scored first.
dbt's mart_ml_vs_llm then compares the model with the LLM on the jobs both scored.

Labels come from the DB only: LLM scores (silver), job_feedback (your decisions and the imported eval
labels), and jobs with no role keyword (weak negatives). Too few labels, a failed check, or missing ML
packages: a log line, and the run continues exactly as before.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

FEEDBACK_SOURCES = ("cli", "eval")


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
        rows = T.labeled_rows(jobs, feedback, profile, feedback_sources=FEEDBACK_SOURCES)
        V.check_unique_rows(rows, v)
        X, y = T.frame(rows, cv_text, profile), rows["y"].to_numpy()
        V.check_features(X, v)
        n = min_positives if min_positives is not None else V.MIN_POSITIVES
        V.check_labels(y, v, min_positives=n, min_negatives=n)
        model = T._fit(T.make_logreg(), X, y)
        version = f"logreg-{T.dataset_hash(rows)[:10]}"

        targets = pd.DataFrame(store.unscored_active())
        if targets.empty:
            return {"n_labels": len(rows), "n_positives": int(y.sum()), "n_scored": 0, "version": version}
        p = T.predict(model, targets, cv_text, profile, v)
        store.save_ml_scores(dict(zip(targets["job_id"], p)), version)
    except V.ValidationError as e:
        log.warning("Relevance model not trained this run (%s); LLM queue order unchanged", e)
        return None
    except Exception as e:  # noqa: BLE001 - shadow mode must never break the daily run
        log.warning("Relevance model failed (%s: %s); continuing without it", type(e).__name__, e)
        return None
    out = {"n_labels": len(rows), "n_positives": int(y.sum()), "n_scored": len(targets), "version": version}
    log.info("Relevance model %s: trained on %d labels (%d positive), scored %d jobs",
             version, out["n_labels"], out["n_positives"], out["n_scored"])
    return out
