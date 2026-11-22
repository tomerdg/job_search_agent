"""Data and model integrity checks for the relevance model. They run inside training and prediction
and fail loudly (ValidationError) instead of letting a bad matrix or a broken model through.

    check_features(X)                    columns, dtypes, no NaN/inf, value ranges
    check_labels(y, min_positives, ...)  enough of each class to train
    check_unique_rows(rows)              one row per labelled job
    check_split(train, test)             per CV split: no company and no job (id, url, company+title) on both sides
    check_no_leakage(train, test)        no job on both sides (by id, url, company+title)
    check_predictions(p, ids)            one probability per job, in [0, 1], unique ids, not all the same
    class_drift(pos_rate, previous)      a warning, not an error: the positive rate moved a lot since last run

Every check records pass/fail in a Validation object, which training logs to MLflow as tags.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jobagent.ml.features import CATEGORICAL, NUMERIC, ROLE_RULES, SENIORITY_RULES

log = logging.getLogger(__name__)

MIN_POSITIVES = 10   # refuse to train on fewer positive examples than this
MIN_NEGATIVES = 10
DRIFT_TOLERANCE = 0.15  # absolute change in the training positive rate that triggers a warning

BINARY = ["role_keyword", "location_ok", "years_missing"]
UNIT_INTERVAL = ["hebrew_share", "skill_coverage"]
COUNTS = ["n_jd_skills", "n_overlap", "n_missing"]
ALLOWED = {
    "role_family": {name for name, _ in ROLE_RULES} | {"other"},
    "seniority": {name for name, _ in SENIORITY_RULES} | {"mid"},
}


class ValidationError(ValueError):
    pass


@dataclass
class Validation:
    """pass/fail per check, and warnings. `require` raises on the first failure."""
    results: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def require(self, name: str, ok: bool, message: str) -> None:
        self.results[name] = bool(ok)
        if not ok:
            raise ValidationError(f"{name}: {message}")

    def warn(self, name: str, message: str) -> None:
        self.warnings.append(f"{name}: {message}")
        log.warning("%s: %s", name, message)

    def tags(self) -> dict[str, str]:
        return {f"validation.{k}": "pass" if v else "fail" for k, v in self.results.items()}


def check_features(X: pd.DataFrame, v: Validation | None = None, prefix: str = "features") -> Validation:
    v = v or Validation()
    expected = CATEGORICAL + NUMERIC + ["text"]
    missing = [c for c in expected if c not in X.columns]
    v.require(f"{prefix}.columns", not missing, f"missing columns {missing}")
    v.require(f"{prefix}.not_empty", len(X) > 0, "no rows")
    bad_types = [c for c in NUMERIC if not pd.api.types.is_numeric_dtype(X[c])]
    bad_types += [c for c in CATEGORICAL + ["text"] if not (pd.api.types.is_object_dtype(X[c])
                                                            or pd.api.types.is_string_dtype(X[c]))]
    v.require(f"{prefix}.dtypes", not bad_types, f"wrong dtype in {bad_types}")

    dense = X[NUMERIC].to_numpy(dtype=float)
    finite = np.isfinite(dense)
    v.require(f"{prefix}.finite", finite.all(),
              f"NaN/inf in {[c for c, ok in zip(NUMERIC, finite.all(axis=0)) if not ok]}")
    nulls = [c for c in CATEGORICAL + ["text"] if X[c].isna().any()]
    v.require(f"{prefix}.no_nulls", not nulls, f"nulls in {nulls}")

    problems = []
    if not X[BINARY].isin([0, 1]).all().all():
        problems.append(f"{BINARY} must be 0/1")
    if not X[UNIT_INTERVAL].apply(lambda s: s.between(0, 1)).all().all():
        problems.append(f"{UNIT_INTERVAL} must be in [0, 1]")
    if (X[COUNTS + ["log_jd_len"]] < 0).any().any():
        problems.append("counts and lengths must be >= 0")
    if not X["years_required"].between(0, 15).all():
        problems.append("years_required must be in [0, 15]")
    if (X["n_overlap"] > X["n_jd_skills"]).any() or not np.allclose(X["n_overlap"] + X["n_missing"], X["n_jd_skills"]):
        problems.append("n_overlap + n_missing must equal n_jd_skills")
    for col, allowed in ALLOWED.items():
        unknown = set(X[col].unique()) - allowed
        if unknown:
            problems.append(f"unknown {col} values {sorted(unknown)}")
    v.require(f"{prefix}.ranges", not problems, "; ".join(problems))
    return v


def check_labels(y, v: Validation | None = None, min_positives: int = MIN_POSITIVES,
                 min_negatives: int = MIN_NEGATIVES) -> Validation:
    v = v or Validation()
    y = np.asarray(y)
    v.require("labels.binary", np.isin(y, [0, 1]).all(), "labels must be 0/1")
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    v.require("labels.min_positives", pos >= min_positives,
              f"{pos} positive examples, need at least {min_positives} (score more jobs or record --applied)")
    v.require("labels.min_negatives", neg >= min_negatives, f"{neg} negative examples, need {min_negatives}")
    return v


def _single_job_urls(df: pd.DataFrame) -> set:
    """Urls that name exactly one row. Some sites give every job the same careers-page url, so a shared
    url is not evidence of the same job."""
    counts = df["url"].dropna().value_counts()
    return set(counts.index[counts == 1])


def check_no_leakage(train: pd.DataFrame, gold: pd.DataFrame, v: Validation | None = None) -> Validation:
    """No job on both sides: same job_id, same single-job url, or same normalised company + title."""
    v = v or Validation()
    if gold.empty or train.empty:
        v.require("leakage.gold_not_in_train", True, "")
        return v

    def keys(df: pd.DataFrame) -> pd.Series:
        norm = lambda s: s.fillna("").astype(str).str.lower().str.split().str.join(" ")
        return norm(df["company"]) + "|" + norm(df["title"])

    hits = {
        "job_id": np.intersect1d(train["job_id"].dropna(), gold["job_id"].dropna()).size,
        "url": len(_single_job_urls(train) & _single_job_urls(gold)),
        "company_title": np.intersect1d(keys(train), keys(gold)).size,
    }
    v.require("leakage.gold_not_in_train", not any(hits.values()),
              f"the same jobs on both sides: {hits}")
    return v


def check_unique_rows(rows: pd.DataFrame, v: Validation | None = None) -> Validation:
    v = v or Validation()
    dupes = int(rows["gid"].duplicated().sum())
    detail = ""
    if dupes and "tier" in rows:  # counts only (no ids or text), so the daily log says which tiers collide
        tiers = rows.loc[rows["gid"].duplicated(keep=False), "tier"].value_counts().sort_index()
        detail = " (" + ", ".join(f"{t}={n}" for t, n in tiers.items()) + ")"
    v.require("labels.unique_rows", dupes == 0, f"{dupes} jobs appear more than once in the labels{detail}")
    return v


def check_split(train: pd.DataFrame, test: pd.DataFrame, v: Validation | None = None) -> Validation:
    """One CV split: the two parts share no company (GroupKFold by company) and no job."""
    v = v or Validation()
    shared = np.intersect1d(train["group"].unique(), test["group"].unique()).size
    v.require("leakage.company_not_split", shared == 0, f"{shared} companies on both sides of a split")
    return check_no_leakage(train, test, v)


def check_predictions(p, ids, v: Validation | None = None, name: str = "predictions") -> Validation:
    v = v or Validation()
    p = np.asarray(p, dtype=float)
    ids = pd.Series(ids)
    v.require(f"{name}.one_per_job", p.ndim == 1 and len(p) == len(ids), f"{p.shape} scores for {len(ids)} jobs")
    v.require(f"{name}.unique_ids", not ids.duplicated().any(), f"{int(ids.duplicated().sum())} duplicate ids")
    v.require(f"{name}.finite", np.isfinite(p).all(), "NaN or inf in scores")
    v.require(f"{name}.in_unit_interval", ((p >= 0) & (p <= 1)).all(), "probabilities outside [0, 1]")
    v.require(f"{name}.not_constant", len(p) < 2 or np.ptp(p) > 1e-9, "every job got the same score")
    return v


def class_drift(pos_rate: float, previous: dict | None, v: Validation | None = None,
                tolerance: float = DRIFT_TOLERANCE) -> Validation:
    """Warn (do not fail) when the training positive rate moved more than `tolerance` since the last run."""
    v = v or Validation()
    prev = (previous or {}).get("train_pos_rate")
    drifted = prev is not None and abs(pos_rate - prev) > tolerance
    v.results["labels.class_drift"] = not drifted
    if drifted:
        v.warn("labels.class_drift", f"positive rate {pos_rate:.3f}, was {prev:.3f} in the previous run")
    return v
