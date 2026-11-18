"""Relevance-model features: regexes (English and Hebrew), role/seniority rules, skill overlap. Synthetic data."""
import numpy as np
import pandas as pd
import pytest

from jobagent.ml.features import (CATEGORICAL, NUMERIC, build_features, hebrew_share, load_jobs, role_family,
                                  seniority, years_required)

PROFILE = {"keywords": ["data engineer", "BI"], "locations_allowed": ["tel aviv"], "locations_blocked": ["haifa"]}


@pytest.mark.parametrize("text, years", [
    ("Requires 3+ years of SQL", 3),
    ("2-5 years of experience", 2),
    ("at least 4 years in data", 4),
    ("לפחות 4 שנות ניסיון ב-Python", 4),
    ("ניסיון של 5 שנים לפחות", 5),
    ("3+ שנים בפיתוח", 3),
    ("1 year with Airflow, 6 years overall", 6),       # the largest requirement wins
    ("founded in 2026, 300 years of tradition", None),  # out of range: ignored
    ("no number here", None),
    (None, None),
])
def test_years_required(text, years):
    got = years_required(pd.Series([text])).iloc[0]
    assert (np.isnan(got) if years is None else got == years)


def test_role_family_mirrors_dbt_rules():
    titles = ["Analytics Engineer", "Senior Data Engineer", "מהנדס/ת נתונים", "BI Developer", "מפתח/ת BI",
              "Data Scientist", "Data Analyst", "Biology lab tech", "Product Manager"]
    assert role_family(pd.Series(titles)).tolist() == [
        "analytics_engineering", "data_engineering", "data_engineering", "bi", "bi",
        "data_science", "analytics", "other", "other"]


def test_seniority():
    assert seniority(pd.Series(["Senior Data Engineer", "Team Lead", "Junior BI", "Head of Data",
                                "מנהל/ת מוצר", "Data Engineer"])).tolist() == [
        "senior", "lead", "junior", "head", "manager", "mid"]


def test_hebrew_share():
    s = hebrew_share(pd.Series(["abcd", "אבגד", "ab אב", "123", ""]))
    assert s.tolist() == [0.0, 1.0, 0.5, 0.0, 0.0]


def test_build_features_skill_overlap_uses_cv_at_runtime():
    df = pd.DataFrame({"title": ["Data Engineer", "Sales Rep"], "location": ["Tel Aviv", "Haifa"],
                       "source": ["greenhouse", "comeet"],
                       "description": ["SQL, Python, Spark and Kafka. 3+ years.", "CRM work"]})
    X, text = build_features(df, cv_text="I write SQL and Python daily", profile=PROFILE)
    assert list(X.columns) == CATEGORICAL + NUMERIC
    a, b = X.iloc[0], X.iloc[1]
    assert (a.n_jd_skills, a.n_overlap, a.n_missing, a.skill_coverage) == (4, 2, 2, 0.5)
    assert (b.n_jd_skills, b.skill_coverage) == (0, 0.0)
    assert (a.role_keyword, a.location_ok, b.role_keyword, b.location_ok) == (1, 1, 0, 0)
    assert a.years_required == 3 and a.years_missing == 0 and b.years_missing == 1
    assert text.iloc[0].startswith("Data Engineer\n")
    # Same jobs, a CV with no overlap: overlap features change, nothing else does.
    X2, _ = build_features(df, cv_text="", profile=PROFILE)
    assert X2["n_overlap"].sum() == 0
    pd.testing.assert_frame_equal(X.drop(columns=["n_overlap", "n_missing", "skill_coverage"]),
                                  X2.drop(columns=["n_overlap", "n_missing", "skill_coverage"]))


def test_load_jobs_from_store(tmp_path):
    from jobagent.db import Store
    from jobagent.models import Job
    s = Store(tmp_path / "j.duckdb")
    s.upsert([Job(source="adam", company="Acme", external_id="1", title="BI Developer", url="u", location="x")])
    df = load_jobs(s)
    assert isinstance(df, pd.DataFrame) and df.loc[0, "title"] == "BI Developer"
    s.close()
    assert len(load_jobs(tmp_path / "j.duckdb")) == 1
