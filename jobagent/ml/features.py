"""Features for the relevance model, built as pandas DataFrames with numpy.

    load_jobs(store_or_path)               jobs table -> DataFrame (DuckDB -> pandas)
    build_features(df, cv_text, profile)   -> (dense feature DataFrame, text Series for TF-IDF)

Dense features (one row per job):
    role_family        same rules as analytics/models/intermediate/int_job_roles.sql
    seniority          junior / mid / senior / lead / head / manager, from the title
    source_type        which ATS the job came from
    role_keyword       the pre-filter's keyword match (title + description)
    location_ok        the pre-filter's location rule
    years_required     largest "N+ years" / "N שנות ניסיון" in the description (0 when none), capped at 15
    years_missing      1 when the description states no years
    log_jd_len         log(1 + characters in the description)
    hebrew_share       share of letters that are Hebrew
    n_jd_skills        skills from SKILLS named in the description
    n_overlap          ... that the CV also names
    n_missing          ... that the CV does not name
    skill_coverage     n_overlap / n_jd_skills (0 when the description names none)

The CV is read at runtime (CV_TEXT or the profile's cv_path). Nothing from it is stored here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from jobagent.filters import PreFilter

JOB_COLUMNS = ["job_id", "source", "company", "title", "location", "department", "url", "description",
               "first_seen", "is_active", "prefilter", "score"]

# Mirrors int_job_roles.sql: first match wins, so the order matters.
ROLE_RULES: list[tuple[str, str]] = [
    ("analytics_engineering", r"analytics engineer"),
    ("data_engineering", r"data engineer|data platform|etl|מהנדס/?ת? נתונים|מהנדס/?ת? דאטה"),
    ("bi", r"(?:^|[^a-z])bi(?:[^a-z]|$)|business intelligence|power bi|tableau"),
    ("data_science", r"data scien|machine learning|(?:^|[^a-z])ml(?:[^a-z]|$)"),
    ("analytics", r"data analyst|analyst|אנליסט|אנליסטית"),
]

SENIORITY_RULES: list[tuple[str, str]] = [
    ("head", r"\bhead\b|\bdirector\b|\bvp\b|ראש צוות|ראש תחום|מנהל/?ת? תחום"),
    ("manager", r"\bmanager\b|מנהל"),
    ("lead", r"\blead\b|\bprincipal\b|\bstaff\b|\barchitect\b|ארכיטקט|מוביל"),
    ("senior", r"\bsenior\b|\bsr\.?\b|בכיר"),
    ("junior", r"\bjunior\b|\bjr\.?\b|\bentry\b|\bintern\b|\bstudent\b|ג'וניור|סטודנט|ללא ניסיון"),
]

# A generic, public list of data-engineering skills: name -> regex (case-insensitive).
SKILLS: dict[str, str] = {
    "sql": r"\bsql\b", "python": r"\bpython\b", "dbt": r"\bdbt\b", "airflow": r"\bairflow\b",
    "spark": r"\bspark\b|pyspark", "kafka": r"\bkafka\b", "snowflake": r"\bsnowflake\b",
    "bigquery": r"\bbig ?query\b", "databricks": r"\bdatabricks\b", "redshift": r"\bredshift\b",
    "power_bi": r"\bpower ?bi\b", "tableau": r"\btableau\b", "looker": r"\blooker\b", "qlik": r"\bqlik",
    "etl": r"\betl\b|\belt\b", "aws": r"\baws\b|amazon web services", "azure": r"\bazure\b",
    "gcp": r"\bgcp\b|google cloud", "docker": r"\bdocker\b", "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "pandas": r"\bpandas\b", "numpy": r"\bnumpy\b", "scikit_learn": r"scikit|sklearn",
    "git": r"\bgit\b|github|gitlab", "ci_cd": r"ci/cd|\bci\b|github actions|jenkins",
    "linux": r"\blinux\b", "postgres": r"postgres", "mongodb": r"mongo", "duckdb": r"duckdb",
    "data_modeling": r"data model|dimensional model|star schema|\bdwh\b|data warehouse",
    "ssis": r"\bssis\b", "informatica": r"informatica", "excel": r"\bexcel\b",
    "terraform": r"terraform", "scala": r"\bscala\b", "java": r"\bjava\b",
}

# "3+ years", "3-5 years", "at least 3 years"; Hebrew "3 שנות ניסיון", "לפחות 3 שנים", "3+ שנים".
YEARS_EN = r"(\d{1,2})\s*\+?\s*(?:-|–|to)?\s*(?:\d{1,2}\s*)?\+?\s*years?"
YEARS_HE = r"(\d{1,2})\s*\+?\s*(?:-|–)?\s*(?:\d{1,2}\s*)?\+?\s*(?:שנות|שנים|שנה)"
HEBREW = r"[֐-׿]"
LETTER = r"[A-Za-z֐-׿]"

CATEGORICAL = ["role_family", "seniority", "source_type"]
NUMERIC = ["role_keyword", "location_ok", "years_required", "years_missing", "log_jd_len", "hebrew_share",
           "n_jd_skills", "n_overlap", "n_missing", "skill_coverage"]


def load_jobs(store_or_path) -> pd.DataFrame:
    """The jobs table as a DataFrame. Accepts a Store, a duckdb connection or a path to the DB file."""
    import duckdb
    if isinstance(store_or_path, (str, Path)):
        con = duckdb.connect(str(store_or_path), read_only=True)
        try:
            return con.execute(f"SELECT {', '.join(JOB_COLUMNS)} FROM jobs").df()
        finally:
            con.close()
    con = getattr(store_or_path, "conn", store_or_path)
    return con.execute(f"SELECT {', '.join(JOB_COLUMNS)} FROM jobs").df()


def _first_match(text: pd.Series, rules: list[tuple[str, str]], default: str) -> pd.Series:
    """Vectorised 'first rule that matches': np.select picks the first true condition per row."""
    conds = [text.str.contains(pat, regex=True, flags=re.I) for _, pat in rules]
    return pd.Series(np.select(conds, [name for name, _ in rules], default=default), index=text.index)


def role_family(title: pd.Series) -> pd.Series:
    return _first_match(title.fillna("").str.lower(), ROLE_RULES, "other")


def seniority(title: pd.Series) -> pd.Series:
    return _first_match(title.fillna(""), SENIORITY_RULES, "mid")


def years_required(text: pd.Series) -> pd.Series:
    """Largest number of years the text asks for (English or Hebrew), NaN when it names none."""
    text = text.fillna("")
    found = pd.concat([text.str.extractall(YEARS_EN, flags=re.I)[0],
                       text.str.extractall(YEARS_HE)[0]]).astype(float)
    found = found[found.between(1, 15)]
    return found.groupby(level=0).max().reindex(text.index)


def skill_matrix(text: pd.Series) -> pd.DataFrame:
    """Boolean DataFrame: one column per skill in SKILLS, True where the text names it."""
    text = text.fillna("")
    return pd.DataFrame({name: text.str.contains(pat, regex=True, flags=re.I) for name, pat in SKILLS.items()},
                        index=text.index)


def cv_skills(cv_text: str) -> np.ndarray:
    """Boolean vector over SKILLS: which ones the CV names."""
    return skill_matrix(pd.Series([cv_text or ""])).to_numpy()[0]


def hebrew_share(text: pd.Series) -> pd.Series:
    text = text.fillna("")
    letters = text.str.count(LETTER).to_numpy(dtype=float)
    hebrew = text.str.count(HEBREW).to_numpy(dtype=float)
    return pd.Series(np.divide(hebrew, letters, out=np.zeros_like(letters), where=letters > 0), index=text.index)


def build_features(df: pd.DataFrame, cv_text: str, profile: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Dense features (CATEGORICAL + NUMERIC columns) and the text for TF-IDF, both indexed like df."""
    title = df["title"].fillna("").astype(str)
    desc = df["description"].fillna("").astype(str)
    jd = title + "\n" + desc
    pf = PreFilter(profile or {})

    jd_skills = skill_matrix(jd).to_numpy()                   # (n_jobs, n_skills) bool
    cv = cv_skills(cv_text)                                   # (n_skills,) bool
    n_jd = jd_skills.sum(axis=1)
    n_overlap = (jd_skills & cv).sum(axis=1)
    years = years_required(desc)

    X = pd.DataFrame({
        "role_family": role_family(title),
        "seniority": seniority(title),
        "source_type": df.get("source", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str),
        "role_keyword": [int(pf.matches_role(t, d)) for t, d in zip(title, desc)],
        "location_ok": [int(pf.location_ok(loc)) for loc in df["location"].fillna("").astype(str)],
        "years_required": years.fillna(0).to_numpy(),
        "years_missing": years.isna().astype(int).to_numpy(),
        "log_jd_len": np.log1p(desc.str.len().to_numpy()),
        "hebrew_share": hebrew_share(jd).to_numpy(),
        "n_jd_skills": n_jd,
        "n_overlap": n_overlap,
        "n_missing": n_jd - n_overlap,
        "skill_coverage": np.divide(n_overlap, n_jd, out=np.zeros(len(df)), where=n_jd > 0),
    }, index=df.index)
    return X[CATEGORICAL + NUMERIC], jd.rename("text")
