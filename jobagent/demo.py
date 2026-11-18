"""Build a demo DuckDB with made-up companies and jobs, so the dbt project can run without real data.

    python -m jobagent.demo --db data/demo.duckdb --days 120

Used by CI to build and test the dbt models, and for showing the project without personal data.
Everything here is synthetic: company names, titles and scores are invented.
"""
from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

from jobagent.db import Store
from jobagent.models import Job

COMPANIES = [  # (name, sector, source type, city, jobs opened per run on average)
    ("Northwind Retail", "retail", "adam", "רמת גן", 1.2),
    ("Blue Harbor Payments", "fintech", "topmatch", "תל אביב", 1.5),
    ("Cedar Logistics", "logistics", "topmatch", "תל אביב", 0.6),
    ("Helix Health", "healthtech", "comeet", "פתח תקווה", 0.8),
    ("Ledgerline Software", "saas", "comeet", "בני ברק", 0.4),
    ("Quanta Ads", "adtech", "greenhouse", "Tel Aviv", 2.0),
    ("Silent Source Ltd", "retail", "adam", "הרצליה", 0.5),  # stops returning jobs: source-health demo
]
# Synthetic CV and profile for the demo and the ML smoke run. Not anyone's real data.
DEMO_CV = "Data engineer. SQL, Python, pandas, numpy, dbt, Airflow, DuckDB, Docker, GitHub Actions, AWS, ETL."
DEMO_PROFILE = {
    "keywords": ["data engineer", "analytics engineer", "BI", "etl", "מהנדס/ת נתונים", "מפתח/ת BI"],
    "locations_allowed": ["תל אביב", "tel aviv", "רמת גן", "פתח תקווה", "בני ברק"],
    "locations_blocked": ["הרצליה"],
}
SKILL_POOL = {  # skills a synthetic description draws from, by kind of role
    "data": ["SQL", "Python", "dbt", "Airflow", "Spark", "Kafka", "Snowflake", "BigQuery", "pandas", "Docker",
             "AWS", "ETL", "Power BI", "Tableau", "Databricks"],
    "other": ["Java", "React", "Excel", "Salesforce", "Figma", "Kubernetes", "Selenium", "CRM"],
}
TITLES = [
    "Data Engineer", "Senior Data Engineer", "Analytics Engineer", "BI Developer", "מפתח/ת BI",
    "מהנדס/ת נתונים", "Data Analyst", "Data Scientist", "Backend Developer", "Product Manager",
    "QA Engineer", "נציג/ת שירות לקוחות", "Marketing Analyst", "DevOps Engineer",
]


def describe(rng: random.Random, title: str, counter: int) -> str:
    """A synthetic job description: skills by role, years in English or Hebrew."""
    data_role = any(k in title for k in ("Data", "BI", "Analytics", "נתונים"))
    pool = SKILL_POOL["data"] if data_role else SKILL_POOL["other"]
    skills = rng.sample(pool, k=rng.randint(2, 6)) + rng.sample(SKILL_POOL["data"] + SKILL_POOL["other"], k=1)
    years = rng.randint(1, 7)
    hebrew = any("\u0590" <= ch <= "\u05FF" for ch in title)
    req = f"לפחות {years} שנות ניסיון" if hebrew else f"{years}+ years of experience"
    return f"Synthetic job {counter}: {title}. {req}. Tools: {', '.join(skills)}."


def demo_score(rng: random.Random, title: str, description: str) -> int:
    """LLM-like score: role fit plus overlap with DEMO_CV's skills, plus noise."""
    data_role = any(k in title for k in ("Data Engineer", "BI", "Analytics", "נתונים", "מפתח/ת BI"))
    overlap = sum(s.lower() in DEMO_CV.lower() for s in SKILL_POOL["data"] if s in description)
    return max(0, min(100, int(25 + 30 * data_role + 8 * overlap + rng.gauss(0, 10))))


def build(db: str | Path, days: int = 120, seed: int = 7) -> None:
    rng = random.Random(seed)
    ml_rng = random.Random(seed + 1)  # separate stream, so the rest of the demo data does not change
    Path(db).unlink(missing_ok=True)
    store = Store(db)
    end = datetime(2026, 11, 1, 8, 0)
    start = end - timedelta(days=days)
    open_jobs: dict[str, tuple[Job, datetime]] = {}  # job_id -> (job, planned close)
    counter = 0
    day = start
    while day <= end:
        if day.weekday() in (6, 1, 3):  # Sun, Tue, Thu like the real schedule
            run_id = store.start_run(ts=day)
            stats, live = [], []
            for name, sector, stype, city, rate in COMPANIES:
                broken = name == "Silent Source Ltd" and day > end - timedelta(days=10)
                if broken:
                    stats.append({"company": name, "source_type": stype, "ok": True, "n_jobs": 0})
                    continue
                for _ in range(rng.choices([0, 1, 2, 3], weights=[1, rate, rate / 2, rate / 4])[0]):
                    counter += 1
                    title = rng.choice(TITLES)
                    j = Job(source=stype, company=name, external_id=str(counter), title=title,
                            url=f"https://example.com/jobs/{counter}", location=city,
                            description=describe(rng, title, counter), sector=sector)
                    open_jobs[j.job_id] = (j, day + timedelta(days=rng.randint(5, 75)))
                company_jobs = [j for j, close in open_jobs.values() if j.company == name and close > day]
                stats.append({"company": name, "source_type": stype, "ok": True, "n_jobs": len(company_jobs),
                              "duration_s": round(rng.uniform(0.3, 4.0), 2)})
                live.extend(company_jobs)
            new_ids = store.upsert(live, ts=day)
            store.mark_inactive({j.company for j in live}, {j.job_id for j in live}, ts=day)
            store.record_observations(run_id, [j.job_id for j in live], ts=day)
            store.record_sources(run_id, stats)
            # Synthetic shadow-mode scores (what jobagent/ml/shadow.py writes): roughly follow the LLM-like score.
            store.save_ml_scores({jid: 1 / (1 + math.exp(-(demo_score(ml_rng, open_jobs[jid][0].title,
                                                                    open_jobs[jid][0].description) - 55) / 10))
                                  for jid in new_ids}, "demo", ts=day)
            for jid in new_ids:
                job = open_jobs[jid][0]
                data_role = any(k in job.title for k in ("Data", "BI", "Analytics", "נתונים"))
                store.set_prefilter(jid, data_role)
                if data_role:
                    score = demo_score(rng, job.title, job.description)
                    store.save_score(jid, score, {"summary": "demo", "strengths": [], "gaps": []})
                # Synthetic owner decisions on a third of all jobs, following the score with noise.
                if rng.random() < 0.33:
                    score = demo_score(rng, job.title, job.description)
                    decision = ("applied" if score >= 65 else "skipped" if score >= 45 else "irrelevant") \
                        if rng.random() > 0.1 else rng.choice(["applied", "skipped", "irrelevant"])
                    store.record_feedback([jid], decision, ts=day + timedelta(days=rng.randint(0, 3)),
                                          source=rng.choice(["cli", "cli", "eval"]))
            store.finish_run(run_id, len(live), len(new_ids), ts=day)
            open_jobs = {k: v for k, v in open_jobs.items() if v[1] > day}
        day += timedelta(days=1)
    store.close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="data/demo.duckdb")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    build(args.db, args.days, args.seed)
    print(f"Demo DB written to {args.db}")


if __name__ == "__main__":
    main()
