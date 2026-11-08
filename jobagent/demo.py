"""Build a demo DuckDB with made-up companies and jobs, so the dbt project can run without real data.

    python -m jobagent.demo --db data/demo.duckdb --days 120

Used by CI to build and test the dbt models, and for showing the project without personal data.
Everything here is synthetic: company names, titles and scores are invented.
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

from jobagent.db import Store
from jobagent.models import Job

COMPANIES = [  # (name, sector, source type, city, jobs opened per run on average)
    ("Northwind Insurance", "insurance", "adam", "רמת גן", 1.2),
    ("Blue Harbor Bank", "banking", "topmatch", "תל אביב", 1.5),
    ("Cedar Investments", "investment", "topmatch", "תל אביב", 0.6),
    ("Helix Pharma", "pharma", "comeet", "פתח תקווה", 0.8),
    ("Ledgerline CPA", "accounting", "comeet", "בני ברק", 0.4),
    ("Quanta Ads", "tech", "greenhouse", "Tel Aviv", 2.0),
    ("Silent Source Ltd", "insurance", "adam", "הרצליה", 0.5),  # stops returning jobs: source-health demo
]
TITLES = [
    "Data Engineer", "Senior Data Engineer", "Analytics Engineer", "BI Developer", "מפתח/ת BI",
    "מהנדס/ת נתונים", "Data Analyst", "Data Scientist", "Backend Developer", "Product Manager",
    "QA Engineer", "נציג/ת שירות לקוחות", "Actuary", "DevOps Engineer",
]


def build(db: str | Path, days: int = 120, seed: int = 7) -> None:
    rng = random.Random(seed)
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
                            description=f"Synthetic job {counter}: {title}. SQL, Python, Airflow.", sector=sector)
                    open_jobs[j.job_id] = (j, day + timedelta(days=rng.randint(5, 75)))
                company_jobs = [j for j, close in open_jobs.values() if j.company == name and close > day]
                stats.append({"company": name, "source_type": stype, "ok": True, "n_jobs": len(company_jobs),
                              "duration_s": round(rng.uniform(0.3, 4.0), 2)})
                live.extend(company_jobs)
            new_ids = store.upsert(live, ts=day)
            store.mark_inactive({j.company for j in live}, {j.job_id for j in live}, ts=day)
            store.record_observations(run_id, [j.job_id for j in live], ts=day)
            store.record_sources(run_id, stats)
            for jid in new_ids:
                job = open_jobs[jid][0]
                data_role = any(k in job.title for k in ("Data", "BI", "Analytics", "נתונים"))
                store.set_prefilter(jid, data_role)
                if data_role:
                    store.save_score(jid, rng.randint(40, 95), {"summary": "demo", "strengths": [], "gaps": []})
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
