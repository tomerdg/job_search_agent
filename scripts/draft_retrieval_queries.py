"""Draft eval/retrieval/queries.jsonl: generic queries with keyword-rule ground truth, for review.

    python scripts/draft_retrieval_queries.py                 # counts on the eval set, writes the draft
    python scripts/draft_retrieval_queries.py --db data/jobs.duckdb --dry-run

Each query's relevant jobs are defined by a rule (see eval/retrieval/__init__.py), so the file holds no job
ids and nothing personal. The script prints how many jobs each rule matches in the corpus and leaves out
rules that match none. Review the printed counts and the rules, then commit the file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (qid, kind, query, rule). Kinds: skill (exact tool names), hebrew (Hebrew phrasing of a skill or role),
# role (role type from the title), paraphrase (no keyword shared with the rule, where vectors must help).
DRAFT = [
    ("skill-dbt", "skill", "dbt", {"skills": ["dbt"]}),
    ("skill-airflow", "skill", "Airflow orchestration", {"skills": ["airflow"]}),
    ("skill-kafka", "skill", "Kafka streaming", {"skills": ["kafka"]}),
    ("skill-snowflake", "skill", "Snowflake", {"skills": ["snowflake"]}),
    ("skill-bigquery", "skill", "BigQuery", {"skills": ["bigquery"]}),
    ("skill-databricks", "skill", "Databricks", {"skills": ["databricks"]}),
    ("skill-redshift", "skill", "Redshift", {"skills": ["redshift"]}),
    ("skill-tableau", "skill", "Tableau dashboards", {"skills": ["tableau"]}),
    ("skill-looker", "skill", "Looker", {"skills": ["looker"]}),
    ("skill-kubernetes", "skill", "Kubernetes", {"skills": ["kubernetes"]}),
    ("skill-terraform", "skill", "Terraform", {"skills": ["terraform"]}),
    ("skill-scala", "skill", "Scala", {"skills": ["scala"]}),
    ("skill-docker", "skill", "Docker containers", {"skills": ["docker"]}),
    ("skill-clickhouse", "skill", "ClickHouse", {"jd": r"clickhouse"}),
    ("skill-dbt-snowflake", "skill", "dbt and Snowflake", {"skills": ["dbt", "snowflake"]}),
    ("skill-spark-kafka", "skill", "Spark and Kafka", {"skills": ["spark", "kafka"]}),
    ("skill-airflow-python", "skill", "Airflow with Python", {"skills": ["airflow", "python"]}),
    ("skill-modeling", "skill", "data modeling and data warehouse design", {"skills": ["data_modeling"]}),
    ("skill-cicd", "skill", "CI/CD", {"skills": ["ci_cd"]}),
    ("he-spark", "hebrew", "מהנדס נתונים עם ניסיון ב-Spark", {"skills": ["spark"]}),
    ("he-sql-python", "hebrew", "ניסיון ב-SQL ובפייתון", {"skills": ["sql", "python"]}),
    ("he-etl", "hebrew", "פיתוח תהליכי ETL", {"skills": ["etl"]}),
    ("he-powerbi", "hebrew", "ניסיון בפיתוח דוחות Power BI", {"skills": ["power_bi"]}),
    ("he-kafka", "hebrew", "עבודה עם Kafka", {"skills": ["kafka"]}),
    ("he-analyst", "hebrew", "אנליסט/ית נתונים", {"role": "analytics"}),
    ("he-data-engineer", "hebrew", "מהנדס/ת דאטה", {"role": "data_engineering"}),
    ("role-de", "role", "Data Engineer", {"role": "data_engineering"}),
    ("role-analyst", "role", "data analyst or product analyst", {"role": "analytics"}),
    ("role-bi", "role", "BI developer", {"role": "bi"}),
    ("role-ds", "role", "data scientist, machine learning", {"role": "data_science"}),
    ("role-ae", "role", "analytics engineer", {"role": "analytics_engineering"}),
    ("para-realtime", "paraphrase", "process events as they happen with low latency",
     {"jd": r"real[- ]?time|streaming"}),
    ("para-from-scratch", "paraphrase", "first data hire, set up the data stack",
     {"jd": r"from scratch|from the ground up|greenfield|first data|founding"}),
    ("para-mentor", "paraphrase", "coach and grow other engineers on the team",
     {"jd": r"mentor|ניהול צוות|team lead|lead a team"}),
    ("para-quality", "paraphrase", "make sure the numbers in reports can be trusted",
     {"jd": r"data quality|data integrity|data validation|איכות ה?נתונים"}),
    ("para-cloud", "paraphrase", "work on Amazon's cloud", {"skills": ["aws"]}),
]
MAX_SHARE = 0.5  # a rule that matches more than half the corpus says little about ranking


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, help="count matches in this jobs DB instead of the eval set")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "retrieval" / "queries.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    import duckdb
    import pandas as pd

    from eval.retrieval import relevant_jobs
    if args.db:
        con = duckdb.connect(str(args.db), read_only=True)
        jobs = con.execute("SELECT job_id, title, description FROM jobs WHERE is_active").df()
        con.close()
    else:
        from eval import dataset as ds
        jobs = pd.DataFrame([{"job_id": c["case_id"], "title": c["title"], "description": c["description"]}
                             for c in ds.load_jobs()])
    kept = []
    print(f"corpus: {len(jobs)} jobs\n{'qid':<22} {'kind':<11} relevant")
    for qid, kind, query, rule in DRAFT:
        n = len(relevant_jobs(rule, jobs))
        ok = 0 < n <= MAX_SHARE * len(jobs)
        print(f"{qid:<22} {kind:<11} {n:>4}{'' if ok else '  (left out)'}")
        if ok:
            kept.append({"qid": qid, "kind": kind, "query": query, "relevant": rule})
    print(f"\n{len(kept)} of {len(DRAFT)} queries kept")
    if not args.dry_run:
        args.out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in kept), encoding="utf-8")
        print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
