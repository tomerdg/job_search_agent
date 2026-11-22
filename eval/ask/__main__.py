"""python -m eval.ask [--db PATH] [--no-mlflow] - see eval/ask/__init__.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval import ask as A
from eval.retrieval import ROOT, relevant_jobs, temp_copy


def main(argv=None) -> int:
    from jobagent.main import load_dotenv
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Evaluate jobagent.ask: hit rate and citation validity")
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "jobs.duckdb", help="copied, not changed")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)

    from jobagent.ask import MODEL, ask
    from jobagent.costs import add_usage, empty_usage, estimate_cost_usd
    from jobagent.rag.embed import get_embedder
    from jobagent.rag.index import index_jobs
    embedder = get_embedder()
    holder, store = temp_copy(args.db)
    rows, usage = [], empty_usage()
    try:
        index_jobs(store, embedder, run_id="ask-eval", scope="all")
        jobs = store.conn.execute("SELECT job_id, company, title, description FROM jobs WHERE is_active").df()
        for q in A.load_questions():
            expected = relevant_jobs(q["expect"], jobs)
            companies = set(jobs.loc[jobs["job_id"].isin(expected), "company"])
            a = ask(store, embedder, q["question"], active_only=True)
            add_usage(usage, a.usage)
            rows.append({"qid": q["qid"], "hit": A.judge(a, expected, companies), "valid": len(a.cited),
                         "invalid": len(a.invalid), "n_expected": len(expected)})
    finally:
        store.close()
        holder.cleanup()
    summary = {**A.summarize(rows), "cost_usd": estimate_cost_usd(usage, MODEL) or 0.0}
    for r in rows:
        print(f"{r['qid']:<18} hit={int(r['hit'])} cited={r['valid']} dropped={r['invalid']} expected_jobs={r['n_expected']}")
    print("\n" + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary.items()))
    if not args.no_mlflow:
        import mlflow
        from eval import tracking
        mlflow.set_tracking_uri(tracking.tracking_uri())
        mlflow.set_experiment("ask")
        with mlflow.start_run(run_name=f"ask-{MODEL}"):
            mlflow.log_params({"model": MODEL, "embedder": embedder.model_id, "k": 12, "n_jobs": len(jobs),
                               "git_sha": tracking.git_sha()})
            mlflow.log_metrics({k: float(v) for k, v in summary.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
