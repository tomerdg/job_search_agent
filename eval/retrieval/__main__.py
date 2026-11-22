"""python -m eval.retrieval [--modes vector bm25 hybrid] [--db PATH] [--gate] - see eval/retrieval/__init__.py."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from eval import retrieval as R

OUT_DIR = R.ROOT / "eval" / "out" / "retrieval"
EXPERIMENT = "retrieval"
FLOOR = {"recall_at_10": 0.7}  # hybrid, all queries. 0.6 at first; the first run (2026-11-22) gave 0.773


def table(results: dict) -> str:
    kinds = [k for k in next(iter(results.values())) if k != "per_query"]
    lines = ["| mode | kind | n | recall@5 | recall@10 | MRR | nDCG@10 |", "|---|---|---|---|---|---|---|"]
    for mode, summary in results.items():
        for kind in kinds:
            s = summary[kind]
            if s.get("n_queries"):
                lines.append(f"| {mode} | {kind} | {s['n_queries']} | {s['recall_at_5']:.3f} | "
                             f"{s['recall_at_10']:.3f} | {s['mrr']:.3f} | {s['ndcg_at_10']:.3f} |")
    return "\n".join(lines)


def log_mlflow(results: dict, params: dict) -> list[str]:
    import mlflow
    from eval import tracking
    mlflow.set_tracking_uri(tracking.tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    ids = []
    for mode, summary in results.items():
        with mlflow.start_run(run_name=f"{mode}-{params['model_id'].split('/')[-1]}") as run:
            mlflow.log_params({**params, "mode": mode, "git_sha": tracking.git_sha()})
            mlflow.log_metrics({f"{kind}.{name}" if kind != "all" else name: float(v)
                                for kind, s in summary.items() if kind != "per_query"
                                for name, v in s.items() if v == v})
            mlflow.set_tags({"trigger": "ci" if tracking.in_ci() else "local"})
            ids.append(run.info.run_id)
    return ids


def main(argv=None) -> int:
    from jobagent.main import load_dotenv
    load_dotenv(R.ROOT / ".env")
    ap = argparse.ArgumentParser(description="Evaluate JD retrieval (recall@k, MRR, nDCG) per search mode")
    ap.add_argument("--modes", nargs="+", default=["vector", "bm25", "hybrid"])
    ap.add_argument("--db", type=Path, help="corpus: active jobs of this jobs DB (copied, not changed); "
                                            "default: the eval set's JDs")
    ap.add_argument("--queries", type=Path, default=R.QUERIES_PATH)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--gate", action="store_true", help=f"exit 1 when hybrid misses the floor {FLOOR}")
    args = ap.parse_args(argv)

    from jobagent.rag.chunking import CHUNKING_VERSION
    from jobagent.rag.embed import get_embedder
    embedder = get_embedder()
    if embedder is None:
        print("No embedder: install requirements-rag.txt (or set RAG_EMBEDDER=fake for a smoke run)")
        return 1
    queries = R.load_queries(args.queries)
    with tempfile.TemporaryDirectory() as tmp:
        if args.db:
            holder, store = R.temp_copy(args.db)
        else:
            holder, store = None, R.eval_set_store(Path(tmp) / "eval.duckdb")
        try:
            stats = R.index_corpus(store, embedder)
            results = R.evaluate(store, embedder, queries, tuple(args.modes))
        finally:
            store.close()
            if holder:
                holder.cleanup()

    params = {"model_id": embedder.model_id, "chunking_version": CHUNKING_VERSION, "k": max(R.KS),
              "per_job": 1, "corpus": "db" if args.db else "eval_set", "n_jobs": stats["jobs_indexed"],
              "n_chunks": stats["chunks_total"], "n_queries": len(queries), "queries_hash": R.queries_hash(args.queries)}
    print(table(results))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{datetime.now():%Y%m%dT%H%M%S}-{params['corpus']}.json"
    out.write_text(json.dumps({"params": params, "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(R.ROOT)}")

    ok = True
    if args.gate:
        if "hybrid" not in results:
            print("--gate needs the hybrid mode")
            return 1
        got = results["hybrid"]["all"]
        fails = [f"{m} {got[m]:.3f} < {floor}" for m, floor in FLOOR.items() if got.get(m, 0) < floor]
        ok = not fails
        print("Gate: " + ("PASS" if ok else "FAIL: " + "; ".join(fails)))
    if not args.no_mlflow:
        try:
            print(f"MLflow runs {', '.join(log_mlflow(results, params))} in experiment '{EXPERIMENT}'")
        except Exception as exc:  # noqa: BLE001 - the evaluation itself succeeded
            print(f"warning: MLflow logging failed: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
