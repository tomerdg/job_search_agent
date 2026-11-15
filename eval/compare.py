"""Compare evaluation runs from MLflow without opening the UI.

    python -m eval.compare                 # last 10 runs, best F1 first
    python -m eval.compare --last 30 --by prompt_version,model   # mean per prompt/model
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from eval import tracking

PARAMS = ["prompt_version", "model", "repeats"]
METRICS = ["f1_at_threshold", "precision_at_threshold", "recall_at_threshold",
           "precision_fit_or_maybe_at_threshold", "recall_fit_or_maybe_at_threshold", "roc_auc_fit", "spearman",
           "precision_at_3", "f1_at_best", "best_threshold", "score_mean_fit", "score_mean_maybe",
           "score_std_mean", "cost_usd_per_job"]


def load_runs(last: int) -> pd.DataFrame:
    import mlflow
    mlflow.set_tracking_uri(tracking.tracking_uri())
    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT], max_results=last,
                              order_by=["attributes.start_time DESC"])
    if runs.empty:
        return runs
    cols = {f"params.{p}": p for p in PARAMS} | {f"metrics.{m}": m for m in METRICS}
    out = runs[["start_time", *[c for c in cols if c in runs.columns]]].rename(columns=cols)
    out["start_time"] = out["start_time"].dt.strftime("%m-%d %H:%M")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--by", help="comma-separated params to group by, e.g. prompt_version,model")
    args = ap.parse_args(argv)
    runs = load_runs(args.last)
    if runs.empty:
        print(f"No runs in experiment '{tracking.EXPERIMENT}' at {tracking.tracking_uri()}")
        return 1
    if args.by:
        keys = args.by.split(",")
        metric_cols = [m for m in METRICS if m in runs.columns]
        table = runs.groupby(keys)[metric_cols].agg("mean")
        table.insert(0, "runs", runs.groupby(keys).size())
        table = table.sort_values("f1_at_threshold", ascending=False)
    else:
        table = runs.sort_values("f1_at_threshold", ascending=False).set_index("start_time")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        print(table.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
