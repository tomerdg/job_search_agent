"""Run the production scoring agent on the labeled set and measure how good the scores are.

    python -m eval.evaluate --prompt-version v1 --model claude-sonnet-5-5 --repeats 1
    python -m eval.evaluate --baseline eval/dataset/baseline.json       # compare with a saved run
    python -m eval.evaluate --write-baseline                            # save this run as the baseline

Each repeat loads the dataset into a fresh temporary DuckDB Store, marks every job as passed
pre-filter, and calls the real jobagent.agent.Agent until nothing is pending, so the evaluation
tests exactly what production runs (same tools, prompt file, batching and save path).

Batches: the agent scores BATCH jobs per conversation and a score can depend on the other jobs in
the conversation, so the order is fixed by --batch-seed. Every run with the same seed sees the same
batches, which keeps prompt-vs-prompt comparisons paired.

Needs ANTHROPIC_API_KEY, the CV (CV_TEXT, or --cv, or cv_path in the profile) and the profile
(PROFILE_YAML or config/profile.yaml). The output is numbers only. Per-case scores go to
eval/out/ (git-ignored); the analysis text the agent writes is never saved.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from eval import dataset as ds
from eval import metrics as M
from jobagent import agent as agent_mod
from jobagent.agent import Agent
from jobagent.costs import add_usage, empty_usage, estimate_cost_usd
from jobagent.db import Store
from jobagent.models import Job

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "eval" / "out" / "runs"
BASELINE_PATH = ds.DATASET_DIR / "baseline.json"
MAX_IDLE_RUNS = 2  # agent.run() calls in a row that score nothing before giving up on the rest


def load_profile() -> dict:
    text = os.environ.get("PROFILE_YAML")
    if not text:
        path = Path(os.environ.get("PROFILE_PATH", ROOT / "config" / "profile.yaml"))
        text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_cv_text(profile: dict, cv_path: str | None) -> str:
    from jobagent.cv import load_cv
    return load_cv(cv_path or ROOT / profile.get("cv_path", "cv.pdf"))


def fill_store(store: Store, jobs: list[dict], batch_seed: int) -> dict[str, str]:
    """Insert the cases so pending_scoring() returns them in a seeded order. Returns job_id -> case_id."""
    order = list(jobs)
    random.Random(batch_seed).shuffle(order)
    # pending_scoring() takes the newest first_seen first, so the first case gets the latest timestamp.
    t0 = datetime(2026, 1, 1)
    mapping = {}
    for i, c in enumerate(order):
        job = Job(source="eval", company=c["company"] or "unknown", external_id=c["case_id"], title=c["title"],
                  url=c["url"], location=c["location"], department=c["department"], description=c["description"])
        store.upsert([job], ts=t0 + timedelta(seconds=len(order) - i))
        store.set_prefilter(job.job_id, True)
        mapping[job.job_id] = c["case_id"]
    return mapping


def score_once(jobs: list[dict], cv: str, profile: dict, model: str, prompt_version: str,
               batch_seed: int, client=None) -> tuple[dict[str, int], dict, int]:
    """One full pass over the cases. Returns (case_id -> score, token usage, API calls)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "eval.duckdb")
        try:
            mapping = fill_store(store, jobs, batch_seed)
            agent = Agent(store, cv, profile, client=client, model=model, prompt_version=prompt_version)
            idle = 0
            while store.pending_scoring(limit=1) and idle < MAX_IDLE_RUNS:
                before = len(agent.saved)
                agent.run()
                idle = idle + 1 if len(agent.saved) == before else 0
            rows = store.conn.execute("SELECT job_id, score FROM jobs WHERE score IS NOT NULL").fetchall()
            return {mapping[j]: s for j, s in rows}, agent.usage, agent.turns
        finally:
            store.close()


def run(jobs: list[dict], labels: dict[str, dict], cv: str, profile: dict, model: str, prompt_version: str,
        repeats: int = 1, batch_seed: int = 0, client=None) -> tuple[pd.DataFrame, dict]:
    """Score every case `repeats` times. Returns (predictions, run info: usage, cost, timing, coverage)."""
    jobs = [j for j in jobs if j["case_id"] in labels and labels[j["case_id"]]["label"] in M.ORDINAL]
    usage, turns, rows = empty_usage(), 0, []
    t0 = time.time()
    for r in range(repeats):
        scores, u, t = score_once(jobs, cv, profile, model, prompt_version, batch_seed, client)
        add_usage(usage, u)
        turns += t
        rows += [{"case_id": cid, "repeat": r, "score": s} for cid, s in scores.items()]
    wall = time.time() - t0
    preds = pd.DataFrame(rows, columns=["case_id", "repeat", "score"])
    preds["label"] = preds["case_id"].map(lambda c: labels[c]["label"])
    n_scored = len(preds)
    cost = estimate_cost_usd(usage, model)
    info = {"prompt_version": prompt_version, "model": model, "repeats": repeats, "batch_seed": batch_seed,
            "n_cases_in_set": len(jobs), "coverage": n_scored / (len(jobs) * repeats) if jobs else 0.0,
            "api_calls": turns, **{f"tokens_{k}": v for k, v in usage.items()},
            "tokens_per_job": sum(usage.values()) / n_scored if n_scored else 0.0,
            "cost_usd": cost, "cost_usd_per_job": (cost / n_scored) if cost is not None and n_scored else None,
            "wall_time_s": wall}
    return preds, info


def load_baseline(path: Path) -> tuple[dict, pd.DataFrame]:
    data = json.loads(ds.read_text(path))  # plain file locally, .enc decrypted in memory in CI
    return data, pd.DataFrame(data["predictions"])


def baseline_json(preds: pd.DataFrame, info: dict, metrics: dict) -> dict:
    """Numbers and case ids only."""
    return {"prompt_version": info["prompt_version"], "model": info["model"], "created": datetime.now().isoformat(),
            "metrics": metrics, "predictions": preds[["case_id", "repeat", "score"]].to_dict("records")}


def markdown_table(metrics: dict, info: dict, comparison: dict | None) -> str:
    def fmt(v):
        if v is None:
            return "n/a"
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    keys = ["n_cases", "n_fit", "n_maybe", "n_no_fit", "threshold", "precision_at_threshold",
            "recall_at_threshold", "f1_at_threshold", "best_threshold", "f1_at_best",
            "recall_fit_or_maybe_at_threshold", "roc_auc_fit", "spearman", "precision_at_3",
            "score_mean_fit", "score_mean_maybe", "score_mean_no_fit", "mae_vs_target",
            "score_std_mean", "decision_flip_rate"]
    lines = [f"### Scoring eval: prompt `{info['prompt_version']}`, model `{info['model']}`, "
             f"{info['repeats']} repeat(s)", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {fmt(metrics[k])} |" for k in keys if k in metrics]
    lines += [f"| {k} | {fmt(info[k])} |" for k in ("coverage", "api_calls", "tokens_per_job", "cost_usd",
                                                     "cost_usd_per_job", "wall_time_s")]
    if comparison:
        lines += ["", f"**vs baseline:** {comparison['verdict']} "
                  f"(Wilcoxon p={comparison['wilcoxon_p']:.3f} on {comparison['n_paired']} cases, "
                  f"MAE delta {comparison['mae_delta']:+.1f}, F1 delta {comparison['f1_delta']:+.3f}, "
                  f"95% CI [{comparison['f1_delta_ci_low']:+.3f}, {comparison['f1_delta_ci_high']:+.3f}])"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from jobagent.main import load_dotenv
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Evaluate the scoring agent on the labeled set")
    ap.add_argument("--prompt-version", default=agent_mod.PROMPT_VERSION)
    ap.add_argument("--model", default=agent_mod.MODEL)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--batch-seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=float(os.environ.get("MIN_SCORE", "50")),
                    help="production threshold (MIN_SCORE, default 50)")
    ap.add_argument("--cv", help="CV file (default: CV_TEXT, else cv_path in the profile)")
    ap.add_argument("--baseline", type=Path, help="compare with this baseline (.json, or its .enc)")
    ap.add_argument("--write-baseline", nargs="?", const=BASELINE_PATH, type=Path,
                    help=f"save this run as the baseline (default {BASELINE_PATH.relative_to(ROOT)}, git-ignored;"
                         " then run `python -m eval.dataset encrypt`)")
    ap.add_argument("--limit", type=int, help="only the first N cases (quick smoke test)")
    args = ap.parse_args(argv)

    jobs, labels = ds.load_jobs(), ds.load_labels()
    if args.limit:
        jobs = jobs[:args.limit]
    profile = load_profile()
    cv = load_cv_text(profile, args.cv)

    preds, info = run(jobs, labels, cv, profile, args.model, args.prompt_version, args.repeats, args.batch_seed)
    if preds.empty:
        print("No case was scored; check the API key and the log above.")
        return 1
    metrics = M.summarize(preds, args.threshold)
    comparison = None
    if args.baseline:
        _, base_preds = load_baseline(args.baseline)
        comparison = M.compare(preds, base_preds, args.threshold)

    run_dir = OUT_DIR / f"{datetime.now():%Y%m%dT%H%M%S}-{args.prompt_version}-{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    preds[["case_id", "repeat", "score"]].to_csv(run_dir / "predictions.csv", index=False)
    M.confusion(M.per_case(preds)).to_csv(run_dir / "confusion.csv")
    (run_dir / "metrics.json").write_text(json.dumps(
        {"info": info, "metrics": metrics, "comparison": comparison}, indent=2), encoding="utf-8")
    if args.write_baseline:
        args.write_baseline.write_text(json.dumps(baseline_json(preds, info, metrics)), encoding="utf-8")

    table = markdown_table(metrics, info, comparison)
    print(table)
    print(M.confusion(M.per_case(preds)).to_string())
    print(f"\nSaved to {run_dir.relative_to(ROOT)}")
    if args.write_baseline:
        print(f"Baseline written to {args.write_baseline}; encrypt it with `python -m eval.dataset encrypt`.")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
