"""MLflow experiment tracking for evaluation runs.

Tracking URI: env MLFLOW_TRACKING_URI, default a local SQLite file (mlflow.db in the repo root,
git-ignored), so no server is needed. Browse it with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

What is logged is numbers, the prompt file, and plots of numbers. The per-case predictions CSV is logged
only on a local run, never in CI: per-case scores are derived from the CV.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from eval import dataset as ds
from eval import metrics as M
from jobagent.agent import PROMPTS_DIR

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "job-scoring"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{ROOT / 'mlflow.db'}"


def in_ci() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def dataset_hash() -> str:
    """SHA-1 of the plain jobs.jsonl + labels.csv content (same value locally and in CI, where they are decrypted)."""
    h = hashlib.sha1()
    for p in (ds.JOBS_PATH, ds.LABELS_PATH):
        h.update(ds.read_text(p).encode("utf-8"))
    return h.hexdigest()


def git_sha() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def pr_number() -> str | None:
    """PR number in a pull_request workflow (GITHUB_REF is refs/pull/<n>/merge)."""
    ref = os.environ.get("GITHUB_REF", "")
    parts = ref.split("/")
    return parts[2] if len(parts) > 2 and parts[1] == "pull" else None


def _plots(per_case: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = M.confusion(per_case)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(conf.to_numpy(), cmap="Blues")
    ax.set_xticks(range(conf.shape[1]), conf.columns)
    ax.set_yticks(range(conf.shape[0]), conf.index)
    for (i, j), v in pd.DataFrame(conf.to_numpy()).stack().items():
        ax.text(j, i, int(v), ha="center", va="center")
    ax.set_xlabel("score band")
    ax.set_title("Label vs score band")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=120)
    plt.close(fig)

    order = [l for l in ("fit", "maybe", "no_fit") if l in set(per_case["label"])]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot([per_case.loc[per_case["label"] == l, "score"] for l in order], tick_labels=order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("score")
    ax.set_title("Score by label")
    fig.tight_layout()
    fig.savefig(out / "score_by_label.png", dpi=120)
    plt.close(fig)
    return [out / "confusion.png", out / "score_by_label.png"]


def log_run(preds: pd.DataFrame, info: dict, metrics: dict, comparison: dict | None, threshold: float) -> str:
    """Log one evaluation as an MLflow run. Returns the run id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        # With a SQLite store, keep artifacts next to the DB file (mlruns/, git-ignored), not wherever the
        # command was run from. Other stores (a server) choose their own artifact location.
        uri = tracking_uri()
        location = (Path(uri.removeprefix("sqlite:///")).parent / "mlruns").as_uri() \
            if uri.startswith("sqlite:///") else None
        mlflow.create_experiment(EXPERIMENT, artifact_location=location)
    mlflow.set_experiment(EXPERIMENT)
    name = f"{info['prompt_version']}-{info['model']}-r{info['repeats']}"
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({
            "prompt_version": info["prompt_version"], "model": info["model"], "repeats": info["repeats"],
            "batch_seed": info["batch_seed"],
            "vary_batches": info.get("vary_batches", False), "n_cases": metrics["n_cases"], "threshold": threshold,
            "dataset_hash": dataset_hash(), "git_sha": git_sha(),
        })
        numeric = {**metrics, **info, **{f"vs_baseline_{k}": v for k, v in (comparison or {}).items()}}
        mlflow.log_metrics({k: float(v) for k, v in numeric.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v})
        tags = {"trigger": "ci" if in_ci() else "local"}
        if comparison:
            tags["vs_baseline"] = comparison["verdict"]
            tags["vs_baseline_paired"] = str(not comparison.get("warnings"))
        if pr_number():
            tags["pr"] = pr_number()
        mlflow.set_tags(tags)

        per_case = M.per_case(preds)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "metrics.json").write_text(json.dumps(
                {"info": info, "metrics": metrics, "comparison": comparison}, indent=2), encoding="utf-8")
            M.confusion(per_case).to_csv(out / "confusion.csv")
            files = [out / "metrics.json", out / "confusion.csv", *_plots(per_case, out)]
            if not in_ci():  # per-case scores are CV-derived: local tracking store only
                preds[["case_id", "repeat", "score", "label"]].to_csv(out / "predictions.csv", index=False)
                files.append(out / "predictions.csv")
            for f in files:
                mlflow.log_artifact(str(f))
        mlflow.log_artifact(str(PROMPTS_DIR / f"{info['prompt_version']}.txt"), artifact_path="prompt")
        return run.info.run_id
