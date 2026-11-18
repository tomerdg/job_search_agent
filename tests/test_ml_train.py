"""Relevance-model training on synthetic data: leakage, metrics, gate, CLI end to end."""
import json

import numpy as np
import pandas as pd
import pytest

from jobagent.demo import DEMO_CV, DEMO_PROFILE, build
from jobagent.ml import train as T
from jobagent.ml import validate as V


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("ml") / "demo.duckdb"
    build(db, days=120)
    return db


@pytest.fixture(scope="module")
def demo_frames(demo_db):
    import duckdb
    con = duckdb.connect(str(demo_db), read_only=True)
    jobs = T.load_jobs(con)
    fb = con.execute("SELECT job_id, decision, source FROM job_feedback").df()
    con.close()
    return jobs, fb


@pytest.fixture(scope="module")
def result(demo_frames):
    jobs, fb = demo_frames
    return T.run(jobs, fb, DEMO_CV, DEMO_PROFILE, cv_repeats=1, n_boot=50)


def test_gold_never_in_training(demo_frames):
    jobs, fb = demo_frames
    gold = T.gold_rows(jobs, fb)
    train = T.training_rows(jobs, gold, DEMO_PROFILE)
    assert len(gold) and len(train)
    assert not set(train["job_id"]) & set(gold["job_id"])
    assert not set(train["url"]) & set(gold["url"])
    assert set(train["tier"]) == {"silver", "weak_negative"}


def test_eval_gold_maps_by_url_and_removes_repost(demo_frames):
    jobs, fb = demo_frames
    j = jobs.dropna(subset=["score"]).iloc[0]
    case = {"case_id": "c1", "company": j.company, "title": j.title, "location": j.location,
            "department": "", "url": j.url, "description": j.description}
    gold = T.gold_rows(jobs, fb.iloc[:0], [case], {"c1": {"label": "maybe"}})
    assert gold.loc[0, "job_id"] == j.job_id and gold.loc[0, "y"] == 1 and gold.loc[0, "is_fit"] == 0
    assert j.job_id not in set(T.training_rows(jobs, gold, DEMO_PROFILE)["job_id"])


def test_run_end_to_end(result):
    m, t = result["metrics"], result["table"]
    assert set(t.index) >= {"logreg", "xgboost", "prefilter", "llm_score"}
    for name in ("logreg", "xgboost"):
        assert 0.6 < m[f"{name}.roc_auc"] <= 1                       # learns the synthetic signal
        assert m[f"{name}.roc_auc_ci_low"] <= m[f"{name}.roc_auc"] <= m[f"{name}.roc_auc_ci_high"]
        assert m[f"{name}.recall_at_r95"] >= 0.95
        assert 0 <= m[f"{name}.filtered_share_at_r95"] <= 1
    assert 0 <= m["logreg_vs_xgboost.auc_diff_p"] <= 1
    assert all(result["validation"].results.values())
    assert len(result["dataset_hash"]) == 40


def test_sparse_matrix(demo_frames):
    from scipy import sparse
    jobs, _ = demo_frames
    X = T.frame(jobs.head(50), DEMO_CV, DEMO_PROFILE)
    f = T.SparseFeatures(k_best=20).fit(X, (jobs.head(50)["score"].fillna(0) > 60).astype(int))
    M = f.transform(X)
    assert sparse.isspmatrix_csr(M) and M.shape == (50, len(f.get_feature_names_out()))


def test_refuses_to_train_on_too_few_positives(demo_frames):
    jobs, fb = demo_frames
    jobs = jobs.assign(score=jobs["score"].where(jobs["score"].isna(), 0))  # no silver positive left
    with pytest.raises(V.ValidationError, match="min_positives"):
        T.run(jobs, fb, DEMO_CV, DEMO_PROFILE, cv_repeats=1, n_boot=10)


def test_threshold_for_recall():
    s, y = np.array([0.9, 0.8, 0.7, 0.2, 0.1]), np.array([1, 1, 1, 0, 1])
    t = T.threshold_for_recall(s, y, 0.75)
    assert t == 0.7 and T.at_threshold(s, y, y, t)["filtered_share"] == 0.4


def test_paired_auc_difference_detects_a_better_scorer():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    good, noise = y + rng.normal(0, 0.5, 200), rng.normal(0, 1, 200)
    out = T.paired_auc_difference(y, good, noise, n_boot=200)
    assert out["auc_diff"] > 0.2 and out["auc_diff_ci_low"] > 0 and out["auc_diff_p"] < 0.05


def test_gate():
    base = {"metrics": {"logreg.roc_auc": 0.8, "logreg.pr_auc": 0.6, "xgboost.roc_auc": 0.8, "xgboost.pr_auc": 0.6}}
    ok, lines = T.gate({"logreg.roc_auc": 0.78, "logreg.pr_auc": 0.6, "xgboost.roc_auc": 0.8, "xgboost.pr_auc": 0.6},
                       base)
    assert ok and all(l.startswith("PASS") for l in lines)
    ok, lines = T.gate({"logreg.roc_auc": 0.7, "logreg.pr_auc": 0.6, "xgboost.roc_auc": 0.8, "xgboost.pr_auc": None},
                       base)
    assert not ok and sum(l.startswith("FAIL") for l in lines) == 2
    assert T.gate({}, None) == (True, ["SKIP: no baseline given"])


def test_cli_writes_aggregates_only(demo_db, tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    out = tmp_path / "out"
    rc = T.main(["--demo", "--db", str(demo_db), "--out", str(out), "--cv-repeats", "1", "--bootstrap", "50",
                 "--report", "--baseline", str(T.ROOT / "jobagent" / "ml" / "baselines" / "demo.json"), "--gate"])
    assert rc == 0
    data = json.loads((out / "metrics.json").read_text())
    assert data["gate"]["passed"] is True and all(data["validation"].values())
    flat = json.dumps(data)
    assert "Synthetic job" not in flat and "example.com" not in flat   # no job text or urls
    assert (out / "relevance_report.md").exists()

    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    run = mlflow.search_runs(experiment_names=[T.EXPERIMENT], output_format="list")[0]
    assert run.data.tags["trigger"] == "ci" and run.data.tags["validation.leakage.gold_not_in_train"] == "pass"
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    assert artifacts == {"comparison.csv", "metrics.json", "pr_curve.png"}   # no model, no weights in CI
    assert "dataset_hash" in run.data.params and "git_sha" in run.data.params
