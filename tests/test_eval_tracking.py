"""MLflow logging and run comparison, against a throwaway local tracking store (no API, no network)."""
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from eval import compare, tracking  # noqa: E402
from eval import dataset as ds  # noqa: E402
from eval import metrics as M  # noqa: E402
from jobagent.agent import load_prompt  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_REF", raising=False)
    jobs, labels = tmp_path / "jobs.jsonl", tmp_path / "labels.csv"
    jobs.write_text('{"case_id": "a", "title": "T", "description": "D"}\n', encoding="utf-8")
    labels.write_text("case_id,label,note\na,fit,\n", encoding="utf-8")
    monkeypatch.setattr(ds, "JOBS_PATH", jobs)
    monkeypatch.setattr(ds, "LABELS_PATH", labels)
    return tmp_path


def _run(prompt="v1", model="m", scores=(95, 80, 70, 20, 10, 5)):
    labels = ["fit", "fit", "maybe", "no_fit", "no_fit", "no_fit"]
    preds = pd.DataFrame({"case_id": list("abcdef"), "repeat": 0, "score": list(scores), "label": labels})
    info = {"prompt_version": prompt, "model": model, "repeats": 1, "batch_seed": 0, "cost_usd": 0.5,
            "cost_usd_per_job": 0.01, "coverage": 1.0}
    return preds, info, M.summarize(preds, 59)


def _artifacts(run_id):
    client = mlflow.tracking.MlflowClient()
    names = {a.path for a in client.list_artifacts(run_id)}
    return names | {a.path for a in client.list_artifacts(run_id, "prompt")}


def test_log_run_params_metrics_artifacts(store):
    preds, info, metrics = _run()
    run_id = tracking.log_run(preds, info, metrics, None, 59)
    run = mlflow.get_run(run_id)
    assert run.data.params["prompt_version"] == "v1" and run.data.params["threshold"] == "59"
    assert len(run.data.params["dataset_hash"]) == 40
    assert run.data.metrics["roc_auc_fit"] == 1.0 and run.data.metrics["cost_usd"] == 0.5
    assert run.data.tags["trigger"] == "local"
    assert {"metrics.json", "confusion.csv", "confusion.png", "score_by_label.png", "predictions.csv",
            "prompt/v1.txt"} <= _artifacts(run_id)


def test_ci_run_never_logs_predictions(store, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
    preds, info, metrics = _run()
    run_id = tracking.log_run(preds, info, metrics, None, 59)
    run = mlflow.get_run(run_id)
    assert run.data.tags["trigger"] == "ci" and run.data.tags["pr"] == "42"
    assert "predictions.csv" not in _artifacts(run_id)


def test_compare_groups_runs(store, capsys):
    for prompt, scores in [("v1", (95, 80, 70, 20, 10, 5)), ("v1", (95, 80, 70, 20, 10, 5)),
                           ("v2", (60, 50, 90, 70, 10, 5))]:
        preds, info, metrics = _run(prompt, scores=scores)
        tracking.log_run(preds, info, metrics, None, 59)
    runs = compare.load_runs(10)
    assert len(runs) == 3 and set(runs["prompt_version"]) == {"v1", "v2"}
    assert compare.main(["--by", "prompt_version,model"]) == 0
    out = capsys.readouterr().out
    assert out.index("v1") < out.index("v2")  # v1 has the higher F1, so it is listed first


def test_v2_prompt_renders():
    text = load_prompt("v2").format(roles="Data Engineer", allowed="Tel Aviv", blocked="Haifa")
    assert "Target roles: Data Engineer." in text and "85-100" in text and text.endswith("DONE.")
