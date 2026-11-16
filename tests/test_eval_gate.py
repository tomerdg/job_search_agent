"""Quality gate decisions and the eval workflow config (no API, no network)."""
from pathlib import Path

import yaml

from eval import gate

GOOD = {"precision_fit_or_maybe_at_threshold": 0.885, "recall_fit_or_maybe_at_threshold": 1.0,
        "recall_at_threshold": 1.0, "roc_auc_fit": 0.948}
INFO = {"coverage": 1.0}
SAME = {"verdict": "no significant difference", "wilcoxon_p": 0.45, "warnings": []}


def test_baseline_like_run_passes_and_says_why():
    ok, lines = gate.decide(GOOD, INFO, SAME)
    assert ok
    assert len(lines) == 1 + len(gate.FLOORS) + 1 and all(l.startswith("PASS") for l in lines)


def test_floor_failure_names_the_metric():
    ok, lines = gate.decide({**GOOD, "precision_fit_or_maybe_at_threshold": 0.70}, INFO, SAME)
    assert not ok
    assert "FAIL: precision_fit_or_maybe_at_threshold 0.700 >= 0.8" in lines


def test_significantly_worse_fails_even_above_floors():
    ok, lines = gate.decide(GOOD, INFO, {"verdict": "worse", "wilcoxon_p": 0.01, "warnings": []})
    assert not ok and any(l.startswith("FAIL: not significantly worse") for l in lines)


def test_better_passes():
    ok, _ = gate.decide(GOOD, INFO, {"verdict": "better", "wilcoxon_p": 0.01, "warnings": []})
    assert ok


def test_low_coverage_and_missing_or_nan_metric_fail():
    ok, lines = gate.decide({**GOOD, "roc_auc_fit": float("nan")}, {"coverage": 0.9}, SAME)
    assert not ok
    assert any("coverage 0.900" in l and l.startswith("FAIL") for l in lines)
    assert "FAIL: roc_auc_fit is missing" in lines


def test_without_baseline_regression_check_is_skipped_not_failed():
    ok, lines = gate.decide(GOOD, INFO, None)
    assert ok and any(l.startswith("SKIP") for l in lines)


def test_unpaired_comparison_is_noted():
    ok, lines = gate.decide(GOOD, INFO, {**SAME, "warnings": ["batch_seed 1 differs from the baseline's 0"]})
    assert ok and any(l.startswith("NOTE") for l in lines)


def test_markdown_states_result():
    assert "**FAILED**" in gate.markdown(False, ["FAIL: x"])
    assert "**passed**" in gate.markdown(True, ["PASS: x"])


def test_eval_workflow_runs_gate_and_never_uploads_predictions():
    wf = yaml.safe_load((Path(__file__).resolve().parent.parent / ".github/workflows/eval.yml").read_text())
    triggers = wf[True]  # YAML 1.1 reads the key `on` as True
    assert "jobagent/prompts/**" in triggers["pull_request"]["paths"]
    steps = wf["jobs"]["eval"]["steps"]
    run = next(s for s in steps if s.get("name") == "Evaluate and gate")["run"]
    assert "--gate" in run and "--baseline" in run
    upload = next(s for s in steps if s.get("name") == "Upload metrics")["with"]["path"]
    assert "predictions" not in upload
    assert all(s.get("if", "").find("HAS_SECRETS") >= 0 for s in steps)  # every step skips without secrets
