"""Eval metrics and harness, with synthetic data and a fake client (no API, no network)."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from eval import evaluate as E
from eval import metrics as M


def preds(scores, labels, repeat=0, ids=None):
    ids = ids or [f"c{i}" for i in range(len(scores))]
    return pd.DataFrame({"case_id": ids, "repeat": repeat, "score": scores, "label": labels})


def test_binary_counts():
    m = M.binary([90, 80, 40, 60, 10], [True, True, True, False, False], threshold=50)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_binary_no_predictions_is_zero_not_error():
    assert M.binary([10, 20], [True, False], 50) == {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                                                     "tp": 0, "fp": 0, "fn": 1}


def test_best_threshold_separates_perfectly():
    best = M.best_threshold([95, 88, 70, 30, 10], [True, True, False, False, False])
    assert best["f1"] == 1.0 and 71 <= best["threshold"] <= 88
    assert best["threshold"] == 71  # lowest threshold among ties


def test_roc_auc_known_values():
    assert M.roc_auc([90, 80, 20, 10], [True, True, False, False]) == 1.0
    assert M.roc_auc([10, 20, 80, 90], [True, True, False, False]) == 0.0
    assert M.roc_auc([50, 50], [True, False]) == 0.5  # a tie counts half
    assert np.isnan(M.roc_auc([50, 60], [True, True]))


def test_spearman_monotone():
    assert M.spearman([10, 50, 90], ["no_fit", "maybe", "fit"]) == pytest.approx(1.0)
    assert M.spearman([90, 50, 10], ["no_fit", "maybe", "fit"]) == pytest.approx(-1.0)


def test_precision_at_3_per_day():
    # 2 days of 15 after a seeded shuffle; fit cases score highest, so each day's top 3 is as fit as possible.
    labels = ["fit"] * 4 + ["no_fit"] * 26
    df = preds([90] * 4 + [10] * 26, labels)
    p = M.precision_at_k(df, k=3, day_size=15, seed=0)
    order = np.random.default_rng(0).permutation(30)
    per_day = [min(3, sum(1 for i in order[d:d + 15] if i < 4)) for d in (0, 15)]
    assert p == pytest.approx(sum(per_day) / 6)


def test_confusion_and_calibration():
    df = preds([90, 70, 65, 40, 5], ["fit", "fit", "maybe", "no_fit", "no_fit"])
    c = M.confusion(df)
    assert c.loc["fit", "85-100"] == 1 and c.loc["fit", "60-84"] == 1
    assert c.loc["maybe", "60-84"] == 1 and c.loc["no_fit", "30-59"] == 1 and c.loc["no_fit", "0-29"] == 1
    assert int(c.to_numpy().sum()) == 5
    cal = M.calibration(df)
    assert cal["score_mean_fit"] == 80 and cal["score_std_fit"] == 10
    # |90-90| + |70-90| + |65-70| + |40-20| + |5-20| = 0+20+5+20+15 = 60 / 5
    assert cal["mae_vs_target"] == pytest.approx(12.0)


def test_stability_across_repeats():
    a = preds([90, 40], ["fit", "no_fit"], repeat=0)
    b = preds([80, 60], ["fit", "no_fit"], repeat=1)
    s = M.stability(pd.concat([a, b]), threshold=50)
    assert s["score_std_mean"] == pytest.approx(7.5)  # std of (90,80) = 5, of (40,60) = 10
    assert s["decision_flip_rate"] == 0.5


def test_summarize_keys():
    df = preds([95, 85, 70, 40, 20, 10], ["fit", "fit", "maybe", "no_fit", "no_fit", "no_fit"])
    m = M.summarize(df, threshold=50)
    assert m["n_cases"] == 6 and m["f1_at_threshold"] == pytest.approx(0.8) and m["roc_auc_fit"] == 1.0
    assert m["spearman"] > 0.9 and "score_std_mean" not in m  # one repeat: no stability block


def test_compare_detects_worse_and_bootstrap_is_seeded():
    labels = ["fit"] * 10 + ["no_fit"] * 10
    base = preds([90] * 10 + [20] * 10, labels)
    worse = preds([40] * 10 + [60] * 10, labels)
    c = M.compare(worse, base, threshold=50)
    assert c["verdict"] == "worse" and c["wilcoxon_p"] < 0.05 and c["f1_delta"] == pytest.approx(-1.0)
    assert c["f1_delta_ci_high"] < 0
    assert c == M.compare(worse, base, threshold=50)  # fixed seed -> identical result
    same = M.compare(base, base, threshold=50)
    assert same["verdict"] == "no significant difference" and same["wilcoxon_p"] == 1.0


class FakeScorer:
    """Plays the agent protocol: list jobs, then save a score for each from a title rule, then DONE.
    Records the job titles of each conversation so tests can check batching."""

    RULE = {"Data Engineer": 92, "BI Developer": 70}

    def __init__(self):
        self.messages = self
        self.batches = []

    def create(self, **kw):
        last = kw["messages"][-1]["content"]
        tu = lambda i, name, inp: NS(type="tool_use", id=f"t{i}", name=name, input=inp)
        usage = NS(input_tokens=100, output_tokens=10, cache_read_input_tokens=50, cache_creation_input_tokens=0)
        if isinstance(last, str):
            return NS(content=[tu(0, "list_new_jobs", {})], usage=usage)
        out = last[0]["content"]
        if out.startswith("["):
            jobs = json.loads(out)
            self.batches.append([j["title"] for j in jobs])
            return NS(content=[tu(i + 1, "save_match", {"job_id": j["job_id"], "score": self.RULE.get(j["title"], 10),
                                                        "summary": "s", "strengths": [], "gaps": []})
                               for i, j in enumerate(jobs)], usage=usage)
        return NS(content=[NS(type="text", text="DONE")], usage=usage)


def _cases(n):
    titles = ["Data Engineer", "BI Developer", "Cook"]
    jobs = [{"case_id": f"c{i}", "company": "Acme", "title": titles[i % 3], "location": "", "department": "",
             "url": "", "description": "d"} for i in range(n)]
    labels = {f"c{i}": {"label": ["fit", "maybe", "no_fit"][i % 3], "note": ""} for i in range(n)}
    return jobs, labels


def test_harness_runs_real_agent_with_fake_client():
    jobs, labels = _cases(40)  # more than BATCH (15): needs several agent.run() calls
    fake = FakeScorer()
    p, info = E.run(jobs, labels, "CV", {}, "claude-sonnet-5-5", "v1", repeats=2, batch_seed=3, client=fake)
    assert len(p) == 80 and info["coverage"] == 1.0
    assert [len(b) for b in fake.batches] == [15, 15, 10, 15, 15, 10]
    assert fake.batches[:3] == fake.batches[3:]  # same seed -> same batches in every repeat
    assert info["api_calls"] == 18 and info["tokens_input_tokens"] == 1800 and info["cost_usd"] > 0
    m = M.summarize(p, threshold=50)
    assert m["roc_auc_fit"] == 1.0 and m["spearman"] == pytest.approx(1.0)
    assert m["score_std_mean"] == 0.0 and m["decision_flip_rate"] == 0.0


def test_harness_batches_depend_on_seed():
    jobs, labels = _cases(30)
    a, b = FakeScorer(), FakeScorer()
    E.run(jobs, labels, "CV", {}, "m", "v1", batch_seed=1, client=a)
    E.run(jobs, labels, "CV", {}, "m", "v1", batch_seed=2, client=b)
    assert a.batches != b.batches


def test_baseline_roundtrip_is_numbers_only(tmp_path):
    jobs, labels = _cases(6)
    p, info = E.run(jobs, labels, "CV", {}, "m", "v1", client=FakeScorer())
    data = E.baseline_json(p, info, M.summarize(p, 50))
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    _, back = E.load_baseline(path)
    assert set(back.columns) == {"case_id", "repeat", "score"} and len(back) == 6
    text = path.read_text(encoding="utf-8")
    assert "Data Engineer" not in text and '"label"' not in text
