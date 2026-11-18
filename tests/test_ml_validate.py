"""Validation layer: every check passes on good data and fails loudly on broken data."""
import numpy as np
import pandas as pd
import pytest

from jobagent.ml import validate as V
from jobagent.ml.features import build_features

PROFILE = {"keywords": ["data engineer"], "locations_allowed": ["tel aviv"]}


@pytest.fixture
def X():
    df = pd.DataFrame({"title": ["Data Engineer", "BI Developer", "Sales"], "location": ["Tel Aviv"] * 3,
                       "source": ["adam"] * 3, "description": ["SQL 3+ years", "Power BI", "CRM"]})
    dense, text = build_features(df, "SQL", PROFILE)
    return dense.assign(text=text)


def test_features_pass(X):
    v = V.check_features(X)
    assert all(v.results.values()) and v.tags()["validation.features.finite"] == "pass"


@pytest.mark.parametrize("breakage, check", [
    (lambda X: X.drop(columns=["n_overlap"]), "features.columns"),
    (lambda X: X.iloc[:0], "features.not_empty"),
    (lambda X: X.assign(log_jd_len=X["log_jd_len"].astype(str)), "features.dtypes"),
    (lambda X: X.assign(hebrew_share=[0.1, np.nan, 0.0]), "features.finite"),
    (lambda X: X.assign(log_jd_len=[1.0, np.inf, 2.0]), "features.finite"),
    (lambda X: X.assign(seniority=["mid", None, "mid"]), "features.no_nulls"),
    (lambda X: X.assign(skill_coverage=[0.5, 1.5, 0.0]), "features.ranges"),
    (lambda X: X.assign(location_ok=[0, 2, 1]), "features.ranges"),
    (lambda X: X.assign(years_required=[0, 40, 3]), "features.ranges"),
    (lambda X: X.assign(n_missing=X["n_missing"] + 1), "features.ranges"),
    (lambda X: X.assign(role_family=["bi", "astronaut", "bi"]), "features.ranges"),
])
def test_features_fail(X, breakage, check):
    v = V.Validation()
    with pytest.raises(V.ValidationError, match=check):
        V.check_features(breakage(X), v)
    assert v.results[check] is False


def test_labels():
    assert all(V.check_labels([1] * 10 + [0] * 10).results.values())
    with pytest.raises(V.ValidationError, match="min_positives"):
        V.check_labels([1] * 3 + [0] * 50)
    with pytest.raises(V.ValidationError, match="min_negatives"):
        V.check_labels([1] * 50 + [0] * 3)
    with pytest.raises(V.ValidationError, match="binary"):
        V.check_labels([1, 2, 0])


def _jobs(ids, urls, titles):
    return pd.DataFrame({"job_id": ids, "url": urls, "company": ["Acme"] * len(ids), "title": titles})


def test_leakage():
    train = _jobs(["a", "b"], ["u/a", "u/b"], ["Data Engineer", "BI Dev"])
    assert V.check_no_leakage(train, _jobs(["c"], ["u/c"], ["Analyst"])).results["leakage.gold_not_in_train"]
    assert V.check_no_leakage(train, pd.DataFrame()).results["leakage.gold_not_in_train"]
    for gold in (_jobs(["a"], ["u/x"], ["X"]),            # same id
                 _jobs([None], ["u/b"], ["X"]),           # same url, eval case without a DB id
                 _jobs(["z"], ["u/z"], ["  data  ENGINEER "])):  # same company + title (a repost)
        with pytest.raises(V.ValidationError, match="leakage"):
            V.check_no_leakage(train, gold)


def test_split_and_unique_rows():
    a = _jobs(["a"], ["u/a"], ["Data Engineer"]).assign(group="acme")
    b = _jobs(["b"], ["u/b"], ["BI Dev"]).assign(company="Other", group="other")
    assert V.check_split(a, b).results["leakage.company_not_split"]
    with pytest.raises(V.ValidationError, match="company_not_split"):
        V.check_split(a, b.assign(group="acme"))
    with pytest.raises(V.ValidationError, match="unique_rows"):
        V.check_unique_rows(pd.DataFrame({"gid": ["x", "x"]}))


def test_predictions():
    assert all(V.check_predictions([0.1, 0.9], ["a", "b"]).results.values())
    for p, ids, check in [([0.1], ["a", "b"], "one_per_job"),
                          ([0.1, 0.2], ["a", "a"], "unique_ids"),
                          ([0.1, np.nan], ["a", "b"], "finite"),
                          ([0.1, 1.2], ["a", "b"], "in_unit_interval"),
                          ([0.3, 0.3, 0.3], ["a", "b", "c"], "not_constant")]:
        with pytest.raises(V.ValidationError, match=check):
            V.check_predictions(p, ids)


def test_class_drift_warns_but_does_not_fail():
    v = V.class_drift(0.5, {"train_pos_rate": 0.1})
    assert v.results["labels.class_drift"] is False and v.warnings
    assert V.class_drift(0.12, {"train_pos_rate": 0.1}).results["labels.class_drift"]
    assert V.class_drift(0.5, None).results["labels.class_drift"]
