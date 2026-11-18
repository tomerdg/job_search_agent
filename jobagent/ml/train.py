"""Offline training and evaluation of the relevance model.

    python -m jobagent.ml.train --report             # local DB + local eval set, MLflow, report
    python -m jobagent.ml.train --demo --db data/demo.duckdb --baseline jobagent/ml/baselines/demo.json --gate
                                                     # synthetic smoke run with the regression gate (CI)

Labels, binary target `fit_or_maybe`:
    gold    the owner's own judgement: eval-set labels (fit/maybe -> 1, no_fit -> 0) and job_feedback rows
            with source='cli' (applied -> 1, skipped/irrelevant -> 0).
    silver  the LLM score on scored jobs (score >= MIN_SCORE -> 1).
    weak    jobs whose title/description miss every pre-filter keyword -> 0 (location misses are not used:
            a job in the wrong city can still be a good role).
A DB job that is also gold (same job id, url, or company + title) appears once, with its gold label.

Evaluation is cross-validated: repeated GroupKFold by company over all labels, so a company is never in both
the training and the test part of a split (validate.check_split enforces it, per split). Every reported
metric is computed on gold rows only, from their out-of-fold predictions (averaged over repeats).
The 95%-recall threshold is chosen inside each training part (inner GroupKFold, gold rows of the training
part) and applied to the test part, so recall and filtered share are not tuned on the jobs they are measured
on. The final models are fit on all labels.

Models, both on one scipy.sparse matrix (dense features one-hot/scaled + TF-IDF):
    logreg   logistic regression, class_weight=balanced, all TF-IDF terms (words + char n-grams for Hebrew)
    xgboost  shallow gradient boosting (max_depth 3) on the 200 TF-IDF terms with the highest chi2
Baselines on gold: the keyword pre-filter and the LLM score (no training, so their thresholds are chosen on
gold itself and marked as such).

Metrics on gold: ROC AUC and PR AUC with bootstrap 95% CIs (scipy.stats.bootstrap), recall, recall of fit and
share of jobs filtered (= LLM calls saved) at the nested threshold. LR vs XGBoost: paired bootstrap of the AUC
difference on the same out-of-fold predictions, and a Wilcoxon signed-rank test on per-fold gold AUCs.

Outputs: metrics.json (aggregates only) and an optional markdown report under eval/out/ (git-ignored),
an MLflow run in experiment "relevance-model". In CI only aggregates are logged: no model, no feature
weights, no per-job scores. Locally the better model is also registered as "relevance-model".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from jobagent.db import DEFAULT_MIN_SCORE
from jobagent.filters import PreFilter
from jobagent.ml import validate as V
from jobagent.ml.features import CATEGORICAL, NUMERIC, build_features, load_jobs

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "relevance-model"
REGISTERED_MODEL = "relevance-model"
OUT_DIR = ROOT / "eval" / "out" / "relevance"
TARGET_RECALL = 0.95
GATE_TOLERANCE = 0.05  # a gold AUC may drop this much below the baseline before the gate fails
GATED = ["roc_auc", "pr_auc"]
MODELS = ("logreg", "xgboost")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------- labels
def _key(company: pd.Series, title: pd.Series) -> pd.Series:
    norm = lambda s: s.fillna("").astype(str).str.lower().str.split().str.join(" ")
    return norm(company) + "|" + norm(title)


def gold_from_eval(cases: list[dict], labels: dict[str, dict], jobs: pd.DataFrame) -> pd.DataFrame:
    """Eval cases as rows with their own text. source and job_id come from the matching DB job (by url)."""
    df = pd.DataFrame(cases)
    if df.empty:
        return df
    df["label"] = df["case_id"].map(lambda c: (labels.get(c) or {}).get("label", ""))
    df = df[df["label"].isin(["fit", "maybe", "no_fit"])].copy()
    by_url = jobs.dropna(subset=["url"]).drop_duplicates("url").set_index("url")
    df["source"] = df["url"].map(by_url["source"]).fillna("unknown")
    df["job_id"] = df["url"].map(by_url["job_id"])
    df["y"] = df["label"].isin(["fit", "maybe"]).astype(int)
    df["is_fit"] = df["label"].eq("fit").astype(int)
    df["origin"] = "eval"
    return df


def gold_from_feedback(feedback: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    fb = feedback[feedback["source"].eq("cli")]
    df = jobs.merge(fb[["job_id", "decision"]], on="job_id")
    df["y"] = df["decision"].eq("applied").astype(int)
    df["is_fit"] = df["y"]
    df["label"] = np.where(df["y"].eq(1), "fit", "no_fit")
    df["origin"] = "feedback"
    return df


def gold_rows(jobs, feedback, eval_cases=None, eval_labels=None) -> pd.DataFrame:
    gold = pd.concat([gold_from_eval(eval_cases or [], eval_labels or {}, jobs),
                      gold_from_feedback(feedback, jobs)], ignore_index=True)
    if gold.empty:
        return gold
    # Eval and feedback can name the same job: keep the owner's later word (feedback, concatenated last).
    gold = gold[~(gold["job_id"].notna() & gold.duplicated("job_id", keep="last"))].reset_index(drop=True)
    gold["gid"] = gold["job_id"].fillna(gold["case_id"]) if "case_id" in gold else gold["job_id"]
    return gold


def training_rows(jobs: pd.DataFrame, gold: pd.DataFrame, profile: dict,
                  min_score: int = DEFAULT_MIN_SCORE) -> pd.DataFrame:
    """Silver (LLM score) + weak negatives (no role keyword), with every gold job removed."""
    pf = PreFilter(profile)
    df = jobs.copy()
    scored = df["score"].notna().to_numpy()
    no_role = ~np.array([pf.matches_role(t or "", d or "") for t, d in zip(df["title"], df["description"])], bool)
    df["tier"] = np.select([scored, no_role], ["silver", "weak_negative"], default="")
    df["y"] = np.where(scored, (df["score"].fillna(0) >= min_score).astype(int), 0)

    in_gold = np.zeros(len(df), bool)
    if not gold.empty:
        in_gold = (df["job_id"].isin(gold["job_id"].dropna()) | df["url"].isin(gold["url"].dropna())
                   | _key(df["company"], df["title"]).isin(_key(gold["company"], gold["title"]))).to_numpy()
    return df[(df["tier"] != "") & ~in_gold].reset_index(drop=True)


def labeled_rows(jobs: pd.DataFrame, feedback: pd.DataFrame, profile: dict, eval_cases=None,
                 eval_labels=None) -> pd.DataFrame:
    """All labels in one frame: gold, then silver and weak negatives that are not gold.
    Columns used later: tier, y, is_fit, gid (unique row id), group (company, the CV group)."""
    gold = gold_rows(jobs, feedback, eval_cases, eval_labels)
    gold = gold.assign(tier="gold") if len(gold) else gold
    proxy = training_rows(jobs, gold, profile)
    proxy = proxy.assign(is_fit=proxy["y"], gid=proxy["job_id"])
    cols = ["gid", "job_id", "url", "company", "title", "location", "department", "description", "source",
            "score", "tier", "y", "is_fit", "case_id"]
    parts = [df.reindex(columns=cols) for df in (gold, proxy) if len(df)]
    df = pd.concat(parts, ignore_index=True)
    df["group"] = df["company"].fillna("").astype(str).str.lower().str.split().str.join(" ")
    return df


# ---------------------------------------------------------------- feature matrix
class SparseFeatures(BaseEstimator, TransformerMixin):
    """Dense features (one-hot categoricals, optionally scaled numerics) next to TF-IDF of the job text,
    stacked into one CSR matrix with scipy.sparse.hstack. k_best keeps only the k TF-IDF columns with the
    highest chi2 against the labels (for the tree model, which does badly on 50k sparse columns)."""

    def __init__(self, scale: bool = True, k_best: int | None = None, min_df: int = 2):
        self.scale, self.k_best, self.min_df = scale, k_best, min_df

    def _vectorizers(self):
        return [TfidfVectorizer(min_df=self.min_df, max_features=20000, ngram_range=(1, 2), sublinear_tf=True),
                # char_wb n-grams cope with Hebrew prefixes (ה, ב, ל, ו) glued to the word.
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=self.min_df, max_features=30000,
                                sublinear_tf=True)]

    def fit(self, X: pd.DataFrame, y=None):
        self.onehot_ = OneHotEncoder(handle_unknown="ignore").fit(X[CATEGORICAL])
        self.scaler_ = StandardScaler().fit(X[NUMERIC]) if self.scale else None
        self.tfidf_ = [v.fit(X["text"]) for v in self._vectorizers()]
        self.text_cols_ = None
        if self.k_best is not None and y is not None:
            tf = sparse.hstack([v.transform(X["text"]) for v in self.tfidf_], format="csr")
            scores = np.nan_to_num(chi2(tf, y)[0])
            k = min(self.k_best, tf.shape[1])
            self.text_cols_ = np.sort(np.argsort(scores, kind="stable")[::-1][:k])
        return self

    def transform(self, X: pd.DataFrame) -> sparse.csr_matrix:
        num = X[NUMERIC].to_numpy(dtype=float)
        if self.scaler_ is not None:
            num = self.scaler_.transform(X[NUMERIC])
        text = sparse.hstack([v.transform(X["text"]) for v in self.tfidf_], format="csr")
        if self.text_cols_ is not None:
            text = text[:, self.text_cols_]
        return sparse.hstack([self.onehot_.transform(X[CATEGORICAL]), sparse.csr_matrix(num), text], format="csr")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        terms = np.concatenate([[f"word:{t}" for t in self.tfidf_[0].get_feature_names_out()],
                                [f"char:{t}" for t in self.tfidf_[1].get_feature_names_out()]])
        if self.text_cols_ is not None:
            terms = terms[self.text_cols_]
        return np.concatenate([self.onehot_.get_feature_names_out(CATEGORICAL), NUMERIC, terms])


def make_logreg() -> Pipeline:
    return Pipeline([("features", SparseFeatures(scale=True)),
                     ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000))])


def make_xgboost(seed: int = 0) -> Pipeline:
    from xgboost import XGBClassifier  # training-only dependency, imported when used
    return Pipeline([("features", SparseFeatures(scale=False, k_best=200)),
                     ("clf", XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.9,
                                           colsample_bytree=0.8, eval_metric="logloss", random_state=seed,
                                           n_jobs=2))])


def make_models(seed: int = 0) -> dict[str, Pipeline]:
    return {"logreg": make_logreg(), "xgboost": make_xgboost(seed)}


def _fit(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    m = clone(model)
    if "xgboost" in type(m.named_steps["clf"]).__module__:
        pos = max(int(y.sum()), 1)
        m.set_params(clf__scale_pos_weight=(len(y) - pos) / pos)
    return m.fit(X, y)


def frame(df: pd.DataFrame, cv_text: str, profile: dict) -> pd.DataFrame:
    dense, text = build_features(df, cv_text, profile)
    return dense.assign(text=text)


def predict(model: Pipeline, df: pd.DataFrame, cv_text: str, profile: dict,
            v: V.Validation | None = None) -> np.ndarray:
    """P(fit or maybe) per job, with input and output validation."""
    v = v or V.Validation()
    X = frame(df, cv_text, profile)
    V.check_features(X, v, prefix="predict_features")
    p = model.predict_proba(X)[:, 1]
    ids = df["gid"] if "gid" in df else df["job_id"]
    V.check_predictions(p, ids, v)
    return p


# ---------------------------------------------------------------- metrics
def threshold_for_recall(scores, y, target: float = TARGET_RECALL) -> float:
    """Highest threshold whose recall (score >= threshold) is still >= target."""
    s, y = np.asarray(scores, float), np.asarray(y, bool)
    if not y.any():
        return float("nan")
    pos = np.sort(s[y])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1])


def at_threshold(scores, y, is_fit, t: float) -> dict:
    s, y, f = np.asarray(scores, float), np.asarray(y, bool), np.asarray(is_fit, bool)
    keep = s >= t
    return {"threshold": t,
            "recall": float(keep[y].mean()) if y.any() else float("nan"),
            "recall_fit": float(keep[f].mean()) if f.any() else float("nan"),
            "precision": float(y[keep].mean()) if keep.any() else float("nan"),
            "filtered_share": float(1 - keep.mean())}


def _roc(y, s) -> float:
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def _pr(y, s) -> float:
    return float(average_precision_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def bootstrap_ci(y, s, metric, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of metric(y, s), resampling gold jobs in pairs (scipy.stats.bootstrap).
    Resamples with a single class give NaN and are ignored."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # DegenerateDataWarning for NaN resamples, handled below
        res = stats.bootstrap((np.asarray(y), np.asarray(s, float)), metric, paired=True, vectorized=False,
                              n_resamples=n_boot, method="percentile", random_state=np.random.default_rng(seed))
    lo, hi = np.nanpercentile(res.bootstrap_distribution, [2.5, 97.5])
    return float(lo), float(hi)


def paired_auc_difference(y, s1, s2, n_boot: int = 1000, seed: int = 0) -> dict:
    """ROC AUC of s1 minus s2 on the same jobs: point estimate, paired bootstrap 95% CI, two-sided p-value."""
    diff = lambda yy, a, b: _roc(yy, a) - _roc(yy, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.bootstrap((np.asarray(y), np.asarray(s1, float), np.asarray(s2, float)), diff, paired=True,
                              vectorized=False, n_resamples=n_boot, method="percentile",
                              random_state=np.random.default_rng(seed))
    d = res.bootstrap_distribution[np.isfinite(res.bootstrap_distribution)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    return {"auc_diff": diff(np.asarray(y), s1, s2), "auc_diff_ci_low": float(lo), "auc_diff_ci_high": float(hi),
            "auc_diff_p": float(p)}


def evaluate_scores(gold: pd.DataFrame, scores, n_boot: int = 1000, seed: int = 0) -> dict:
    s = pd.Series(np.asarray(scores, float), index=gold.index)
    ok = s.notna()
    g, s = gold[ok], s[ok].to_numpy()
    y = g["y"].to_numpy()
    out = {"n_gold": int(len(g)), "n_pos": int(y.sum()), "roc_auc": _roc(y, s), "pr_auc": _pr(y, s)}
    out["roc_auc_ci_low"], out["roc_auc_ci_high"] = bootstrap_ci(y, s, _roc, n_boot, seed)
    out["pr_auc_ci_low"], out["pr_auc_ci_high"] = bootstrap_ci(y, s, _pr, n_boot, seed)
    t = threshold_for_recall(s, y)
    out.update({f"{k}_at_r95": v for k, v in at_threshold(s, y, g["is_fit"], t).items()})
    return out


def _splits(groups: pd.Series, n_splits: int, seed: int):
    n_splits = min(n_splits, groups.nunique())
    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(groups, groups=groups)


def nested_threshold(model: Pipeline, X: pd.DataFrame, rows: pd.DataFrame, n_splits: int = 3,
                     seed: int = 0) -> float:
    """The 95%-recall threshold, chosen only from a training part: inner GroupKFold out-of-fold scores of
    its gold rows (all its rows when it has fewer than 5 gold positives)."""
    y = rows["y"].to_numpy()
    oof = np.full(len(rows), np.nan)
    for tr, te in _splits(rows["group"], n_splits, seed):
        if len(np.unique(y[tr])) == 2:
            oof[te] = _fit(model, X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    use = np.isfinite(oof)
    gold = (rows["tier"] == "gold").to_numpy()
    if (y[use & gold] == 1).sum() >= 5:
        use &= gold
    return threshold_for_recall(oof[use], y[use])


def cross_validate(models: dict[str, Pipeline], X: pd.DataFrame, rows: pd.DataFrame, v: V.Validation,
                   n_splits: int = 5, repeats: int = 3, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Repeated GroupKFold by company over all labels.
    Returns: out-of-fold scores of gold rows (mean over repeats, one column per model); per-fold gold AUCs
    (paired across models); and pooled at-threshold counts per model using each fold's nested threshold."""
    y = rows["y"].to_numpy()
    gold = (rows["tier"] == "gold").to_numpy()
    oof = {name: np.zeros(len(rows)) for name in models}
    hits = {name: np.zeros(len(rows)) for name in models}         # how often a row was kept (>= threshold)
    seen = np.zeros(len(rows))
    folds = []
    for r in range(repeats):
        for k, (tr, te) in enumerate(_splits(rows["group"], n_splits, seed + r)):
            V.check_split(rows.iloc[tr], rows.iloc[te], v)
            if len(np.unique(y[tr])) < 2:
                continue
            fold = {}
            for name, m in models.items():
                fitted = _fit(m, X.iloc[tr], y[tr])
                p = fitted.predict_proba(X.iloc[te])[:, 1]
                V.check_predictions(p, rows["gid"].iloc[te], v, name="cv_predictions")
                t = nested_threshold(m, X.iloc[tr], rows.iloc[tr], seed=seed + r)
                oof[name][te] += p
                hits[name][te] += p >= t
                g = gold[te]
                fold[name] = _roc(y[te][g], p[g]) if g.any() else np.nan
            seen[te] += 1
            folds.append(fold)
    ok = gold & (seen > 0)
    scores = pd.DataFrame({name: oof[name][ok] / seen[ok] for name in models}, index=rows.index[ok])
    kept = pd.DataFrame({name: hits[name][ok] / seen[ok] for name in models}, index=rows.index[ok])
    return scores, pd.DataFrame(folds, columns=list(models)), kept


def nested_at_threshold(gold: pd.DataFrame, kept: pd.Series) -> dict:
    """Recall, recall of fit, precision and filtered share, where kept = share of repeats in which the
    row scored above its fold's nested threshold (averaging over repeats)."""
    y, f, k = gold["y"].to_numpy(bool), gold["is_fit"].to_numpy(bool), kept.to_numpy(float)
    return {"threshold_nested": True,
            "recall": float(k[y].mean()) if y.any() else float("nan"),
            "recall_fit": float(k[f].mean()) if f.any() else float("nan"),
            "precision": float((k * y).sum() / k.sum()) if k.sum() else float("nan"),
            "filtered_share": float(1 - k.mean())}


def evaluate_scores(gold: pd.DataFrame, scores, n_boot: int = 1000, seed: int = 0,
                    kept: pd.Series | None = None) -> dict:
    """AUCs with bootstrap CIs; at-threshold numbers from the nested `kept` shares when given, else at the
    threshold chosen on these same scores (marked threshold_nested = False)."""
    s = pd.Series(np.asarray(scores, float), index=gold.index)
    ok = s.notna()
    g, s = gold[ok], s[ok].to_numpy()
    y = g["y"].to_numpy()
    out = {"n_gold": int(len(g)), "n_pos": int(y.sum()), "roc_auc": _roc(y, s), "pr_auc": _pr(y, s)}
    out["roc_auc_ci_low"], out["roc_auc_ci_high"] = bootstrap_ci(y, s, _roc, n_boot, seed)
    out["pr_auc_ci_low"], out["pr_auc_ci_high"] = bootstrap_ci(y, s, _pr, n_boot, seed)
    if kept is not None:
        at = nested_at_threshold(g, kept[ok.to_numpy()])
    else:
        at = {"threshold_nested": False, **at_threshold(s, y, g["is_fit"], threshold_for_recall(s, y))}
    out.update({f"{k}_at_r95": val for k, val in at.items()})
    return out


def compare_folds(folds: pd.DataFrame) -> dict:
    """Mean/std of per-fold gold AUC per model, and a Wilcoxon signed-rank test of logreg vs xgboost on them
    (same folds, so paired)."""
    folds = folds.dropna()
    out = {"cv_folds": int(len(folds))}
    for name in folds.columns:
        out[f"{name}.fold_roc_auc_mean"] = float(folds[name].mean()) if len(folds) else float("nan")
        out[f"{name}.fold_roc_auc_std"] = float(folds[name].std(ddof=1)) if len(folds) > 1 else float("nan")
    d = (folds["logreg"] - folds["xgboost"]).to_numpy() if len(folds) else np.array([])
    out["cv_wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if np.count_nonzero(d) >= 2 else float("nan")
    return out


def dataset_hash(rows: pd.DataFrame) -> str:
    """SHA-1 of the sorted (row id, tier, label) triples: same labels in, same hash."""
    triples = rows[["gid", "tier", "y"]].astype(str).sort_values(["gid", "tier"]).to_numpy()
    return hashlib.sha1("\n".join(map(",".join, triples)).encode()).hexdigest()


# ---------------------------------------------------------------- pipeline
def run(jobs: pd.DataFrame, feedback: pd.DataFrame, cv_text: str, profile: dict,
        eval_cases: list[dict] | None = None, eval_labels: dict | None = None,
        llm_eval_scores: dict[str, float] | None = None, cv_repeats: int = 3, n_boot: int = 1000,
        seed: int = 0, min_positives: int = V.MIN_POSITIVES, previous: dict | None = None,
        n_splits: int = 5) -> dict:
    """Cross-validate both models on all labels, report on gold rows, then fit the final models on all
    labels. Validation failures raise V.ValidationError. Returns metrics (aggregates), the comparison table,
    validation results, and the fitted models and gold predictions (those two stay local)."""
    v = V.Validation()
    rows = labeled_rows(jobs, feedback, profile, eval_cases, eval_labels)
    V.check_unique_rows(rows, v)
    X, y = frame(rows, cv_text, profile), rows["y"].to_numpy()
    V.check_features(X, v)
    V.check_labels(y, v, min_positives=min_positives, min_negatives=min_positives)
    V.class_drift(float(y.mean()), previous, v)

    tier = rows["tier"]
    gold = rows[tier == "gold"]
    metrics = {
        "n_labeled": len(rows), "n_gold": int(len(gold)), "n_gold_pos": int(gold["y"].sum()),
        "n_gold_eval": int(gold["case_id"].notna().sum()), "n_gold_feedback": int(gold["case_id"].isna().sum()),
        "n_silver": int((tier == "silver").sum()), "n_silver_pos": int(((tier == "silver") & (rows["y"] == 1)).sum()),
        "n_weak_negative": int((tier == "weak_negative").sum()), "n_companies": int(rows["group"].nunique()),
        "train_pos_rate": float(y.mean()),
    }
    models = make_models(seed)
    preds, folds, kept = cross_validate(models, X, rows, v, n_splits=n_splits, repeats=cv_repeats, seed=seed)
    metrics.update(compare_folds(folds))

    table, gold_eval = pd.DataFrame(), gold.loc[preds.index]
    if len(gold_eval) and gold_eval["y"].nunique() == 2:
        out = {name: evaluate_scores(gold_eval, preds[name], n_boot, seed, kept[name]) for name in models}
        pf = PreFilter(profile)
        preds["prefilter"] = [float(pf(t, d or "", l or "")) for t, d, l
                              in zip(gold_eval["title"], gold_eval["description"], gold_eval["location"])]
        llm = gold_eval["case_id"].map(llm_eval_scores or {}).astype(float)
        db_score = jobs.drop_duplicates("job_id").set_index("job_id")["score"].astype(float)
        llm = llm.fillna(gold_eval["job_id"].map(db_score))
        if llm.notna().any():
            preds["llm_score"] = llm.astype(float)
        for name in ("prefilter", "llm_score"):
            if name in preds:
                out[name] = evaluate_scores(gold_eval, preds[name], n_boot, seed)
        paired = paired_auc_difference(gold_eval["y"], preds["logreg"], preds["xgboost"], n_boot, seed)
        metrics.update({f"logreg_vs_xgboost.{k}": val for k, val in paired.items()})
        for name, row in out.items():
            metrics.update({f"{name}.{k}": val for k, val in row.items()})
        table = pd.DataFrame.from_dict(out, orient="index")
        for name in MODELS:
            for k in ("fold_roc_auc_mean", "fold_roc_auc_std"):
                table.loc[name, k] = metrics[f"{name}.{k}"]
    else:
        log.warning("No gold rows with both classes: no gold metrics")

    fitted = {name: _fit(m, X, y) for name, m in models.items()}           # final models: all labels
    return {"metrics": metrics, "table": table, "validation": v, "models": fitted, "gold": gold_eval,
            "preds": preds, "dataset_hash": dataset_hash(rows),
            "params": {f"{n}.{k}": val for n, m in models.items()
                       for k, val in m.named_steps["clf"].get_params().items()
                       if isinstance(val, (int, float, str, bool))}}


# ---------------------------------------------------------------- gate
def gate(metrics: dict, baseline: dict | None, tolerance: float = GATE_TOLERANCE) -> tuple[bool, list[str]]:
    """Fail when a model's gold ROC/PR AUC drops more than `tolerance` below the baseline. Like eval/gate.py,
    every rule prints one PASS/FAIL line."""
    if baseline is None:
        return True, ["SKIP: no baseline given"]
    base_metrics = baseline.get("metrics", baseline)
    ok, lines = True, []
    for model in MODELS:
        for m in GATED:
            key = f"{model}.{m}"
            cur, base = metrics.get(key), base_metrics.get(key)
            if base is None:
                lines.append(f"SKIP: {key} not in baseline")
                continue
            passed = cur is not None and cur == cur and cur >= base - tolerance
            ok &= passed
            lines.append(f"{'PASS' if passed else 'FAIL'}: {key} {cur if cur is not None else float('nan'):.3f}"
                         f" >= {base:.3f} - {tolerance}")
    return ok, lines


def metrics_json(result: dict, gate_lines: list[str] | None = None, passed: bool | None = None) -> dict:
    def clean(x):
        x = x.item() if isinstance(x, np.generic) else x
        return None if isinstance(x, float) and not np.isfinite(x) else x
    return {"metrics": {k: clean(v) for k, v in result["metrics"].items()},
            "validation": result["validation"].results, "warnings": result["validation"].warnings,
            "dataset_hash": result["dataset_hash"], "gate": {"passed": passed, "lines": gate_lines or []}}


# ---------------------------------------------------------------- outputs
def pr_curve_plot(gold: pd.DataFrame, preds: pd.DataFrame, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    for name in preds.columns:
        ok = preds[name].notna()
        if gold.loc[ok, "y"].nunique() == 2:
            p, r, _ = precision_recall_curve(gold.loc[ok, "y"], preds.loc[ok, name])
            ax.step(r, p, where="post", label=name)
    ax.axvline(TARGET_RECALL, color="grey", ls=":", lw=1)
    ax.set_xlabel("recall (fit or maybe)")
    ax.set_ylabel("precision")
    ax.set_title("Gold set: precision-recall")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "pr_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def report_markdown(result: dict, gate_lines: list[str]) -> str:
    m, t = result["metrics"], result["table"]
    counts = [k for k in m if k.startswith("n_") or k == "train_pos_rate"]
    paired = [k for k in m if k.startswith(("logreg_vs_xgboost", "cv_")) or ".fold_" in k]
    v = result["validation"]
    lines = ["# Relevance model report", "",
             "Aggregates only. Labels = the owner's (gold) + LLM score (silver) + keyword negatives. Metrics are "
             "on gold rows only, from out-of-fold predictions of repeated GroupKFold by company.",
             "", "| | |", "|---|---|", *[f"| {k} | {m[k]:.3g} |" for k in counts], "",
             f"Threshold = highest score that keeps recall (fit or maybe) >= {TARGET_RECALL}; for the two models it is "
             "chosen inside each training fold (`threshold_nested_at_r95` = 1), for the baselines on gold itself. "
             "`filtered_share_at_r95` = jobs below it = LLM calls saved. CIs: bootstrap 95%.", "",
             "```", t.round(3).T.to_string(), "```", "",
             "LR vs XGBoost:", "", *[f"- {k}: {m[k]:.3g}" for k in paired], "",
             "Validation:", "", *[f"- {k}: {'pass' if ok else 'FAIL'}" for k, ok in v.results.items()],
             *[f"- warning: {w}" for w in v.warnings], "",
             "Gate:", "", *[f"- {l}" for l in gate_lines]]
    return "\n".join(lines) + "\n"


def top_features(lr: Pipeline, k: int = 30) -> pd.DataFrame:
    names = lr.named_steps["features"].get_feature_names_out()
    coef = lr.named_steps["clf"].coef_[0]
    idx = np.r_[np.argsort(coef)[:k], np.argsort(coef)[-k:]]
    return pd.DataFrame({"feature": names[idx], "coef": coef[idx]})


def log_mlflow(result: dict, params: dict, gate_passed: bool | None) -> str:
    import mlflow
    from eval.tracking import git_sha, in_ci, tracking_uri

    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        uri = tracking_uri()
        location = (Path(uri.removeprefix("sqlite:///")).parent / "mlruns").as_uri() \
            if uri.startswith("sqlite:///") else None
        mlflow.create_experiment(EXPERIMENT, artifact_location=location)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=params.get("run_name", "relevance")) as run:
        mlflow.log_params({**params, **result["params"], "dataset_hash": result["dataset_hash"],
                           "git_sha": git_sha()})
        mlflow.log_metrics({k: float(v) for k, v in result["metrics"].items()
                            if isinstance(v, (int, float, np.number)) and np.isfinite(v)})
        mlflow.set_tags({"trigger": "ci" if in_ci() else "local", **result["validation"].tags(),
                         "gate": "skip" if gate_passed is None else "pass" if gate_passed else "fail"})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result["table"].to_csv(out / "comparison.csv")                       # aggregates only
            (out / "metrics.json").write_text(json.dumps(metrics_json(result), indent=2), encoding="utf-8")
            files = [out / "comparison.csv", out / "metrics.json"]
            if len(result["preds"]):
                files.append(pr_curve_plot(result["gold"], result["preds"], out))
            if not in_ci():  # feature weights can echo personal data: local store only
                top_features(result["models"]["logreg"]).to_csv(out / "logreg_top_features.csv", index=False)
                files.append(out / "logreg_top_features.csv")
            for f in files:
                mlflow.log_artifact(str(f))
        if not in_ci():  # models: local store and registry only, never in CI
            best = max(MODELS, key=lambda n: np.nan_to_num(result["metrics"].get(f"{n}.pr_auc", np.nan), nan=-1.0))
            for name, m in result["models"].items():
                mlflow.sklearn.log_model(
                    m, name=name, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    registered_model_name=REGISTERED_MODEL if name == best and not params.get("demo") else None)
            mlflow.set_tag("registered_model", best)
        return run.info.run_id


# ---------------------------------------------------------------- CLI
def _load_eval() -> tuple[list[dict], dict, dict[str, float]]:
    """Eval cases, labels and the LLM baseline's mean score per case; empty when not readable here."""
    from eval import dataset as ds
    try:
        cases, labels = ds.load_jobs(), ds.load_labels()
    except (FileNotFoundError, RuntimeError, OSError) as e:
        log.info("No eval set (%s); gold comes from feedback only", type(e).__name__)
        return [], {}, {}
    try:
        preds = pd.DataFrame(json.loads(ds.read_text(ds.DATASET_DIR / "baseline.json"))["predictions"])
        llm = preds.groupby("case_id")["score"].mean().to_dict()
    except (FileNotFoundError, RuntimeError, OSError, KeyError):
        llm = {}
    return cases, labels, llm


def _read_json(path: Path | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else None


def main(argv=None) -> int:
    from jobagent.main import load_dotenv
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Train and evaluate the relevance model")
    ap.add_argument("--db", default=os.environ.get("JOBS_DB", str(ROOT / "data" / "jobs.duckdb")))
    ap.add_argument("--demo", action="store_true", help="synthetic CV and profile, no eval set (for demo DBs)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="folder for metrics.json and the report")
    ap.add_argument("--report", action="store_true", help="also write relevance_report.md")
    ap.add_argument("--baseline", type=Path, help="metrics.json of an earlier run to gate against")
    ap.add_argument("--gate", action="store_true", help="exit 1 when the gate fails")
    ap.add_argument("--write-baseline", type=Path, help="save this run's metrics.json here too")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--min-positives", type=int, default=V.MIN_POSITIVES)
    ap.add_argument("--cv-repeats", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore", message=r"k=\d+ is greater than n_features")

    if args.demo:
        from jobagent.demo import DEMO_CV, DEMO_PROFILE
        cv_text, profile, (cases, labels, llm) = DEMO_CV, DEMO_PROFILE, ([], {}, {})
    else:
        import yaml
        from eval.evaluate import load_cv_text, load_profile
        has_profile = os.environ.get("PROFILE_YAML") or (ROOT / "config" / "profile.yaml").exists()
        profile = load_profile() if has_profile else \
            yaml.safe_load((ROOT / "config" / "profile.example.yaml").read_text(encoding="utf-8"))
        cv_text = load_cv_text(profile, None)
        cases, labels, llm = _load_eval()

    import duckdb
    con = duckdb.connect(args.db, read_only=True)
    try:
        jobs = load_jobs(con)
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        feedback = con.execute("SELECT job_id, decision, source FROM job_feedback").df() \
            if "job_feedback" in tables else pd.DataFrame(columns=["job_id", "decision", "source"])
    finally:
        con.close()

    args.out.mkdir(parents=True, exist_ok=True)
    previous = _read_json(args.out / "metrics.json")
    try:
        result = run(jobs, feedback, cv_text, profile, cases, labels, llm, args.cv_repeats, args.bootstrap,
                     args.seed, args.min_positives, (previous or {}).get("metrics"))
    except V.ValidationError as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        return 2

    passed, lines = gate(result["metrics"], _read_json(args.baseline))
    out = metrics_json(result, lines, passed if args.baseline else None)
    text = json.dumps(out, indent=2) + "\n"
    (args.out / "metrics.json").write_text(text, encoding="utf-8")
    if args.write_baseline:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(text, encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(result["table"].round(3).T)
    print("\n".join(lines))
    if not args.no_mlflow:
        params = {"run_name": "demo" if args.demo else "relevance", "seed": args.seed, "demo": args.demo,
                  "cv_repeats": args.cv_repeats, "bootstrap": args.bootstrap, "target_recall": TARGET_RECALL,
                  "min_positives": args.min_positives}
        print("mlflow run:", log_mlflow(result, params, passed if args.baseline else None))
    if args.report:
        (args.out / "relevance_report.md").write_text(report_markdown(result, lines), encoding="utf-8")
        print("report:", args.out / "relevance_report.md")
    return 1 if args.gate and not passed else 0


if __name__ == "__main__":
    sys.exit(main())
