"""Scoring-quality metrics. Pure functions over numbers: no API, no files, unit-testable.

Inputs are a predictions DataFrame with columns case_id, repeat, score (0-100) and label
(fit / maybe / no_fit). "Positive" means fit unless a function says otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ORDINAL = {"no_fit": 0, "maybe": 1, "fit": 2}
# What a perfect scorer would give each label: the middle of its band in the prompt.
TARGET_SCORE = {"fit": 90, "maybe": 70, "no_fit": 20}
BANDS = [(0, 29), (30, 59), (60, 84), (85, 100)]  # the bands in prompts/v1.txt


def binary(scores, positive, threshold: float) -> dict:
    """Precision, recall and F1 of `score >= threshold` against a boolean `positive`."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    pred = s >= threshold
    tp, fp, fn = int(np.sum(pred & y)), int(np.sum(pred & ~y)), int(np.sum(~pred & y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def best_threshold(scores, positive) -> dict:
    """Sweep thresholds 0..100 and return the one with the highest F1 (lowest threshold on ties)."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    thresholds = np.arange(0, 101)
    pred = s[None, :] >= thresholds[:, None]            # (101, n)
    tp = (pred & y).sum(1)
    fp = (pred & ~y).sum(1)
    fn = (~pred & y).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    i = int(np.argmax(f1))
    return {"threshold": int(thresholds[i]), **binary(s, y, thresholds[i])}


def roc_auc(scores, positive) -> float:
    """P(score of a random positive > score of a random negative), ties count half (Mann-Whitney U)."""
    s, y = np.asarray(scores, float), np.asarray(positive, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def spearman(scores, labels) -> float:
    rho = stats.spearmanr(np.asarray(scores, float), [ORDINAL[l] for l in labels]).statistic
    return float(rho)


def precision_at_k(df: pd.DataFrame, k: int = 3, day_size: int = 15, seed: int = 0) -> float:
    """Split the cases into random "days" of day_size (fixed seed) and take the top k by score per day,
    like the digest does. Returns the share of those picks that are fit."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    picks = []
    for start in range(0, len(df), day_size):
        day = df.iloc[order[start:start + day_size]]
        picks.append(day.sort_values("score", ascending=False, kind="stable").head(k))
    top = pd.concat(picks)
    return float((top["label"] == "fit").mean()) if len(top) else float("nan")


def band_of(score: float) -> str:
    lo, hi = next(b for b in BANDS if b[0] <= score <= b[1])
    return f"{lo}-{hi}"


def confusion(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: label; columns: the prompt's score bands; values: case counts."""
    bands = df["score"].map(band_of)
    table = pd.crosstab(pd.Categorical(df["label"], list(ORDINAL)[::-1]),
                        pd.Categorical(bands, [f"{lo}-{hi}" for lo, hi in BANDS]), dropna=False)
    table.index.name, table.columns.name = "label", "score_band"
    return table


def calibration(df: pd.DataFrame) -> dict:
    """Mean and std of score per label, and mean absolute error against TARGET_SCORE."""
    out = {}
    for label, g in df.groupby("label"):
        out[f"score_mean_{label}"] = float(g["score"].mean())
        out[f"score_std_{label}"] = float(g["score"].std(ddof=0))
    out["mae_vs_target"] = float(abs_error(df).mean())
    return out


def abs_error(df: pd.DataFrame) -> pd.Series:
    return (df["score"] - df["label"].map(TARGET_SCORE)).abs()


def stability(preds: pd.DataFrame, threshold: float) -> dict:
    """Across repeats: mean per-case score std, and share of cases whose >= threshold decision flips."""
    if preds["repeat"].nunique() < 2:
        return {}
    g = preds.groupby("case_id")["score"]
    above = preds.assign(above=preds["score"] >= threshold).groupby("case_id")["above"]
    return {"score_std_mean": float(g.std(ddof=0).mean()),
            "decision_flip_rate": float((above.nunique() > 1).mean())}


def per_case(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per case: mean score over repeats, and the label."""
    return preds.groupby("case_id", as_index=False).agg(score=("score", "mean"), label=("label", "first"))


def summarize(preds: pd.DataFrame, threshold: float, seed: int = 0) -> dict:
    """All quality metrics for one evaluation run (repeats averaged per case)."""
    df = per_case(preds)
    fit = df["label"].eq("fit")
    considered = df["label"].isin(["fit", "maybe"])
    m = {"n_cases": len(df), "n_fit": int(fit.sum()), "n_maybe": int(df["label"].eq("maybe").sum()),
         "n_no_fit": int(df["label"].eq("no_fit").sum()), "threshold": threshold}
    m.update({f"{k}_at_threshold": v for k, v in binary(df["score"], fit, threshold).items()})
    best = best_threshold(df["score"], fit)
    m.update({"best_threshold": best["threshold"], "f1_at_best": best["f1"]})
    m["recall_fit_or_maybe_at_threshold"] = binary(df["score"], considered, threshold)["recall"]
    m["roc_auc_fit"] = roc_auc(df["score"], fit)
    m["spearman"] = spearman(df["score"], df["label"])
    m["precision_at_3"] = precision_at_k(df, 3, 15, seed)
    m.update(calibration(df))
    m.update(stability(preds, threshold))
    return m


def compare(current: pd.DataFrame, baseline: pd.DataFrame, threshold: float,
            n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired comparison on the cases both runs share.

    - Wilcoxon signed-rank test on per-case absolute error vs TARGET_SCORE (lower is better).
    - Bootstrap 95% CI of the difference in F1 at the threshold (current - baseline), resampling cases.
    verdict is "better" / "worse" when the Wilcoxon test is significant, else "no significant difference".
    """
    cur, base = per_case(current), per_case(baseline)
    both = cur.merge(base[["case_id", "score"]], on="case_id", suffixes=("", "_base"))
    err_cur = abs_error(both)
    err_base = abs_error(both.assign(score=both["score_base"]))
    diff = err_cur - err_base
    if (diff != 0).any():
        p = float(stats.wilcoxon(err_cur, err_base, zero_method="wilcox").pvalue)
    else:
        p = 1.0
    y = both["label"].eq("fit").to_numpy()
    s_cur, s_base = both["score"].to_numpy(), both["score_base"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(both), size=(n_boot, len(both)))
    deltas = np.array([binary(s_cur[i], y[i], threshold)["f1"] - binary(s_base[i], y[i], threshold)["f1"]
                       for i in idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    if p < alpha:
        verdict = "better" if diff.mean() < 0 else "worse"
    else:
        verdict = "no significant difference"
    return {"n_paired": len(both), "mae_delta": float(diff.mean()), "wilcoxon_p": p,
            "f1_delta": binary(s_cur, y, threshold)["f1"] - binary(s_base, y, threshold)["f1"],
            "f1_delta_ci_low": float(lo), "f1_delta_ci_high": float(hi), "verdict": verdict}
