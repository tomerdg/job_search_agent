"""The CI quality gate: does a prompt or model change keep scoring quality?

Two kinds of rules, both configurable here:

1. Floors on absolute metrics at the production threshold. They catch a change that breaks scoring.
   "Wanted" means fit or maybe: the digest should show both, so a maybe above the threshold is a hit.
   Set below the v1 baseline (precision 0.885, recall 1.0, AUC 0.948 on 82 cases, 2026-11-15) with room
   for noise: one case moves precision by ~0.04 and fit recall by ~0.07 on this set.
2. No significant regression against the baseline: fails when the paired Wilcoxon test on per-case
   error says "worse" (p < 0.05). A non-significant difference passes; with this set size, small real
   changes are not detectable either way.
"""
from __future__ import annotations

FLOORS = {
    "precision_fit_or_maybe_at_threshold": 0.80,  # share of jobs above the threshold that are fit or maybe
    "recall_fit_or_maybe_at_threshold": 0.90,     # share of fit or maybe jobs that reach the threshold
    "recall_at_threshold": 0.85,                  # share of fit jobs that reach the threshold
    "roc_auc_fit": 0.90,                          # ranking: fit above the rest
}
MIN_COVERAGE = 0.98  # share of cases the agent actually scored (a crash or a skipped job is a failure)
FAIL_ON_SIGNIFICANT_REGRESSION = True


def decide(metrics: dict, info: dict, comparison: dict | None) -> tuple[bool, list[str]]:
    """Return (passed, reasons). Every rule adds one line, passed or failed, so the log always says why."""
    lines, ok = [], True

    def check(passed: bool, text: str) -> None:
        nonlocal ok
        ok &= passed
        lines.append(f"{'PASS' if passed else 'FAIL'}: {text}")

    cov = info.get("coverage", 0.0)
    check(cov >= MIN_COVERAGE, f"coverage {cov:.3f} >= {MIN_COVERAGE}")
    for name, floor in FLOORS.items():
        value = metrics.get(name)
        if value is None or value != value:  # missing or NaN
            check(False, f"{name} is missing")
        else:
            check(value >= floor, f"{name} {value:.3f} >= {floor}")
    if comparison is None:
        lines.append("SKIP: no baseline given, regression check not run")
    elif FAIL_ON_SIGNIFICANT_REGRESSION:
        check(comparison["verdict"] != "worse",
              f"not significantly worse than the baseline (verdict: {comparison['verdict']}, "
              f"Wilcoxon p={comparison['wilcoxon_p']:.3f})")
        for w in comparison.get("warnings", []):
            lines.append(f"NOTE: comparison not batch-paired: {w}")
    return ok, lines


def markdown(passed: bool, lines: list[str]) -> str:
    head = "#### Quality gate: **passed**" if passed else "#### Quality gate: **FAILED**"
    return "\n".join([head, "", *[f"- {l}" for l in lines]]) + "\n"
