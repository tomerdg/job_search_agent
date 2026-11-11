"""Draft an evaluation set from the local job database, for labeling by hand.

    python scripts/export_eval_candidates.py --db data/jobs.duckdb --n 60

Writes eval/out/jobs.jsonl and eval/out/labels.csv (blank labels). eval/out/ is git-ignored.
Review the jobs, fill in the labels, then copy both files into eval/dataset/ and run
`python -m eval.dataset encrypt` (see eval/README.md).

The sample is stratified so the set has hard cases, not just easy negatives:
  scored_*          jobs the agent already scored, by score band (the bands in the prompt)
  role_bad_location matches the role keywords but the pre-filter dropped it for its location
  data_title        a data-ish title that the role keywords missed
  other             everything else (easy negatives)
Within a stratum, jobs are picked round-robin across companies so no company dominates.

The draft deliberately leaves out the agent's score and analysis: seeing the model's answer
while labeling anchors the label to it, and then the evaluation measures nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jobagent.agent import JD_MAX_CHARS  # noqa: E402
from jobagent.filters import PreFilter  # noqa: E402

BANDS = [(0, 29), (30, 59), (60, 84), (85, 100)]
DATA_TITLE = re.compile(r"data|\bBI\b|analy|SQL|ETL|נתונים|דאטה|אנליסט|ביג", re.I)
# Share of the set per stratum before redistribution of what a stratum cannot fill.
QUOTAS = {"scored_85_100": .15, "scored_60_84": .15, "scored_30_59": .15, "scored_0_29": .10,
          "role_bad_location": .15, "data_title": .15, "other": .15}


def load_profile() -> dict:
    text = os.environ.get("PROFILE_YAML")
    if not text:
        path = ROOT / "config" / "profile.yaml"
        if not path.exists():
            path = ROOT / "config" / "profile.example.yaml"
        text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def stratum(row: dict, pf: PreFilter) -> str:
    if row["score"] is not None:
        lo, hi = next(b for b in BANDS if b[0] <= row["score"] <= b[1])
        return f"scored_{lo}_{hi}"
    if pf.matches_role(row["title"], row["description"]) and not pf.location_ok(row["location"] or ""):
        return "role_bad_location"
    if DATA_TITLE.search(row["title"]):
        return "data_title"
    return "other"


def round_robin(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    by_company = defaultdict(list)
    for r in rows:
        by_company[r["company"]].append(r)
    queues = [rng.sample(v, len(v)) for v in by_company.values()]
    rng.shuffle(queues)
    out = []
    while len(out) < k and any(queues):
        for q in queues:
            if q and len(out) < k:
                out.append(q.pop())
    return out


def sample(rows: list[dict], pf: PreFilter, n: int, seed: int) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    pools = defaultdict(list)
    for r in rows:
        pools[stratum(r, pf)].append(r)
    want = {s: round(q * n) for s, q in QUOTAS.items()}
    picked: dict[str, list[dict]] = {s: round_robin(pools[s], want[s], rng) for s in QUOTAS}
    # Strata that ran short leave room; fill it from the hard strata first, easy negatives last.
    for s in QUOTAS:
        room = n - sum(len(v) for v in picked.values())
        if room <= 0:
            break
        rest = [r for r in pools[s] if r not in picked[s]]
        picked[s] += round_robin(rest, room, rng)
    return [(s, r) for s in QUOTAS for r in picked[s]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(ROOT / "data" / "jobs.duckdb"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(ROOT / "eval" / "out"))
    ap.add_argument("--min-description", type=int, default=200,
                    help="skip jobs whose description is shorter than this (nothing to judge)")
    args = ap.parse_args(argv)

    con = duckdb.connect(args.db, read_only=True)
    cur = con.execute(
        "SELECT job_id, company, title, location, department, url, description, score FROM jobs "
        "WHERE length(coalesce(description, '')) >= ? ORDER BY job_id", (args.min_description,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()

    picked = sample(rows, PreFilter(load_profile()), args.n, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "jobs.jsonl", "w", encoding="utf-8") as f:
        for _, r in picked:
            f.write(json.dumps({
                "case_id": r["job_id"], "company": r["company"], "title": r["title"],
                "location": r["location"] or "", "department": r["department"] or "", "url": r["url"] or "",
                "description": (r["description"] or "")[:JD_MAX_CHARS],
            }, ensure_ascii=False) + "\n")
    with open(out / "labels.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        # company/title/location/url are only there to make labeling easy; loaders read case_id, label, note.
        w.writerow(["case_id", "label", "note", "company", "title", "location", "url"])
        for _, r in picked:
            w.writerow([r["job_id"], "", "", r["company"], r["title"], r["location"] or "", r["url"] or ""])

    print(f"{len(picked)} of {len(rows)} jobs written to {out}/ (jobs.jsonl, labels.csv)")
    for s, k in Counter(s for s, _ in picked).items():
        print(f"  {s}: {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
