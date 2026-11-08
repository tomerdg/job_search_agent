"""Daily run: fetch -> store (dedup, incremental) -> pre-filter -> Claude scoring -> digest."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

from jobagent.db import Store
from jobagent.fetchers import fetch_all
from jobagent.filters import PreFilter
from jobagent.notify import format_digest, notify

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("jobagent")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("JOBS_DB", str(ROOT / "data" / "jobs.db")))
    ap.add_argument("--sources", default=str(ROOT / "config" / "sources.yaml"))
    ap.add_argument("--profile", default=os.environ.get("PROFILE_PATH", str(ROOT / "config" / "profile.yaml")))
    ap.add_argument("--no-llm", action="store_true", help="fetch and filter only, skip Claude")
    ap.add_argument("--no-notify", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sources = yaml.safe_load(open(args.sources, encoding="utf-8"))["sources"]
    # Personal preferences: PROFILE_YAML env (a GitHub secret) wins over the local git-ignored file.
    profile_path = Path(args.profile)
    if not os.environ.get("PROFILE_YAML") and not profile_path.exists():
        log.warning("No PROFILE_YAML secret and no %s; using config/profile.example.yaml", profile_path)
        profile_path = ROOT / "config" / "profile.example.yaml"
    profile_text = os.environ.get("PROFILE_YAML") or profile_path.read_text(encoding="utf-8")
    profile = yaml.safe_load(profile_text)
    store = Store(args.db)

    jobs = fetch_all(sources)
    new_ids = store.upsert(jobs)
    store.mark_inactive({j.company for j in jobs}, {j.job_id for j in jobs})
    log.info("Fetched %d jobs, %d new", len(jobs), len(new_ids))

    # Re-check every unscored job, not just new ones, so a changed profile applies to old jobs too.
    pf = PreFilter(profile)
    passed = 0
    for row in store.unscored_active():
        ok = pf(row["title"], row["description"] or "", row["location"] or "")
        store.set_prefilter(row["job_id"], ok)
        passed += ok
    log.info("%d unscored jobs pass the pre-filter", passed)

    if not args.no_llm and os.environ.get("ANTHROPIC_API_KEY"):
        from jobagent.agent import Agent
        from jobagent.cv import load_cv
        cv = load_cv(os.environ.get("CV_PATH") or ROOT / profile.get("cv_path", "cv.pdf"))
        Agent(store, cv, profile).run()
    elif not args.no_llm:
        log.warning("ANTHROPIC_API_KEY not set, skipping scoring")

    top = store.top_unsent(n=3, min_score=int(os.environ.get("MIN_SCORE", "50")))
    digest = format_digest(top, len(new_ids))
    if args.no_notify:
        print(digest)
    else:
        if notify(digest):
            store.mark_notified([r["job_id"] for r in top])


if __name__ == "__main__":
    main()
