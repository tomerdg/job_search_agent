"""Daily run: fetch -> DuckDB store (dedup, incremental, history) -> pre-filter -> Claude scoring -> digest."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

from jobagent.db import min_score, open_store
from jobagent.fetchers import fetch_all
from jobagent.filters import PreFilter
from jobagent.notify import format_digest, notify

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("jobagent")


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a local, git-ignored .env file. Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def main(argv=None) -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("JOBS_DB", str(ROOT / "data" / "jobs.duckdb")))
    ap.add_argument("--sources", default=str(ROOT / "config" / "sources.yaml"))
    ap.add_argument("--profile", default=os.environ.get("PROFILE_PATH", str(ROOT / "config" / "profile.yaml")))
    ap.add_argument("--no-llm", action="store_true", help="fetch and filter only, skip Claude")
    ap.add_argument("--no-ml", action="store_true", help="skip the relevance model (shadow mode)")
    ap.add_argument("--no-rag", action="store_true", help="skip chunking and embedding JDs for retrieval")
    ap.add_argument("--no-notify", action="store_true", help="print the digest instead of sending it")
    ap.add_argument("--mark-seen", nargs="+", metavar="ID",
                    help="mark jobs as seen (ids from the digest) so they are never sent again, then exit")
    for decision in ("applied", "skipped", "irrelevant"):
        ap.add_argument(f"--{decision}", nargs="+", metavar="ID",
                        help=f"record that you {decision if decision != 'irrelevant' else 'found irrelevant'}"
                             " these jobs (ids from the digest), then exit")
    ap.add_argument("--note", help="with --applied/--skipped/--irrelevant: a private note (never exported)")
    ap.add_argument("--mark-printed", action="store_true",
                    help="with --no-notify: mark the printed jobs as seen")
    args = ap.parse_args(argv)
    chosen = [d for d in ("applied", "skipped", "irrelevant") if getattr(args, d)]
    if len(chosen) > 1:
        ap.error("use only one of --applied / --skipped / --irrelevant at a time")
    if args.note and not chosen:
        ap.error("--note needs --applied, --skipped or --irrelevant")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sources = yaml.safe_load(open(args.sources, encoding="utf-8"))["sources"]
    # Personal preferences: PROFILE_YAML env (a GitHub secret) wins over the local git-ignored file.
    profile_path = Path(args.profile)
    if not os.environ.get("PROFILE_YAML") and not profile_path.exists():
        log.warning("No PROFILE_YAML secret and no %s; using config/profile.example.yaml", profile_path)
        profile_path = ROOT / "config" / "profile.example.yaml"
    profile_text = os.environ.get("PROFILE_YAML") or profile_path.read_text(encoding="utf-8")
    profile = yaml.safe_load(profile_text)
    store = open_store(args.db)
    try:
        run(store, args, sources, profile)
    finally:
        store.close()


def cv_text(profile: dict) -> str | None:
    """The CV for the relevance model, or None (logged) when there is none."""
    from jobagent.cv import load_cv
    try:
        return load_cv(os.environ.get("CV_PATH") or ROOT / profile.get("cv_path", "cv.pdf")) or None
    except Exception as e:  # noqa: BLE001 - a missing or unreadable CV only skips the model
        log.info("No CV for the relevance model (%s)", type(e).__name__)
        return None


def resolve_for_feedback(store, prefixes: list[str]) -> list[str]:
    """Like resolve_ids, but strict: a short id must match exactly one job. A full 16-char id
    that is not in this DB (e.g. a digest from the scheduled run) is kept as is."""
    out = []
    for p in prefixes:
        p = p.strip()
        matches = store.resolve_ids([p])
        if len(matches) == 1:
            out.append(matches[0])
        elif len(matches) > 1:
            log.warning("%s matches %d jobs; use more characters. Skipped.", p, len(matches))
        elif len(p) == 16 and all(c in "0123456789abcdef" for c in p):
            log.warning("%s is not in this DB; recorded anyway (full id)", p)
            out.append(p)
        else:
            log.warning("%s matches no job in this DB. Skipped.", p)
    return out


def run(store, args, sources: list[dict], profile: dict) -> None:
    if args.mark_seen:
        ids = store.resolve_ids(args.mark_seen)
        store.mark_notified(ids)
        print(f"Marked {len(ids)} job(s) as seen")
        return
    for decision in ("applied", "skipped", "irrelevant"):
        if getattr(args, decision, None):
            ids = resolve_for_feedback(store, getattr(args, decision))
            n = store.record_feedback(ids, decision, note=args.note) if ids else 0
            print(f"Recorded {n} job(s) as {decision}")
            return

    from jobagent.feedback import sync_at_start
    sync_at_start(store)

    run_id = store.start_run()
    source_stats: list[dict] = []
    jobs = fetch_all(sources, stats=source_stats)
    new_ids = store.upsert(jobs)
    store.mark_inactive({j.company for j in jobs}, {j.job_id for j in jobs})
    store.record_observations(run_id, [j.job_id for j in jobs])
    store.record_sources(run_id, source_stats)
    store.finish_run(run_id, len(jobs), len(new_ids))
    log.info("Fetched %d jobs, %d new", len(jobs), len(new_ids))

    # Re-check every unscored job, not just new ones, so a changed profile applies to old jobs too.
    pf = PreFilter(profile)
    passed = 0
    for row in store.unscored_active():
        ok = pf(row["title"], row["description"] or "", row["location"] or "")
        store.set_prefilter(row["job_id"], ok)
        passed += ok
    log.info("%d unscored jobs pass the pre-filter", passed)

    # Retrieval index: chunk and embed the data-ish jobs incrementally (only new or changed chunks).
    if not getattr(args, "no_rag", False):
        from jobagent.rag.index import index_step
        index_step(store, run_id)

    # Shadow mode: score and reorder the LLM queue, never filter.
    if not getattr(args, "no_ml", False):
        cv = cv_text(profile)
        if cv:
            from jobagent.ml.shadow import shadow_score
            shadow_score(store, cv, profile)

    if not args.no_llm and os.environ.get("ANTHROPIC_API_KEY"):
        from jobagent.agent import Agent
        from jobagent.cv import load_cv
        cv = load_cv(os.environ.get("CV_PATH") or ROOT / profile.get("cv_path", "cv.pdf"))
        Agent(store, cv, profile).run()
    elif not args.no_llm:
        log.warning("ANTHROPIC_API_KEY not set, skipping scoring")

    top = store.top_unsent(n=3, min_score=min_score())
    digest = format_digest(top, len(new_ids))
    if args.no_notify:
        print(digest)
        if args.mark_printed:
            store.mark_notified([r["job_id"] for r in top])
    else:
        if notify(digest):
            store.mark_notified([r["job_id"] for r in top])

if __name__ == "__main__":
    main()
