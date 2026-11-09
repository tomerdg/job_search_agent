from __future__ import annotations

import logging
import time

from jobagent.models import Job

from . import adam, comeet, greenhouse, topmatch

log = logging.getLogger(__name__)

FETCHERS = {
    "adam": adam.fetch,
    "topmatch": topmatch.fetch,
    "comeet": comeet.fetch,
    "greenhouse": greenhouse.fetch,
}


def fetch_all(sources: list[dict], stats: list[dict] | None = None) -> list[Job]:
    """Run every configured source. A failing source is logged and skipped.

    If `stats` is given, one dict per source is appended to it (company, source_type, ok, n_jobs,
    duration_s, error), which the store keeps as source-health history.
    """
    jobs: list[Job] = []
    for src in sources:
        t0 = time.monotonic()
        stat = {"company": src["company"], "source_type": src["type"], "ok": False, "n_jobs": 0}
        fn = FETCHERS.get(src["type"])
        if fn is None:
            log.warning("No fetcher for type %s (%s)", src["type"], src["company"])
            continue
        if stats is not None:
            stats.append(stat)
        try:
            found = fn(src)
            for j in found:
                j.sector = src.get("sector", "")
                # Some boards only say "Israel"; use the office city we know instead.
                if src.get("city") and j.location.strip().lower() in ("", "israel", "il"):
                    j.location = src["city"]
            log.info("%-25s %-10s %3d jobs", src["company"], src["type"], len(found))
            jobs.extend(found)
            stat.update(ok=True, n_jobs=len(found))
        except Exception as exc:  # noqa: BLE001 - one bad site must not stop the run
            log.error("%s (%s) failed: %s", src["company"], src["type"], exc)
            stat["error"] = f"{type(exc).__name__}: {exc}"
        stat["duration_s"] = round(time.monotonic() - t0, 2)
    return jobs
