from __future__ import annotations

import logging

from jobagent.models import Job

from . import adam, comeet, greenhouse, topmatch

log = logging.getLogger(__name__)

FETCHERS = {
    "adam": adam.fetch,
    "topmatch": topmatch.fetch,
    "comeet": comeet.fetch,
    "greenhouse": greenhouse.fetch,
}


def fetch_all(sources: list[dict]) -> list[Job]:
    """Run every configured source. A failing source is logged and skipped."""
    jobs: list[Job] = []
    for src in sources:
        fn = FETCHERS.get(src["type"])
        if fn is None:
            log.warning("No fetcher for type %s (%s)", src["type"], src["company"])
            continue
        try:
            found = fn(src)
            for j in found:
                j.sector = src.get("sector", "")
                # Some boards only say "Israel"; use the office city we know instead.
                if src.get("city") and j.location.strip().lower() in ("", "israel", "il"):
                    j.location = src["city"]
            log.info("%-25s %-10s %3d jobs", src["company"], src["type"], len(found))
            jobs.extend(found)
        except Exception as exc:  # noqa: BLE001 - one bad site must not stop the run
            log.error("%s (%s) failed: %s", src["company"], src["type"], exc)
    return jobs
