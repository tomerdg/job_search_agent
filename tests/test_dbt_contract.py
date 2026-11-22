"""The dbt staging test lists the accepted source types by hand. Every fetcher must be in it, or the
first real job from a new source fails the daily run's dbt build."""

from pathlib import Path

import yaml

from jobagent.fetchers import FETCHERS

STAGING_YML = Path(__file__).resolve().parent.parent / "analytics" / "models" / "staging" / "_staging.yml"


def _accepted_source_types() -> set[str]:
    spec = yaml.safe_load(STAGING_YML.read_text())
    stg_jobs = next(m for m in spec["models"] if m["name"] == "stg_jobs")
    column = next(c for c in stg_jobs["columns"] if c["name"] == "source_type")
    for test in column["data_tests"]:
        if isinstance(test, dict) and "accepted_values" in test:
            return set(test["accepted_values"]["arguments"]["values"])
    raise AssertionError("stg_jobs.source_type has no accepted_values test")


def test_every_fetcher_is_an_accepted_source_type():
    missing = set(FETCHERS) - _accepted_source_types()
    assert not missing, f"add {sorted(missing)} to stg_jobs.source_type accepted_values in {STAGING_YML.name}"
