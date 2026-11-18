"""Sync the owner's applied/skipped/irrelevant decisions between the local DB and the scheduled run.

The scheduled run keeps its own encrypted DB in the Actions cache, so decisions recorded locally
(`python -m jobagent.main --applied ID ...`) reach it through one committed, encrypted file:

    data/feedback.csv.enc   job_id,decision,decided_at,source, encrypted with DB_PASSPHRASE (committed)
    data/feedback.csv       plain copy, local only (git-ignored by data/*.csv)

No titles, companies or notes go into the file: only hashed job ids and the decision.

    python -m jobagent.feedback export       local DB -> data/feedback.csv -> data/feedback.csv.enc
    python -m jobagent.feedback import       data/feedback.csv(.enc) -> local DB (latest decided_at wins)
    python -m jobagent.feedback import-eval  one-time: eval labels -> local DB, for eval jobs found by url
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "feedback.csv"
ENC_PATH = ROOT / "data" / "feedback.csv.enc"
DBCRYPT = ROOT / "scripts" / "dbcrypt.sh"
FIELDS = ("job_id", "decision", "decided_at", "source")
EVAL_TO_DECISION = {"fit": "applied", "maybe": "applied", "no_fit": "irrelevant"}

log = logging.getLogger(__name__)


def _crypt(mode: str, src: Path, dst: Path) -> None:
    if not os.environ.get("DB_PASSPHRASE"):
        raise RuntimeError("DB_PASSPHRASE is not set")
    subprocess.run(["bash", str(DBCRYPT), mode, str(src), str(dst)], check=True)


def decrypt_text(enc: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "feedback.csv"
        _crypt("decrypt", enc, plain)
        return plain.read_text(encoding="utf-8")


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in sorted(rows, key=lambda r: r["job_id"]):
        w.writerow({**r, "decided_at": r["decided_at"].isoformat(sep=" ")})
    return buf.getvalue()


def parse_csv(text: str) -> list[dict]:
    return [{"job_id": r["job_id"].strip(), "decision": r["decision"].strip(),
             "decided_at": datetime.fromisoformat(r["decided_at"].strip()),
             "source": (r.get("source") or "").strip() or "cli", "note": None}
            for r in csv.DictReader(io.StringIO(text)) if (r.get("job_id") or "").strip()]


def export(store, csv_path: Path = CSV_PATH, enc_path: Path | None = ENC_PATH) -> int:
    """Write the local decisions to csv_path and, when DB_PASSPHRASE is set, to enc_path.
    The .enc file is rewritten only when its content changed (each encryption has a new salt,
    so re-encrypting unchanged data would make a meaningless git diff). Returns the row count."""
    rows = store.feedback()
    text = to_csv(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(text, encoding="utf-8")
    if enc_path is not None and os.environ.get("DB_PASSPHRASE"):
        if not (enc_path.exists() and decrypt_text(enc_path) == text):
            _crypt("encrypt", csv_path, enc_path)
    return len(rows)


def import_text(store, text: str) -> int:
    """Merge feedback rows into the store; a row wins only if it is newer. Returns rows changed."""
    return store.upsert_feedback(parse_csv(text), when="newer")


def import_file(store, csv_path: Path | None = None, enc_path: Path | None = ENC_PATH) -> int | None:
    """Import the plain file if it exists, else the encrypted one if DB_PASSPHRASE is set.
    Returns rows changed, or None when there was nothing to import."""
    if csv_path is not None and Path(csv_path).exists():
        return import_text(store, Path(csv_path).read_text(encoding="utf-8"))
    if enc_path is not None and Path(enc_path).exists() and os.environ.get("DB_PASSPHRASE"):
        return import_text(store, decrypt_text(Path(enc_path)))
    return None


def sync_at_start(store) -> None:
    """Called by the daily run. FEEDBACK_CSV (a plain file, decrypted by the workflow and mounted
    read-only) wins; otherwise FEEDBACK_CSV_ENC / data/feedback.csv.enc when DB_PASSPHRASE is set.
    Never fails the run: a broken feedback file only costs labels, not the digest."""
    plain = os.environ.get("FEEDBACK_CSV")
    enc = Path(os.environ.get("FEEDBACK_CSV_ENC") or ENC_PATH)
    try:
        n = import_file(store, Path(plain) if plain else None, enc)
    except Exception as e:  # noqa: BLE001
        log.warning("Feedback import failed (%s); continuing without it", type(e).__name__)
        return
    if n is not None:
        log.info("Feedback: %d decision(s) imported", n)


def import_eval(store, jobs: list[dict], labels: dict[str, dict]) -> dict[str, int]:
    """Bootstrap labels from the eval set: eval cases whose url matches a job in the DB.
    fit/maybe -> applied, no_fit -> irrelevant, source='eval'. Never overwrites an existing decision.
    Returns counts only."""
    by_url: dict[str, list[str]] = {}
    for job_id, url in store.conn.execute("SELECT job_id, url FROM jobs WHERE url IS NOT NULL").fetchall():
        by_url.setdefault(url.strip(), []).append(job_id)
    from jobagent.db import now
    ts = now()
    rows, counts = [], {"cases": len(jobs), "matched": 0, "unlabeled": 0}
    for j in jobs:
        decision = EVAL_TO_DECISION.get((labels.get(j["case_id"]) or {}).get("label", ""))
        if decision is None:
            counts["unlabeled"] += 1
            continue
        ids = by_url.get((j.get("url") or "").strip(), [])
        if ids:
            counts["matched"] += 1
        rows += [{"job_id": i, "decision": decision, "decided_at": ts, "source": "eval"} for i in ids]
    counts["recorded"] = store.upsert_feedback(rows, when="missing")
    return counts


def main(argv=None) -> int:
    from jobagent.main import load_dotenv
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Sync applied/skipped decisions (labels)")
    ap.add_argument("command", choices=["export", "import", "import-eval"])
    ap.add_argument("--db", default=os.environ.get("JOBS_DB", str(ROOT / "data" / "jobs.duckdb")))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from jobagent.db import open_store
    store = open_store(args.db)
    try:
        if args.command == "export":
            n = export(store)
            enc = "encrypted to " + str(ENC_PATH.relative_to(ROOT)) if os.environ.get("DB_PASSPHRASE") \
                else "NOT encrypted (DB_PASSPHRASE not set)"
            print(f"{n} decision(s) -> {CSV_PATH.relative_to(ROOT)} (git-ignored), {enc}")
        elif args.command == "import":
            n = import_file(store, CSV_PATH)
            print("nothing to import" if n is None else f"{n} decision(s) imported")
        else:
            from eval.dataset import load_jobs, load_labels
            c = import_eval(store, load_jobs(), load_labels())
            print(f"{c['cases']} eval cases, {c['matched']} matched a job in the DB by url, "
                  f"{c['recorded']} decision(s) recorded, {c['unlabeled']} without a usable label")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
