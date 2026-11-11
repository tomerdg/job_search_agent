"""The labeled evaluation set: loading, checking, and keeping the labels encrypted.

    eval/dataset/jobs.jsonl        public job postings, one JSON object per line (committed)
    eval/dataset/labels.csv.enc    the owner's labels, encrypted with DB_PASSPHRASE (committed)
    eval/dataset/labels.csv        the plain labels, local only (git-ignored)

The labels say which jobs the owner would apply to, so they are personal and never committed in
plain text. CI decrypts them with the DB_PASSPHRASE secret.

    python -m eval.dataset check              counts per label, and any problems (numbers only)
    python -m eval.dataset encrypt            labels.csv -> labels.csv.enc (skipped if unchanged)
    python -m eval.dataset decrypt            labels.csv.enc -> labels.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "eval" / "dataset"
JOBS_PATH = DATASET_DIR / "jobs.jsonl"
LABELS_PATH = DATASET_DIR / "labels.csv"
LABELS_ENC_PATH = DATASET_DIR / "labels.csv.enc"
DBCRYPT = ROOT / "scripts" / "dbcrypt.sh"

LABELS = ("fit", "maybe", "no_fit")
JOB_FIELDS = ("case_id", "company", "title", "location", "department", "url", "description")


def load_jobs(path: str | Path = JOBS_PATH) -> list[dict]:
    jobs = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in ("case_id", "title", "description") if not row.get(k)]
            if missing:
                raise ValueError(f"{path}:{n}: missing {', '.join(missing)}")
            jobs.append({k: row.get(k) or "" for k in JOB_FIELDS})
    return jobs


def parse_labels(text: str) -> dict[str, dict]:
    """case_id -> {label, note}. Blank labels are kept as '' so check() can report them."""
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        cid = (row.get("case_id") or "").strip()
        if cid:
            out[cid] = {"label": (row.get("label") or "").strip(), "note": (row.get("note") or "").strip()}
    return out


def _crypt(mode: str, src: Path, dst: Path) -> None:
    if not os.environ.get("DB_PASSPHRASE"):
        raise RuntimeError("DB_PASSPHRASE is not set")
    subprocess.run(["bash", str(DBCRYPT), mode, str(src), str(dst)], check=True)


def _decrypt_to_text(enc: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "labels.csv"
        _crypt("decrypt", enc, plain)
        return plain.read_text(encoding="utf-8")


def load_labels(plain: Path = LABELS_PATH, enc: Path = LABELS_ENC_PATH) -> dict[str, dict]:
    """Plain labels.csv when present (local), otherwise decrypt labels.csv.enc in memory (CI)."""
    if plain.exists():
        return parse_labels(plain.read_text(encoding="utf-8"))
    if enc.exists():
        return parse_labels(_decrypt_to_text(enc))
    raise FileNotFoundError(f"no labels: neither {plain} nor {enc} exists")


def check(jobs: list[dict], labels: dict[str, dict]) -> tuple[Counter, list[str]]:
    """Label counts and a list of problems. Problems hold case ids and counts only, never label text."""
    problems = []
    ids = [j["case_id"] for j in jobs]
    dupes = [cid for cid, n in Counter(ids).items() if n > 1]
    if dupes:
        problems.append(f"{len(dupes)} duplicate case_id in jobs.jsonl: {', '.join(dupes[:5])}")
    job_ids = set(ids)
    unlabeled = sorted(job_ids - set(labels))
    if unlabeled:
        problems.append(f"{len(unlabeled)} jobs have no row in labels.csv: {', '.join(unlabeled[:5])}")
    orphans = sorted(set(labels) - job_ids)
    if orphans:
        problems.append(f"{len(orphans)} labels have no job: {', '.join(orphans[:5])}")
    counts = Counter()
    bad = []
    for cid in sorted(job_ids & set(labels)):
        lab = labels[cid]["label"]
        if lab in LABELS:
            counts[lab] += 1
        else:
            bad.append(cid)
    if bad:
        problems.append(f"{len(bad)} labels are blank or not one of {'/'.join(LABELS)}: {', '.join(bad[:5])}")
    return counts, problems


def encrypt(plain: Path = LABELS_PATH, enc: Path = LABELS_ENC_PATH) -> bool:
    """Encrypt labels.csv. Each encryption uses a new salt, so skip it when the content is unchanged
    (otherwise every run would produce a meaningless git diff). Returns True if the file was written."""
    if enc.exists() and _decrypt_to_text(enc) == plain.read_text(encoding="utf-8"):
        return False
    _crypt("encrypt", plain, enc)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluation dataset tools")
    ap.add_argument("command", choices=["check", "encrypt", "decrypt"])
    args = ap.parse_args(argv)
    if args.command == "encrypt":
        print("encrypted" if encrypt() else "unchanged, not re-encrypted")
        return 0
    if args.command == "decrypt":
        _crypt("decrypt", LABELS_ENC_PATH, LABELS_PATH)
        print(f"wrote {LABELS_PATH.relative_to(ROOT)} (git-ignored)")
        return 0
    jobs = load_jobs()
    counts, problems = check(jobs, load_labels())
    print(f"{len(jobs)} jobs; " + ", ".join(f"{k}={counts.get(k, 0)}" for k in LABELS))
    for p in problems:
        print("problem:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
