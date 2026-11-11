"""The labeled evaluation set: loading, checking, and keeping it encrypted.

    eval/dataset/jobs.jsonl.enc    job postings, one JSON object per line, encrypted (committed)
    eval/dataset/labels.csv.enc    the owner's labels, encrypted (committed)
    eval/dataset/jobs.jsonl        plain copies, local only (git-ignored)
    eval/dataset/labels.csv

Both files are personal: the labels say which jobs the owner would apply to, and the job list includes
jobs the owner actually applied to. So neither is committed in plain text. Both are encrypted with
DB_PASSPHRASE (scripts/dbcrypt.sh); CI decrypts them in memory with the same secret.

    python -m eval.dataset check      counts per label, and any problems (numbers and case ids only)
    python -m eval.dataset encrypt    plain -> .enc for both files (each skipped if unchanged)
    python -m eval.dataset decrypt    .enc -> plain for both files
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
DBCRYPT = ROOT / "scripts" / "dbcrypt.sh"

LABELS = ("fit", "maybe", "no_fit")
JOB_FIELDS = ("case_id", "company", "title", "location", "department", "url", "description")


def enc_path(plain: Path) -> Path:
    return plain.with_name(plain.name + ".enc")


def _crypt(mode: str, src: Path, dst: Path) -> None:
    if not os.environ.get("DB_PASSPHRASE"):
        raise RuntimeError("DB_PASSPHRASE is not set")
    subprocess.run(["bash", str(DBCRYPT), mode, str(src), str(dst)], check=True)


def _decrypt_to_text(enc: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        plain = Path(tmp) / "plain"
        _crypt("decrypt", enc, plain)
        return plain.read_text(encoding="utf-8")


def read_text(plain: Path) -> str:
    """The plain file when present (local), otherwise its .enc decrypted in memory (CI)."""
    plain = Path(plain)
    if plain.exists():
        return plain.read_text(encoding="utf-8")
    if enc_path(plain).exists():
        return _decrypt_to_text(enc_path(plain))
    raise FileNotFoundError(f"neither {plain} nor {enc_path(plain)} exists")


def parse_jobs(text: str, source: str = "jobs.jsonl") -> list[dict]:
    jobs = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [k for k in ("case_id", "title", "description") if not row.get(k)]
        if missing:
            raise ValueError(f"{source}:{n}: missing {', '.join(missing)}")
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


def load_jobs(path: str | Path = JOBS_PATH) -> list[dict]:
    return parse_jobs(read_text(Path(path)), str(path))


def load_labels(path: str | Path = LABELS_PATH) -> dict[str, dict]:
    return parse_labels(read_text(Path(path)))


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


def encrypt(plain: Path) -> bool:
    """Encrypt one file. Each encryption uses a new salt, so skip it when the content is unchanged
    (otherwise every run would produce a meaningless git diff). Returns True if the file was written."""
    plain, enc = Path(plain), enc_path(Path(plain))
    if enc.exists() and _decrypt_to_text(enc) == plain.read_text(encoding="utf-8"):
        return False
    _crypt("encrypt", plain, enc)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluation dataset tools")
    ap.add_argument("command", choices=["check", "encrypt", "decrypt"])
    args = ap.parse_args(argv)
    files = (JOBS_PATH, LABELS_PATH)
    if args.command == "encrypt":
        for f in files:
            print(f"{enc_path(f).name}: {'encrypted' if encrypt(f) else 'unchanged, not re-encrypted'}")
        return 0
    if args.command == "decrypt":
        for f in files:
            _crypt("decrypt", enc_path(f), f)
            print(f"wrote {f.relative_to(ROOT)} (git-ignored)")
        return 0
    jobs = load_jobs()
    counts, problems = check(jobs, load_labels())
    print(f"{len(jobs)} jobs; " + ", ".join(f"{k}={counts.get(k, 0)}" for k in LABELS))
    for p in problems:
        print("problem:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
