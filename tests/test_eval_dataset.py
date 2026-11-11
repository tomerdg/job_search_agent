import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from eval import dataset as ds
from jobagent import demo
from jobagent.filters import PreFilter


def _write_jobs(path, ids):
    path.write_text("".join(json.dumps({"case_id": i, "title": "T", "description": "D"}) + "\n" for i in ids),
                    encoding="utf-8")


def test_check_reports_problems(tmp_path):
    _write_jobs(tmp_path / "jobs.jsonl", ["a", "b", "b", "c"])
    labels = ds.parse_labels("case_id,label,note\na,fit,\nb,meh,\nz,no_fit,\n")
    counts, problems = ds.check(ds.load_jobs(tmp_path / "jobs.jsonl"), labels)
    assert counts == {"fit": 1}
    text = "\n".join(problems)
    assert "duplicate case_id" in text and "no row in labels.csv: c" in text
    assert "labels have no job: z" in text and "blank or not one of" in text


def test_check_clean_set(tmp_path):
    _write_jobs(tmp_path / "jobs.jsonl", ["a", "b", "c"])
    labels = ds.parse_labels("case_id,label,note\na,fit,\nb,maybe,x\nc,no_fit,\n")
    counts, problems = ds.check(ds.load_jobs(tmp_path / "jobs.jsonl"), labels)
    assert problems == [] and counts == {"fit": 1, "maybe": 1, "no_fit": 1}


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")
def test_encrypt_roundtrip_and_skip_when_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PASSPHRASE", "test-passphrase")
    plain, enc = tmp_path / "labels.csv", tmp_path / "labels.csv.enc"
    plain.write_text("case_id,label,note\na,fit,\n", encoding="utf-8")
    assert ds.encrypt(plain, enc) is True
    first = enc.read_bytes()
    assert b"fit" not in first
    assert ds.encrypt(plain, enc) is False and enc.read_bytes() == first  # no new salt, no git diff
    plain.unlink()
    assert ds.load_labels(plain, enc) == {"a": {"label": "fit", "note": ""}}  # decrypted in memory
    assert not plain.exists()


def test_export_stratifies_demo_db(tmp_path):
    db = tmp_path / "demo.duckdb"
    demo.build(db, days=60)
    out = tmp_path / "out"
    script = Path(__file__).resolve().parent.parent / "scripts" / "export_eval_candidates.py"
    subprocess.run([sys.executable, str(script), "--db", str(db), "--n", "20", "--out", str(out),
                    "--min-description", "1"], check=True, capture_output=True,
                   env={"PATH": os.environ["PATH"], "PROFILE_YAML": "keywords: [Data]\n"})
    jobs = ds.load_jobs(out / "jobs.jsonl")
    labels = ds.parse_labels((out / "labels.csv").read_text(encoding="utf-8"))
    assert len(jobs) == 20 and set(labels) == {j["case_id"] for j in jobs}
    assert all(v["label"] == "" for v in labels.values())  # blank, for the owner to fill
    assert "score" not in json.loads((out / "jobs.jsonl").read_text(encoding="utf-8").splitlines()[0])


def test_stratum_buckets():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export", Path(__file__).resolve().parent.parent / "scripts" / "export_eval_candidates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pf = PreFilter({"keywords": ["Data Engineer"], "locations_allowed": ["Tel Aviv"], "locations_blocked": ["Haifa"]})
    row = lambda **kw: {"score": None, "title": "x", "description": "", "location": "Tel Aviv", **kw}
    assert mod.stratum(row(score=90), pf) == "scored_85_100"
    assert mod.stratum(row(score=29), pf) == "scored_0_29"
    assert mod.stratum(row(title="Data Engineer", location="Haifa"), pf) == "role_bad_location"
    assert mod.stratum(row(title="BI Developer"), pf) == "data_title"
    assert mod.stratum(row(title="Chef"), pf) == "other"


def _fetch_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_postings", Path(__file__).resolve().parent.parent / "scripts" / "fetch_eval_postings.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_reads_jsonld_and_url_list(tmp_path):
    mod = _fetch_module()
    page = ('<html><script type="application/ld+json">{"@type": "JobPosting", "title": "Data Engineer",'
            ' "description": "&lt;p&gt;Build pipelines&lt;/p&gt;",'
            ' "jobLocation": {"address": {"addressLocality": "Tel Aviv"}}}</script></html>')
    got = mod._jsonld_posting(page)
    assert got["title"] == "Data Engineer" and got["location"] == "Tel Aviv" and got["description"] == "Build pipelines"
    urls = tmp_path / "urls.txt"
    urls.write_text("# comment\nAcme\thttps://a.example/job/1\nhttps://b.example/job/2\n", encoding="utf-8")
    assert mod.read_urls(urls) == [("Acme", "https://a.example/job/1"), ("", "https://b.example/job/2")]
    assert mod.fetch_posting("https://www.linkedin.com/jobs/view/1/") is None  # skipped, no request made
