from pathlib import Path

import numpy as np

from jobagent.rag.chunking import (MIN_CHARS, WINDOW, chunk_cv, chunk_jd, embed_text, heading_section,
                                   windows)
from jobagent.rag.embed import DIM, FakeEmbedder

FIX = Path(__file__).parent / "fixtures" / "rag"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_english_jd_by_section_and_boilerplate_dropped():
    res = chunk_jd(_read("jd_en.txt"), "Acme Data", "Data Engineer")
    sections = [c.section for c in res.chunks]
    assert sections == ["about", "about", "responsibilities", "requirements", "advantage"]
    text = " ".join(c.text for c in res.chunks)
    assert "Airflow and Kafka" in text and "ClickHouse" in text
    assert "gym" not in text and "equal opportunity" not in text and "#LI" not in text
    assert res.dropped == 1  # the benefits section
    assert [c.seq for c in res.chunks] == list(range(len(res.chunks)))


def test_hebrew_jd_by_section():
    res = chunk_jd(_read("jd_he.txt"), "חברה", "מהנדס/ת נתונים")
    by = {c.section: c.text for c in res.chunks}
    assert set(by) == {"about", "responsibilities", "requirements"}
    assert "Spark" in by["responsibilities"]
    assert "Databricks – יתרון" in by["requirements"]  # an inline 'יתרון' stays with its requirement
    assert "לנשים ולגברים" not in " ".join(by.values())
    assert res.dropped == 1  # היקף משרה


def test_no_headings_falls_back_to_overlapping_windows():
    sentences = [f"Sentence number {i} talks about pipelines and warehouses in some detail." for i in range(60)]
    res = chunk_jd(" ".join(sentences))
    assert len(res.chunks) > 3 and {c.section for c in res.chunks} == {"other"}
    assert all(len(c.text) <= WINDOW for c in res.chunks)
    for a, b in zip(res.chunks, res.chunks[1:]):  # each window starts with the end of the one before
        assert b.text.split(". ")[0] in a.text


def test_windows_cut_a_sentence_longer_than_the_window():
    parts = windows("x" * 2000)
    assert len(parts) >= 3 and all(len(p) <= WINDOW for p in parts)


def test_short_and_empty_text():
    assert chunk_jd(None).chunks == [] and chunk_jd("").dropped == 0
    assert chunk_jd("Data Engineer").dropped == 1  # shorter than MIN_CHARS
    assert len("Data Engineer") < MIN_CHARS


def test_heading_detection():
    assert heading_section("Requirements") == "requirements"
    assert heading_section("What you’ll do:") == "responsibilities"
    assert heading_section("Bonus points if you have:") == "advantage"
    assert heading_section("Preferred qualifications:") == "advantage"
    assert heading_section("מה אנחנו מחפשים?") == "requirements"
    assert heading_section("אנחנו על המפה:") == "benefits"
    assert heading_section("Why Acme?") == "about"
    assert heading_section("Strong SQL and Python. Experience with Snowflake or BigQuery.") is None
    assert heading_section("• Requirements:") is None  # a bullet is content


def test_inline_heading_keeps_its_text():
    res = chunk_jd("Some intro about the company and what it builds for its customers.\n"
                   "Requirements: 5+ years of SQL and Python in production data systems.")
    assert res.chunks[-1].section == "requirements" and res.chunks[-1].text.startswith("5+ years")


def test_embed_text_has_context():
    assert embed_text("Acme", "Data Engineer", "requirements", "SQL").startswith("Acme | Data Engineer | requirements")


def test_cv_chunks_by_heading_and_job():
    chunks = chunk_cv(_read("cv.txt"))
    sections = [c.section for c in chunks]
    assert sections == ["summary", "experience", "experience", "skills", "education"]
    assert "Example Corp" in chunks[1].text and "Sample Ltd" in chunks[2].text


def test_cv_from_pdf_word_per_line():
    words = " ".join(_read("cv.txt").split()).split(" ")
    chunks = chunk_cv("\n\n".join(words))
    assert {"summary", "experience", "skills", "education"} <= {c.section for c in chunks}
    assert sum(c.section == "experience" for c in chunks) == 2


def test_fake_embedder_is_deterministic_and_normalised():
    e = FakeEmbedder()
    v = e.embed(["dbt and Snowflake", "dbt and Snowflake", "customer service"])
    assert v.shape == (3, DIM) and e.dim == DIM and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1)
    assert np.allclose(v[0], v[1]) and v[0] @ v[2] < 0.5
    assert e.embed([]).shape == (0, DIM)
