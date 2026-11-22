"""Prompt v3's retrieval tools, and the eval's leakage rule for find_similar_jobs."""
import json
import shutil
from types import SimpleNamespace as NS

import pytest

from eval.evaluate import EXCLUDED_FEEDBACK_SOURCES, fill_store, load_history
from jobagent.agent import RAG_TOOL_NAMES, TOOLS, Agent, load_prompt, prompt_tools
from jobagent.db import Store
from jobagent.models import Job
from jobagent.rag.embed import FakeEmbedder
from jobagent.rag.index import index_jobs

CV = ("SUMMARY\nData engineer, six years of Python and SQL.\nEXPERIENCE\nAcme | Data Engineer 2020–Present\n"
      "Built Kafka streaming pipelines and a dbt warehouse on Snowflake.\nSKILLS\nPython, SQL, dbt, Kafka, Airflow.")


def _job(ext, title="Data Engineer", company="Acme", desc="Requirements:\nKafka streaming and dbt on Snowflake."):
    return Job(source="greenhouse", company=company, external_id=ext, title=title, url=f"u/{ext}", description=desc)


class ScriptClient:
    """Calls the given tool batches in order, then says DONE. Records the tools offered on each call."""

    def __init__(self, script):
        self.script, self.calls, self.offered, self.results = script, 0, [], []
        self.messages = self

    def create(self, **kw):
        self.offered.append({t["name"] for t in kw["tools"]})
        last = kw["messages"][-1]["content"]
        if isinstance(last, list):
            self.results.extend(b["content"] for b in last if isinstance(b, dict) and b.get("type") == "tool_result")
        self.calls += 1
        batch = self.script[self.calls - 1] if self.calls <= len(self.script) else []
        if not batch:
            return NS(content=[NS(type="text", text="DONE")])
        return NS(content=[NS(type="tool_use", id=f"t{self.calls}{i}", name=n, input=a) for i, (n, a) in enumerate(batch)])


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    s.upsert([_job("new"), _job("old1", company="Beta"), _job("old2", company="Gamma", title="Analytics Engineer"),
              _job("dup")])  # same company + title as "new"
    ids = {e: s.conn.execute("SELECT job_id FROM jobs WHERE external_id=?", [e]).fetchone()[0]
           for e in ("new", "old1", "old2", "dup")}
    s.set_prefilter(ids["new"], True)
    for e in ("old1", "old2", "dup"):
        s.save_score(ids[e], 70, {})
    s.record_feedback([ids["old1"]], "applied", note="private: liked the team", source="cli")
    s.record_feedback([ids["old2"]], "skipped", source="eval")
    s.record_feedback([ids["dup"]], "applied", source="cli")
    index_jobs(s, FakeEmbedder(), scope="all")
    yield s, ids
    s.close()


def test_v1_and_v2_offer_exactly_the_old_tools():
    for v in ("v1", "v2"):
        assert prompt_tools(v) == set()
    assert prompt_tools("v3") == RAG_TOOL_NAMES
    assert not load_prompt("v3").startswith("#tools")
    assert load_prompt("v3").format(roles="DE", allowed="any", blocked="none").endswith("DONE.")


def test_v1_agent_does_not_need_an_embedder(store, monkeypatch):
    s, ids = store
    monkeypatch.setattr("jobagent.rag.embed.get_embedder", lambda: pytest.fail("v1 must not load an embedder"))
    client = ScriptClient([])
    Agent(s, CV, {}, client=client, prompt_version="v1").run()
    assert client.offered[0] == {t["name"] for t in TOOLS}


def test_v3_tools_return_cv_evidence_and_similar_decisions(store):
    s, ids = store
    client = ScriptClient([
        [("search_cv", {"query": "Kafka streaming"}), ("find_similar_jobs", {"job_id": ids["new"]})],
        [("save_match", {"job_id": ids["new"], "score": 80, "summary": "x", "strengths": [], "gaps": []})],
    ])
    agent = Agent(s, CV, {}, client=client, prompt_version="v3", embedder=FakeEmbedder())
    assert agent.run() == 1
    assert client.offered[0] == {t["name"] for t in TOOLS} | RAG_TOOL_NAMES
    cv_hits, similar = json.loads(client.results[0]), json.loads(client.results[1])
    assert len(cv_hits) <= 3 and "Kafka" in cv_hits[0]["text"]
    assert {r["company"] for r in similar} == {"Beta", "Gamma"}  # not itself, not its duplicate
    assert all(set(r) == {"title", "company", "score", "decision", "similarity"} for r in similar)
    assert "private" not in client.results[1]


def test_v3_needs_an_embedder(store, monkeypatch):
    s, _ = store
    monkeypatch.setattr("jobagent.rag.embed.get_embedder", lambda: None)
    with pytest.raises(RuntimeError):
        Agent(s, CV, {}, client=object(), prompt_version="v3")


def test_eval_mode_never_returns_eval_labels(store):
    s, ids = store
    client = ScriptClient([[("find_similar_jobs", {"job_id": ids["new"]})]])
    Agent(s, CV, {}, client=client, prompt_version="v3", embedder=FakeEmbedder(),
          exclude_feedback_sources=EXCLUDED_FEEDBACK_SOURCES).run()
    assert [r["company"] for r in json.loads(client.results[0])] == ["Beta"]


def test_history_for_the_eval_skips_eval_cases_and_eval_labels(store, tmp_path):
    hist, ids = store
    hist.conn.execute("CHECKPOINT")
    hist_copy = tmp_path / "hist.duckdb"  # the fixture keeps the original open
    shutil.copy(hist.path, hist_copy)
    cases = [{"case_id": ids["old1"], "company": "Beta", "title": "Data Engineer", "location": "", "department": "",
              "url": "u", "description": "Kafka"},  # an eval case that is also in the history: must not leak
             {"case_id": "c2", "company": "Zeta", "title": "BI Developer", "location": "", "department": "",
              "url": "u", "description": "Power BI"}]
    ev = Store(tmp_path / "eval.duckdb")
    fill_store(ev, cases, batch_seed=0)
    n = load_history(ev, hist_copy, cases)
    copied = {r[0] for r in ev.conn.execute("SELECT job_id FROM job_feedback").fetchall()}
    assert n == 1 and copied == {ids["dup"]}  # old1 is an eval case, old2 is an eval label
    assert ev.conn.execute("SELECT count(*) FROM job_feedback WHERE note IS NOT NULL").fetchone()[0] == 0
    assert len(ev.pending_scoring()) == 2  # history jobs are never scored
    ev.close()
