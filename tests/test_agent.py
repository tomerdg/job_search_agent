from types import SimpleNamespace as NS

from jobagent.agent import Agent
from jobagent.db import Store
from jobagent.models import Job


class FakeClient:
    """Plays a scripted conversation: read_cv + list, then read JD, then save, then stop."""

    def __init__(self, job_id):
        self.calls = 0
        self.job_id = job_id
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        tu = lambda i, name, inp: NS(type="tool_use", id=f"t{i}", name=name, input=inp)
        script = {
            1: [tu(1, "read_cv", {}), tu(2, "list_new_jobs", {})],
            2: [tu(3, "read_job_description", {"job_id": self.job_id})],
            3: [tu(4, "save_match", {"job_id": self.job_id, "score": 88, "summary": "מתאים",
                                     "strengths": ["Airflow"], "gaps": []})],
        }
        return NS(content=script.get(self.calls, [NS(type="text", text="DONE")]))


def test_agent_scores_job(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    j = Job(source="adam", company="Acme", external_id="1", title="Data Engineer", url="u",
            location="רמת גן", description="Airflow, Python")
    s.upsert([j])
    s.set_prefilter(j.job_id, True)
    n = Agent(s, "CV text", {"target_roles": ["Data Engineer"]}, client=FakeClient(j.job_id)).run()
    assert n == 1
    assert s.get(j.job_id)["score"] == 88


class UsageClient(FakeClient):
    """Same script, with usage on every response. Turn 2 omits the cache fields, turn 3 sets one to None."""

    USAGE = {
        1: NS(input_tokens=1000, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=900),
        2: NS(input_tokens=20, output_tokens=30),
        3: NS(input_tokens=10, output_tokens=40, cache_read_input_tokens=950, cache_creation_input_tokens=None),
        4: NS(input_tokens=5, output_tokens=2, cache_read_input_tokens=1000, cache_creation_input_tokens=0),
    }

    def create(self, **kw):
        resp = super().create(**kw)
        resp.usage = self.USAGE[self.calls]
        return resp


def _one_job_store(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    j = Job(source="adam", company="Acme", external_id="1", title="Data Engineer", url="u",
            location="רמת גן", description="Airflow, Python")
    s.upsert([j])
    s.set_prefilter(j.job_id, True)
    return s, j


def test_agent_sums_usage_and_records_provenance(tmp_path):
    s, j = _one_job_store(tmp_path)
    agent = Agent(s, "CV text", {}, client=UsageClient(j.job_id), model="claude-sonnet-5-5", prompt_version="v1")
    agent.run()
    assert agent.turns == 4
    assert agent.usage == {"input_tokens": 1035, "output_tokens": 122,
                           "cache_read_input_tokens": 1950, "cache_creation_input_tokens": 900}
    # sonnet-5-5: $2 in, $10 out, $0.10 cache read, cache write 1.25 x $2 = $2.50
    expected = (1035 * 2 + 122 * 10 + 1950 * 0.10 + 900 * 2.50) / 1e6
    assert abs(agent.cost_usd - expected) < 1e-12
    row = s.get(j.job_id)
    assert (row["prompt_version"], row["model"]) == ("v1", "claude-sonnet-5-5")


def test_cost_unknown_model_is_none():
    from jobagent.costs import estimate_cost_usd
    assert estimate_cost_usd({"input_tokens": 10}, "no-such-model") is None


def test_v1_prompt_renders(tmp_path):
    from jobagent.agent import load_prompt
    text = load_prompt("v1").format(roles="Data Engineer", allowed="any", blocked="none")
    assert "Target roles: Data Engineer." in text and text.endswith("DONE.")


def test_bad_prompt_version_rejected(tmp_path):
    import pytest
    s, _ = _one_job_store(tmp_path)
    for bad in ("../secrets", "nope"):
        with pytest.raises((ValueError, FileNotFoundError)):
            Agent(s, "CV", {}, client=object(), prompt_version=bad)


def test_old_db_gets_new_columns(tmp_path):
    import duckdb
    path = tmp_path / "old.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE jobs (job_id VARCHAR PRIMARY KEY, source VARCHAR NOT NULL, company VARCHAR NOT NULL,"
                " sector VARCHAR, external_id VARCHAR, title VARCHAR NOT NULL, location VARCHAR, department VARCHAR,"
                " url VARCHAR, description VARCHAR, posted_at VARCHAR, first_seen TIMESTAMP NOT NULL,"
                " last_seen TIMESTAMP NOT NULL, is_active BOOLEAN NOT NULL DEFAULT TRUE, closed_at TIMESTAMP,"
                " prefilter BOOLEAN, score INTEGER, analysis VARCHAR, notified_at TIMESTAMP)")
    con.execute("INSERT INTO jobs (job_id, source, company, title, first_seen, last_seen, score)"
                " VALUES ('a', 'adam', 'Acme', 'DE', now(), now(), 70)")
    con.close()
    s = Store(path)
    row = s.get("a")
    assert row["score"] == 70 and row["prompt_version"] is None and row["model"] is None
