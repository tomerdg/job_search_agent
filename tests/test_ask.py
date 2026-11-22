from types import SimpleNamespace as NS

import pytest

from eval import ask as A
from jobagent.ask import ask, check_citations, skill_facts, skills_in
from jobagent.db import Store
from jobagent.models import Job
from jobagent.rag.embed import FakeEmbedder
from jobagent.rag.index import index_jobs


class EchoClient:
    """Answers by citing the first source's job id, plus one id that was never retrieved."""

    def __init__(self):
        self.messages = self
        self.prompt = None

    def create(self, **kw):
        self.prompt = kw["messages"][0]["content"]
        first = self.prompt.split("job_id=")[1][:16]
        return NS(content=[NS(type="text", text=f"Acme asks for Kafka [{first}] and so does X [deadbeefdeadbeef].")],
                  usage=NS(input_tokens=100, output_tokens=20))


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "jobs.duckdb")
    s.upsert([Job(source="greenhouse", company=c, external_id=c, title="Data Engineer", url="u",
                  description=f"Requirements:\n{d}") for c, d in
              [("Acme", "Kafka streaming and Spark for real time data."), ("Beta", "Kafka and dbt models in Snowflake."),
               ("Gamma", "Tableau dashboards and SQL reporting.")]])
    index_jobs(s, FakeEmbedder(), scope="all")
    yield s
    s.close()


def test_citations_outside_the_retrieved_set_are_dropped():
    text, valid, invalid = check_citations("a [abc123] b [ffff00001111aaaa] c [abc123]", ["abc1234567890def"])
    assert valid == ["abc1234567890def"] and invalid == ["ffff00001111aaaa"]
    assert "[abc1234567890def]" in text and "citation removed" in text


def test_aggregate_question_gets_sql_counts(store):
    assert skills_in("Which companies asked for Kafka and dbt?") == ["dbt", "kafka"]
    facts = skill_facts(store, "Which companies asked for Kafka?")
    assert facts == ["2 job(s) mention kafka across 2 company(ies); by company: Acme (1), Beta (1)"]
    assert skill_facts(store, "Tell me about Kafka jobs") == []  # not an aggregate question
    assert skill_facts(store, "כמה משרות מבקשות Kafka?")[0].startswith("2 job(s)")


def test_ask_end_to_end_with_post_check(store):
    client = EchoClient()
    a = ask(store, FakeEmbedder(), "Which companies asked for Kafka?", client=client)
    assert "FACTS FROM SQL" in client.prompt and "SOURCES" in client.prompt
    assert len(a.cited) == 1 and a.invalid == ["deadbeefdeadbeef"]
    assert "citation removed" in a.text
    assert A.judge(a, {a.cited[0]}, {"Acme"})


def test_ask_with_no_match_does_not_call_the_model(store):
    a = ask(store, FakeEmbedder(), "zzzz qqqq", client=object(), since="2100-01-01")
    assert a.cited == [] and "No collected job" in a.text


def test_ask_eval_judge_and_summary():
    none = NS(text="The sources do not say.", cited=[], invalid=[])
    assert A.judge(none, set(), set()) and not A.judge(none, {"x"}, {"Acme"})
    s = A.summarize([{"hit": True, "valid": 2, "invalid": 0}, {"hit": False, "valid": 1, "invalid": 1}])
    assert s["hit_rate"] == 0.5 and s["citation_validity"] == 0.75
    assert len(A.load_questions()) == 10
