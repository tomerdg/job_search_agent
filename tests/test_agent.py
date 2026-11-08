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
    s = Store(tmp_path / "jobs.db")
    j = Job(source="adam", company="Harel", external_id="1", title="Data Engineer", url="u",
            location="רמת גן", description="Airflow, Python")
    s.upsert([j])
    s.set_prefilter(j.job_id, True)
    n = Agent(s, "CV text", {"target_roles": ["Data Engineer"]}, client=FakeClient(j.job_id)).run()
    assert n == 1
    assert s.get(j.job_id)["score"] == 88
