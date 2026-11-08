from jobagent.demo import build
from jobagent.db import Store


def test_demo_db_has_history(tmp_path):
    db = tmp_path / "demo.duckdb"
    build(db, days=30)
    s = Store(db)
    count = lambda t: s.conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    assert count("runs") >= 10
    assert count("jobs") > 0 and count("job_observations") > count("jobs")
    assert s.conn.execute("SELECT count(*) FROM jobs WHERE NOT is_active").fetchone()[0] > 0
    s.close()
