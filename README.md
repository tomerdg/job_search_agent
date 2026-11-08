# Job Search Agent

A daily agent that checks the careers sites of Israeli banks, insurance companies, investment houses,
pharma companies and accounting firms (the ones that often skip LinkedIn), keeps every job and its history
in DuckDB, scores new ones against a CV with Claude, and sends the 3 best matches to Telegram or email.
A dbt project on top of the same data models the hiring market: who hires most, how long jobs stay open,
and which sources stopped working.

## How it works

```
GitHub Actions (cron, Sun–Thu) → Docker image
  └─ fetchers (Adam, TopMatch, Comeet)   → jobs
  └─ fetchers (Adam, TopMatch, Comeet, Greenhouse) → jobs + per-source fetch stats
  └─ DuckDB data/jobs.duckdb (raw)       → dedup by job_id, incremental: only first-seen jobs are new;
                                           append-only history of runs, job observations, source health
  └─ pre-filter (config/profile.yaml)    → role keywords + Tel Aviv / Ramat Gan / Bnei Brak / Petah Tikva / Herzliya
  └─ Claude tool-calling agent           → tools: list_new_jobs, read_job_description, read_cv, save_match
  └─ digest                              → top 3 unsent matches → Telegram / email
  └─ dbt build (analytics/)              → staging → intermediate → marts in data/warehouse.duckdb + data tests
  └─ data/jobs.duckdb kept in the Actions cache between runs (never committed)
```

| Part | File |
|---|---|
| Sources (companies + public tokens) | `config/sources.yaml` |
| Target roles, keywords, locations | `config/profile.yaml` (git-ignored; template in `config/profile.example.yaml`) |
| Fetchers | `jobagent/fetchers/{adam,topmatch,comeet}.py` |
| Storage (DuckDB raw layer) | `jobagent/db.py` |
| dbt models and tests | `analytics/` |
| Synthetic demo data | `jobagent/demo.py` |
| Agent | `jobagent/agent.py` |

Sources wired today: Harel, Migdal, Altshuler Shaham (Adam); Meitav, More, Analyst (TopMatch);
ONE ZERO, Medison (Comeet). Custom sites still to add: Leumi, Hapoalim, Discount, Mizrahi, Clal, Phoenix and others.

## Data layer (DuckDB + dbt)

The agent writes a small raw layer to `data/jobs.duckdb`; dbt builds everything else from it into a separate
`data/warehouse.duckdb`, with the raw file attached read-only. The warehouse can be deleted at any time.

**Raw tables** (written by `jobagent/db.py`)

| Table | Grain | Why |
|---|---|---|
| `jobs` | one job | current state: first/last seen, open or closed, pre-filter, score, sent |
| `runs` | one agent run | run time, jobs fetched, new jobs |
| `job_observations` | job × run (append-only) | which jobs were live in each run, for open-duration history |
| `source_runs` | source × run (append-only) | fetch ok?, jobs returned, duration: source health over time |

**dbt models** (`analytics/models`)

```
staging       stg_jobs, stg_runs, stg_job_observations, stg_source_runs       (views: rename, cast, clean)
intermediate  int_job_roles (title → role family, EN + HE), int_job_activity  (views)
marts         fct_job_postings      one row per job: opened, closed, days open (censoring flagged), role, match band
              dim_companies         company, sector, source, postings
              mart_company_hiring   who hires most (30/90 days), open now, median days to close, rank
              mart_hiring_weekly    opened / closed / open at week end, per sector and week
              mart_time_to_close    p50 / p75 days open by sector and role family
              mart_source_health    last fetch per source, empty runs in a row, ok / degraded / down
```

**Tests:** `unique`, `not_null`, `accepted_values` and `relationships` on keys and enums, a custom generic
`unique_combination` test for composite grains, singular tests (dates in order; open jobs were seen in the
latest good run), a dbt **unit test** for the Hebrew/English role classifier, and a `warn` test that fires when a
source returned nothing 3 runs in a row.

**Design choices**

- *Two files, raw read-only.* The app owns `jobs.duckdb`; dbt can never corrupt it, and the warehouse is disposable.
- *History as append-only facts.* `job_observations` and `source_runs` are never updated, so durations and
  source trends can be recomputed with new logic at any time.
- *Reproducible "now".* Marts use the latest run time, not the wall clock, so the same DB always gives the same result.
- *Closed means gone from a source that worked.* A job is closed only when its company was fetched fine and the job
  was missing, so a broken site does not close all its jobs.
- *Upgrade path.* An existing SQLite `jobs.db` is imported into `jobs.duckdb` once, automatically.

**Run it on demo data** (made-up companies, no personal data):

```bash
pip install -r requirements.txt -r requirements-dbt.txt
python -m jobagent.demo --db data/demo.duckdb
DBT_RAW_DB=data/demo.duckdb DBT_WAREHOUSE_DB=data/demo_warehouse.duckdb \
  dbt build --project-dir analytics --profiles-dir analytics
python -c "import duckdb; print(duckdb.connect('data/demo_warehouse.duckdb').sql('from marts.mart_company_hiring order by hiring_rank_90d'))"
```

On your real data, drop the two env vars: the defaults are `data/jobs.duckdb` and `data/warehouse.duckdb`.
CI runs the same `dbt build` on demo data for every change, and on the real cached DB after every scheduled run.

## Personal data stays out of git

| What | Locally | In GitHub Actions |
|---|---|---|
| CV | any PDF/MD path, set `cv_path` in `config/profile.yaml` or `CV_PATH` | secret `CV_TEXT` |
| Preferences (roles, keywords, locations) | `config/profile.yaml` | secret `PROFILE_YAML` (paste the whole file) |
| Job history + scores | `data/jobs.duckdb` | Actions cache, never committed |
| dbt marts | `data/warehouse.duckdb` | built on the runner and discarded; the log shows only test pass/warn counts |
| Daily digest | printed with `--no-notify` | Telegram / email only, never written to the log |

`.gitignore` blocks `*.pdf`, `*.docx`, `cv.*`, `config/profile.yaml`, `data/*.db`, `data/*.duckdb` and `data/*.csv`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m jobagent.main --no-llm --no-notify          # fetch + filter only
ANTHROPIC_API_KEY=... python -m jobagent.main --no-notify
```

Docker:

```bash
docker build -t job-search-agent .
docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/data:/app/data" -v "$PWD:/repo:ro" \
  -e PROFILE_PATH=/repo/config/profile.yaml -e CV_PATH=/repo/<your-cv>.pdf job-search-agent --no-notify
```

## GitHub setup

Repository secrets (Settings → Secrets and variables → Actions):

| Secret | Needed | What |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Claude API key |
| `PROFILE_YAML` | yes | full contents of your `config/profile.yaml` |
| `CV_TEXT` | yes | CV as plain text (phone and email are stripped before it reaches the model) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | for Telegram | bot from @BotFather; chat id from `getUpdates` |
| `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_TO` | for email | e.g. Gmail with an app password |

Run it by hand from the Actions tab (`daily-job-search` → Run workflow).

## Cost control

Only new jobs that pass the keyword and location filter reach the model, at most 15 per run (`BATCH` in
`jobagent/agent.py`), with prompt caching on. Model is set by `CLAUDE_MODEL` (default `claude-sonnet-5-5`).
