# Job Search Agent

A daily agent that checks the careers sites of Israeli banks, insurance companies, investment houses,
pharma companies and accounting firms (the ones that often skip LinkedIn), keeps new jobs in SQLite,
scores them against a CV with Claude, and sends the 3 best matches to Telegram or email.

## How it works

```
GitHub Actions (cron, Sun–Thu) → Docker image
  └─ fetchers (Adam, TopMatch, Comeet)   → jobs
  └─ SQLite data/jobs.db                 → dedup by job_id, incremental: only first-seen jobs are new
  └─ pre-filter (config/profile.yaml)    → role keywords + Tel Aviv / Ramat Gan / Bnei Brak / Petah Tikva / Herzliya
  └─ Claude tool-calling agent           → tools: list_new_jobs, read_job_description, read_cv, save_match
  └─ digest                              → top 3 unsent matches → Telegram / email
  └─ data/jobs.db kept in the Actions cache between runs (never committed)
```

| Part | File |
|---|---|
| Sources (companies + public tokens) | `config/sources.yaml` |
| Target roles, keywords, locations | `config/profile.yaml` (git-ignored; template in `config/profile.example.yaml`) |
| Fetchers | `jobagent/fetchers/{adam,topmatch,comeet}.py` |
| Storage | `jobagent/db.py` |
| Agent | `jobagent/agent.py` |

Sources wired today: Harel, Migdal, Altshuler Shaham (Adam); Meitav, More, Analyst (TopMatch);
ONE ZERO, Medison (Comeet). Custom sites still to add: Leumi, Hapoalim, Discount, Mizrahi, Clal, Phoenix and others.

## Personal data stays out of git

| What | Locally | In GitHub Actions |
|---|---|---|
| CV | any PDF/MD path, set `cv_path` in `config/profile.yaml` or `CV_PATH` | secret `CV_TEXT` |
| Preferences (roles, keywords, locations) | `config/profile.yaml` | secret `PROFILE_YAML` (paste the whole file) |
| Job history + scores | `data/jobs.db` | Actions cache, never committed |
| Daily digest | printed with `--no-notify` | Telegram / email only, never written to the log |

`.gitignore` blocks `*.pdf`, `*.docx`, `cv.*`, `config/profile.yaml`, `data/*.db` and `data/*.csv`.

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
