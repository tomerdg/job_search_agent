# Evaluation

Measures whether the scoring agent (`jobagent/agent.py`) gives high scores to the right jobs, so a prompt
or model change can be judged by numbers instead of by eye.

## The dataset

| File | In git | What |
|---|---|---|
| `dataset/jobs.jsonl.enc` | yes, encrypted | Job postings, one JSON object per line: `case_id, company, title, location, department, url, description` (description trimmed to `JD_MAX_CHARS`). |
| `dataset/labels.csv.enc` | yes, encrypted | The owner's labels: `case_id,label,note` (extra columns are ignored). |
| `dataset/jobs.jsonl`, `dataset/labels.csv` | **no** | The plain copies. Local only. |

Both files are personal: the labels say which jobs the owner would apply to, and the job list includes
jobs the owner applied to. They are committed only encrypted with the `DB_PASSPHRASE` secret
(`scripts/dbcrypt.sh`), and the tools print counts and case ids, never labels next to job text. CI
decrypts them in memory with the same secret.

## Label definitions

Label each job as the owner, from the posting alone, before looking at any model score.

| Label | Meaning | Rough score the prompt should give |
|---|---|---|
| `fit` | I would apply: a target role, in an allowed location, and I meet most of the must-haves. | 85-100 |
| `maybe` | I might apply: a target role with real gaps (seniority, a missing core tool), or an adjacent data role I would consider (e.g. analytics engineer, BI developer with heavy SQL/ETL). | 60-84 |
| `no_fit` | I would not apply: not a data role, a data role in a different family (e.g. data scientist, data analyst without engineering), or a location I exclude. | 0-59 |

Rules that keep labels consistent:

- **Location counts.** A perfect role in an excluded location is `no_fit`, because the digest should not send it.
- **Judge the job, not the title.** "Data Engineer" that is really Excel reporting is `no_fit`; "BI Developer" that is mostly pipelines may be `maybe` or `fit`.
- **Seniority:** a lead or senior role you could realistically get is `maybe` at most if the gap is real; one clearly out of reach is `no_fit`.
- **When torn between two labels, pick the lower one and write why in `note`.** Notes are for you; they are encrypted with the labels.
- Aim for 40-60 jobs, with at least 10 `fit` and 10 `maybe`. Too few positives makes precision and recall meaningless.

## Building or extending the set

```bash
python scripts/export_eval_candidates.py --db data/jobs.duckdb --n 60   # draft into eval/out/ (git-ignored)
```

The export samples by stratum (already-scored jobs by score band, jobs the pre-filter dropped for
location, data-ish titles the keywords missed, and plain negatives) and round-robins across companies.
It leaves out the agent's score on purpose, so it does not anchor your labels.

To add postings by URL (e.g. jobs you applied to), list them in a git-ignored file and run
`python scripts/fetch_eval_postings.py eval/out/urls.txt --label fit --note applied`. It appends to the draft;
check the result, because closed postings and JavaScript-only pages come back empty or as site menus.

Then:

1. Review `eval/out/jobs.jsonl`; remove jobs with empty or useless descriptions.
2. Fill in `label` (and optionally `note`) in `eval/out/labels.csv`.
3. Copy both into `eval/dataset/`. To add more positives, append a line to `jobs.jsonl` by hand for a public
   posting from any company (use a unique `case_id`, e.g. `manual-001`) and a matching row in `labels.csv`.
4. Check and encrypt:

```bash
python -m eval.dataset check      # counts per label and any missing/blank/orphan rows
DB_PASSPHRASE=... python -m eval.dataset encrypt   # writes jobs.jsonl.enc and labels.csv.enc; each skipped when unchanged
```

Commit the two `.enc` files. On another machine, `DB_PASSPHRASE=... python -m eval.dataset decrypt`
restores the plain copies.

## Running an evaluation

```bash
pip install -r requirements.txt -r requirements-eval.txt
python -m eval.evaluate --prompt-version v1 --model claude-sonnet-5-5 --repeats 1
python -m eval.evaluate --prompt-version v2 --repeats 3 --baseline eval/dataset/baseline.json
```

Each run scores every labeled job with the production `Agent` (in a temporary database), prints a
numbers-only metrics table, saves per-case scores to `eval/out/runs/` (git-ignored), and logs an MLflow
run. One pass over the set costs about $0.80 with Sonnet; `--limit 15` is a cheap smoke test.

The threshold defaults to the production `MIN_SCORE` (59). `--batch-seed` fixes which jobs share a
conversation, so two prompts compared with the same seed see the same batches. By default every repeat
uses that same seed, so `score_std_mean` and `decision_flip_rate` measure sampling noise only.
`--vary-batches` gives repeat r the seed `batch-seed + r`, so they also include the effect of which jobs
share a batch (closer to production, where batches change daily). Use it for stability runs, not for
baseline comparisons: the baseline records its `batch_seed`, and `--baseline` warns when the run is not
batch-paired with it.

## Experiment tracking (MLflow)

Every run is logged to the `job-scoring` experiment in a local SQLite store (`mlflow.db`, git-ignored;
set `MLFLOW_TRACKING_URI` to use a server): parameters (prompt version, model, repeats, dataset hash,
threshold, git commit), all metrics, and artifacts (the prompt file, metrics JSON, the label x score-band
confusion matrix as CSV and PNG, a score-by-label box plot). The per-case predictions CSV is logged only
locally, never in CI.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db          # browse runs at http://127.0.0.1:5000
python -m eval.compare                                     # last 10 runs, best F1 first
python -m eval.compare --last 30 --by prompt_version,model # mean per prompt and model
```
