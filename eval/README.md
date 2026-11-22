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

## CI quality gate

`.github/workflows/eval.yml` runs on every PR that changes `jobagent/agent.py`, `jobagent/costs.py`,
`jobagent/prompts/`, `eval/` (not its Markdown) or `requirements-eval.txt`, and on demand from the Actions
tab (choose prompt version, model and repeats). It runs:

```bash
python -m eval.evaluate --repeats 1 --baseline eval/dataset/baseline.json --gate
```

The gate (`eval/gate.py`, thresholds at the top of the file) fails the PR when:

- coverage is below 98% (the agent did not score every job), or
- a floor is missed at the production threshold: precision for fit-or-maybe 0.80, recall for fit-or-maybe
  0.90, recall for fit 0.85, ROC AUC 0.90, or
- the run is significantly worse than the baseline (paired Wilcoxon on per-case error, p < 0.05).

It always prints which rule passed or failed, in the log and in the Job Summary. A run costs about $0.70.
The dataset, labels and baseline are decrypted in memory with `DB_PASSPHRASE`; the CV and profile come from
the `CV_TEXT` and `PROFILE_YAML` secrets. Without them (a PR from a fork) the job is skipped, not failed.
Only numbers are uploaded (`eval-metrics` artifact: metrics.json, info.json, confusion.csv); per-case
scores are not.

After merging an intentional improvement, refresh the baseline so later PRs are compared with it:

```bash
python -m eval.evaluate --prompt-version <new> --repeats 3 --write-baseline
DB_PASSPHRASE=... python -m eval.dataset encrypt    # then commit eval/dataset/baseline.json.enc
```

## Experiment: prompt v1 vs v2, and a cheaper model (2026-11-12 to 2026-11-15)

82 labeled jobs (15 fit, 8 maybe, 59 no_fit), threshold 59, 3 repeats per arm, same batch seed. v2
(`jobagent/prompts/v2.txt`) defines the score bands by "would the candidate apply", to separate fit
from maybe better. The comparison test is a paired Wilcoxon on per-case error against the v1 baseline.

| | v1, Sonnet 5.5 | v2, Sonnet 5.5 | v1, Haiku 5.5 |
|---|---|---|---|
| ROC AUC (fit vs rest) | 0.948 | 0.961 | 0.949 |
| Spearman | 0.754 | 0.774 | 0.752 |
| Precision at 59, fit or maybe counts as a hit | 0.885 | 0.917 | 0.846 |
| Recall at 59, fit or maybe | 1.000 | 0.957 | 0.957 |
| Recall at 59, fit only | 1.000 | 1.000 | 0.933 |
| Precision@3 per 15-job day | 0.500 | 0.556 | 0.611 |
| Mean score: fit / maybe | 78.5 / 72.9 | not logged | 74.9 / 70.5 |
| Decision flip rate across repeats | 2.4% | 2.4% | 4.9% |
| Cost per pass over the set | ~$0.70 | not logged | ~$0.06 |
| vs v1 baseline | - | no significant difference (p=0.45) | no significant difference (p=0.56) |

Result: **v1 on Sonnet stays in production.**

- v2 is not worse and pushes no_fit jobs a little lower, but the whole gain is two jobs out of 82 and it
  misses one maybe that v1 keeps. Not significant; not a reason to switch.
- Haiku costs about 12x less, ranks as well (same AUC), but drops one fit job below the threshold and is
  twice as unstable. A missed job you would apply to is the most expensive error here, and at 15 new jobs
  per run the saving is about $0.10 per run, so it is not worth it.
- The limit is the data, not the repeats: run-to-run noise is small (mean score spread 1.3 points), and the
  uncertainty comes from having only 15 fit and 8 maybe cases. More labeled fit/maybe jobs will make the
  next comparison decisive.
- The v2 run crashed after scoring (since fixed), so its tokens, cost and fit/maybe means were not kept and
  it is not in MLflow; its metrics were recomputed from the saved per-case scores.

## Retrieval evaluation

`python -m eval.retrieval` measures the JD search in `jobagent/rag/retrieve.py`, separately from scoring.
It indexes the eval set's JDs (82 jobs, decrypted in memory in CI) with the real embedding model and runs
the queries in `eval/retrieval/queries.jsonl` in each mode: `vector`, `bm25` and `hybrid` (reciprocal rank
fusion, k=60). Ranking is per job (one chunk per job), and the metrics are recall@5, recall@10, MRR and
nDCG@10 (`eval/metrics.py`). recall@k divides by min(k, relevant jobs), so a query with more than k relevant
jobs can still reach 1. One MLflow run per mode goes to the experiment `retrieval` (params: model id,
chunking version, k, corpus size, queries hash; metrics only, no text).

The queries are generic (skills, Hebrew phrasings, role types, paraphrases). Each has a **rule** for which
jobs are relevant (skills named in the JD, the title's role family, or a regex), not a list of job ids, so
the file is public and runs on any corpus. `scripts/draft_retrieval_queries.py` drafts it and prints how
many jobs each rule matches. The catch: a skill rule is a keyword match, which favours BM25 by construction,
so results are reported per kind, and the paraphrase queries are the ones where vectors have to earn it.

`.github/workflows/retrieval-eval.yml` runs it on PRs that touch the RAG code or the queries (no API cost)
and fails when hybrid recall@10 drops below 0.7. `--db data/jobs.duckdb` runs it on a copy of a jobs DB.

### First run (2026-11-22, `paraphrase-multilingual-MiniLM-L12-v2`, chunking c1, 36 queries)

| mode | all: recall@10 | MRR | nDCG@10 | skill (19) | Hebrew (7) | role (5) | paraphrase (5) |
|---|---|---|---|---|---|---|---|
| vector | 0.436 | 0.683 | 0.441 | 0.348 | 0.329 | 0.900 | 0.460 |
| bm25 | 0.755 | 0.898 | 0.764 | 0.920 | 0.414 | 0.860 | 0.500 |
| hybrid | **0.773** | **0.910** | **0.773** | 0.843 | **0.586** | 0.860 | **0.680** |

(per-kind columns are recall@10)

- Hybrid is best overall and on the Hebrew and paraphrase queries, and beats vector-only on skill names
  by a wide margin (0.843 vs 0.348), which is the reason for keeping BM25.
- BM25 alone is best on skill queries, partly by construction (keyword rules).
- Vector-only is weak. One known cause: this model truncates its input at 128 tokens, so most of an
  ~800-character chunk is never seen (Hebrew uses more tokens per character). Next things to try, measured
  with this same eval: smaller chunks, or a model with a longer input (`paraphrase-multilingual-mpnet-base-v2`,
  512 tokens, 768 dims, ~1 GB).
