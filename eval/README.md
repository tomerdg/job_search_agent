# Evaluation

Measures whether the scoring agent (`jobagent/agent.py`) gives high scores to the right jobs, so a prompt
or model change can be judged by numbers instead of by eye.

## The dataset

| File | In git | What |
|---|---|---|
| `dataset/jobs.jsonl` | yes | Public job postings, one JSON object per line: `case_id, company, title, location, department, url, description` (description trimmed to `JD_MAX_CHARS`). |
| `dataset/labels.csv.enc` | yes, encrypted | The owner's labels, encrypted with the `DB_PASSPHRASE` secret (`scripts/dbcrypt.sh`). |
| `dataset/labels.csv` | **no** | The plain labels: `case_id,label,note`. Local only. |

The labels say which jobs the owner would apply to, so they are personal: they are committed only
encrypted, and the tools print counts and case ids, never labels next to job text. CI decrypts them in
memory with the same secret.

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
DB_PASSPHRASE=... python -m eval.dataset encrypt   # writes labels.csv.enc; skipped when unchanged
```

Commit `jobs.jsonl` and `labels.csv.enc`. On another machine, `DB_PASSPHRASE=... python -m eval.dataset decrypt`
restores `labels.csv`.
