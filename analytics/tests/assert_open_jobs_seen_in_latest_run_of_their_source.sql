-- An open job should have been seen in the last run where its company's source returned jobs.
-- Otherwise the agent failed to close it. Warn only: a source can be legitimately skipped.
{{ config(severity='warn') }}

with last_ok_run as (
    select company, max(run_id) as run_id
    from {{ ref('stg_source_runs') }}
    where fetch_ok and n_jobs > 0
    group by company
)

select f.job_id
from {{ ref('fct_job_postings') }} f
join last_ok_run l using (company)
left join {{ ref('stg_job_observations') }} o on o.job_id = f.job_id and o.run_id = l.run_id
where not f.is_closed and o.job_id is null
