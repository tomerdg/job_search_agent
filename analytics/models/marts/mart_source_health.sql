-- Is each source still working? A source that keeps returning 0 jobs probably changed its site.
with runs as (
    select
        *,
        row_number() over (partition by company, source_type order by run_at desc) as recency
    from {{ ref('stg_source_runs') }}
),

last_good as (
    select company, source_type, max(run_at) as last_nonzero_at
    from runs
    where fetch_ok and n_jobs > 0
    group by all
),

streak as (
    -- runs since the last run that returned jobs
    select r.company, r.source_type, count(*) as empty_or_failed_runs_in_a_row
    from runs r
    left join last_good g using (company, source_type)
    where g.last_nonzero_at is null or r.run_at > g.last_nonzero_at
    group by all
)

select
    r.company,
    r.source_type,
    r.run_at                                                      as last_run_at,
    r.fetch_ok                                                    as last_fetch_ok,
    r.n_jobs                                                      as last_n_jobs,
    g.last_nonzero_at,
    coalesce(s.empty_or_failed_runs_in_a_row, 0)                  as empty_or_failed_runs_in_a_row,
    (select avg(n_jobs) from runs x
      where x.company = r.company and x.source_type = r.source_type and x.recency <= 10) as avg_jobs_last_10_runs,
    case
        when coalesce(s.empty_or_failed_runs_in_a_row, 0) >= 3 then 'down'
        when not r.fetch_ok or r.n_jobs = 0                    then 'degraded'
        else 'ok'
    end                                                           as status
from runs r
left join last_good g using (company, source_type)
left join streak s using (company, source_type)
where r.recency = 1
