-- How long jobs stay open, by sector and role family. Only closed jobs count toward the percentiles.
select
    coalesce(sector, 'unknown')                             as sector,
    role_family,
    count(*)                                                as postings,
    count(*) filter (where is_closed)                       as closed_postings,
    quantile_cont(days_open, 0.5)  filter (where is_closed) as p50_days_open,
    quantile_cont(days_open, 0.75) filter (where is_closed) as p75_days_open,
    max(days_open) filter (where not is_closed)             as oldest_open_days
from {{ ref('fct_job_postings') }}
group by all
