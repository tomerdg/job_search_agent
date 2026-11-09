select
    company,
    any_value(sector)           as sector,
    any_value(source_type)      as source_type,
    min(first_seen_at)          as first_seen_at,
    count(*)                    as total_postings,
    count(*) filter (where not is_closed) as open_postings
from {{ ref('fct_job_postings') }}
group by company
