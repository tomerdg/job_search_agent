select
    s.run_id,
    r.started_at                 as run_at,
    trim(s.company)              as company,
    s.source_type,
    s.ok                         as fetch_ok,
    s.n_jobs,
    s.duration_s,
    s.error is not null          as has_error
from {{ source('raw', 'source_runs') }} s
join {{ ref('stg_runs') }} r using (run_id)
