select
    run_id,
    started_at,
    finished_at,
    date_diff('second', started_at, finished_at) as run_seconds,
    n_fetched,
    n_new
from {{ source('raw', 'runs') }}
