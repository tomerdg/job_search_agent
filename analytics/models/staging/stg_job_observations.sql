select
    run_id,
    job_id,
    observed_at
from {{ source('raw', 'job_observations') }}
