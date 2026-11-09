-- How often and when each job was seen live on the careers site.
select
    job_id,
    count(*)          as observed_runs,
    min(observed_at)  as first_observed_at,
    max(observed_at)  as last_observed_at
from {{ ref('stg_job_observations') }}
group by job_id
