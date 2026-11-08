-- A job cannot be last seen or closed before it was first seen.
select job_id
from {{ ref('fct_job_postings') }}
where last_seen_at < first_seen_at
   or (closed_at is not null and closed_at < first_seen_at)
   or days_open < 0
