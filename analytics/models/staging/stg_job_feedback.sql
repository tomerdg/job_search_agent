-- The note column stays out on purpose: free text, personal, not a feature.
select
    job_id,
    decision,
    decision = 'applied'          as is_positive,
    decided_at,
    source                        as feedback_source
from {{ source('raw', 'job_feedback') }}
