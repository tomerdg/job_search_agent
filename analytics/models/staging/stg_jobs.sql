with source as (
    select * from {{ source('raw', 'jobs') }}
)

select
    job_id,
    source                                         as source_type,
    trim(company)                                  as company,
    nullif(trim(sector), '')                       as sector,
    external_id,
    trim(title)                                    as title,
    nullif(trim(location), '')                     as location,
    nullif(trim(department), '')                   as department,
    url,
    try_cast(nullif(posted_at, '') as date)        as posted_on,
    first_seen                                     as first_seen_at,
    last_seen                                      as last_seen_at,
    is_active,
    case when not is_active then coalesce(closed_at, last_seen) end as closed_at,
    prefilter                                      as passed_prefilter,
    score                                          as match_score,
    score is not null                              as is_scored,
    notified_at,
    notified_at is not null                        as was_notified
from source
