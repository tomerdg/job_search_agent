-- One row per job posting with its lifecycle: when it opened, if/when it closed, how long it stayed open.
with jobs as (
    select * from {{ ref('stg_jobs') }}
),

roles as (
    select * from {{ ref('int_job_roles') }}
),

activity as (
    select * from {{ ref('int_job_activity') }}
),

as_of as (
    select {{ as_of() }} as as_of_at
)

select
    j.job_id,
    j.company,
    j.sector,
    j.source_type,
    j.title,
    r.role_family,
    r.role_family <> 'other'                                         as is_data_role,
    j.location,
    j.first_seen_at,
    cast(j.first_seen_at as date)                                    as first_seen_on,
    j.last_seen_at,
    not j.is_active                                                  as is_closed,
    j.closed_at,
    -- open jobs are right-censored: their duration so far is a lower bound
    date_diff('day', j.first_seen_at, coalesce(j.closed_at, a.as_of_at)) as days_open,
    j.is_active                                                      as is_duration_censored,
    coalesce(act.observed_runs, 0)                                   as observed_runs,
    j.passed_prefilter,
    j.match_score,
    case
        when j.match_score is null then 'not_scored'
        when j.match_score >= 85 then 'strong'
        when j.match_score >= 60 then 'good'
        when j.match_score >= 30 then 'weak'
        else 'no_fit'
    end                                                              as match_band,
    j.was_notified
from jobs j
join roles r using (job_id)
left join activity act using (job_id)
cross join as_of a
