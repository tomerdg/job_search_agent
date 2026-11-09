-- Who is hiring most, and how fast do their jobs close. One row per company.
with p as (
    select f.*, {{ as_of() }} as as_of_at from {{ ref('fct_job_postings') }} f
)

select
    company,
    any_value(sector)                                                             as sector,
    count(*) filter (where not is_closed)                                         as open_now,
    count(*) filter (where not is_closed and is_data_role)                        as data_roles_open_now,
    count(*) filter (where first_seen_at >= as_of_at - interval 30 day)           as posted_last_30d,
    count(*) filter (where first_seen_at >= as_of_at - interval 90 day)           as posted_last_90d,
    count(*) filter (where is_data_role and first_seen_at >= as_of_at - interval 90 day)
                                                                                  as data_roles_posted_last_90d,
    count(*) filter (where is_closed and closed_at >= as_of_at - interval 90 day) as closed_last_90d,
    median(days_open) filter (where is_closed)                                    as median_days_to_close,
    rank() over (order by count(*) filter (where first_seen_at >= as_of_at - interval 90 day) desc)
                                                                                  as hiring_rank_90d
from p
group by company
