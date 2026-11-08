-- Weekly hiring trend per sector: postings opened, closed, and open at the end of each week.
with p as (
    select * from {{ ref('fct_job_postings') }}
),

weeks as (
    select unnest(generate_series(
        date_trunc('week', (select min(first_seen_at) from p)),
        date_trunc('week', {{ as_of() }}),
        interval 7 day
    )) as week_start
),

sectors as (
    select distinct coalesce(sector, 'unknown') as sector from p
)

select
    cast(w.week_start as date)                                                    as week_start,
    s.sector,
    count(p.job_id) filter (where date_trunc('week', p.first_seen_at) = w.week_start) as opened,
    count(p.job_id) filter (where date_trunc('week', p.first_seen_at) = w.week_start and p.is_data_role)
                                                                                      as data_roles_opened,
    count(p.job_id) filter (where p.is_closed and date_trunc('week', p.closed_at) = w.week_start) as closed,
    count(p.job_id) filter (where p.first_seen_at < w.week_start + interval 7 day
                              and (p.closed_at is null or p.closed_at >= w.week_start + interval 7 day))
                                                                                      as open_at_week_end
from weeks w
cross join sectors s
left join p on coalesce(p.sector, 'unknown') = s.sector
group by all
order by week_start, sector
