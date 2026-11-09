-- Classify each job title into a role family. Titles are English or Hebrew.
with jobs as (
    select job_id, title, lower(title) as t from {{ ref('stg_jobs') }}
)

select
    job_id,
    case
        when regexp_matches(t, 'analytics engineer')                                  then 'analytics_engineering'
        when regexp_matches(t, 'data engineer|data platform|etl|מהנדס/?ת? נתונים|מהנדס/?ת? דאטה')
                                                                                      then 'data_engineering'
        when regexp_matches(t, '(^|[^a-z])bi([^a-z]|$)|business intelligence|power bi|tableau')
                                                                                      then 'bi'
        when regexp_matches(t, 'data scien|machine learning|(^|[^a-z])ml([^a-z]|$)')  then 'data_science'
        when regexp_matches(t, 'data analyst|analyst|אנליסט|אנליסטית')                then 'analytics'
        else 'other'
    end as role_family
from jobs
