-- Alert: a source returned no jobs (or failed) 3 runs in a row. Shows as a WARN in the dbt output.
{{ config(severity='warn') }}

select company, source_type, empty_or_failed_runs_in_a_row
from {{ ref('mart_source_health') }}
where status = 'down'
