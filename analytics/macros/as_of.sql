{# "Now" for the marts = the start of the latest run, so results are reproducible for a given DB. #}
{% macro as_of() -%}
    (select max(started_at) from {{ ref('stg_runs') }})
{%- endmacro %}
