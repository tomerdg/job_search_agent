-- Alert: in a week with enough jobs scored by both, the relevance model and the LLM agreed on too few.
{{ config(severity='warn') }}

select week_start, n_both_scored, agreement
from {{ ref('mart_ml_vs_llm') }}
where n_both_scored >= {{ var('ml_min_jobs_per_week') }}
  and agreement < {{ var('ml_min_agreement') }}
