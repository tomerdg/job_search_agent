-- Shadow mode check: how the relevance model's guess (before the LLM saw the job) compares with the LLM score.
-- One row per week of ML scoring. "keep" = the model would send the job to the LLM (ml_score >= threshold).
with scored as (
    select
        date_trunc('week', ml_scored_at)::date                          as week_start,
        ml_score,
        ml_score >= {{ var('ml_keep_threshold') }}                     as ml_keep,
        match_score,
        match_score >= {{ var('llm_match_threshold') }}                as llm_match
    from {{ ref('stg_jobs') }}
    where ml_score is not null
)

select
    week_start,
    count(*)                                                            as n_ml_scored,
    count(match_score)                                                  as n_both_scored,
    avg(case when not ml_keep then 1.0 else 0.0 end)                    as would_be_filtered_share,
    avg(case when match_score is not null
             then (ml_keep = llm_match)::int end)                       as agreement,
    sum(llm_match::int)                                                 as n_llm_matches,
    -- of the jobs the LLM rated a match, the share the model would have kept
    avg(case when llm_match then ml_keep::int end)                      as ml_recall_vs_llm,
    corr(ml_score, match_score)                                         as score_correlation
from scored
group by week_start
order by week_start
