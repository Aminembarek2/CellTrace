# Evaluation exp_repeat_20261012_trial_3

> Live experiment. Inspect completed and unavailable counts before interpreting scores.

Scheduled runs 27, dispatched 27, completed 27, unscored 0, source sessions 3.

| model_id | arm | runs | completed | partial | exhausted | failed | unavailable | runs_with_claims | mean_factual_support | fabricated_evidence_claims | mean_fact_completeness | pattern_recovery_rate | corrected_after_rejection | mean_provider_attempts | mean_input_tokens | total_estimated_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek/deepseek-v4.1-flash | investigator_gated | 27 | 27 | 0 | 0 | 0 | 0 | 27 | 1.000 | 0 | 0.963 | 0.870 | 15 | 6.963 | 6488.333 | 0.593 |

## Paired differences

- deepseek/deepseek-v4.1-flash fact_completeness:investigator_gated_minus_investigator_raw: mean NA over 0 pairs from 0 sessions, interval NA
- deepseek/deepseek-v4.1-flash fact_completeness:investigator_reviewer_gated_minus_investigator_gated: mean NA over 0 pairs from 0 sessions, interval NA
- deepseek/deepseek-v4.1-flash pattern_recovered:investigator_gated_minus_investigator_raw: mean NA over 0 pairs from 0 sessions, interval NA
- deepseek/deepseek-v4.1-flash pattern_recovered:investigator_reviewer_gated_minus_investigator_gated: mean NA over 0 pairs from 0 sessions, interval NA
- deepseek/deepseek-v4.1-flash factual_support:investigator_gated_minus_investigator_raw: mean NA over 0 pairs from 0 sessions, interval NA
- deepseek/deepseek-v4.1-flash factual_support:investigator_reviewer_gated_minus_investigator_gated: mean NA over 0 pairs from 0 sessions, interval NA

## Notes

- Constructed variants are observation patterns with a known construction, not physical causes.
- Cases built from one base window are correlated. Source sessions are the bootstrap unit.
- cause_status is constrained by the schema, so its appropriateness is not evidence of reasoning.
- Scores cover code checked claims and observation labels. Causal language and usefulness are not scored here.
- Review corrections and damage compare retained claim IDs in the immediate revision after a review. Initial failed and passed retained claims are the respective denominators. New failed claims are reported over all new claims, and dropped claims are counted separately. These changes do not establish reviewer causation.
