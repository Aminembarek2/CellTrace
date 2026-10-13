# Evaluation exp_fresh_heldout_20261009

> Live experiment. Inspect completed and unavailable counts before interpreting scores.

Scheduled runs 81, dispatched 81, completed 73, unscored 0, source sessions 3.

| model_id | arm | runs | completed | partial | exhausted | failed | unavailable | runs_with_claims | mean_factual_support | fabricated_evidence_claims | mean_fact_completeness | pattern_recovery_rate | corrected_after_rejection | mean_provider_attempts | mean_input_tokens | total_estimated_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek/deepseek-v4.1-flash | investigator_gated | 27 | 27 | 0 | 0 | 0 | 0 | 27 | 1.000 | 0 | 1.000 | 0.913 | 15 | 7.333 | 7358.741 | 0.303 |
| deepseek/deepseek-v4.1-flash | investigator_raw | 27 | 27 | 0 | 0 | 0 | 0 | 27 | 0.987 | 0 | 1.000 | 0.913 | 0 | 6.148 | 6381.593 | 0.260 |
| deepseek/deepseek-v4.1-flash | investigator_reviewer_gated | 27 | 19 | 0 | 8 | 0 | 0 | 27 | 0.998 | 0 | 1.000 | 0.826 | 7 | 10.778 | 14273.704 | 0.389 |

## Paired differences

- deepseek/deepseek-v4.1-flash fact_completeness:investigator_gated_minus_investigator_raw: mean 0.000 over 27 pairs from 3 sessions, interval NA
- deepseek/deepseek-v4.1-flash fact_completeness:investigator_reviewer_gated_minus_investigator_gated: mean 0.000 over 27 pairs from 3 sessions, interval NA
- deepseek/deepseek-v4.1-flash pattern_recovered:investigator_gated_minus_investigator_raw: mean 0.000 over 23 pairs from 3 sessions, interval NA
- deepseek/deepseek-v4.1-flash pattern_recovered:investigator_reviewer_gated_minus_investigator_gated: mean -0.087 over 23 pairs from 3 sessions, interval NA
- deepseek/deepseek-v4.1-flash factual_support:investigator_gated_minus_investigator_raw: mean 0.013 over 27 pairs from 3 sessions, interval NA
- deepseek/deepseek-v4.1-flash factual_support:investigator_reviewer_gated_minus_investigator_gated: mean -0.002 over 27 pairs from 3 sessions, interval NA

## Notes

- Constructed variants are observation patterns with a known construction, not physical causes.
- Cases built from one base window are correlated. Source sessions are the bootstrap unit.
- cause_status is constrained by the schema, so its appropriateness is not evidence of reasoning.
- Scores cover code checked claims and observation labels. Causal language and usefulness are not scored here.
- Review corrections and damage compare retained claim IDs in the immediate revision after a review. Initial failed and passed retained claims are the respective denominators. New failed claims are reported over all new claims, and dropped claims are counted separately. These changes do not establish reviewer causation.
