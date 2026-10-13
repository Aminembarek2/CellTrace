# Evaluation exp_template_s4a

> OFFLINE HARNESS CHECK with the deterministic template policy. These numbers describe the scripted baseline and the harness, not model quality.

Scheduled runs 27, dispatched 27, completed 24, unscored 0, source sessions 3.

| model_id | arm | runs | completed | partial | exhausted | failed | unavailable | runs_with_claims | mean_factual_support | fabricated_evidence_claims | mean_fact_completeness | pattern_recovery_rate | corrected_after_rejection | mean_provider_attempts | mean_input_tokens | total_estimated_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scripted-fixture | investigator_gated | 27 | 24 | 3 | 0 | 0 | 0 | 27 | 1.000 | 0 | 0.963 | 0.739 | 0 | 6.222 | 34623.111 | NA |

## Paired differences

- scripted-fixture fact_completeness:investigator_gated_minus_investigator_raw: mean NA over 0 pairs from 0 sessions, interval NA
- scripted-fixture fact_completeness:investigator_reviewer_gated_minus_investigator_gated: mean NA over 0 pairs from 0 sessions, interval NA
- scripted-fixture pattern_recovered:investigator_gated_minus_investigator_raw: mean NA over 0 pairs from 0 sessions, interval NA
- scripted-fixture pattern_recovered:investigator_reviewer_gated_minus_investigator_gated: mean NA over 0 pairs from 0 sessions, interval NA
- scripted-fixture factual_support:investigator_gated_minus_investigator_raw: mean NA over 0 pairs from 0 sessions, interval NA
- scripted-fixture factual_support:investigator_reviewer_gated_minus_investigator_gated: mean NA over 0 pairs from 0 sessions, interval NA

## Notes

- Constructed variants are observation patterns with a known construction, not physical causes.
- Cases built from one base window are correlated. Source sessions are the bootstrap unit.
- cause_status is constrained by the schema, so its appropriateness is not evidence of reasoning.
- Scores cover code checked claims and observation labels. Causal language and usefulness are not scored here.
- Review corrections and damage compare retained claim IDs in the immediate revision after a review. Initial failed and passed retained claims are the respective denominators. New failed claims are reported over all new claims, and dropped claims are counted separately. These changes do not establish reviewer causation.
