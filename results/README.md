# Results

Curated outputs behind the [README](../README.md) and the
[technical report](../docs/TECHNICAL_REPORT.md). Each folder holds what the command or script
named below wrote: a report or a JSON summary, and figures where the analysis has them. Raw
measurements are not included, see the README for the public sources.

| Folder | What it shows | Produced by |
| --- | --- | --- |
| [`population/`](population/population_report.md) | Service levels, drop rates, chance checks, drop triage, monitoring coverage and the cell replication figure (Q0 to Q7) | `celltrace population` |
| [`persistence_and_order/`](persistence_and_order/persistence_and_order.md) | Low rate stretches against autocorrelation preserving surrogates, and whether drops precede logged cell changes | `scripts/strengthen_population_tests.py` |
| [`cells/raca_2019_2020/`](cells/raca_2019_2020/cell_study_report.md) | Slow cells across days on the Irish 5G archive, and the false discovery check of the slow cell selection under three prior shapes (`prior_robustness.json`) | `celltrace cells` |
| [`cells/lte_2018_operator_a/`](cells/lte_2018_operator_a/cell_study_report.md), [`cells/lte_2018_operator_b/`](cells/lte_2018_operator_b/cell_study_report.md) | The same study replicated on the 2017 and 2018 LTE dataset, per operator, with the power check run before the data was used (`power_check.json`) | `celltrace cells`, `scripts/cell_replication_power.py` |
| [`forecasting/cross_validation/`](forecasting/cross_validation/cross_validation_report.md) | Nested session cross validation of the 5 s rate forecast, intervals and early warning, and the comparison with simple baselines (`simple_baselines.json`, recomputed from the saved predictions) | `celltrace predict cross-validate` |
| [`forecasting/external_validation/`](forecasting/external_validation/external_validation_report.md) | The external test on US drives, run once | `celltrace predict external-validate` |
| [`agents/repeatability/`](agents/repeatability/repeatability.json) | Three isolated trials on the 27 constructed cases of the unseen session study, rerun with the later prompts | `celltrace benchmark`, `celltrace score`, `scripts/agent_repeatability.py` |
| [`agents/unseen_sessions/`](agents/unseen_sessions/summary.md) | The first run on those 27 cases, built from sessions no earlier case used | `celltrace benchmark`, `celltrace score` |
| [`agents/template_baseline/`](agents/template_baseline/comparison.json) | The scripted template on the same 27 cases, compared case by case with the agent's three trials by an exact McNemar test | `celltrace benchmark --mode offline`, `celltrace score` |
| [`agents/planted_errors/`](agents/planted_errors/planted_finding_level.json) | What the Reviewer catches when errors are planted, per distinct finding with exact intervals, and the per conversation summary of the latest study | `celltrace planted-errors`, `scripts/planted_finding_level.py` |
| [`agents/incident_queue/`](agents/incident_queue/live_queue_report.md) | Live investigation of the 18 real low rate incidents | `celltrace investigate-queue --live` |
| [`agents/blind_rating/`](agents/blind_rating/summary.json) | Blind paired rating of agent and template reports by two model raters | `scripts/blind_review.py` |
