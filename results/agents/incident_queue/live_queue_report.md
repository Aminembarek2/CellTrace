# Real incident queue: live investigation

All 18 distinct incidents in the population queue (19 sustained low rate episodes from eight
sessions, mapped to their earliest overlapping unexplained drop) were investigated live on
12 October 2026 by `deepseek/deepseek-v4.1-flash`, with the Investigator, the Reviewer and code
verification, under the live allowances in the technical report and a USD 1 spending cap.
Measurements were not modified. Reports and transcripts stay private, and this page publishes
aggregates only ([`live_summary.json`](live_summary.json)).

| Outcome | Count |
| --- | ---: |
| Incidents attempted | 18 of 18 |
| Completed reports | 16 |
| Runs that reached the shared tool request limit before submitting findings | 2 |
| Completed reports with passing code checks and a checked active rate drop | 16 |
| Reviewer decisions | 15 completed reports accepted, 1 left unresolved, 1 limited review of a run that reached the tool limit |
| Reviewer queries of its own per review | 4 to 11, median 6 |
| Runs whose first findings failed code checks and were repaired | 4 |
| Verified typed claims | 290 (232 in the completed reports, 36 and 22 in the two runs that reached the limit) |
| Provider attempts, unknown usage | 181, 0 |
| Active time | 1915 s, about 106 s per incident |
| Cost, known from provider responses | USD 0.277 plus USD 0.002 for the capability check |

The two runs that reached the tool limit close with reports assembled by code from the
evidence they had collected. The Reviewer shares the run's tool pool, which is why this limit
appears.

## What the investigations show together

- 13 of the 18 incidents come from three sessions recorded as static, eight of them from one
  session, and all 13 are on the same 5G serving cell, 10805003. The other five come from five
  driving sessions on four cells.
- Of those 13, 11 completed. In all 11, median reported RSRP moved by at most 1 dB while the
  rate fell. Reported SNR fell by 6.0 and 8.25 dB in two of them, and every completed report
  kept radio degradation as a consistent explanation. Mobility was marked contradicted in 4,
  not testable in 5 and consistent in 2. The static label comes from the archive's folder name.
- Every report keeps the cause unresolved and names the measurements that would separate the
  remaining explanations: cell load or PRB utilization, server and transport timing, and NR
  radio quality (SINR, MCS).

Reading: repeated drops at a stationary site while reported RSRP stays steady point the next
measurement at cell load, the transport path and NR radio quality. The archive cannot measure
any of them, so the reports leave the cause open.

## Scope

There are no truth labels for these incidents, so this measures completion, code checked
facts, review behaviour and cost, not diagnostic accuracy. The queue was selected
retrospectively from the archive used throughout the project, with one trial per incident. The
repeatability of the Investigator with code checks was measured separately on constructed
cases (23 of 27 pass all three trials).
