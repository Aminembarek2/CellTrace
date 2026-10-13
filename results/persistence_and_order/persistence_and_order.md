# Persistence and order checks

Snapshot `snap_dd48e8cf739fd2be`, source `1a47c02edbc0c873…`. Rules fixed in an internal plan before these results, summarized in `docs/TECHNICAL_REPORT.md`, appendix. Exploratory: this archive was analysed before.

## Low rate persistence against rank Gaussianized IAAFT surrogates

Controls first, seed 20261013, 20 synthetic archives per family with one empty bin in seven and 99 surrogates each. Rejections of the long share at 0.05: negative controls with lag one coefficient 0.88: 1, negative controls with coefficient 0.95: 1, positive controls: 20. Required: at most 3 in each negative family and at least 16 in the positive family. Controls passed.

| Statistic | Observed | IAAFT null mean | 95% null range | p | Holm p |
| --- | ---: | ---: | --- | ---: | ---: |
| Episodes of 10 s or more under 1.5 Mbps | 74 | 58 | 46 to 68 | 0.015 | 0.015 |
| Share of low seconds inside such episodes | 0.517 | 0.253 | 0.207 to 0.300 | 0.005 | 0.010 |

Decision by the frozen rule: low rate stretches are longer than the rate's own autocorrelation and distribution predict.

## Order of drop onset and reported cell change, driving sessions

337 candidates have a reported change within 10 s of onset. The change comes after onset in 61.4% of them (ties count half), median lag 1.0 s. With change times shifted within eligible runs (1000 shifts) the share averages 52.9% (95% of shifts 48.9% to 57.1%), two sided p = 0.002.

Decision by the frozen rule: drops tend to begin before the reported change. A logged identity can trail the radio handover, so this is the order of logged events at one second resolution.

![Rate around a reported cell change](cell_change_profile.png)

| Contrast, after over before | Mean log10 ratio | Factor | 95% interval | Sessions |
| --- | ---: | ---: | --- | ---: |
| 1 s after against 1 s before | -0.012 | 0.97 | 0.53 to 1.77 | 13 |
| 5 s after against 5 s before | 0.127 | 1.34 | 0.86 to 2.08 | 15 |

118 changes with no other change within 20 s, inside one segment, with at least 80% active samples and a baseline of at least 0.5 Mbps from 20 to 10 s before. Intervals delete one session at a time (jackknife, Student t).

Limits: one operator in one city, a transition proxy rather than a handover record, and route and technology confounding unadjusted.
