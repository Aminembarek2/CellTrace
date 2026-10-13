# Nested session cross validation of delivered rate forecasts

21 file download sessions and 15714 forecast rows. Two outer schemes keep every session whole. Leave one session out tests transfer to an unseen session. Forward chaining trains only on sessions that ended before each test session started (14 test sessions, each after at least 7 earlier sessions), the rolling origin evaluation of Hyndman and Athanasopoulos applied to whole sessions.

Inside every outer fold, five inner folds of whole training sessions select one of the six predeclared configurations by out of fold mean absolute error, supply out of fold quantiles for interval calibration and fit the sigmoid probability calibrator. No outer test session influences its own configuration, calibrator or benchmark. Feature design and the configuration grid were developed earlier on this archive, so the results remain exploratory, not an independent held out estimate. Bootstrap intervals resample sessions, conditional on the saved predictions.

## Configuration selected inside each outer fold

| Scheme | Feature set | Folds per configuration index | Mean inner MAE by index |
| --- | --- | --- | --- |
| leave_one_session_out | history_only | 0: 8, 1: 2, 3: 11 | 19.78, 19.83, 20.00, 19.78, 20.12, 20.11 |
| leave_one_session_out | combined | 0: 14, 3: 7 | 20.15, 20.44, 20.63, 20.24, 20.42, 20.39 |
| forward_chaining | history_only | 0: 11, 1: 1, 3: 1, 4: 1 | 19.55, 19.71, 19.85, 19.72, 19.81, 19.80 |
| forward_chaining | combined | 0: 8, 5: 6 | 20.13, 20.54, 20.68, 20.37, 20.23, 20.13 |

## Point forecasts of the next 5 s mean rate

Leave one session out:

| Forecast | MAE (Mbps) | Skill against persistence | Sessions better than persistence | Median session MAE |
| --- | ---: | --- | ---: | ---: |
| persistence | 27.07 | 0.0% (0.0 to 0.0) | 0 of 21 | 19.46 |
| trailing_mean | 26.96 | 0.4% (-7.8 to 7.3) | 12 of 21 | 21.80 |
| history_only_mean | 19.51 | 27.9% (19.7 to 33.2) | 14 of 21 | 16.81 |
| combined_mean | 19.94 | 26.3% (17.6 to 32.0) | 16 of 21 | 17.83 |

Forward chaining:

| Forecast | MAE (Mbps) | Skill against persistence | Sessions better than persistence | Median session MAE |
| --- | ---: | --- | ---: | ---: |
| persistence | 29.27 | 0.0% (0.0 to 0.0) | 0 of 14 | 22.21 |
| trailing_mean | 27.65 | 5.5% (-2.0 to 11.4) | 9 of 14 | 23.05 |
| history_only_mean | 20.34 | 30.5% (22.5 to 36.1) | 10 of 14 | 17.70 |
| combined_mean | 20.77 | 29.0% (20.7 to 34.7) | 11 of 14 | 17.97 |

## Quantile forecasts (0.1, 0.5, 0.9), leave one session out

Scored with the quantile score, interval coverage and Winkler score. The benchmark adds the training sessions' residual quantiles to persistence.

| Forecast | Quantile score skill | 80% interval coverage | Mean width (Mbps) | Winkler | Missed by the 10th percentile | False warnings by the 10th percentile |
| --- | --- | --- | ---: | ---: | --- | --- |
| naive | 0.0% (0.0 to 0.0) | 79.2% (72.5 to 86.3) | 59.9 | 189.8 | 0.3% (0.0 to 0.8) | 71.5% (61.7 to 81.9) |
| history_only | 44.5% (40.5 to 47.7) | 75.6% (73.9 to 77.4) | 54.6 | 89.1 | 18.2% (12.6 to 26.1) | 6.0% (3.1 to 10.0) |
| combined | 43.8% (40.4 to 46.9) | 75.2% (73.9 to 76.6) | 57.2 | 90.6 | 19.3% (11.8 to 30.6) | 5.1% (2.9 to 8.0) |

"Missed by the 10th percentile" is the share of forecast rows whose future mean was below 1.5 Mbps while the cautious forecast was not. "False warnings" is the share of future means at or above 1.5 Mbps whose cautious forecast was below it. These are overlapping forecast windows, not independent seconds or incidents.

## Intervals calibrated with equal session weights

Conformalized quantile regression (Romano, Patterson and Candès) shifts both quantile bounds by a correction learned from out of fold predictions on every training session, with equal weight per session, the CDF pooling method of Dunn, Wasserman and Ramdas for grouped data. Global CQR uses one correction per outer fold. Mondrian CQR uses one per quartile of the predicted level, the midpoint of the 0.1 and 0.9 forecasts, because throughput spread grows with its level and one additive shift over covers low forecasts while under covering high ones. Both target 80% coverage of a new row from a new session, approximately, assuming exchangeable sessions. Coverage on average can hide groups below target, so the lowest predicted level quartile, the lowest current technology and the lowest session are shown. Values are empirical.

Leave one session out:

| Forecast | Interval | Row coverage | Lowest predicted level quartile | Lowest technology | Lowest session | Width (Mbps) | Winkler |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| history_only | raw quantiles | 75.6% (73.9 to 77.4) | 73.8% | 50.0% | 69.1% | 54.6 | 89.1 |
| history_only | global CQR | 80.0% (78.2 to 82.2) | 74.6% | 75.0% | 70.1% | 55.5 | 89.0 |
| history_only | Mondrian CQR | 81.9% (80.4 to 83.6) | 80.2% | 68.8% | 74.6% | 58.5 | 88.6 |
| combined | raw quantiles | 75.2% (73.9 to 76.6) | 73.1% | 62.5% | 64.0% | 57.2 | 90.6 |
| combined | global CQR | 79.8% (78.4 to 81.7) | 74.5% | 71.4% | 71.2% | 58.2 | 90.5 |
| combined | Mondrian CQR | 81.8% (80.5 to 83.3) | 79.3% | 71.4% | 74.0% | 60.9 | 90.3 |

Forward chaining:

| Forecast | Interval | Row coverage | Lowest predicted level quartile | Lowest technology | Lowest session | Width (Mbps) | Winkler |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| history_only | raw quantiles | 77.0% (73.9 to 79.7) | 75.3% | 54.5% | 68.0% | 59.0 | 95.1 |
| history_only | global CQR | 81.5% (77.9 to 85.0) | 76.2% | 72.7% | 69.7% | 60.2 | 95.1 |
| history_only | Mondrian CQR | 83.1% (79.4 to 86.2) | 80.4% | 72.7% | 71.3% | 63.0 | 95.1 |
| combined | raw quantiles | 75.7% (73.9 to 77.8) | 75.2% | 36.4% | 65.8% | 61.8 | 97.3 |
| combined | global CQR | 80.4% (77.8 to 83.5) | 75.8% | 72.7% | 72.0% | 62.7 | 97.2 |
| combined | Mondrian CQR | 81.7% (78.4 to 85.0) | 77.6% | 79.5% | 69.3% | 64.1 | 97.2 |

## Probability that the next 5 s mean is below 1.5 Mbps, leave one session out

Base rate 6.0% (948 rows). The sigmoid calibrator is fitted on out of fold scores from the inner session folds. When an inner fold lacks one outcome, the forecast falls back to the training prevalence and the fallback is recorded. A reliability diagram measures empirical calibration; it does not guarantee it.

| Forecast | Average precision | Brier | Brier skill against climatology | Precision at 0.5 | Recall at 0.5 |
| --- | ---: | ---: | --- | ---: | ---: |
| climatology_warning | 0.040 | 0.0569 | 0.0% (0.0 to 0.0) | n/a | 0.0% |
| naive_warning | 0.187 | 0.0466 | 18.2% (10.2 to 24.1) | n/a | 0.0% |
| history_only_warning | 0.661 | 0.0304 | 46.6% (33.1 to 56.6) | 72.7% | 53.1% |
| combined_warning | 0.687 | 0.0288 | 49.3% (36.7 to 59.7) | 71.8% | 62.2% |

The two-valued naive probability benchmark discards continuous rate rankings. Average precision is also compared with negative persistence and negative trailing mean, which are ranking scores rather than probabilities. The subset whose current rate is at least 1.5 Mbps asks whether a new low future mean is foreseen, rather than an already low one continuing.

| Ranking forecast | AP, all rows | AP, current rate at least 1.5 Mbps | Session mean AP difference against continuous persistence (descriptive) | Forward chaining AP, all rows | Forward chaining AP, current at least 1.5 Mbps |
| --- | ---: | ---: | --- | ---: | ---: |
| persistence_warning_score | 0.261 | 0.139 | 0.000 | 0.244 | 0.170 |
| trailing_mean_warning_score | 0.477 | 0.142 | 0.120 | 0.464 | 0.152 |
| history_only_warning | 0.661 | 0.200 | 0.326 | 0.597 | 0.197 |
| combined_warning | 0.687 | 0.234 | 0.337 | 0.636 | 0.213 |

The current-not-low subset has 13290 rows and 217 future low means under leave one session out. The target is a 5 s mean, not an instantaneous crossing, a playback stall or a physical fault.

## Could it run on a device?

| Model | Size | Single prediction (median) |
| --- | ---: | ---: |
| cautious_quantile_0.1 | 403 KiB | 3.13 ms |
| warning_sigmoid | 1019 KiB | 3.80 ms |

Fitted on every session with the configuration selected most often under leave one session out. Measured estimator prediction on precomputed encoded features in Python on the development laptop; excludes data loading, features, encoders and interval construction. This is not end-to-end device latency. Conversion was not attempted.

## Limits

21 sessions from one 2019 to 2020 archive, recorded on few days and routes. The target is delivered application rate, not capacity. The kbps unit is inferred. Rows inside a session overlap in time. Session start times are naive local wall clock values. The session-paired AP analysis excludes sessions without both classes, with identities saved; pooled metrics retain every forecast row.
