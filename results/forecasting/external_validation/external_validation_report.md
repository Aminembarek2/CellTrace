# External validation on PAM 2025

Rendered from the saved summary with a persistence column added. No model was refitted and no number changed.

The history only forecast was trained on all 21 Raca file download sessions from Ireland, 2019 to 2020, and applied unchanged to AT&T downlink traces from three drives in the PAM 2025 dataset, recorded in the United States with a different phone and tool. Configuration, interval corrections and probability calibration came from inner Raca session folds only. The protocol was frozen (sha256 `02126c5515ef3c3c…`) before any PAM outcome was used, and the evaluation ran once.

| Data | Rows | Skill against persistence | Trailing mean skill against persistence | Raw 80% coverage | Global CQR coverage | Mondrian CQR coverage | Warning AP, model / persistence / trailing mean | Same, current rate at least 1.5 Mbps |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| All three drives | 17319 | -147.7% (-167.7 to -125.4) | -100.9% (-111.8 to -89.0) | 46.6% (42.5 to 50.5) | 48.0% (43.7 to 52.1) | 48.5% (44.3 to 52.7) | 0.682 / 0.820 / 0.583 | 0.054 / 0.127 / 0.063 |
| atnt_d1_dl | 5224 | -146.9% (-178.9 to -104.1) | -106.5% (-130.8 to -81.6) | 42.0% (33.2 to 51.0) | 43.9% (34.6 to 53.2) | 44.4% (35.0 to 53.7) | 0.784 / 0.864 / 0.670 | 0.087 / 0.151 / 0.065 |
| atnt_day_1_d2_dl | 6255 | -166.9% (-215.2 to -127.2) | -99.2% (-116.5 to -80.3) | 49.7% (43.6 to 55.2) | 50.9% (44.5 to 56.5) | 51.1% (44.8 to 56.9) | 0.121 / 0.267 / 0.154 | 0.025 / 0.170 / 0.135 |
| atnt_day_1_d3_dl | 5840 | -135.2% (-162.1 to -107.6) | -95.9% (-109.4 to -82.5) | 47.3% (42.0 to 52.4) | 48.5% (43.0 to 53.7) | 49.5% (43.9 to 54.8) | 0.249 / 0.295 / 0.246 | 0.086 / 0.156 / 0.168 |

Decisions fixed in advance:

- Point forecast transfers (MAE skill against persistence positive in all three drives): **no**.
- Global CQR intervals transfer (pooled calibrated coverage between 75 and 85 percent): **no**.
- Mondrian CQR intervals transfer: **no**.

Coverage by predicted level quartile, all drives:

| Interval | Quartile 0 | Quartile 1 | Quartile 2 | Quartile 3 |
| --- | ---: | ---: | ---: | ---: |
| Raw quantiles | 53.3% | 54.9% | 43.6% | 31.1% |
| Global CQR | 57.5% | 56.4% | 44.2% | 31.2% |
| Mondrian CQR | 59.9% | 55.7% | 44.1% | 34.1% |

Bootstrap intervals resample test segments. Segments of one drive share a route and a day, so they are descriptive, and the per drive rows are the primary view. Units differ in origin: PAM publishes Mbps, Raca units are inferred as kbps. PAM has no activity state, and cell transitions were not derived, so neither enters the model.

Exclusions per drive: `{"atnt_d1_dl": {"bins": 5645, "candidate_rows": 5651, "excluded_no_valid_downlink_in_history": 78, "excluded_target_bin_absent": 28, "excluded_target_bin_incomplete": 1, "excluded_target_bin_without_valid_downlink": 0, "excluded_target_window_extends_beyond_segment": 320, "feature_rows": 5224, "rejected_negative_or_nonfinite": 0, "rows": 56168, "rows_with_rate": 11185, "segments": 80}, "atnt_day_1_d2_dl": {"bins": 6526, "candidate_rows": 6526, "excluded_no_valid_downlink_in_history": 54, "excluded_target_bin_absent": 0, "excluded_target_bin_incomplete": 1, "excluded_target_bin_without_valid_downlink": 0, "excluded_target_window_extends_beyond_segment": 216, "feature_rows": 6255, "rejected_negative_or_nonfinite": 0, "rows": 74457, "rows_with_rate": 12996, "segments": 54}, "atnt_day_1_d3_dl": {"bins": 6096, "candidate_rows": 6096, "excluded_no_valid_downlink_in_history": 51, "excluded_target_bin_absent": 0, "excluded_target_bin_incomplete": 1, "excluded_target_bin_without_valid_downlink": 0, "excluded_target_window_extends_beyond_segment": 204, "feature_rows": 5840, "rejected_negative_or_nonfinite": 0, "rows": 68235, "rows_with_rate": 12128, "segments": 51}}`.
