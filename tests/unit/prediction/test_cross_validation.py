"""Leave one session out scoring: textbook formulas, no leakage, valid quantiles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from celltrace.prediction import cross_validation as validation
from celltrace.prediction.models import PERSISTENCE_FEATURE, TRAILING_MEAN_FEATURE


def test_pinball_matches_the_textbook_definition() -> None:
    actual = np.array([10.0, 14.0])
    forecast = np.array([12.0, 12.0])
    # Below the forecast: 2(1 - p)(f - y). Above it: 2p(y - f).
    assert validation.pinball(actual, forecast, 0.1) == pytest.approx([3.6, 0.4])
    # At the median the score is the absolute error.
    assert validation.pinball(actual, forecast, 0.5) == pytest.approx([2.0, 2.0])


def test_winkler_is_width_inside_and_penalised_outside() -> None:
    lower, upper = np.array([2.0, 2.0, 2.0]), np.array([6.0, 6.0, 6.0])
    actual = np.array([4.0, 1.0, 7.0])
    assert validation.winkler(lower, upper, actual, 0.2) == pytest.approx([4.0, 14.0, 14.0])


def test_calibration_cannot_report_crossed_or_negative_bounds() -> None:
    lower, upper, effective = validation.calibrated_bounds(
        np.array([0.0, 2.0, 0.0]), np.array([1.0, 12.0, 2.0]), np.array([-4.0, -1.0, 3.0])
    )
    assert lower == pytest.approx([0.5, 3.0, 0.0])
    assert upper == pytest.approx([0.5, 11.0, 5.0])
    assert effective == pytest.approx([-0.5, -1.0, 3.0])
    assert np.all(lower >= 0) and np.all(lower <= upper)


@pytest.mark.parametrize("bounds", [([-1.0], [1.0]), ([2.0], [1.0])])
def test_calibration_rejects_invalid_input_bounds(bounds: tuple[list[float], list[float]]) -> None:
    with pytest.raises(ValueError):
        validation.calibrated_bounds(np.array(bounds[0]), np.array(bounds[1]), 1.0)


def synthetic_frame(sessions: int = 6, rows: int = 60) -> pd.DataFrame:
    generator = np.random.default_rng(7)
    parts = []
    for index in range(sessions):
        level = 2.0 + index
        history = generator.gamma(2.0, level, rows)
        parts.append(
            pd.DataFrame(
                {
                    "session_id": f"ses_{index}",
                    "as_of_ms": np.arange(rows) * 1000,
                    "mobility_cohort": "driving",
                    "transition_in_window": False,
                    "target_mbps": np.clip(history + generator.normal(0, 1, rows), 0, None),
                    PERSISTENCE_FEATURE: history,
                    TRAILING_MEAN_FEATURE: np.full(rows, history.mean()),
                    "feature_a": history,
                    "feature_b": generator.normal(0, 1, rows),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_no_fold_trains_on_its_test_session(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = synthetic_frame()
    seen: list[tuple[set[str], set[str]]] = []

    def recording_fit(
        train: pd.DataFrame, test: pd.DataFrame, names: Sequence[str]
    ) -> dict[str, np.ndarray]:
        seen.append((set(train["session_id"]), set(test["session_id"])))
        return {"mean": np.zeros(len(test))}

    monkeypatch.setattr(validation, "fit_fold", recording_fit)
    rows = validation.cross_validate(
        frame, {"history_only": ["feature_a"], "combined": ["feature_a"]}
    )
    assert len(rows) == len(frame)
    for train_sessions, test_sessions in seen:
        assert len(test_sessions) == 1 and not train_sessions & test_sessions
    assert {session for _, test in seen for session in test} == set(frame["session_id"])


def test_benchmarks_are_estimated_from_training_sessions_only() -> None:
    frame = synthetic_frame()
    train = frame[frame["session_id"] != "ses_0"]
    test = frame[frame["session_id"] == "ses_0"]
    shifted = test.copy()
    shifted["target_mbps"] = shifted["target_mbps"] + 1000
    # Changing the test targets must not change any benchmark forecast.
    first = validation.baseline_fold(train, test)
    second = validation.baseline_fold(train, shifted)
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_quantile_forecasts_never_cross_and_stay_non_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validation,
        "CONFIGURATIONS",
        ({"max_iter": 5, "min_samples_leaf": 5}, {"max_iter": 20, "min_samples_leaf": 5}),
    )
    frame = synthetic_frame()
    train = frame[frame["session_id"] != "ses_1"]
    test = frame[frame["session_id"] == "ses_1"]
    result = validation.fit_fold(train, test, ["feature_a", "feature_b"])
    assert np.all(result["q0.1"] <= result["q0.5"]) and np.all(result["q0.5"] <= result["q0.9"])
    assert np.all(result["q0.1"] >= 0) and np.all(result["mean"] >= 0)
    assert np.all((result["warning"] >= 0) & (result["warning"] <= 1))


def test_warning_benchmarks_preserve_continuous_ranking_and_train_only_climatology() -> None:
    train = synthetic_frame()
    train["target_mbps"] = np.tile([0.1, 3.0], len(train) // 2)
    test = train.iloc[:4].copy()
    test[PERSISTENCE_FEATURE] = [2.0, 4.0, 6.0, 8.0]
    test[TRAILING_MEAN_FEATURE] = [3.0, 5.0, 7.0, 9.0]
    forecasts = validation.baseline_fold(train, test)
    assert forecasts["climatology_warning"] == pytest.approx([0.5] * 4)
    assert forecasts["persistence_warning_score"] == pytest.approx([-2, -4, -6, -8])
    assert forecasts["trailing_mean_warning_score"] == pytest.approx([-3, -5, -7, -9])
    # Identical threshold category retains four continuous scores rather than one tie.
    assert len(set(forecasts["naive_warning"])) == 1
    assert len(set(forecasts["persistence_warning_score"])) == 4


def test_future_low_subset_excludes_already_low_rows_and_scores_do_not_claim_crossings() -> None:
    rows = pd.DataFrame(
        {
            "session_id": ["a"] * 4,
            "target_mbps": [0.1, 0.4, 3.0, 4.0],
            "persistence": [0.1, 2.0, 3.0, 4.0],
            "naive_warning": [0.9, 0.2, 0.2, 0.2],
            "climatology_warning": [0.5] * 4,
            "persistence_warning_score": [-0.1, -2.0, -3.0, -4.0],
            "trailing_mean_warning_score": [-0.2, -2.1, -3.1, -4.1],
        }
    )
    scores = validation.warning_scores(rows)
    assert scores["rows"] == 4 and scores["positive_rows"] == 2
    subset = scores["future_low_given_current_not_low"]
    assert subset["rows"] == 3 and subset["positive_rows"] == 1
    assert subset["persistence_warning_score"]["average_precision"] == 1.0
    assert "not a stall or threshold crossing" in scores["target"]


def test_no_positive_session_ap_is_explicitly_undefined() -> None:
    rows = pd.DataFrame(
        {
            "session_id": ["a", "a"],
            "target_mbps": [3.0, 4.0],
            "persistence": [3.0, 4.0],
            "naive_warning": [0.0, 0.0],
            "climatology_warning": [0.0, 0.0],
            "persistence_warning_score": [-3.0, -4.0],
            "trailing_mean_warning_score": [-3.0, -4.0],
        }
    )
    scores = validation.warning_scores(rows)
    assert scores["naive_warning"]["average_precision"] is None
    paired = scores["naive_warning"]["paired_session_ap_difference_vs_continuous_persistence"]
    assert paired["value"] is None and paired["interval_95"] is None
    assert paired["excluded_no_positive_sessions"] == ["a"]


def test_empty_future_low_subset_remains_serializable() -> None:
    import json

    rows = pd.DataFrame(
        {
            "session_id": ["a", "a"],
            "target_mbps": [0.1, 0.4],
            "persistence": [0.1, 0.4],
            "naive_warning": [1.0, 1.0],
            "climatology_warning": [1.0, 1.0],
            "persistence_warning_score": [-0.1, -0.4],
            "trailing_mean_warning_score": [-0.1, -0.4],
        }
    )
    scores = validation.warning_scores(rows)
    assert scores["future_low_given_current_not_low"]["rows"] == 0
    json.dumps(scores, allow_nan=False)
