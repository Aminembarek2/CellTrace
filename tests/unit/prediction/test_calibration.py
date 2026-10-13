"""Nested selection and calibration use training sessions only, never test outcomes."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from celltrace.prediction import cross_validation as validation
from celltrace.prediction.models import PERSISTENCE_FEATURE, TRAILING_MEAN_FEATURE

# Two small configurations keep fits fast while selection still has a real choice.
SMALL_CONFIGURATIONS = (
    {"max_iter": 2, "min_samples_leaf": 2},
    {"max_iter": 12, "min_samples_leaf": 2},
)


def frame(sessions: int = 8) -> pd.DataFrame:
    parts = []
    for session in range(sessions):
        rate = np.linspace(0.1, 6.0, 40)
        parts.append(
            pd.DataFrame(
                {
                    "session_id": f"session_{session}",
                    "as_of_ms": np.arange(40) * 1000,
                    "target_mbps": rate,
                    "feature_a": rate + 0.5,
                    PERSISTENCE_FEATURE: rate + 0.5,
                    TRAILING_MEAN_FEATURE: rate + 0.25,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def session_times(sessions: int = 8) -> dict[str, tuple[datetime, datetime]]:
    origin = datetime(2020, 1, 1, 8)
    return {
        f"session_{index}": (
            origin + timedelta(hours=index),
            origin + timedelta(hours=index, minutes=30),
        )
        for index in range(sessions)
    }


def test_inner_folds_keep_sessions_whole_and_ignore_outcomes_and_row_order() -> None:
    original = frame()
    splits = validation.inner_folds(original)
    changed = original.sample(frac=1, random_state=13).copy()
    changed["target_mbps"] = 100000 - changed["target_mbps"]
    changed_splits = validation.inner_folds(changed)

    def held_sessions(data: pd.DataFrame, folds: validation.Splits) -> list[set[str]]:
        return [set(data.iloc[held]["session_id"]) for _, held in folds]

    assert held_sessions(original, splits) == held_sessions(changed, changed_splits)
    seen: set[str] = set()
    for fit_index, held_index in splits:
        fit, held = (
            set(original.iloc[fit_index]["session_id"]),
            set(original.iloc[held_index]["session_id"]),
        )
        assert not fit & held and not seen & held
        seen |= held
    assert seen == set(original["session_id"]) and len(splits) == validation.INNER_FOLDS


def test_test_targets_never_change_any_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "CONFIGURATIONS", SMALL_CONFIGURATIONS)
    data = frame()
    train = data.loc[data["session_id"] != "session_0"]
    test = data.loc[data["session_id"] == "session_0"]
    corrupted = test.copy()
    corrupted["target_mbps"] = 100000.0
    first = validation.fit_fold(train, test, ["feature_a"])
    second = validation.fit_fold(train, corrupted, ["feature_a"])
    for name in first:
        if np.issubdtype(first[name].dtype, np.number):
            np.testing.assert_allclose(first[name], second[name], equal_nan=True)
        else:
            assert np.array_equal(first[name], second[name])
    assert set(first["warning_calibration_status"]) == {"sigmoid_inner_session_folds"}
    assert set(first["interval_calibration_status"]) == {"pooled_session_cqr"}


def test_selection_and_encoders_see_training_sessions_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "CONFIGURATIONS", SMALL_CONFIGURATIONS)
    data = frame()
    data["network_mode_last"] = np.where(data["session_id"] == "session_0", "test_only", "LTE")
    train = data.loc[data["session_id"] != "session_0"]
    test = data.loc[data["session_id"] == "session_0"]
    folded: list[set[str]] = []
    original_folds = validation.inner_folds

    def record_folds(training: pd.DataFrame) -> validation.Splits:
        folded.append(set(training["session_id"]))
        return original_folds(training)

    seen_categories: list[set[str]] = []
    original_fit = validation.CategoryEncoder.fit

    def record_fit(values: pd.Series) -> validation.CategoryEncoder:
        seen_categories.append(set(values.dropna()))
        return original_fit(values)

    monkeypatch.setattr(validation, "inner_folds", record_folds)
    monkeypatch.setattr(validation.CategoryEncoder, "fit", record_fit)
    validation.fit_fold(train, test, ["feature_a", "network_mode_last"])
    assert folded == [set(train["session_id"])]
    assert seen_categories and all(categories == {"LTE"} for categories in seen_categories)


def test_selection_prefers_the_configuration_with_lower_inner_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "CONFIGURATIONS", SMALL_CONFIGURATIONS)
    data = frame()
    target = data["target_mbps"].to_numpy(dtype=float)
    chosen, errors = validation.select_configuration(
        data, ["feature_a"], target, validation.inner_folds(data)
    )
    assert len(errors) == 2 and errors[1] < errors[0] and chosen == 1


def test_each_inner_transform_excludes_categories_unique_to_its_held_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "CONFIGURATIONS", SMALL_CONFIGURATIONS)
    data = frame()
    data["network_mode_last"] = data["session_id"]
    train = data.loc[data["session_id"] != "session_0"]
    test = data.loc[data["session_id"] == "session_0"]
    allowed = {
        frozenset(train.iloc[fit_index]["session_id"])
        for fit_index, _ in validation.inner_folds(train)
    }
    seen: list[frozenset[str]] = []
    original = validation.CategoryEncoder.fit

    def record(values: pd.Series) -> validation.CategoryEncoder:
        seen.append(frozenset(values.dropna()))
        return original(values)

    monkeypatch.setattr(validation.CategoryEncoder, "fit", record)
    validation.fit_fold(train, test, ["feature_a", "network_mode_last"])
    assert allowed.issubset(set(seen))
    assert all(
        categories in allowed or categories == frozenset(train["session_id"]) for categories in seen
    )
    assert all("session_0" not in categories for categories in seen)
    # More than selection/quantile fitting: sigmoid calibration also fits the transform.
    assert len(seen) >= (len(SMALL_CONFIGURATIONS) + 3) * len(allowed) + 2


def test_pooled_correction_gives_every_session_equal_weight() -> None:
    # One long session of perfect forecasts and four one-row sessions with large errors.
    scores = np.array([0.0] * 100 + [10.0, 20.0, 30.0, 40.0])
    sessions = np.array(["long"] * 100 + ["b", "c", "d", "e"])
    # Each session weighs 0.2. Row weighting would return 0, letting one session decide.
    assert validation.pooled_conformal_correction(scores, sessions, alpha=0.2) == 30.0
    assert float(np.quantile(scores, 0.8)) == 0.0


def test_pooled_correction_narrows_intervals_that_are_too_wide() -> None:
    scores = np.array([-4.0, -3.0, -2.0, -1.0])
    sessions = np.array(list("abcd"))
    assert validation.pooled_conformal_correction(scores, sessions, alpha=0.25) == -2.0
    with pytest.raises(ValueError, match="strictly between"):
        validation.pooled_conformal_correction(scores, sessions, alpha=1.0)
    with pytest.raises(ValueError, match="finite"):
        validation.pooled_conformal_correction(np.array([np.nan]), np.array(["a"]), 0.2)


def test_forward_chaining_trains_only_on_sessions_that_ended_earlier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "MINIMUM_FORWARD_TRAINING_SESSIONS", 3)
    times = session_times()
    # Session 2 runs long and overlaps session 3, so session 3 cannot train on it.
    times["session_2"] = (times["session_2"][0], times["session_3"][0] + timedelta(minutes=5))
    folds = validation.forward_chaining(frame(), times)
    as_dictionary = dict(folds)
    # Only sessions 0 and 1 ended before session 3 started, fewer than the minimum of three.
    assert "session_3" not in as_dictionary
    assert as_dictionary["session_4"] == ["session_0", "session_1", "session_2", "session_3"]
    for test_session, training in folds:
        start = times[test_session][0]
        assert len(training) >= 3 and all(times[other][1] <= start for other in training)
    with pytest.raises(ValueError, match="start and end times"):
        validation.forward_chaining(frame(), {"session_0": times["session_0"]})


def test_one_class_training_has_named_constant_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "CONFIGURATIONS", SMALL_CONFIGURATIONS)
    data = frame()
    data["target_mbps"] = 0.1
    result = validation.fit_fold(
        data.loc[data["session_id"] != "session_0"],
        data.loc[data["session_id"] == "session_0"],
        ["feature_a"],
    )
    assert (result["warning"] == 1.0).all()
    assert set(result["warning_calibration_status"]) == {
        "constant_training_prevalence_fit_one_class"
    }


def test_inner_fold_without_both_outcomes_falls_back_to_prevalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "CONFIGURATIONS", SMALL_CONFIGURATIONS)
    data = frame()
    # Only one training session has any low rows, so some inner fit folds lack them.
    data.loc[data["session_id"] != "session_1", "target_mbps"] += 10
    train = data.loc[data["session_id"] != "session_0"]
    result = validation.fit_fold(train, data.loc[data["session_id"] == "session_0"], ["feature_a"])
    prevalence = float((train["target_mbps"] < validation.WARNING_THRESHOLD_MBPS).mean())
    assert set(result["warning_calibration_status"]) == {
        "constant_training_prevalence_inner_fold_one_class"
    }
    np.testing.assert_allclose(result["warning"], prevalence)


@pytest.mark.parametrize("name", list(validation.FIGURES.values()))
def test_existing_figures_are_protected_before_reading_or_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / name).write_text("keep this existing artifact")

    def forbidden_open(*args: object, **kwargs: object) -> None:
        pytest.fail("existing output protection must happen before opening the snapshot")

    monkeypatch.setattr(validation, "open_snapshot", forbidden_open)
    with pytest.raises(FileExistsError):
        validation.run_cross_validation(tmp_path / "snapshot", public, tmp_path / "private")
    assert (public / name).read_text() == "keep this existing artifact"


@pytest.mark.parametrize("private", ["public", "public/private"])
def test_public_and_private_output_overlap_is_rejected(tmp_path: Path, private: str) -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        validation.run_cross_validation(
            tmp_path / "snapshot", tmp_path / "public", tmp_path / private
        )


def patch_snapshot(monkeypatch: pytest.MonkeyPatch, data: pd.DataFrame) -> None:
    build = SimpleNamespace(
        frame=data,
        feature_groups={"history_only": ("feature_a",), "combined": ("feature_a",)},
        specification_hash="feature_identity",
        exclusions={"feature_rows": len(data)},
        exclusions_by_session={},
    )
    connection = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(validation, "open_snapshot", lambda _: (connection, {"snapshot_id": "s"}))
    monkeypatch.setattr(validation, "build_feature_frame", lambda _: build)

    def times(_connection: Any, sessions: list[str]) -> dict[str, tuple[datetime, datetime]]:
        return session_times(len(sessions))

    monkeypatch.setattr(validation, "load_session_times", times)
    monkeypatch.setattr(validation, "MINIMUM_FORWARD_TRAINING_SESSIONS", 3)


def test_failed_fits_preserve_identity_plan_and_failure_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_snapshot(monkeypatch, frame())

    def fail_fit(*args: object, **kwargs: object) -> None:
        raise ValueError("intentional failed model fit")

    monkeypatch.setattr(validation, "cross_validate", fail_fit)
    private = tmp_path / "private"
    with pytest.raises(ValueError, match="intentional failed model fit"):
        validation.run_cross_validation(tmp_path / "snapshot", tmp_path / "public", private)
    artifact = json.loads((private / validation.FOLDS_NAME).read_text())
    assert artifact["status"] == "failed_fit"
    assert artifact["error"] == {"type": "ValueError", "message": "intentional failed model fit"}
    assert len(artifact["folds"][validation.LEAVE_ONE_OUT]) == 8
    assert len(artifact["folds"][validation.FORWARD]) == 5
    for folds in artifact["folds"].values():
        for fold in folds:
            training, test = set(fold["training_sessions"]), set(fold["test_sessions"])
            inner = [set(sessions) for sessions in fold["inner_fold_sessions"]]
            assert not training & test and set().union(*inner) == training
    assert not (private / validation.PREDICTIONS_NAME).exists()


def test_nested_run_writes_both_schemes_folds_and_support_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(validation, "CONFIGURATIONS", SMALL_CONFIGURATIONS)
    data = frame()
    data["mobility_cohort"] = "driving"
    data["transition_in_window"] = False
    data["target_start_ms"] = data["as_of_ms"]
    data["target_end_ms"] = data["as_of_ms"] + 5000
    patch_snapshot(monkeypatch, data)
    public, private = tmp_path / "public", tmp_path / "private"
    summary = validation.run_cross_validation(tmp_path / "snapshot", public, private)
    saved = json.loads((public / validation.SUMMARY_NAME).read_text())
    folds = json.loads((private / validation.FOLDS_NAME).read_text())
    assert (
        saved == summary and saved["claim_scope"] == "exploratory_nested_session_cross_validation"
    )
    assert saved["fold_manifest_sha256"] == validation.content_hash(folds)
    assert folds["status"] == "complete"
    predictions = pd.read_csv(private / validation.PREDICTIONS_NAME)
    forward = pd.read_csv(private / validation.FORWARD_PREDICTIONS_NAME)
    assert predictions["target_end_ms"].equals(predictions["as_of_ms"] + 5000)
    assert "combined_warning_raw" in predictions and "combined_configuration_index" in predictions
    assert set(forward["session_id"]) == set(saved[validation.FORWARD]["test_sessions"])
    assert (forward["training_sessions"] >= 3).all()
    selection = saved["configuration_selection"][validation.LEAVE_ONE_OUT]["combined"]
    assert sum(selection["folds_by_index"].values()) == 8
    assert "persistence_warning_score" in saved["warning"]
    interval = saved["calibrated_interval"]["combined"]
    assert interval["global_cqr"]["row_coverage"]["value"] is not None
    assert set(interval["mondrian_cqr"]["coverage_by_group"]["predicted_level_quartile"]) <= {
        "0",
        "1",
        "2",
        "3",
    }
    report = (public / validation.REPORT_NAME).read_text()
    assert "not an independent held out estimate" in report
    assert "Forward chaining" in report and "not end-to-end device latency" in report


def test_mondrian_gives_each_predicted_level_quartile_its_own_correction() -> None:
    levels = np.tile(np.arange(40, dtype=float), 5)
    sessions = np.repeat([f"s{index}" for index in range(5)], 40)
    # Forecasts in the top quartile miss by 10 Mbps, the others are exact.
    scores = np.where(levels >= 30, 10.0, 0.0)
    test_levels = np.array([1.0, 35.0])
    quartiles, corrections, fallback = validation.mondrian_corrections(
        levels, scores, sessions, test_levels, global_correction=3.0
    )
    assert list(quartiles) == [0, 3] and fallback == 0
    assert list(corrections) == [0.0, 10.0]


def test_mondrian_falls_back_to_global_when_a_quartile_has_few_sessions() -> None:
    levels = np.arange(40, dtype=float)
    sessions = np.array(["only_session"] * 40)
    _, corrections, fallback = validation.mondrian_corrections(
        levels, np.zeros(40), sessions, np.array([5.0, 35.0]), global_correction=3.0
    )
    assert fallback == 4 and list(corrections) == [3.0, 3.0]
