"""Temporal availability, known targets, session splits and training boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy
import pandas
import pytest

from celltrace.prediction.features import (
    PastOnlyViolation,
    WindowFeature,
    assert_past_only,
    build_feature_frame,
    default_feature_specs,
)
from celltrace.prediction.models import CategoryEncoder, fit_model, load_model, save_model
from celltrace.prediction.splits import (
    SplitManifestError,
    assign_forward_time,
    check_forward_purge,
    check_session_groups_disjoint,
    session_split,
)


def test_features_use_closed_past_bins_and_target_preserves_zero_and_missing(
    fixture_connection: duckdb.DuckDBPyConnection,
) -> None:
    features = build_feature_frame(fixture_connection)
    assert_past_only(features.frame, 5000)
    indexed_bins = {
        (session_id, int(bin_start_ms)): downlink_mbps
        for session_id, bin_start_ms, downlink_mbps in fixture_connection.execute(
            "SELECT session_id, bin_start_ms, downlink_mbps FROM bins_1s"
        ).fetchall()
    }
    for row in features.frame.itertuples():
        values = [
            indexed_bins.get((row.session_id, row.as_of_ms + offset))
            for offset in (0, 1000, 2000, 3000, 4000)
        ]
        assert all(value is not None for value in values)
        assert row.target_mbps == pytest.approx(sum(values) / 5)
        previous_value = indexed_bins.get((row.session_id, row.as_of_ms - 1000))
        if previous_value is None:
            assert pandas.isna(row.downlink_lag_1_mbps)
        else:
            assert row.downlink_lag_1_mbps == previous_value
    assert (features.frame["target_mbps"] == 0).any()
    assert features.exclusions["excluded_target_bin_without_valid_downlink"] > 0


def test_future_feature_and_missing_availability_are_rejected(
    fixture_connection: duckdb.DuckDBPyConnection,
) -> None:
    future_feature = WindowFeature("future_rate_mbps", "downlink_mbps", "mean", 0, 1, "history")
    features = build_feature_frame(
        fixture_connection, specs=(*default_feature_specs(), future_feature)
    )
    with pytest.raises(PastOnlyViolation, match="available after"):
        assert_past_only(features.frame, 5000)
    frame = build_feature_frame(fixture_connection).frame.copy()
    frame.loc[0, "latest_available_ms"] = numpy.nan
    with pytest.raises(PastOnlyViolation, match="missing or nonfinite"):
        assert_past_only(frame, 5000)


def test_session_groups_are_intact_and_overlap_is_rejected() -> None:
    cohorts = {f"session_{session_index}": "driving" for session_index in range(10)}
    assignment = session_split(cohorts)
    manifest = {"session_split": {"sessions": assignment}}
    check_session_groups_disjoint(manifest)
    assert {member for members in assignment.values() for member in members} == set(cohorts)
    assignment["test"].append(assignment["train"][0])
    with pytest.raises(SplitManifestError, match="both"):
        check_session_groups_disjoint(manifest)


def test_target_boundary_purge_uses_interval_support() -> None:
    frame = pandas.DataFrame(
        {
            "session_id": ["session_one"] * 5,
            "as_of_ms": [50_000, 60_000, 100_000, 110_000, 200_000],
            "target_end_ms": [55_000, 65_000, 105_000, 115_000, 205_000],
            "earliest_support_ms": [20_000, 30_000, 70_000, 80_000, 170_000],
        }
    )
    boundaries = {"session_one": {"dev_start_ms": 90_000, "test_start_ms": 190_000}}
    labels = assign_forward_time(frame, boundaries, maximum_history_ms=30_000)
    assert labels.tolist() == ["train", "purged", "dev", "dev", "test"]
    check_forward_purge(frame, labels, boundaries, maximum_history_ms=30_000)
    frame.loc[0, "target_end_ms"] = 80_000
    with pytest.raises(SplitManifestError, match="purge violated"):
        check_forward_purge(frame, labels, boundaries, maximum_history_ms=30_000)


def test_training_encoder_and_estimator_do_not_create_hidden_validation_split(
    tmp_path: Path,
) -> None:
    train = pandas.DataFrame(
        {
            "network_mode_last": ["LTE", "5G"] * 20,
            "past_rate_mbps": numpy.arange(40, dtype=float),
            "target_mbps": numpy.arange(40, dtype=float),
        }
    )
    encoder = CategoryEncoder.fit(train["network_mode_last"])
    assert numpy.isnan(encoder.transform(pandas.Series(["UNSEEN"]))[0])
    model = fit_model(
        train, ["past_rate_mbps", "network_mode_last"], {"max_iter": 2, "min_samples_leaf": 2}
    )
    assert model.estimator.get_params()["early_stopping"] is False
    assert model.estimator.get_params()["random_state"] == 42
    path = save_model(model, tmp_path / "forecast.pkl", {"data": "original test values"})
    restored, sidecar = load_model(path)
    numpy.testing.assert_allclose(restored.predict(train), model.predict(train))
    assert sidecar["training_row_count"] == 40
    record = json.loads(path.with_suffix(".json").read_text())
    assert record["model_sha256"]
