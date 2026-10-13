"""Regime labels, resampling boundaries and provenance checks for saved forecast errors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy
import pandas
import pytest

from celltrace.canonical import file_sha256
from celltrace.config import DetectorSection
from celltrace.prediction import error_analysis
from celltrace.prediction.error_analysis import (
    ISSUANCE_ORDER,
    REALIZED_ORDER,
    PredictionHashMismatch,
    RegimeRule,
    analyze_split,
    block_totals,
    classify_change,
    run_error_analysis,
    within_session_block_indices,
)

MODELS = ("persistence", "trailing_mean_30s", "hist_gradient_boosting_combined")
RULE = RegimeRule.from_detector(DetectorSection())


def synthetic_predictions(session_count: int, rows_per_session: int, seed: int) -> pandas.DataFrame:
    generator = numpy.random.default_rng(seed)
    frames = []
    for session_index in range(session_count):
        target = generator.uniform(5.0, 50.0, rows_per_session)
        frames.append(
            pandas.DataFrame(
                {
                    "session_id": f"ses_{session_index:02d}",
                    "mobility_cohort": "driving" if session_index % 2 else "static",
                    "as_of_ms": numpy.arange(rows_per_session) * 1000,
                    "target_mbps": target,
                    "prediction_persistence": target + generator.normal(0, 8, rows_per_session),
                    "prediction_trailing_mean_30s": numpy.full(rows_per_session, 25.0),
                    "prediction_hist_gradient_boosting_combined": target
                    + generator.normal(0, 4, rows_per_session),
                }
            )
        )
    return pandas.concat(frames, ignore_index=True)


def test_regime_rule_reuses_the_detector_thresholds_and_mirrors_drops() -> None:
    reference = numpy.array([10.0, 10.0, 10.0, 10.0, 0.5, 1.0])
    value = numpy.array([4.0, 4.1, 25.0, 24.9, 0.0, 0.45])
    labels = classify_change(reference, value, RULE, REALIZED_ORDER)
    assert labels.tolist() == ["drop", "steady", "rise", "steady", "low_reference", "steady"]
    assert RULE.minimum_relative_change == DetectorSection().minimum_relative_drop
    assert RULE.rise_factor == pytest.approx(1.5)


def test_absolute_threshold_keeps_small_changes_steady_near_the_minimum_reference() -> None:
    labels = classify_change(numpy.array([1.0]), numpy.array([0.55]), RULE, ISSUANCE_ORDER)
    assert labels.tolist() == ["level"]


def test_block_resampling_never_mixes_sessions() -> None:
    sessions = numpy.array(["a", "a", "a", "b", "b"], dtype=object)
    indices = within_session_block_indices(sessions, numpy.random.default_rng(0), 200)
    assert indices.shape == (200, 5)
    assert set(indices[:, :3].ravel()) <= {0, 1, 2}
    assert set(indices[:, 3:].ravel()) <= {3, 4}


def test_blocks_group_sixty_seconds_of_issuance_time_inside_a_session() -> None:
    rows = synthetic_predictions(2, 150, seed=1)
    rows["time_block"] = rows["as_of_ms"] // error_analysis.BLOCK_MS
    totals = block_totals(rows, MODELS)
    assert totals.block_count == 6
    assert totals.row_counts.sum() == len(rows)
    point = totals.absolute_error_sums.sum(axis=0) / len(rows)
    direct = (rows["prediction_persistence"] - rows["target_mbps"]).abs().mean()
    assert point[0] == pytest.approx(direct)


def test_regime_totals_keep_zero_occupancy_blocks_and_segment_boundaries() -> None:
    rows = synthetic_predictions(1, 120, seed=1)
    rows["time_block"] = rows["as_of_ms"] // error_analysis.BLOCK_MS
    rows["segment_id"] = numpy.where(rows["as_of_ms"] < 30_000, "first", "second")
    rare = rows.iloc[:2]
    totals = block_totals(rare, MODELS, universe=rows)
    assert totals.block_count == 3
    assert sorted(totals.row_counts.tolist()) == [0, 0, 2]
    direct = (rare["prediction_persistence"] - rare["target_mbps"]).abs().mean()
    assert totals.absolute_error_sums[:, 0].sum() / totals.row_counts.sum() == pytest.approx(direct)


def test_per_session_intervals_are_withheld_for_short_sessions() -> None:
    short = analyze_split(synthetic_predictions(1, 120, seed=1), MODELS, RULE, 42)
    long = analyze_split(synthetic_predictions(1, 660, seed=1), MODELS, RULE, 42)
    assert short["per_session"][0]["mae_interval_mbps"][MODELS[2]] == {}
    assert long["per_session"][0]["difference_interval_mbps"][MODELS[2]]["within_session_blocks"]


def test_cluster_interval_is_withheld_for_few_sessions_and_reported_for_many() -> None:
    few = analyze_split(synthetic_predictions(4, 300, seed=2), MODELS, RULE, 42)
    many = analyze_split(synthetic_predictions(12, 300, seed=3), MODELS, RULE, 42)
    assert few["overall"]["interval_methods"]["session_clusters"].startswith("not reported")
    assert many["overall"]["interval_methods"]["session_clusters"] == "reported"
    combined = many["overall"]["models"]["hist_gradient_boosting_combined"]
    low, high = combined["mae_interval_mbps"]["within_session_blocks"]
    assert low <= combined["mae_mbps"] <= high
    difference_high = combined["difference_interval_mbps"]["session_clusters"][1]
    assert difference_high < 0
    assert many["sessions_with_lower_mae_than_persistence"]["hist_gradient_boosting_combined"] == {
        "sessions": 12,
        "of": 12,
    }


def test_resampling_is_reproducible_with_the_declared_seed() -> None:
    rows = synthetic_predictions(3, 400, seed=4)
    first = analyze_split(rows, MODELS, RULE, 42)
    second = analyze_split(rows, MODELS, RULE, 42)
    assert first == second


def test_changed_prediction_file_is_refused(tmp_path: Path) -> None:
    rows = synthetic_predictions(2, 120, seed=5)
    path = tmp_path / "session_predictions.csv"
    rows.to_csv(path, index=False)
    manifest = {
        "files": {"session": {"path": str(path), "sha256": file_sha256(path)}},
        "model_sha256": "m" * 64,
        "split_manifest_hash": "s" * 64,
        "snapshot_logical_hash": "l" * 64,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    payload = run_error_analysis(manifest_path, tmp_path / "out")
    assert (tmp_path / "out" / "error_analysis.md").is_file()
    assert Path(payload["figures"]["error_by_regime"]).is_file()
    rows.iloc[:1].to_csv(path, index=False)
    with pytest.raises(PredictionHashMismatch):
        run_error_analysis(manifest_path, tmp_path / "again")
