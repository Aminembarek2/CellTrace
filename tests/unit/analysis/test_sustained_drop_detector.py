"""Sustained drop detector on the fixture with independently computed expectations."""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import duckdb
import pytest

from celltrace.analysis.detector import (
    DETECTOR_VERSION,
    BinView,
    detect_candidates,
    detector_config_hash,
    evaluate_times,
    interpolated_median,
    run_detect,
)
from celltrace.config import DetectorSection
from celltrace.contracts.evidence import CandidateEvent
from celltrace.data import fixture

DEFAULT = DetectorSection()


def _independent_flags(
    connection: duckdb.DuckDBPyConnection, session_id: str, config: DetectorSection
) -> dict[int, tuple[float, float]]:
    """Recompute flags from samples for a gap free, fully active session."""
    rows = connection.execute(
        "SELECT elapsed_ms, downlink_mbps FROM samples WHERE session_id = ?", [session_id]
    ).fetchall()
    per_second: dict[int, list[float]] = defaultdict(list)
    for elapsed, rate in rows:
        if rate is not None:
            per_second[elapsed // 1000].append(rate)
    means = {second: statistics.fmean(values) for second, values in per_second.items()}
    last_second = max(means)
    reference_bins = config.reference_ms // 1000
    event_bins = config.event_ms // 1000
    flags: dict[int, tuple[float, float]] = {}
    # The final second is incomplete, so the last usable confirmation is last_second seconds.
    for confirmation in range(reference_bins + event_bins, last_second + 1):
        reference = [
            means[second]
            for second in range(
                confirmation - reference_bins - event_bins, confirmation - event_bins
            )
        ]
        event = [means[second] for second in range(confirmation - event_bins, confirmation)]
        reference_median = statistics.median(reference)
        event_median = statistics.median(event)
        drop = reference_median - event_median
        if (
            reference_median >= config.minimum_reference_mbps
            and drop / reference_median >= config.minimum_relative_drop
            and drop >= config.minimum_absolute_drop_mbps
        ):
            flags[confirmation * 1000] = (reference_median, event_median)
    return flags


def test_interpolated_median_matches_duckdb_and_statistics() -> None:
    generator = random.Random(11)
    connection = duckdb.connect()
    try:
        for size in (1, 2, 5, 6, 29, 30):
            values = [generator.uniform(0, 600) for _ in range(size)]
            row = connection.execute(
                "SELECT median(value) FROM unnest(?::DOUBLE[]) AS series(value)", [values]
            ).fetchone()
            assert row is not None
            assert interpolated_median(values) == row[0]
            assert interpolated_median(values) == statistics.median(values)
    finally:
        connection.close()
    with pytest.raises(ValueError):
        interpolated_median([])


def test_flags_the_constructed_active_drop_at_the_expected_time(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    session_id = fixture_session_ids[fixture.STABLE_RADIO_DROP_MEMBER]
    result = detect_candidates(fixture_connection, DEFAULT, session_ids=[session_id])
    expected = _independent_flags(fixture_connection, session_id, DEFAULT)
    assert sorted(expected) == list(range(43000, 61000, 1000))
    assert [flag["confirmation_ms"] for flag in result.flags] == sorted(expected)
    for flag in result.flags:
        reference, event = expected[flag["confirmation_ms"]]
        assert flag["reference_median_mbps"] == pytest.approx(reference)
        assert flag["event_median_mbps"] == pytest.approx(event)
    [candidate] = result.candidates
    assert candidate.first_confirmation_ms == candidate.detection_time_ms == 43000
    assert candidate.last_confirmation_ms == 60000
    assert (candidate.event_window.start_ms, candidate.event_window.end_ms) == (38000, 60000)
    assert (
        candidate.representative_reference_window.start_ms,
        candidate.representative_reference_window.end_ms,
    ) == (8000, 38000)
    assert (
        candidate.representative_event_window.start_ms,
        candidate.representative_event_window.end_ms,
    ) == (38000, 43000)
    assert candidate.reference_median_mbps == pytest.approx(statistics.median([20.0] * 30))
    assert candidate.event_median_mbps == pytest.approx(
        statistics.median([20.0, 20.0, 3.0, 3.0, 3.0])
    )
    assert candidate.absolute_drop_mbps == pytest.approx(17.0)
    assert candidate.relative_drop == pytest.approx(0.85)
    assert candidate.flag_count == 18
    assert re.fullmatch(r"evt_[0-9a-f]{12}", candidate.event_id)
    last_flag = result.flags[-1]
    assert last_flag["reference_median_mbps"] == pytest.approx(
        statistics.median([20.0] * 15 + [3.0] * 15)
    )
    [details] = result.candidate_details
    assert details["detection_delay_ms"] == 5000 >= DEFAULT.event_ms


def test_drop_with_radio_fall_and_cell_change_is_flagged(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    session_id = fixture_session_ids[fixture.RADIO_DROP_WITH_CELL_CHANGE_MEMBER]
    result = detect_candidates(fixture_connection, DEFAULT, session_ids=[session_id])
    [candidate] = result.candidates
    assert candidate.first_confirmation_ms == 43000
    assert candidate.relative_drop == pytest.approx((25.0 - 2.0) / 25.0)


@pytest.mark.parametrize(
    "member",
    [
        fixture.STEADY_MEMBER,
        fixture.IDLE_DECLINE_MEMBER,
        fixture.NEAR_ZERO_REFERENCE_MEMBER,
        fixture.MISSING_VALUES_MEMBER,
        fixture.GAP_MEMBER,
        fixture.SHORT_MEMBER,
        fixture.ORDER_AND_TECHNOLOGY_MEMBER,
    ],
)
def test_sessions_without_candidates(
    fixture_connection: duckdb.DuckDBPyConnection,
    fixture_session_ids: dict[str, str],
    member: str,
) -> None:
    result = detect_candidates(
        fixture_connection, DEFAULT, session_ids=[fixture_session_ids[member]]
    )
    assert result.candidates == [] and result.flags == []


def test_idle_decline_is_rejected_by_activity(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    session_id = fixture_session_ids[fixture.IDLE_DECLINE_MEMBER]
    counts = detect_candidates(
        fixture_connection, DEFAULT, session_ids=[session_id]
    ).eligibility_counts
    assert counts["rejected_event_active_fraction_below_minimum"] > 0
    assert counts["rejected_reference_active_fraction_below_minimum"] > 0
    assert counts["flagged_times"] == 0


def test_near_zero_reference_is_rejected_by_the_reference_minimum(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    session_id = fixture_session_ids[fixture.NEAR_ZERO_REFERENCE_MEMBER]
    counts = detect_candidates(
        fixture_connection, DEFAULT, session_ids=[session_id]
    ).eligibility_counts
    assert counts["rejected_reference_below_minimum_mbps"] == counts["after_activity"] > 0
    assert counts["eligible_times"] == 0
    # The drop itself would pass both drop thresholds, only the reference minimum stops it.
    relative = (0.6 - 0.05) / 0.6
    assert relative >= DEFAULT.minimum_relative_drop
    assert DEFAULT.minimum_absolute_drop_mbps <= 0.6 - 0.05


def _constructed_bins(
    event_rate: float, late_bin_start_ms: int | None = None
) -> dict[int, BinView]:
    """Forty complete, fully active one second bins: 10 Mbps, then event_rate from 30 s."""
    bins = {}
    for start in range(0, 40_000, 1000):
        available = start + 1000
        if start == late_bin_start_ms:
            available = start + 3000
        bins[start] = BinView(
            segment_id=0,
            bin_start_ms=start,
            available_at_ms=available,
            is_complete=True,
            sample_count=1,
            active_count=1,
            downlink_mbps=10.0 if start < 30_000 else event_rate,
            downlink_valid_count=1,
        )
    return bins


def _outcome_at(bins: dict[int, BinView], confirmation_ms: int) -> str:
    for decision in evaluate_times("constructed", bins, [(0, 0, 40_000)], DEFAULT):
        if decision.confirmation_ms == confirmation_ms:
            return decision.outcome
    raise AssertionError("confirmation time not evaluated")


def test_a_decline_below_the_relative_threshold_is_not_flagged() -> None:
    # Expert audit mutation check: ignoring the 60 percent rule left every test passing.
    # 10 to 6 Mbps passes the 0.5 Mbps absolute rule but is only a 40 percent decline.
    assert _outcome_at(_constructed_bins(6.0), 35_000) == "not_flagged_relative_drop_below_minimum"
    assert _outcome_at(_constructed_bins(3.0), 35_000) == "flagged_times"


def test_a_bin_available_after_confirmation_rejects_the_window() -> None:
    # Expert audit mutation check: tolerating late bins left every test passing.
    late = _constructed_bins(3.0, late_bin_start_ms=34_000)
    assert _outcome_at(late, 35_000) == "rejected_not_yet_available"
    assert _outcome_at(late, 37_000) == "flagged_times"


def test_short_and_gap_sessions_have_no_valid_window(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    for member in (fixture.SHORT_MEMBER, fixture.GAP_MEMBER):
        counts = detect_candidates(
            fixture_connection, DEFAULT, session_ids=[fixture_session_ids[member]]
        ).eligibility_counts
        assert counts["evaluated_times"] == counts["rejected_window_not_inside_one_segment"] > 0


def test_eligibility_counts_form_a_consistent_funnel(
    fixture_connection: duckdb.DuckDBPyConnection,
) -> None:
    counts = detect_candidates(fixture_connection, DEFAULT).eligibility_counts
    assert counts["evaluated_times"] == (
        counts["rejected_window_not_inside_one_segment"] + counts["after_segment_containment"]
    )
    assert counts["after_segment_containment"] == (
        counts["rejected_incomplete_bin"]
        + counts["rejected_not_yet_available"]
        + counts["after_completeness_and_availability"]
    )
    assert counts["after_completeness_and_availability"] == (
        counts["rejected_reference_coverage_below_minimum"]
        + counts["rejected_event_coverage_below_minimum"]
        + counts["after_coverage"]
    )
    assert counts["after_coverage"] == (
        counts["rejected_reference_active_fraction_below_minimum"]
        + counts["rejected_event_active_fraction_below_minimum"]
        + counts["after_activity"]
    )
    assert counts["after_activity"] == (
        counts["rejected_reference_below_minimum_mbps"] + counts["eligible_times"]
    )
    assert counts["eligible_times"] == (
        counts["not_flagged_relative_drop_below_minimum"]
        + counts["not_flagged_absolute_drop_below_minimum"]
        + counts["flagged_times"]
    )
    assert counts["sessions_evaluated"] == 9
    assert counts["flagged_times"] == 36 and counts["candidates"] == 2


def test_output_is_deterministic(fixture_connection: duckdb.DuckDBPyConnection) -> None:
    first = detect_candidates(fixture_connection, DEFAULT)
    second = detect_candidates(fixture_connection, DEFAULT)
    assert [item.model_dump() for item in first.candidates] == [
        item.model_dump() for item in second.candidates
    ]
    assert first.flags == second.flags
    assert first.config_hash == detector_config_hash(DEFAULT)
    assert first.config_hash != detector_config_hash(DetectorSection(minimum_relative_drop=0.5))


def test_split_at_maximum_event_duration(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    config = DetectorSection(maximum_event_ms=10_000)
    session_id = fixture_session_ids[fixture.STABLE_RADIO_DROP_MEMBER]
    candidates = detect_candidates(fixture_connection, config, session_ids=[session_id]).candidates
    assert [(item.event_window.start_ms, item.event_window.end_ms) for item in candidates] == [
        (38000, 48000),
        (44000, 54000),
        (50000, 60000),
    ]
    assert [item.flag_count for item in candidates] == [6, 6, 6]
    assert candidates[0].continues_event_id is None
    assert candidates[1].continues_event_id == candidates[0].event_id
    assert candidates[2].continues_event_id == candidates[1].event_id
    assert all(item.event_window.length_ms <= config.maximum_event_ms for item in candidates)
    # Each continuation keeps its own earliest flag as representative.
    assert candidates[1].first_confirmation_ms == 49000
    assert candidates[1].representative_reference_window.start_ms == 14000


def test_zero_merge_gap_keeps_flags_separate(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    config = DetectorSection(merge_gap_ms=0)
    session_id = fixture_session_ids[fixture.STABLE_RADIO_DROP_MEMBER]
    candidates = detect_candidates(fixture_connection, config, session_ids=[session_id]).candidates
    assert len(candidates) == 18
    assert all(item.continues_event_id is None and item.flag_count == 1 for item in candidates)


def test_unknown_session_is_rejected(fixture_connection: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(ValueError):
        detect_candidates(fixture_connection, DEFAULT, session_ids=["ses_000000000000"])


def test_run_detect_writes_candidates(fixture_snapshot_dir: Path, tmp_path: Path) -> None:
    path = run_detect(fixture_snapshot_dir, tmp_path, DEFAULT)
    payload = json.loads(path.read_text())
    assert payload["detector_version"] == DETECTOR_VERSION
    assert payload["snapshot_id"] == fixture_snapshot_dir.name
    assert payload["config_hash"] == detector_config_hash(DEFAULT)
    candidates = [CandidateEvent.model_validate(item) for item in payload["candidates"]]
    assert len(candidates) == 2
    assert sum(payload["session_flag_counts"].values()) == len(payload["flags"]) == 36
    assert {item["event_id"] for item in payload["candidate_details"]} == {
        item.event_id for item in candidates
    }
    assert (
        path.read_bytes()
        == run_detect(fixture_snapshot_dir, tmp_path / "again", DEFAULT).read_bytes()
    )
