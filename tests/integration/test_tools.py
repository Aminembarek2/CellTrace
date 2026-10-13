"""Typed tools against the fixture snapshot: scope, statistics and evidence."""

from __future__ import annotations

import statistics
from typing import Any

import numpy

from celltrace.contracts.common import AggregationBasis, MetricName, TimeWindow
from celltrace.tools.dispatch import evidence_content_hash


def window_arguments(harness: Any) -> dict[str, int]:
    candidate = harness.context.candidate
    return {
        "reference_start_ms": candidate.representative_reference_window.start_ms,
        "reference_end_ms": candidate.representative_reference_window.end_ms,
        "event_start_ms": candidate.representative_event_window.start_ms,
        "event_end_ms": candidate.representative_event_window.end_ms,
    }


def test_compare_windows_reproduces_detector_medians(stable_radio_harness: Any) -> None:
    harness = stable_radio_harness
    outcome = harness.dispatcher.dispatch(
        "compare_windows",
        {
            **window_arguments(harness),
            "metrics": ["downlink_mbps", "reported_rsrp_dbm"],
            "basis": "bin_1s",
            "statistic": "median",
        },
    )
    assert outcome.error is None and outcome.evidence is not None
    comparisons = outcome.evidence.result["comparisons"]
    candidate = harness.context.candidate
    rate = comparisons["downlink_mbps"]
    assert rate["reference_value"] == candidate.reference_median_mbps
    assert rate["event_value"] == candidate.event_median_mbps
    assert rate["relative_decrease"] == (
        (candidate.reference_median_mbps - candidate.event_median_mbps)
        / candidate.reference_median_mbps
    )
    radio = comparisons["reported_rsrp_dbm"]
    # dBm differences are additive. Relative change is never computed for them.
    assert radio["relative_decrease"] is None
    assert radio["relative_decrease_status"] == "not_a_rate_metric"
    assert radio["delta_event_minus_reference"] == radio["event_value"] - radio["reference_value"]


def test_native_statistics_match_an_independent_calculation(stable_radio_harness: Any) -> None:
    harness = stable_radio_harness
    window = harness.context.allowed_window
    end = min(window.end_ms, window.start_ms + 60_000)
    rows = harness.connection.execute(
        "SELECT downlink_mbps FROM samples WHERE session_id = ? AND elapsed_ms >= ? "
        "AND elapsed_ms < ?",
        [harness.context.session_id, window.start_ms, end],
    ).fetchall()
    values = [row[0] for row in rows if row[0] is not None]
    summary = harness.repository.summarize(
        harness.context.session_id,
        TimeWindow(start_ms=window.start_ms, end_ms=end),
        MetricName.downlink_mbps,
        AggregationBasis.native,
    )
    assert summary.valid_count == len(values)
    assert summary.median == statistics.median(values)
    assert summary.mean == numpy.mean(values)
    assert summary.p10 == numpy.percentile(values, 10)
    assert summary.p90 == numpy.percentile(values, 90)


def test_scope_and_argument_errors_are_never_success_shaped(stable_radio_harness: Any) -> None:
    harness = stable_radio_harness
    allowed = harness.context.allowed_window
    cases = {
        "outside": {"start_ms": allowed.end_ms, "end_ms": allowed.end_ms + 1000},
        "reversed": {"start_ms": allowed.start_ms + 5000, "end_ms": allowed.start_ms},
        "unknown_metric": {"start_ms": allowed.start_ms, "end_ms": allowed.start_ms + 1000},
    }
    outside = harness.dispatcher.dispatch(
        "summarize_window", {**cases["outside"], "metrics": ["downlink_mbps"], "basis": "native"}
    )
    assert outside.error is not None and outside.error.code.value == "OUT_OF_SCOPE"
    assert outside.evidence is None
    reversed_window = harness.dispatcher.dispatch(
        "summarize_window", {**cases["reversed"], "metrics": ["downlink_mbps"], "basis": "native"}
    )
    assert reversed_window.error is not None
    assert reversed_window.error.code.value == "INVALID_ARGUMENT"
    unknown = harness.dispatcher.dispatch(
        "summarize_window",
        {
            **cases["unknown_metric"],
            "metrics": ["downlink_mbps; DROP TABLE samples"],
            "basis": "native",
        },
    )
    assert unknown.error is not None and unknown.error.code.value == "INVALID_ARGUMENT"
    extra = harness.dispatcher.dispatch(
        "summarize_window",
        {**cases["unknown_metric"], "metrics": ["downlink_mbps"], "basis": "native", "sql": "x"},
    )
    assert extra.error is not None and extra.error.code.value == "INVALID_ARGUMENT"
    other_session = harness.dispatcher.dispatch("get_session_context", {"session_id": "ses_other"})
    assert other_session.error is not None and other_session.error.code.value == "OUT_OF_SCOPE"
    unknown_tool = harness.dispatcher.dispatch("run_sql", {"query": "select 1"})
    assert unknown_tool.error is not None and unknown_tool.error.code.value == "UNKNOWN_TOOL"


def test_window_longer_than_the_tool_maximum_is_refused(stable_radio_harness: Any) -> None:
    from celltrace.tools.dispatch import ToolDispatcher

    harness = stable_radio_harness
    narrow_context = harness.context.model_copy(update={"maximum_tool_window_ms": 5_000})
    dispatcher = ToolDispatcher(
        narrow_context, harness.repository, harness.store, harness.dispatcher.clock
    )
    allowed = harness.context.allowed_window
    too_long = dispatcher.dispatch(
        "get_activity_state",
        {"start_ms": allowed.start_ms, "end_ms": allowed.start_ms + 6_000, "basis": "native"},
    )
    assert too_long.error is not None and too_long.error.code.value == "OUT_OF_SCOPE"
    within = dispatcher.dispatch(
        "get_activity_state",
        {"start_ms": allowed.start_ms, "end_ms": allowed.start_ms + 5_000, "basis": "native"},
    )
    assert within.error is None


def test_evidence_is_hashed_cached_and_scoped_to_the_run(stable_radio_harness: Any) -> None:
    harness = stable_radio_harness
    arguments = {
        **window_arguments(harness),
        "metrics": ["downlink_mbps"],
        "basis": "bin_1s",
        "statistic": "median",
    }
    first = harness.dispatcher.dispatch("compare_windows", arguments)
    assert first.evidence is not None
    harness.store.save_evidence(first.evidence.model_dump(mode="json"))
    second = harness.dispatcher.dispatch("compare_windows", arguments)
    assert second.cache_hit and second.evidence is not None
    assert second.evidence.evidence_id == first.evidence.evidence_id
    record = first.evidence.model_dump(mode="json")
    assert evidence_content_hash(record) == record["content_sha256"]
    retrieved = harness.dispatcher.dispatch(
        "get_evidence", {"evidence_id": first.evidence.evidence_id}
    )
    assert retrieved.retrieved is not None
    missing = harness.dispatcher.dispatch("get_evidence", {"evidence_id": "ev_does_not_exist"})
    assert missing.error is not None and missing.error.code.value == "OUT_OF_SCOPE"


def test_missing_values_are_null_counts_not_zero(stable_radio_harness: Any) -> None:
    harness = stable_radio_harness
    allowed = harness.context.allowed_window
    outcome = harness.dispatcher.dispatch(
        "summarize_window",
        {
            "start_ms": allowed.start_ms,
            "end_ms": allowed.start_ms + 10_000,
            "metrics": ["ping_average_ms"],
            "basis": "native",
        },
    )
    assert outcome.evidence is not None
    summary = outcome.evidence.result["summaries"]["ping_average_ms"]
    assert summary["valid_count"] == 0
    assert summary["mean"] is None and summary["median"] is None
    assert summary["missing_count"] == summary["row_count"] > 0


def test_cell_transition_tool_reports_the_qualified_change(radio_drop_harness: Any) -> None:
    harness = radio_drop_harness
    candidate = harness.context.candidate
    outcome = harness.dispatcher.dispatch(
        "get_cell_transitions",
        {
            "start_ms": candidate.representative_reference_window.start_ms,
            "end_ms": candidate.representative_event_window.end_ms,
        },
    )
    assert outcome.evidence is not None
    transitions = outcome.evidence.result["transitions"]
    assert [item["transition_type"] for item in transitions] == ["serving_cell_change"]
    assert transitions[0]["earlier_time_ms"] == 39_000
    assert transitions[0]["later_time_ms"] == 40_000
    assert "identity_scope_unverified" in transitions[0]["uncertainty"]
