"""Populated radio fields cannot establish unsupported serving-radio facts."""

from typing import Any

import duckdb
import pytest

from celltrace.contracts.common import AggregationBasis, MetricName, TimeWindow
from celltrace.contracts.findings import MetricStatisticClaim, ObservationPattern
from celltrace.reporting.fallback import claims_from_results
from celltrace.verification.verifier import Verifier

from .test_verifier import collect, evidence_set, findings_from


@pytest.fixture
def mutable_measurements(stable_radio_harness: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Copy invented fixture measurements into RAM; the frozen snapshot stays untouched."""
    harness = stable_radio_harness
    connection = duckdb.connect()
    for table in ("samples", "bins_1s", "sessions"):
        connection.register(
            "fixture_rows", harness.connection.execute(f"SELECT * FROM {table}").df()
        )
        connection.execute(f"CREATE TABLE {table} AS SELECT * FROM fixture_rows")
        connection.unregister("fixture_rows")
    monkeypatch.setattr(harness.repository, "connection", connection)
    yield harness
    connection.close()


@pytest.mark.parametrize("mode", ["UMTS", "HSPA+", None, "unknown"])
def test_populated_inapplicable_radio_is_visible_but_cannot_pass_as_a_fact(
    mutable_measurements: Any,
    mode: str | None,
) -> None:
    harness = mutable_measurements
    harness.repository.connection.execute("UPDATE samples SET network_mode = ?", [mode])
    records = evidence_set(harness)
    comparison = records[0]["result"]["comparisons"]["reported_rsrp_dbm"]
    reference = comparison["reference"]
    assert reference["valid_count"] > 0 and reference["median"] is not None
    assert not reference["measurement_applicability"]["supported"]
    bundle = claims_from_results(records)
    assert bundle.rate_change is not None and bundle.radio_change is None
    window = harness.context.candidate.representative_reference_window
    summary_record = collect(
        harness,
        "summarize_window",
        {
            **window.model_dump(),
            "metrics": ["reported_rsrp_dbm", "reported_rsrq_db"],
            "basis": "native",
        },
    )
    for metric, summary in summary_record["result"]["summaries"].items():
        claim = MetricStatisticClaim.model_validate(
            {
                "claim_type": "metric_statistic",
                "claim_id": "radio",
                "metric": metric,
                "statistic": "median",
                "basis": "native",
                "window": window.model_dump(),
                "value": summary["median"],
                "unit": summary["unit"],
                "valid_count": summary["valid_count"],
                "missing_count": summary["missing_count"],
                "evidence_ids": [summary_record["evidence_id"]],
            }
        )
        check = Verifier(harness.context, harness.store, harness.repository).check_claim(claim)
        assert not check.passed and check.code.value == "METRIC_NOT_APPLICABLE"
    # A legacy or malicious agent can still submit the preserved numeric change.
    claim = {
        "claim_type": "metric_change",
        "claim_id": "legacy_radio",
        "metric": "reported_rsrp_dbm",
        "statistic": "median",
        "basis": "bin_1s",
        "reference_window": records[0]["result"]["reference_window"],
        "event_window": records[0]["result"]["event_window"],
        "reference_value": comparison["reference_value"],
        "event_value": comparison["event_value"],
        "delta": comparison["delta_event_minus_reference"],
        "unit": "dBm",
        "evidence_ids": [records[0]["evidence_id"]],
    }
    findings = findings_from(harness, [claim]).model_copy(
        update={
            "observation_patterns": [ObservationPattern.stable_reported_radio],
        }
    )
    result = Verifier(harness.context, harness.store, harness.repository).verify(findings)
    assert result.claim_checks[0].code.value == "METRIC_NOT_APPLICABLE"
    assert not result.pattern_checks[0].passed
    from celltrace.config import ObservationSection
    from celltrace.verification.patterns import check_reported_radio

    pattern = check_reported_radio(
        harness.context,
        findings,
        ObservationPattern.stable_reported_radio,
        {"legacy_radio"},
        harness.repository,
        ObservationSection(),
        lambda _: None,
    )
    assert not pattern.passed and pattern.code.value == "METRIC_NOT_APPLICABLE"
    # Availability describes logger presence and remains valid, despite unknown semantics.
    availability = claims_from_results([summary_record]).claims
    assert (
        Verifier(harness.context, harness.store, harness.repository)
        .verify(findings_from(harness, availability))
        .passed
    )


def test_mixed_supported_modes_pass_but_one_unknown_contributing_row_blocks(
    mutable_measurements: Any,
) -> None:
    harness = mutable_measurements
    connection = harness.repository.connection
    connection.execute(
        "UPDATE samples SET network_mode = CASE WHEN elapsed_ms % 2000 = 0 THEN 'LTE' ELSE '5G' END"
    )
    claims = claims_from_results(evidence_set(harness)).claims
    assert (
        Verifier(harness.context, harness.store, harness.repository)
        .verify(findings_from(harness, claims))
        .passed
    )
    window = harness.context.candidate.representative_event_window
    connection.execute(
        "UPDATE samples SET network_mode = NULL WHERE elapsed_ms = ?", [window.start_ms]
    )
    radio = next(claim for claim in claims if claim.get("metric") == "reported_rsrp_dbm")
    result = Verifier(harness.context, harness.store, harness.repository).verify(
        findings_from(harness, [radio])
    )
    assert result.claim_checks[0].code.value == "METRIC_NOT_APPLICABLE"


def test_bin_applicability_only_checks_contributing_rows_and_half_open_bounds(
    mutable_measurements: Any,
) -> None:
    harness = mutable_measurements
    connection = harness.repository.connection
    connection.execute("UPDATE samples SET network_mode = 'LTE'")
    connection.execute("UPDATE samples SET network_mode = 'UMTS' WHERE elapsed_ms IN (0, 3000)")
    window = TimeWindow(start_ms=500, end_ms=3000)
    binned = harness.repository.summarize(
        harness.context.session_id, window, MetricName.reported_rsrp_dbm, AggregationBasis.bin_1s
    )
    assert binned.applicability.supported
    native = harness.repository.summarize(
        harness.context.session_id,
        TimeWindow(start_ms=0, end_ms=3000),
        MetricName.reported_rsrp_dbm,
        AggregationBasis.native,
    )
    assert native.applicability.unsupported_sample_count == 1
    assert not native.applicability.supported
