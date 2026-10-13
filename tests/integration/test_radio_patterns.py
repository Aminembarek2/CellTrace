"""Reported radio labels require retrievable same-window measurements and structural support."""

from dataclasses import replace
from typing import Any

import pytest

from celltrace.contracts.common import MetricName
from celltrace.contracts.findings import ObservationPattern
from celltrace.reporting.fallback import claims_from_results
from celltrace.verification.verifier import Verifier

from .test_verifier import evidence_set, findings_from


def labeled(harness: Any, pattern: ObservationPattern) -> Any:
    return findings_from(harness, claims_from_results(evidence_set(harness)).claims).model_copy(
        update={"observation_patterns": [pattern]}
    )


def test_radio_truth_changes_with_measurements_not_the_candidate_label(
    stable_radio_harness: Any,
    radio_drop_harness: Any,
) -> None:
    for harness, correct, wrong in (
        (
            stable_radio_harness,
            ObservationPattern.stable_reported_radio,
            ObservationPattern.reported_radio_degradation,
        ),
        (
            radio_drop_harness,
            ObservationPattern.reported_radio_degradation,
            ObservationPattern.stable_reported_radio,
        ),
    ):
        verified = Verifier(harness.context, harness.store, harness.repository).verify(
            labeled(harness, correct)
        )
        assert verified.passed, verified.failed_checks
        assert verified.pattern_checks[0].supporting_claim_ids
        rejected = Verifier(harness.context, harness.store, harness.repository).verify(
            labeled(harness, wrong)
        )
        assert all(check.passed for check in rejected.claim_checks)
        assert not rejected.pattern_checks[0].passed


def test_radio_claim_must_be_verified_and_labels_cannot_conflict(stable_radio_harness: Any) -> None:
    harness = stable_radio_harness
    findings = labeled(harness, ObservationPattern.stable_reported_radio)
    findings = findings.model_copy(
        update={
            "factual_claims": [
                claim
                for claim in findings.factual_claims
                if getattr(claim, "metric", None) != MetricName.reported_rsrp_dbm
            ]
        }
    )
    result = Verifier(harness.context, harness.store, harness.repository).verify(findings)
    assert not result.pattern_checks[0].passed
    findings = labeled(harness, ObservationPattern.stable_reported_radio)
    findings.observation_patterns.append(ObservationPattern.reported_radio_degradation)
    result = Verifier(harness.context, harness.store, harness.repository).verify(findings)
    assert not result.passed
    assert all(
        check.code.value == "OBSERVATION_PATTERN_CONFLICT" for check in result.pattern_checks
    )


@pytest.mark.parametrize("field", ["complete_bins", "available_bins", "same_segment_bins"])
def test_radio_structure_failure_cannot_pass(
    stable_radio_harness: Any, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = stable_radio_harness
    findings = labeled(harness, ObservationPattern.stable_reported_radio)
    original = harness.repository.drop_window
    monkeypatch.setattr(
        harness.repository, "drop_window", lambda *args: replace(original(*args), **{field: 0})
    )
    result = Verifier(harness.context, harness.store, harness.repository).verify(findings)
    assert not result.pattern_checks[0].passed


def test_failed_radio_recomputation_is_a_failed_check(
    stable_radio_harness: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = stable_radio_harness
    findings = labeled(harness, ObservationPattern.stable_reported_radio)

    def failed(*args: Any) -> Any:
        raise RuntimeError("unavailable")

    monkeypatch.setattr(harness.repository, "drop_window", failed)
    result = Verifier(harness.context, harness.store, harness.repository).verify(findings)
    assert not result.passed
    assert result.pattern_checks[0].code.value == "RECOMPUTATION_FAILED"


@pytest.mark.parametrize(
    "change", [{"coverage": 0.79}, {"median": None}, {"outside_documented_range_count": 1}]
)
def test_missing_sparse_or_out_of_range_radio_is_rejected(
    stable_radio_harness: Any,
    change: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = stable_radio_harness
    findings = labeled(harness, ObservationPattern.stable_reported_radio)
    claim_ids = {claim.claim_id for claim in findings.factual_claims}
    original = harness.repository.summarize
    monkeypatch.setattr(
        harness.repository, "summarize", lambda *args: replace(original(*args), **change)
    )
    from celltrace.config import ObservationSection
    from celltrace.verification.patterns import check_reported_radio

    check = check_reported_radio(
        harness.context,
        findings,
        ObservationPattern.stable_reported_radio,
        claim_ids,
        harness.repository,
        ObservationSection(),
        lambda _: None,
    )
    assert not check.passed


@pytest.mark.parametrize(
    "delta, stable, degraded",
    [(-3.0, True, False), (-3.01, False, True), (3.0, True, False), (3.01, False, False)],
)
def test_radio_threshold_boundary_and_improvement_are_not_degradation(
    stable_radio_harness: Any,
    delta: float,
    stable: bool,
    degraded: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = stable_radio_harness
    findings = labeled(harness, ObservationPattern.stable_reported_radio)
    event = harness.context.candidate.representative_event_window
    original = harness.repository.summarize

    def summary(session: Any, window: Any, metric: Any, basis: Any) -> Any:
        value = original(session, window, metric, basis)
        return replace(value, median=-90.0 + (delta if window == event else 0.0))

    monkeypatch.setattr(harness.repository, "summarize", summary)
    from celltrace.config import ObservationSection
    from celltrace.verification.patterns import check_reported_radio

    for pattern, expected in (
        (ObservationPattern.stable_reported_radio, stable),
        (ObservationPattern.reported_radio_degradation, degraded),
    ):
        updated = findings.model_copy(update={"observation_patterns": [pattern]})
        check = check_reported_radio(
            harness.context,
            updated,
            pattern,
            {c.claim_id for c in findings.factual_claims},
            harness.repository,
            ObservationSection(),
            lambda _: None,
        )
        assert check.passed is expected


def test_radio_claim_on_another_window_is_insufficient(stable_radio_harness: Any) -> None:
    from celltrace.config import ObservationSection
    from celltrace.verification.patterns import check_reported_radio

    harness = stable_radio_harness
    findings = labeled(harness, ObservationPattern.stable_reported_radio)
    claims = [
        claim.model_copy(update={"reference_window": claim.event_window})
        if getattr(claim, "metric", None) == MetricName.reported_rsrp_dbm
        else claim
        for claim in findings.factual_claims
    ]
    updated = findings.model_copy(update={"factual_claims": claims})
    check = check_reported_radio(
        harness.context,
        updated,
        ObservationPattern.stable_reported_radio,
        {c.claim_id for c in claims},
        harness.repository,
        ObservationSection(),
        lambda _: None,
    )
    assert not check.passed


def test_simultaneous_snapshot_readers_keep_external_access_locked(
    fixture_snapshot_dir: Any,
) -> None:
    from celltrace.data.snapshot import open_snapshot

    first, _ = open_snapshot(fixture_snapshot_dir)
    try:
        second, _ = open_snapshot(fixture_snapshot_dir)
        try:
            assert second.execute(
                "SELECT current_setting('enable_external_access')"
            ).fetchone() == (False,)
            with pytest.raises(Exception, match="locked"):
                second.execute("SET enable_external_access=true")
            assert first.execute("SELECT 1").fetchone() == (1,)
        finally:
            second.close()
    finally:
        first.close()
