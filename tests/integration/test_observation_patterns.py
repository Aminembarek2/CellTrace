"""Observation labels cannot bypass verified measurements and detector support."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from celltrace.config import DetectorSection
from celltrace.contracts.common import Arm, TimeWindow
from celltrace.contracts.findings import ObservationPattern
from celltrace.data.fixture import IDLE_DECLINE_MEMBER, STABLE_RADIO_DROP_MEMBER
from celltrace.reporting.fallback import claims_from_results
from celltrace.reporting.render import render_markdown
from celltrace.verification.verifier import Verifier

from .conftest import FixtureSnapshot
from .test_verifier import evidence_set, findings_from
from .test_workflow import (
    CORRECTION_ALLOWANCE,
    PromptRecordingInvestigator,
    PromptRecordingReviewer,
    runner_for,
    start,
)


def labeled_findings(harness: Any) -> Any:
    claims = claims_from_results(evidence_set(harness)).claims
    return findings_from(harness, claims).model_copy(
        update={"observation_patterns": [ObservationPattern.active_rate_drop]}
    )


def test_active_drop_requires_verified_measurements(stable_radio_harness: Any) -> None:
    harness = stable_radio_harness
    findings = labeled_findings(harness)
    result = Verifier(harness.context, harness.store, harness.repository).verify(findings)
    assert result.passed, result.failed_checks
    pattern_check = result.pattern_checks[0]
    assert pattern_check.passed
    assert pattern_check.actual["relative_drop"] >= 0.6
    assert pattern_check.actual["event"]["active_count"] == 5
    assert len(pattern_check.supporting_claim_ids) == 2
    assert result.verification_queries == len(findings.factual_claims) + 2

    rate_claim = next(
        claim for claim in findings.factual_claims if claim.claim_type == "metric_change"
    )
    invented_rate = rate_claim.model_copy(update={"evidence_ids": ["ev_invented"]})
    invalid_findings = findings.model_copy(
        update={
            "factual_claims": [
                invented_rate if claim.claim_id == rate_claim.claim_id else claim
                for claim in findings.factual_claims
            ]
        }
    )
    result = Verifier(harness.context, harness.store, harness.repository).verify(invalid_findings)
    assert not result.passed
    assert not result.pattern_checks[0].passed
    assert result.pattern_checks[0].supporting_claim_ids == []


def test_candidate_drop_numbers_do_not_establish_a_drop(stable_radio_harness: Any) -> None:
    harness = stable_radio_harness
    candidate = harness.context.candidate.model_copy(
        update={
            "representative_reference_window": TimeWindow(start_ms=0, end_ms=30000),
            "representative_event_window": TimeWindow(start_ms=30000, end_ms=35000),
            "reference_median_mbps": 100.0,
            "event_median_mbps": 1.0,
            "relative_drop": 0.99,
        }
    )
    harness.context = harness.context.model_copy(update={"candidate": candidate})
    harness.dispatcher.context = harness.context
    result = Verifier(harness.context, harness.store, harness.repository).verify(
        labeled_findings(harness)
    )
    assert all(check.passed for check in result.claim_checks)
    assert not result.passed
    pattern_check = result.pattern_checks[0]
    assert pattern_check.code is not None
    assert pattern_check.code.value == "OBSERVATION_PATTERN_UNSUPPORTED"
    assert pattern_check.actual["relative_drop"] == 0.0


def test_combined_activity_cannot_hide_an_idle_event(
    stable_radio_harness: Any, fixture_snapshot: FixtureSnapshot
) -> None:
    harness = stable_radio_harness
    session_id = fixture_snapshot.session_by_member[IDLE_DECLINE_MEMBER]
    candidate = harness.context.candidate.model_copy(update={"session_id": session_id})
    harness.context = harness.context.model_copy(
        update={"session_id": session_id, "candidate": candidate}
    )
    harness.dispatcher.context = harness.context
    findings = labeled_findings(harness)
    activity_claim = next(
        claim for claim in findings.factual_claims if claim.claim_type == "activity"
    )
    assert activity_claim.fraction >= 0.8
    result = Verifier(harness.context, harness.store, harness.repository).verify(findings)
    assert all(check.passed for check in result.claim_checks)
    assert not result.pattern_checks[0].passed
    support = result.pattern_checks[0].actual["event"]
    assert support["active_count"] / support["sample_count"] < 0.8


@pytest.mark.parametrize(
    "changed_support",
    [
        {"valid_bins": 3},
        {"complete_bins": 4},
        {"available_bins": 4},
        {"same_segment_bins": 4},
    ],
)
def test_drop_with_inadequate_bin_support_is_rejected(
    stable_radio_harness: Any, monkeypatch: pytest.MonkeyPatch, changed_support: dict[str, int]
) -> None:
    harness = stable_radio_harness
    findings = labeled_findings(harness)
    original_query = harness.repository.drop_window

    def query_with_gap(
        session_id: str, segment_id: int, window: TimeWindow, available_by_ms: int
    ) -> Any:
        support = original_query(session_id, segment_id, window, available_by_ms)
        if window == harness.context.candidate.representative_event_window:
            return replace(support, **changed_support)
        return support

    monkeypatch.setattr(harness.repository, "drop_window", query_with_gap)
    result = Verifier(harness.context, harness.store, harness.repository).verify(findings)
    assert all(check.passed for check in result.claim_checks)
    assert not result.passed
    assert not result.pattern_checks[0].passed


def test_pattern_recomputation_failure_never_passes(
    stable_radio_harness: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = stable_radio_harness
    findings = labeled_findings(harness)

    def unavailable(*arguments: Any) -> Any:
        raise RuntimeError("measurement service unavailable")

    monkeypatch.setattr(harness.repository, "drop_window", unavailable)
    result = Verifier(harness.context, harness.store, harness.repository).verify(findings)
    assert not result.passed
    assert result.pattern_checks[0].code is not None
    assert result.pattern_checks[0].code.value == "RECOMPUTATION_FAILED"


def test_configured_threshold_and_conflicting_labels_are_enforced(
    stable_radio_harness: Any,
) -> None:
    harness = stable_radio_harness
    findings = labeled_findings(harness)
    result = Verifier(
        harness.context,
        harness.store,
        harness.repository,
        detector_configuration=DetectorSection(minimum_relative_drop=0.95),
    ).verify(findings)
    assert all(check.passed for check in result.claim_checks)
    assert not result.pattern_checks[0].passed
    assert result.pattern_checks[0].expected["minimum_relative_drop"] == 0.95

    contradictory = findings.model_copy(
        update={
            "observation_patterns": [
                ObservationPattern.active_rate_drop,
                ObservationPattern.no_clear_symptom,
            ]
        }
    )
    result = Verifier(harness.context, harness.store, harness.repository).verify(contradictory)
    assert not result.passed
    assert result.pattern_checks[0].code is not None
    assert result.pattern_checks[0].code.value == "OBSERVATION_PATTERN_CONFLICT"


def test_raw_and_gated_reports_distinguish_checked_labels(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    raw_runner = runner_for(tmp_path / "raw")
    gated_runner = runner_for(tmp_path / "gated")
    try:
        raw_report = start(
            raw_runner, fixture_snapshot, STABLE_RADIO_DROP_MEMBER, Arm.investigator_raw, "run_raw"
        )
        gated_report = start(
            gated_runner,
            fixture_snapshot,
            STABLE_RADIO_DROP_MEMBER,
            Arm.investigator_gated,
            "run_gated",
        )
        assert raw_report.verified_observation_patterns == []
        assert "active_rate_drop" in raw_report.unverified_observation_patterns
        assert "active_rate_drop" in gated_report.verified_observation_patterns
        assert "stable_reported_radio" in gated_report.verified_observation_patterns
        assert "Unverified observation labels" in render_markdown(raw_report)
    finally:
        raw_runner.close()
        gated_runner.close()


def test_pattern_failure_routes_revision_and_cannot_be_reviewed_away(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    runner = runner_for(
        tmp_path,
        overrides={**CORRECTION_ALLOWANCE, ("analysis", "detector", "minimum_relative_drop"): 0.95},
    )
    try:
        report = start(
            runner,
            fixture_snapshot,
            STABLE_RADIO_DROP_MEMBER,
            Arm.investigator_reviewer_gated,
            "run_wrong_pattern",
        )
        reviews = runner.store.reviews("run_wrong_pattern")
        assert reviews and reviews[0]["review"]["decision"] == "accept"
        assert reviews[0]["independent_query_count"] > 0
        assert report.fact_check_status.value == "failed"
        assert report.workflow_outcome.value != "completed"
        assert report.usage.revision_rounds == 2
        assert "active_rate_drop" not in report.observation_patterns
        assert "Observation labels rejected by code" in render_markdown(report)
        feedback = runner.store.messages("run_wrong_pattern", "investigator")
        assert any("OBSERVATION_PATTERN_UNSUPPORTED" in str(message) for message in feedback)
        reviewer_messages = runner.store.messages("run_wrong_pattern", "reviewer")
        assert any("submit_review" in str(message) for message in reviewer_messages)
    finally:
        runner.close()


def test_both_roles_receive_the_actual_observation_rule(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> None:
    investigator = PromptRecordingInvestigator()
    reviewer = PromptRecordingReviewer()
    runner = runner_for(
        tmp_path,
        investigator=investigator,
        reviewer=reviewer,
        overrides={**CORRECTION_ALLOWANCE, ("analysis", "detector", "minimum_relative_drop"): 0.95},
    )
    try:
        start(
            runner,
            fixture_snapshot,
            STABLE_RADIO_DROP_MEMBER,
            Arm.investigator_reviewer_gated,
            "run_rule",
        )
        assert investigator.prompts and reviewer.prompts
        assert all(
            '"minimum_relative_drop":0.95' in prompt
            for prompt in investigator.prompts + reviewer.prompts
        )
        assert len(set(investigator.prompts)) == 1
        assert len(set(reviewer.prompts)) == 1
    finally:
        runner.close()
