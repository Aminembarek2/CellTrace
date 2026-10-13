"""Code verification rejects wrong, invented, altered and unsupported claims."""

from __future__ import annotations

import json
from typing import Any

import pytest

from celltrace.contracts.common import Arm, ExecutionMode
from celltrace.contracts.findings import InvestigationFindings
from celltrace.reporting.build import build_report
from celltrace.reporting.fallback import claims_from_results
from celltrace.verification.verifier import Verifier, numbers_match


def collect(harness: Any, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    outcome = harness.dispatcher.dispatch(tool, arguments)
    assert outcome.evidence is not None, outcome.error
    record = outcome.evidence.model_dump(mode="json")
    harness.store.save_evidence(record)
    return record


def evidence_set(harness: Any) -> list[dict[str, Any]]:
    candidate = harness.context.candidate
    reference = candidate.representative_reference_window
    event = candidate.representative_event_window
    windows = {
        "reference_start_ms": reference.start_ms,
        "reference_end_ms": reference.end_ms,
        "event_start_ms": event.start_ms,
        "event_end_ms": event.end_ms,
    }
    span = {"start_ms": reference.start_ms, "end_ms": event.end_ms}
    return [
        collect(
            harness,
            "compare_windows",
            {
                **windows,
                "metrics": ["downlink_mbps", "reported_rsrp_dbm"],
                "basis": "bin_1s",
                "statistic": "median",
            },
        ),
        collect(harness, "get_activity_state", {**span, "basis": "native"}),
        collect(harness, "get_cell_transitions", span),
    ]


def findings_from(harness: Any, claims: list[dict[str, Any]]) -> InvestigationFindings:
    return InvestigationFindings.model_validate(
        {
            "event_id": harness.context.candidate.event_id,
            "eligibility": {"status": "unknown", "supporting_claim_ids": [], "reason": "test"},
            "factual_claims": claims,
            "observation_patterns": [],
            "hypotheses": [],
            "missing_evidence": [],
            "next_measurements": [],
            "decision_summary": "test",
            "cause_status": "unresolved",
        }
    )


def verifier(harness: Any) -> Verifier:
    return Verifier(harness.context, harness.store, harness.repository)


def test_correct_claims_pass(radio_drop_harness: Any) -> None:
    claims = claims_from_results(evidence_set(radio_drop_harness)).claims
    assert {claim["claim_type"] for claim in claims} == {"metric_change", "activity", "transition"}
    result = verifier(radio_drop_harness).verify(findings_from(radio_drop_harness, claims))
    assert result.passed, result.failed_checks
    assert result.verification_queries == len(claims)


def mutate(claims: list[dict[str, Any]], claim_type: str, **changes: Any) -> list[dict[str, Any]]:
    mutated = json.loads(json.dumps(claims))
    target = next(claim for claim in mutated if claim["claim_type"] == claim_type)
    target.update(changes)
    return mutated


@pytest.mark.parametrize(
    ("claim_type", "changes", "code"),
    [
        ("metric_change", {"reference_value": 30.0}, "VALUE_MISMATCH"),
        ("metric_change", {"evidence_ids": ["ev_invented0000000"]}, "UNKNOWN_EVIDENCE"),
        ("metric_change", {"unit": "dB"}, "UNIT_MISMATCH"),
        ("activity", {"denominator": 99}, "COUNT_MISMATCH"),
        ("transition", {"later_cell_id": "123"}, "TRANSITION_NOT_FOUND"),
        ("transition", {"transition_type": "technology_change"}, "TRANSITION_NOT_FOUND"),
    ],
)
def test_wrong_claims_are_rejected(
    radio_drop_harness: Any, claim_type: str, changes: dict[str, Any], code: str
) -> None:
    claims = mutate(
        claims_from_results(evidence_set(radio_drop_harness)).claims, claim_type, **changes
    )
    result = verifier(radio_drop_harness).verify(findings_from(radio_drop_harness, claims))
    assert not result.passed
    assert [check.code.value for check in result.failed_checks if check.code] == [code]


def test_relative_change_on_dbm_is_rejected(radio_drop_harness: Any) -> None:
    claims = claims_from_results(evidence_set(radio_drop_harness)).claims
    radio = next(
        claim
        for claim in claims
        if claim["metric"] == "reported_rsrp_dbm"
        if claim["claim_type"] == "metric_change"
    )
    radio["relative_decrease"] = 0.4
    result = verifier(radio_drop_harness).verify(findings_from(radio_drop_harness, [radio]))
    assert result.failed_checks[0].code.value == "RELATIVE_CHANGE_NOT_ALLOWED"


def test_evidence_that_does_not_cover_the_claim_is_rejected(radio_drop_harness: Any) -> None:
    records = evidence_set(radio_drop_harness)
    claims = claims_from_results(records).claims
    activity_id = next(
        record["evidence_id"] for record in records if record["tool_name"] == "get_activity_state"
    )
    claims = mutate(claims, "metric_change", evidence_ids=[activity_id])
    result = verifier(radio_drop_harness).verify(findings_from(radio_drop_harness, claims))
    assert result.failed_checks[0].code.value == "EVIDENCE_DOES_NOT_SUPPORT"


def test_altered_wrong_snapshot_and_other_run_evidence_are_rejected(
    radio_drop_harness: Any,
) -> None:
    harness = radio_drop_harness
    records = evidence_set(harness)
    claims = claims_from_results(records).claims
    change = next(claim for claim in claims if claim["claim_type"] == "metric_change")
    original = harness.store.get_evidence(change["evidence_ids"][0])
    assert original is not None

    def replace(record: dict[str, Any]) -> None:
        harness.store.connection.execute(
            "UPDATE evidence SET record_json = ? WHERE evidence_id = ?",
            (json.dumps(record), record["evidence_id"]),
        )

    altered = json.loads(json.dumps(original))
    altered["result"]["comparisons"]["downlink_mbps"]["reference_value"] = 99.0
    replace(altered)
    check = verifier(harness).check_claim(findings_from(harness, [change]).factual_claims[0])
    assert check.code is not None and check.code.value == "EVIDENCE_CONTENT_ALTERED"

    wrong_snapshot = dict(original, snapshot_hash="0" * 64)
    replace(wrong_snapshot)
    check = verifier(harness).check_claim(findings_from(harness, [change]).factual_claims[0])
    assert check.code is not None and check.code.value == "EVIDENCE_WRONG_SNAPSHOT"

    other_run = dict(original, run_id="run_someone_else")
    replace(other_run)
    check = verifier(harness).check_claim(findings_from(harness, [change]).factual_claims[0])
    assert check.code is not None and check.code.value == "EVIDENCE_WRONG_RUN"

    wrong_dictionary = dict(original, dictionary_hash="f" * 64)
    replace(wrong_dictionary)
    check = verifier(harness).check_claim(findings_from(harness, [change]).factual_claims[0])
    assert check.code is not None and check.code.value == "EVIDENCE_WRONG_DICTIONARY"


def test_availability_may_cite_session_context_for_the_allowed_window(
    radio_drop_harness: Any,
) -> None:
    context = collect(radio_drop_harness, "get_session_context", {})
    result = context["result"]
    metric, counts = next(iter(result["metric_availability_in_allowed_window"].items()))
    claim = {
        "claim_id": "availability_from_context",
        "claim_type": "availability",
        "metric": metric,
        "basis": "native",
        "window": result["allowed_window"],
        "valid_count": counts["valid_count"],
        "missing_count": counts["missing_count"],
        "evidence_ids": [context["evidence_id"]],
    }
    passed = verifier(radio_drop_harness).verify(findings_from(radio_drop_harness, [claim]))
    assert passed.passed, passed.failed_checks
    wrong_count = {**claim, "valid_count": counts["valid_count"] + 1}
    failed = verifier(radio_drop_harness).verify(findings_from(radio_drop_harness, [wrong_count]))
    assert failed.failed_checks[0].code.value == "COUNT_MISMATCH"
    narrower = {
        **claim,
        "window": {
            "start_ms": result["allowed_window"]["start_ms"],
            "end_ms": result["allowed_window"]["end_ms"] - 1000,
        },
    }
    failed = verifier(radio_drop_harness).verify(findings_from(radio_drop_harness, [narrower]))
    assert failed.failed_checks[0].code.value == "EVIDENCE_DOES_NOT_SUPPORT"


def test_claim_outside_the_allowed_window_is_rejected(radio_drop_harness: Any) -> None:
    harness = radio_drop_harness
    claims = claims_from_results(evidence_set(harness)).claims
    allowed = harness.context.allowed_window
    outside = mutate(
        claims, "activity", window={"start_ms": allowed.end_ms, "end_ms": allowed.end_ms + 1000}
    )
    result = verifier(harness).verify(findings_from(harness, outside))
    assert result.failed_checks[0].code.value == "OUT_OF_SCOPE"


def test_review_acceptance_without_a_query_fails_policy(radio_drop_harness: Any) -> None:
    harness = radio_drop_harness
    claims = claims_from_results(evidence_set(harness)).claims
    review = {
        "review": {"decision": "accept", "review_limitation": None},
        "independent_query_count": 0,
    }
    result = verifier(harness).verify(findings_from(harness, claims), review)
    assert not result.passed
    assert result.policy_errors[0].code.value == "REVIEW_INCOMPLETE"


@pytest.mark.parametrize(
    ("hypothesis_id", "status", "rejected"),
    [
        ("radio_degradation", "contradicted", True),
        ("shared_resource_load", "contradicted", True),
        ("transport_or_server", "contradicted", True),
        ("radio_degradation", "not_testable", False),
        ("radio_degradation", "consistent", False),
        ("application_demand", "contradicted", False),
    ],
)
def test_hypotheses_this_archive_cannot_rule_out_are_never_contradicted(
    radio_drop_harness: Any, hypothesis_id: str, status: str, rejected: bool
) -> None:
    harness = radio_drop_harness
    claims = claims_from_results(evidence_set(harness)).claims
    findings = findings_from(harness, claims).model_dump(mode="json")
    findings["hypotheses"] = [
        {
            "hypothesis_id": hypothesis_id,
            "status": status,
            "supporting_claim_ids": [],
            "contradicting_claim_ids": [],
            "assumptions": [],
            "discriminating_measurement": "Block error rate and scheduling counters.",
        }
    ]
    result = verifier(harness).verify(InvestigationFindings.model_validate(findings))
    codes = {check.code.value for check in result.policy_errors if check.code}
    assert ("EXCLUSION_NOT_OBSERVABLE" in codes) is rejected


def test_tolerance_is_render_independent() -> None:
    assert numbers_match(20.0, 20.0 + 1e-7)
    assert numbers_match(1000.0, 1000.05)
    assert not numbers_match(1000.0, 1000.5)
    assert not numbers_match(None, 0.0)
    assert numbers_match(None, None)


def test_structured_whole_session_scope_fails_even_when_numbers_pass(
    radio_drop_harness: Any,
) -> None:
    harness = radio_drop_harness
    claims = claims_from_results(evidence_set(harness)).claims
    payload = findings_from(harness, claims).model_dump(mode="json")
    payload["assessment_scope"] = "whole_session"
    result = verifier(harness).verify(InvestigationFindings.model_validate(payload))
    assert all(check.passed for check in result.claim_checks)
    assert not result.passed
    assert result.policy_errors[0].claim_id == "assessment_scope"
    assert result.policy_errors[0].code.value == "OUT_OF_SCOPE"


def test_failed_contradicting_claim_cannot_support_hypothesis(
    radio_drop_harness: Any,
) -> None:
    harness = radio_drop_harness
    claims = mutate(
        claims_from_results(evidence_set(harness)).claims, "metric_change", reference_value=999.0
    )
    payload = findings_from(harness, claims).model_dump(mode="json")
    bad_id = next(c["claim_id"] for c in claims if c["claim_type"] == "metric_change")
    payload["hypotheses"] = [
        {
            "hypothesis_id": "application_demand",
            "status": "contradicted",
            "supporting_claim_ids": [],
            "contradicting_claim_ids": [bad_id],
            "assumptions": [],
            "discriminating_measurement": "Application demand logs.",
        }
    ]
    result = verifier(harness).verify(InvestigationFindings.model_validate(payload))
    assert any(c.claim_id == "hypothesis:application_demand" for c in result.policy_errors)


@pytest.mark.parametrize("use_fallback", [False, True])
def test_policy_rejected_exclusion_is_not_delivered_as_qualified_hypothesis(
    radio_drop_harness: Any,
    use_fallback: bool,
) -> None:
    harness = radio_drop_harness
    claims = claims_from_results(evidence_set(harness)).claims
    payload = findings_from(harness, [] if use_fallback else claims).model_dump(mode="json")
    payload["hypotheses"] = [
        {
            "hypothesis_id": "radio_degradation",
            "status": "contradicted",
            "supporting_claim_ids": [],
            "contradicting_claim_ids": [],
            "assumptions": [],
            "discriminating_measurement": "Block error measurements.",
        }
    ]
    findings = InvestigationFindings.model_validate(payload)
    gate = verifier(harness).verify(findings)
    findings_id, _ = harness.store.save_findings(harness.context.run_id, payload, 0)
    harness.store.save_verification(
        harness.context.run_id, findings_id, gate.passed, gate.model_dump(mode="json")
    )
    report = build_report(
        harness.store,
        harness.context,
        Arm.investigator_gated,
        ExecutionMode.scripted_fixture,
        None,
        "verification_failed",
        verifier(harness),
    )
    assert report.verified_claims
    assert (report.report_basis.value == "deterministic_fallback") is use_fallback
    assert report.fact_check_status.value == "failed"
    assert report.qualified_hypotheses == []
    assert report.final_findings is not None
    assert report.final_findings.hypotheses[0].status.value == "contradicted"
    assert any("failed a policy check" in text for text in report.limitations)


def test_policy_identifier_cannot_reject_a_valid_factual_claim(radio_drop_harness: Any) -> None:
    harness = radio_drop_harness
    claims = claims_from_results(evidence_set(harness)).claims
    claims[0]["claim_id"] = "assessment_scope"
    payload = findings_from(harness, claims).model_dump(mode="json")
    payload["assessment_scope"] = "whole_session"
    gate = verifier(harness).verify(InvestigationFindings.model_validate(payload))
    findings_id, _ = harness.store.save_findings(harness.context.run_id, payload, 0)
    harness.store.save_verification(
        harness.context.run_id, findings_id, gate.passed, gate.model_dump(mode="json")
    )
    report = build_report(
        harness.store,
        harness.context,
        Arm.investigator_gated,
        ExecutionMode.scripted_fixture,
        None,
        "verification_failed",
    )
    assert any(c.claim_id == "assessment_scope" for c in report.verified_claims)
    assert report.rejected_claims == []
    assert report.fact_check_status.value == "failed"


def test_legacy_findings_do_not_invent_an_explicit_scope_declaration(
    radio_drop_harness: Any,
) -> None:
    assert findings_from(radio_drop_harness, []).assessment_scope is None
