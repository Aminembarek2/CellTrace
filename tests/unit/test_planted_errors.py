"""No-credit cases at the planted-error scoring boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from celltrace.contracts.findings import ReviewResult, finding_path_value
from celltrace.evaluation.planted_errors import score_item, summarize


def review(paths: list[str], decision: str = "unresolved") -> dict:
    return {
        "decision": decision,
        "disputed_finding_paths": paths,
        "disputed_claim_ids": [],
        "independent_evidence_ids": ["ev_own"],
        "alternative_checks": [],
        "requested_queries": [],
        "reason_summary": "An interpretation exceeds the measurements.",
    }


@pytest.mark.parametrize(
    ("paths", "independent", "cited", "expected"),
    [
        (["/decision_summary"], 1, True, True),
        ([], 1, True, False),
        (["/missing_evidence"], 1, True, False),
        (["/absent"], 1, True, False),
        (["/decision_summary"], 0, True, False),
        (["/decision_summary"], 1, False, False),
    ],
)
def test_detection_requires_exact_path_and_cited_successful_query(
    paths: list[str], independent: int, cited: bool, expected: bool
) -> None:
    score = score_item(
        {"family": "physical_cause", "target_path": "/decision_summary"},
        {"decision_summary": "unsupported", "factual_claims": [], "missing_evidence": []},
        {
            "status": "submitted",
            "review": review(paths),
            "successful_independent_queries": independent,
            "cited_independent_query": cited,
        },
    )
    assert score["localized_challenge"] is expected


def test_control_objection_is_false_alarm_and_missing_results_stay_in_denominator() -> None:
    findings = {"decision_summary": "qualified", "factual_claims": []}
    items = [
        {
            "item_id": "a",
            "findings": findings,
            "label": {"family": "correct_control", "target_path": None},
        },
        {
            "item_id": "b",
            "findings": findings,
            "label": {"family": "physical_cause_new", "target_path": "/decision_summary"},
        },
    ]
    outcomes = {"a": {"status": "submitted", "review": review(["/decision_summary"])}}
    result = summarize(items, outcomes, "scripted_fixture")
    assert result["scheduled"] == 2
    assert result["families"]["correct_control"]["decision_objections"] == 1
    assert result["families"]["physical_cause_new"]["scheduled"] == 1
    assert result["families"]["physical_cause_new"]["statuses"] == {"not_run": 1}


def test_json_pointer_and_contradictory_acceptance_contract() -> None:
    assert finding_path_value({"a/b": {"~": ["value"]}}, "/a~1b/~0/0") == "value"
    for path in ("", "/", "/a~2", "/a~", "/array/01", "/array/-1"):
        with pytest.raises(ValueError):
            finding_path_value({"array": [1, 2]}, path)
    with pytest.raises(ValidationError):
        ReviewResult.model_validate(review(["/decision_summary"], "accept"))
    with pytest.raises(ValidationError):
        ReviewResult.model_validate(review(["/decision_summary", "/decision_summary"]))


def test_secondary_endpoint_ignores_only_the_query_rule() -> None:
    label = {"family": "physical_cause", "target_path": "/decision_summary"}
    findings = {"decision_summary": "unsupported", "factual_claims": [], "missing_evidence": []}
    without_query = score_item(
        label,
        findings,
        {
            "status": "submitted",
            "review": review(["/decision_summary"]),
            "successful_independent_queries": 0,
            "cited_independent_query": False,
        },
    )
    assert without_query["localized_without_query"] and not without_query["localized_challenge"]
    wrong_section = score_item(
        label,
        findings,
        {
            "status": "submitted",
            "review": review(["/missing_evidence"]),
            "successful_independent_queries": 1,
            "cited_independent_query": True,
        },
    )
    assert not wrong_section["localized_without_query"]


def test_controls_report_which_planted_paths_were_disputed_and_draw_consistency() -> None:
    findings = {
        "decision_summary": "qualified",
        "factual_claims": [],
        "hypotheses": [{"status": "not_testable"}],
    }
    items = [
        {
            "item_id": f"control_{draw}",
            "finding_id": "control",
            "findings": findings,
            "label": {"family": "correct_control", "target_path": None},
        }
        for draw in range(2)
    ]
    outcomes = {
        "control_0": {
            "status": "submitted",
            "review": review(["/decision_summary"]),
            "successful_independent_queries": 1,
            "cited_independent_query": True,
        },
        "control_1": {"status": "submitted", "review": review([], "accept")},
    }
    control = summarize(items, outcomes, "scripted_fixture")["families"]["correct_control"]
    assert control["decision_objections"] == 1
    assert control["disputed_target_paths"] == {"/decision_summary": 1}
    assert control["draw_consistency"] == {"0_of_2": 1, "1_of_2": 0, "2_of_2": 0}
