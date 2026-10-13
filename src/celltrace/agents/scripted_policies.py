"""Scripted investigator and reviewer policies for offline runs.

These replay a fixed plan and read numbers back from their own tool results.
They exist to exercise orchestration, verification and recovery without a
model. They are not evidence of adaptive model behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from celltrace.contracts.common import MetricName
from celltrace.providers.base import ProviderRequest
from celltrace.providers.scripted import ScriptedTurn, tool_results
from celltrace.reporting.fallback import claims_from_results

FABRICATED_EVIDENCE_ID = "ev_fabricated00000000"


def _candidate(messages: list[dict[str, Any]]) -> dict[str, Any]:
    text = messages[0]["content"][0]["text"]
    return dict(json.loads(text.split("Candidate: ", 1)[1].split("\n", 1)[0]))


def _assistant_tools(messages: list[dict[str, Any]]) -> list[str]:
    names = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if block.get("type") == "tool_use":
                names.append(block["name"])
    return names


def _last_feedback(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    last = messages[-1]
    if last["role"] != "user":
        return None
    for block in last["content"]:
        if block.get("type") == "tool_result":
            payload = json.loads(block["content"])
            if payload.get("status") == "revision requested":
                return dict(payload)
    return None


def claims_by_type(claims: list[dict[str, Any]], claim_type: str) -> list[str]:
    return [claim["claim_id"] for claim in claims if claim["claim_type"] == claim_type]


def _results_by_tool(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in tool_results(messages) if "evidence_id" in result]


@dataclass
class ScriptedInvestigator:
    """Fixed plan: context, activity, rate and RSRP comparisons, transitions, submit."""

    corrupt_first_submission: bool = False
    radio_metric: MetricName = MetricName.reported_rsrp_dbm

    def plan(self, candidate: dict[str, Any]) -> list[ScriptedTurn]:
        detector = candidate["detector_output_to_check"]
        reference = detector["representative_reference_window"]
        event = detector["representative_event_window"]
        windows = {
            "reference_start_ms": reference["start_ms"],
            "reference_end_ms": reference["end_ms"],
            "event_start_ms": event["start_ms"],
            "event_end_ms": event["end_ms"],
        }
        span = {"start_ms": reference["start_ms"], "end_ms": event["end_ms"]}
        return [
            ScriptedTurn(
                "Read the allowed bounds and supported measurements.", "get_session_context", {}
            ),
            ScriptedTurn(
                "Check application activity across the reference and event windows.",
                "get_activity_state",
                {**span, "basis": "native"},
            ),
            ScriptedTurn(
                "Recompute the detector comparison of median downlink on one second bins.",
                "compare_windows",
                {**windows, "metrics": ["downlink_mbps"], "basis": "bin_1s", "statistic": "median"},
            ),
            ScriptedTurn(
                "Compare reported radio power over the same windows.",
                "compare_windows",
                {
                    **windows,
                    "metrics": [self.radio_metric.value],
                    "basis": "bin_1s",
                    "statistic": "median",
                },
            ),
            ScriptedTurn(
                "Look for qualified serving cell changes around the drop.",
                "get_cell_transitions",
                span,
            ),
        ]

    def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
        messages = request.messages
        candidate = _candidate(messages)
        plan = self.plan(candidate)
        used = _assistant_tools(messages)
        submissions = used.count("submit_findings")
        if submissions == 0 and len(used) < len(plan):
            return plan[len(used)]
        feedback = _last_feedback(messages)
        if feedback is not None and feedback.get("reviewer") and used[-1] == "submit_findings":
            event = candidate["detector_output_to_check"]["representative_event_window"]
            return ScriptedTurn(
                "Reviewer asked for uplink context in the event window.",
                "summarize_window",
                {**event, "metrics": ["uplink_mbps"], "basis": "native"},
            )
        corrupt = self.corrupt_first_submission and submissions == 0
        return ScriptedTurn(
            "Submitting findings from the recorded evidence.",
            "submit_findings",
            build_findings(candidate, messages, corrupt=corrupt),
        )


def build_findings(
    candidate: dict[str, Any], messages: list[dict[str, Any]], corrupt: bool = False
) -> dict[str, Any]:
    bundle = claims_from_results(_results_by_tool(messages))
    claims = bundle.claims
    rate_change = bundle.rate_change
    radio_change = bundle.radio_change
    active_fraction = bundle.active_fraction
    transitions = bundle.transitions
    activity_ids = [claim["claim_id"] for claim in claims if claim["claim_type"] == "activity"]

    if corrupt and rate_change is not None:
        # Deliberately wrong number and an invented citation, to exercise the code gate.
        rate_change["reference_value"] = rate_change["reference_value"] * 1.25
        claims.append(
            {
                "claim_type": "metric_statistic",
                "claim_id": "invented_statistic",
                "metric": "downlink_mbps",
                "statistic": "median",
                "window": rate_change["event_window"],
                "basis": "bin_1s",
                "value": 0.0,
                "unit": "Mbps",
                "valid_count": 5,
                "missing_count": 0,
                "evidence_ids": [FABRICATED_EVIDENCE_ID],
            }
        )

    rate_dropped = rate_change is not None and (rate_change["relative_decrease"] or 0) >= 0.5
    radio_fell = radio_change is not None and radio_change["delta"] <= -3.0
    active = active_fraction is not None and active_fraction >= 0.8
    patterns = []
    if rate_dropped and active:
        patterns.append("active_rate_drop")
    if active_fraction is not None and active_fraction < 0.8:
        patterns.append("idle_or_inactive_traffic")
    if radio_change is not None:
        patterns.append("reported_radio_degradation" if radio_fell else "stable_reported_radio")
    if transitions:
        patterns.append("qualified_cell_transition")
    if not patterns:
        patterns.append("no_clear_symptom")

    radio_claim_ids = [radio_change["claim_id"]] if radio_change else []
    hypotheses = [
        {
            "hypothesis_id": "radio_degradation",
            # Stable RSRP cannot rule out interference or link problems, so without a
            # fall the hypothesis stays untestable rather than contradicted.
            "status": "consistent" if radio_fell else "not_testable",
            "supporting_claim_ids": radio_claim_ids if radio_fell else [],
            "contradicting_claim_ids": [],
            "assumptions": ["Reported RSRP reflects the serving link during the window."],
            "discriminating_measurement": "Block error rate and link adaptation (MCS) logs.",
        },
        {
            "hypothesis_id": "shared_resource_load",
            "status": "not_testable",
            "supporting_claim_ids": [],
            "contradicting_claim_ids": [],
            "assumptions": [],
            "discriminating_measurement": "Cell load and scheduled resource block counters.",
        },
        {
            "hypothesis_id": "application_demand",
            "status": "contradicted" if active else "consistent",
            "supporting_claim_ids": [] if active or active_fraction is None else activity_ids[:1],
            "contradicting_claim_ids": activity_ids[:1] if active else [],
            "assumptions": ["Active download state means the application wanted data."],
            "discriminating_measurement": "Application request log and server send rate.",
        },
        {
            "hypothesis_id": "transport_or_server",
            "status": "not_testable",
            "supporting_claim_ids": [],
            "contradicting_claim_ids": [],
            "assumptions": [],
            "discriminating_measurement": "TCP retransmissions, round trip time and server logs.",
        },
    ]
    if transitions:
        hypotheses.append(
            {
                "hypothesis_id": "mobility_related",
                "status": "consistent",
                "supporting_claim_ids": [claims_by_type(claims, "transition")[0]],
                "contradicting_claim_ids": [],
                "assumptions": ["The identity change reflects a cell change, which is unverified."],
                "discriminating_measurement": "Handover signalling events with timestamps.",
            }
        )
    eligibility_claims = activity_ids[:1]
    return {
        "event_id": candidate["event_id"],
        "eligibility": {
            "status": "eligible"
            if active
            else ("ineligible" if active_fraction is not None else "unknown"),
            "supporting_claim_ids": eligibility_claims,
            "reason": "Activity share of the reference and event windows against the 0.8 rule.",
        },
        "factual_claims": claims,
        "observation_patterns": patterns,
        "hypotheses": hypotheses,
        "missing_evidence": [
            "No cell load or scheduling counters.",
            "No transport layer measurements.",
        ],
        "next_measurements": [
            "Scheduled resource blocks and cell load for the serving cell.",
            "TCP retransmissions and round trip time during the event window.",
        ],
        "decision_summary": (
            "Checked activity, the detector's rate comparison, reported radio power and cell "
            "identity changes. The physical cause is not identifiable from these measurements."
        ),
        "cause_status": "unresolved",
        "assessment_scope": "cited_windows",
    }


@dataclass
class ScriptedReviewer:
    """Checks the rate comparison with a different statistic, then decides."""

    request_more_in_first_round: bool = False

    def next_turn(self, request: ProviderRequest) -> ScriptedTurn:
        messages = request.messages
        opening = messages[0]["content"][0]["text"]
        candidate = json.loads(opening.split("Candidate: ", 1)[1].split("\n", 1)[0])
        detector = candidate["detector_output_to_check"]
        reference = detector["representative_reference_window"]
        event = detector["representative_event_window"]
        used = _assistant_tools(messages)
        rounds_submitted = used.count("submit_review")
        since_last = (
            used[len(used) - used[::-1].index("submit_review") :] if rounds_submitted else used
        )
        if not since_last:
            return ScriptedTurn(
                "Independently check the rate drop using means over logged samples.",
                "compare_windows",
                {
                    "reference_start_ms": reference["start_ms"],
                    "reference_end_ms": reference["end_ms"],
                    "event_start_ms": event["start_ms"],
                    "event_end_ms": event["end_ms"],
                    "metrics": ["downlink_mbps"],
                    "basis": "native",
                    "statistic": "mean",
                },
            )
        own_results = [
            result
            for result in _results_by_tool(messages)
            if result["tool_name"] == "compare_windows"
        ]
        evidence_ids = [result["evidence_id"] for result in own_results][-1:]
        if not evidence_ids:
            # The independent check failed, so record a limited review instead of accepting.
            return ScriptedTurn(
                "My independent measurement failed, so the review is limited.",
                "submit_review",
                {
                    "decision": "unresolved",
                    "disputed_claim_ids": [],
                    "independent_evidence_ids": [],
                    "alternative_checks": [],
                    "requested_queries": [],
                    "reason_summary": "No independent measurement was possible.",
                    "review_limitation": "The comparison tool returned errors for every attempt.",
                },
            )
        if self.request_more_in_first_round and rounds_submitted == 0:
            return ScriptedTurn(
                "Uplink context is missing for the event window.",
                "submit_review",
                {
                    "decision": "request_more_evidence",
                    "disputed_claim_ids": [],
                    "independent_evidence_ids": evidence_ids,
                    "alternative_checks": ["Check uplink traffic during the event window."],
                    "requested_queries": [
                        "summarize_window uplink_mbps native over the event window"
                    ],
                    "reason_summary": "The rate drop is confirmed, but uplink context is absent.",
                    "review_limitation": None,
                },
            )
        return ScriptedTurn(
            "The independent check agrees with the reported drop.",
            "submit_review",
            {
                "decision": "accept",
                "disputed_claim_ids": [],
                "independent_evidence_ids": evidence_ids,
                "alternative_checks": ["Application inactivity was checked by the investigator."],
                "requested_queries": [],
                "reason_summary": "Mean based check agrees with the median based drop.",
                "review_limitation": None,
            },
        )
