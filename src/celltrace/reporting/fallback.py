"""Typed claims derived mechanically from evidence records.

Used by the deterministic fallback report and by the scripted policies. The
input is a list of tool results in the shape the model receives (or stored
evidence records, which carry the same keys). Nothing is inferred beyond the
numbers already in those results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClaimBundle:
    claims: list[dict[str, Any]] = field(default_factory=list)
    rate_change: dict[str, Any] | None = None
    radio_change: dict[str, Any] | None = None
    active_fraction: float | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)


def claims_from_results(results: list[dict[str, Any]]) -> ClaimBundle:
    bundle = ClaimBundle()
    seen: set[str] = set()

    def add(claim: dict[str, Any]) -> None:
        if claim["claim_id"] not in seen:
            seen.add(claim["claim_id"])
            bundle.claims.append(claim)

    for result in results:
        tool = result["tool_name"]
        body = result["result"]
        if tool == "compare_windows":
            for metric, comparison in body["comparisons"].items():
                if any(
                    summary.get("measurement_applicability", {}).get("supported") is False
                    for summary in (comparison["reference"], comparison["event"])
                ):
                    continue
                if comparison["reference_value"] is None or comparison["event_value"] is None:
                    continue
                basis = result["aggregation_basis"]
                claim = {
                    "claim_type": "metric_change",
                    "claim_id": f"change_{metric}_{body['statistic']}_{basis}",
                    "metric": metric,
                    "statistic": body["statistic"],
                    "basis": result["aggregation_basis"],
                    "reference_window": body["reference_window"],
                    "event_window": body["event_window"],
                    "reference_value": comparison["reference_value"],
                    "event_value": comparison["event_value"],
                    "delta": comparison["delta_event_minus_reference"],
                    "relative_decrease": comparison["relative_decrease"],
                    "unit": comparison["unit"],
                    "evidence_ids": [result["evidence_id"]],
                }
                add(claim)
                if metric == "downlink_mbps" and bundle.rate_change is None:
                    bundle.rate_change = claim
                elif metric != "downlink_mbps" and bundle.radio_change is None:
                    bundle.radio_change = claim
        elif tool == "get_activity_state":
            fraction = body["fractions"]["active"]
            if fraction is not None:
                window = body["window"]
                add(
                    {
                        "claim_type": "activity",
                        "claim_id": f"activity_{window['start_ms']}_{window['end_ms']}",
                        "window": window,
                        "basis": body["basis"],
                        "state": "active",
                        "fraction": fraction,
                        "denominator": body["denominator"],
                        "evidence_ids": [result["evidence_id"]],
                    }
                )
                if bundle.active_fraction is None:
                    bundle.active_fraction = fraction
        elif tool == "get_cell_transitions":
            for transition in body["transitions"][:2]:
                bundle.transitions.append(transition)
                add(
                    {
                        "claim_type": "transition",
                        "claim_id": f"transition_{transition['later_time_ms']}",
                        "transition_type": transition["transition_type"],
                        "earlier_cell_id": transition["earlier_cell_id"],
                        "later_cell_id": transition["later_cell_id"],
                        "earlier_time_ms": transition["earlier_time_ms"],
                        "later_time_ms": transition["later_time_ms"],
                        "gap_ms": transition["gap_ms"],
                        "identity_basis": transition["identity_basis"],
                        "evidence_ids": [result["evidence_id"]],
                    }
                )
        elif tool == "summarize_window":
            window = body["window"]
            for metric, summary in body["summaries"].items():
                add(
                    {
                        "claim_type": "availability",
                        "claim_id": f"availability_{metric}_{window['start_ms']}",
                        "metric": metric,
                        "window": window,
                        "basis": result["aggregation_basis"],
                        "valid_count": summary["valid_count"],
                        "missing_count": summary["missing_count"],
                        "evidence_ids": [result["evidence_id"]],
                    }
                )
    return bundle
