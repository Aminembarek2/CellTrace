"""Synthetic control reports with construction expectations.

Each control states one wording on purpose, so a judge that follows the rubric should return a
known verdict. The expectations are rubric assertions written for this project, not labels of
real incidents. They stay in the plan and never reach a judge.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CONTROL_CRITERIA = ("unsupported_causal_claim", "useful_next_measurement", "missed_alternative")
VARIANTS = ("canonical", "repeat", "order")

_DEMAND = (
    "Measure offered application demand alongside delivered rate. Low offered demand "
    "supports an application limit; high demand with low delivery motivates a bottleneck check."
)
_RADIO = (
    "Acquire time-aligned BLER and retransmission counters. Elevated errors support a radio "
    "impairment; unchanged errors leave scheduler or transport explanations open."
)
_TRANSITION = (
    "Acquire RRC event logs with interruption times. A logged handover overlapping the "
    "drop supports the transition hypothesis; no event leaves other explanations open."
)
_TRANSPORT = (
    "Acquire TCP loss and RTT alongside server send rate. Rising loss or RTT with steady "
    "server output supports transport limitation; falling server output supports demand."
)
# Summary wording, proposed measurement, expected causal overreach, expected usefulness.
_SPECIFICATIONS: tuple[tuple[str, str, bool, bool], ...] = (
    ("Stable RSRP proves congestion caused the drop.", "Collect more data.", True, False),
    (
        "Stable reported RSRP accompanies lower delivered rate. Congestion is only a "
        "hypothesis; other radio dimensions and offered demand are unmeasured.",
        _DEMAND,
        False,
        True,
    ),
    (
        "The cell ID change proves a successful handover caused the drop.",
        "Measure battery temperature for more detail.",
        True,
        False,
    ),
    (
        "A cell identity change is a transition proxy. Handover timing and physical cause "
        "are unverified.",
        _TRANSITION,
        False,
        True,
    ),
    (
        "The delivered application bitrate proves available network capacity fell.",
        "Repeat the already complete rate summary without changing its window.",
        True,
        False,
    ),
    (
        "Lower delivered application rate does not measure capacity. Offered demand is "
        "unknown, and physical cause remains unresolved.",
        _DEMAND,
        False,
        True,
    ),
    (
        "Stable RSRP definitively excludes every radio impairment.",
        "Investigate further.",
        True,
        False,
    ),
    (
        "RSRP is stable, but missing BLER and retransmission counters prevent exclusion "
        "of radio impairment. Transport is also unmeasured.",
        _RADIO,
        False,
        True,
    ),
    (
        "Delivered rate fell while the application was active throughout. "
        "Physical cause remains unresolved.",
        "Collect more data.",
        False,
        False,
    ),
    (
        "The observed application is idle during the low rate interval. We cannot infer "
        "a capacity fault; independent offered demand would distinguish inactivity.",
        _TRANSPORT,
        False,
        True,
    ),
)


def legacy_control_items() -> list[dict[str, Any]]:
    items = []
    for index, (summary, measurement, causal, useful) in enumerate(_SPECIFICATIONS):
        idle = index >= 8
        items.append(
            {
                "item_id": f"control_{index:02d}",
                "phase": "control",
                "model_submission_available": True,
                "candidate": {"event_id": "synthetic_event", "start_ms": 30000, "end_ms": 35000},
                "findings": {
                    "decision_summary": summary,
                    "cause_status": "unresolved",
                    "hypotheses": [],
                    "next_measurements": [measurement, "Collect more data."],
                },
                "cited_measurement_evidence": {
                    "evidence_rate": {
                        "tool_name": "compare_windows",
                        "units": {"downlink_mbps": "Mbps"},
                        "result": {
                            "reference_median_mbps": 20.0,
                            "event_median_mbps": 3.0,
                            "coverage": 1.0,
                        },
                    },
                    "evidence_radio": {
                        "tool_name": "compare_windows",
                        "units": {"rsrp_dbm": "dBm"},
                        "result": {
                            "reference_median_rsrp_dbm": -85.0,
                            "event_median_rsrp_dbm": -85.0,
                            "coverage": 1.0,
                            "bler_available": False,
                            "retransmissions_available": False,
                        },
                    },
                    "evidence_cell": {
                        "tool_name": "get_cell_transitions",
                        "result": {
                            "qualified_identity_changes": 1,
                            "rrc_event_logs_available": False,
                        },
                    },
                    "evidence_activity": {
                        "tool_name": "get_activity_state",
                        "result": {
                            "active_fraction": 0.0 if idle else 1.0,
                            "idle_fraction": 1.0 if idle else 0.0,
                            "unknown_fraction": 0.0,
                            "coverage": 1.0,
                        },
                    },
                },
                "expected": {
                    "unsupported_causal_claim": str(causal).lower(),
                    "useful_next_measurement": str(useful).lower(),
                },
            }
        )
    return items


def presentation_variant(item: dict[str, Any]) -> dict[str, Any]:
    """Same content in reversed list order, to detect verdicts that depend on presentation."""
    selected = deepcopy(item)
    for field in ("next_measurements", "hypotheses", "factual_claims", "missing_evidence"):
        if isinstance(selected["findings"].get(field), list):
            selected["findings"][field].reverse()
    return selected


def control_items() -> list[dict[str, Any]]:
    """Twenty additional assertions. Ten calibrate alternatives; all calibrate scope/usefulness."""
    themes = (
        (
            "The RSRP readings establish that scheduler overload caused this slowdown.",
            "Scheduler overload is a candidate explanation, but RSRP cannot establish it.",
        ),
        (
            "This serving-identity transition establishes a completed handover as the cause.",
            "Serving identity is a transition proxy; a successful handover and cause are unknown.",
        ),
        (
            "The application-rate measurement establishes the network's available capacity.",
            "Application-rate measurements describe delivered traffic, not available capacity.",
        ),
        (
            "Unchanged reported signal power rules out all possible radio impairments.",
            "Unchanged signal power does not rule out unmeasured radio errors.",
        ),
        (
            "The lower delivery rate establishes a physical transport fault.",
            "The lower delivery rate alone cannot establish a physical transport fault.",
        ),
    )
    specifications: list[tuple[str, dict[str, Any], dict[str, str]]] = []
    for pair in themes:
        for asserted, summary in enumerate(pair):
            specifications.append(
                (
                    summary + " Offered demand, radio errors and transport counters "
                    "are missing; the physical cause remains unresolved.",
                    {"reference_median_mbps": 24.0, "event_median_mbps": 4.0},
                    {"unsupported_causal_claim": "true" if asserted == 0 else "false"},
                )
            )
    alternatives = (
        (
            "Lower delivered traffic was observed; activity was not considered.",
            "Lower traffic coincides with application inactivity. A capacity fault is unverified.",
            {"active_fraction": 0.0, "idle_fraction": 1.0, "coverage": 1.0},
        ),
        (
            "An active-rate drop was reported without checking representative rate windows.",
            "Delivery medians are unchanged; a meaningful rate drop is not supported.",
            {"reference_median_mbps": 18.0, "event_median_mbps": 18.0, "coverage": 1.0},
        ),
        (
            "A complete event window was assumed; missing samples were not considered.",
            "Only 20 percent of event bins are observed. A full comparison is unsupported.",
            {"event_valid_fraction": 0.2, "reference_valid_fraction": 1.0},
        ),
        (
            "The radio comparison was applied without checking its timestamps.",
            "The radio query covers another interval and cannot support this event.",
            {
                "query_start_ms": 90000,
                "query_end_ms": 95000,
                "event_start_ms": 30000,
                "event_end_ms": 35000,
            },
        ),
    )
    for ignored, addressed, evidence in alternatives:
        for text, verdict in ((ignored, "true"), (addressed, "false")):
            specifications.append(
                (
                    text + " Physical cause is unresolved.",
                    evidence,
                    {"unsupported_causal_claim": "false", "missed_alternative": verdict},
                )
            )
    for text in (
        "No report measurements are retrievable for this interval.",
        "The cited measurement could not be retrieved; its outcome is unknown.",
    ):
        specifications.append(
            (
                text + " Physical cause is unresolved.",
                {},
                {"unsupported_causal_claim": "false", "missed_alternative": "unknown"},
            )
        )
    items = []
    for index, (summary, result, expected) in enumerate(specifications):
        useful = (index // 2) % 2 == 0 if index < 10 else index in (10, 11, 14, 15)
        measurement = (
            (
                "Measure offered application demand and delivery in the same future interval; "
                "low demand supports inactivity; sustained demand with low delivery motivates "
                "a network or transport investigation."
            )
            if useful
            else "Gather some additional information."
        )
        if useful and index in (14, 15):
            measurement = (
                "In a future comparable episode acquire complete synchronized one-second "
                "delivery samples in both windows. Dense observations with a rate change support "
                "a real traffic change; a disappearing difference supports a sampling artifact."
            )
        available = index < 18
        items.append(
            {
                "item_id": f"contract_control_{index:02d}",
                "phase": "control",
                "model_submission_available": True,
                "candidate": {"event_id": "synthetic_event", "start_ms": 30000, "end_ms": 35000},
                "findings": {
                    "decision_summary": summary,
                    "cause_status": "unresolved",
                    "hypotheses": [],
                    "next_measurements": [measurement, "Explore the situation further."],
                },
                "cited_measurement_evidence": {
                    "ev_measurement": {
                        "tool_name": "compare_windows",
                        "result": result,
                        "units": {"downlink_mbps": "Mbps"},
                        "quality_flags": [],
                    }
                    if available
                    else None
                },
                "expected": expected | {"useful_next_measurement": str(useful).lower()},
            }
        )
    return items
