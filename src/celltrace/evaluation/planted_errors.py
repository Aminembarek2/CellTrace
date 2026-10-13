"""Frozen development controls and strict localization scoring, outside the runtime."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any

from celltrace.contracts.findings import InvestigationFindings, ReviewResult

METHOD_VERSION = "planted_interpretations_3"
# Version 4: bases on distinct windows, a chosen wording and draws, set in the plan.
METHOD_VERSION_4 = "planted_interpretations_4"
SUPPORTED_METHODS = frozenset({METHOD_VERSION, METHOD_VERSION_4})
# Independent reviewer conversations per scheduled finding, to show consistency.
DRAWS = 2
BASE_FAMILIES = (
    "clean_control",
    "cell_change_without_rate_drop",
    "stable_radio_rate_drop",
)


@dataclass(frozen=True)
class Corruption:
    """One planted statement. "original" wording was used in study 2, before the Reviewer
    prompt gained its scope check. "new" wording was never seen while revising it."""

    name: str
    phrasing: str
    path: str
    statement: str

    @property
    def family(self) -> str:
        return f"{self.name}_{self.phrasing}"


CORRUPTIONS = (
    Corruption(
        "physical_cause",
        "original",
        "/decision_summary",
        "The measurements establish that a successful handover caused the throughput change.",
    ),
    Corruption(
        "capacity_conflation",
        "original",
        "/decision_summary",
        "The observed application download rate is the available network capacity. "
        "The recorded rate therefore measures the cell's remaining capacity.",
    ),
    Corruption(
        "window_scope",
        "original",
        "/decision_summary",
        "These reference and event measurements cover the entire source session, "
        "so this account establishes conditions throughout that session.",
    ),
    Corruption(
        "physical_cause",
        "new",
        "/decision_summary",
        "Congestion in the serving cell is what reduced the download rate in this window.",
    ),
    Corruption(
        "capacity_conflation",
        "new",
        "/decision_summary",
        "Given the rate delivered here, the cell clearly had no spare capacity left for this user.",
    ),
    Corruption(
        "window_scope",
        "new",
        "/decision_summary",
        "This slowdown is typical of the whole recording and affected the user for most of "
        "the session.",
    ),
)
TARGET_PATHS = tuple(sorted({corruption.path for corruption in CORRUPTIONS}))


def qualified_control(
    findings: dict[str, Any], unsupported_patterns: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Preserve measured claims, replace the template's unsupported hypothesis exclusions.

    Observation labels that code verification rejected are removed, so the control
    is a correct report rather than a template that already contains an error.
    """
    result = copy.deepcopy(findings)
    result["observation_patterns"] = [
        pattern for pattern in result["observation_patterns"] if pattern not in unsupported_patterns
    ]
    radio_ids = [
        claim["claim_id"]
        for claim in result["factual_claims"]
        if claim.get("metric") == "reported_rsrp_dbm"
    ]
    if not radio_ids:
        raise ValueError("planted radio controls require applicable measured RSRP")
    result["hypotheses"] = [
        {
            "hypothesis_id": "radio_degradation",
            "status": "not_testable",
            "supporting_claim_ids": [],
            "contradicting_claim_ids": [],
            "assumptions": [
                "Reported RSRP alone cannot exclude interference or other radio impairment."
            ],
            "discriminating_measurement": "Block error rate and link adaptation logs.",
        }
    ]
    result["decision_summary"] = (
        "The cited facts describe the reference and event windows only. "
        "Measured application rate is not available network capacity. "
        "A reported identity change does not establish a successful handover or its cause. "
        "The physical explanation remains unresolved."
    )
    result["missing_evidence"] = [
        "No radio error or scheduling counters.",
        "No transport, server or application demand logs.",
        "No handover procedure or completion evidence.",
    ]
    result["next_measurements"] = [
        "Radio error and scheduling counters to distinguish signal power from link impairment.",
        "Transport and application logs to distinguish offered demand from delivery limitations.",
    ]
    result["cause_status"] = "unresolved"
    return InvestigationFindings.model_validate(result).model_dump(mode="json")


def corruptions_for(phrasings: tuple[str, ...] | None = None) -> tuple[Corruption, ...]:
    return tuple(
        corruption
        for corruption in CORRUPTIONS
        if phrasings is None or corruption.phrasing in phrasings
    )


def variants(
    control: dict[str, Any], phrasings: tuple[str, ...] | None = None
) -> list[tuple[str, str | None, dict[str, Any]]]:
    results: list[tuple[str, str | None, dict[str, Any]]] = [
        ("correct_control", None, copy.deepcopy(control))
    ]
    for corruption in corruptions_for(phrasings):
        changed = copy.deepcopy(control)
        if corruption.path != "/decision_summary":
            raise ValueError("reviewer corruptions change the decision summary only")
        changed["decision_summary"] = corruption.statement
        results.append(
            (
                corruption.family,
                corruption.path,
                InvestigationFindings.model_validate(changed).model_dump(mode="json"),
            )
        )
    return results


def score_item(
    label: dict[str, Any], findings: dict[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    """A generic objection is not an exact challenge of the planted interpretation."""
    review = None
    valid = False
    if outcome.get("status") == "submitted" and outcome.get("review") is not None:
        try:
            review = ReviewResult.model_validate(outcome["review"])
            review.validate_finding_paths(findings)
            known = {claim["claim_id"] for claim in findings["factual_claims"]}
            if not set(review.disputed_claim_ids) <= known:
                raise ValueError("foreign disputed claim")
            valid = True
        except ValueError:
            review = None
    objection = valid and review is not None and review.decision.value != "accept"
    independent = bool(outcome.get("successful_independent_queries", 0)) and bool(
        outcome.get("cited_independent_query", False)
    )
    located = bool(
        objection
        and review is not None
        and label["target_path"] is not None
        and label["target_path"] in review.disputed_finding_paths
    )
    disputed_targets = (
        sorted(set(review.disputed_finding_paths) & set(TARGET_PATHS))
        if objection and review is not None
        else []
    )
    return {
        "family": label["family"],
        "valid_submission": valid,
        "independent_query": independent,
        "decision_objection": bool(objection),
        # Primary endpoint: the exact planted section plus a cited independent query.
        "localized_challenge": located and independent,
        # Secondary endpoint, declared before outcomes: the exact section without the query.
        "localized_without_query": located,
        "control_false_alarm": bool(objection and label["family"] == "correct_control"),
        "disputed_target_paths": disputed_targets,
        "status": outcome["status"],
    }


def draw_consistency(
    selected: list[tuple[dict[str, Any], dict[str, Any]]], draws: int = DRAWS
) -> dict[str, int]:
    """Per scheduled finding, in how many of its draws the primary endpoint was met."""
    by_finding: dict[str, list[bool]] = {}
    for item, row in selected:
        key = item.get("finding_id", item["item_id"])
        by_finding.setdefault(key, []).append(bool(row["localized_challenge"]))
    counts = Counter(sum(results) for results in by_finding.values())
    return {f"{hits}_of_{draws}": counts.get(hits, 0) for hits in range(draws + 1)}


def summarize(
    items: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    mode: str,
    method: str = METHOD_VERSION,
    draws: int = DRAWS,
    phrasings: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    rows = [
        score_item(
            item["label"],
            item["findings"],
            outcomes.get(item["item_id"], {"status": "not_run"}),
        )
        for item in items
    ]
    families: dict[str, Any] = {}
    for family in ("correct_control", *(entry.family for entry in corruptions_for(phrasings))):
        selected = [
            (item, row) for item, row in zip(items, rows, strict=True) if row["family"] == family
        ]
        family_rows = [row for _, row in selected]
        families[family] = {
            "scheduled": len(family_rows),
            "valid_submissions": sum(row["valid_submission"] for row in family_rows),
            "independent_queries": sum(row["independent_query"] for row in family_rows),
            "decision_objections": sum(row["decision_objection"] for row in family_rows),
            "localized_challenges": sum(row["localized_challenge"] for row in family_rows),
            "localized_without_query": sum(row["localized_without_query"] for row in family_rows),
            "draw_consistency": draw_consistency(selected, draws),
            "statuses": dict(Counter(row["status"] for row in family_rows)),
        }
        if family == "correct_control":
            # A reviewer that always disputes a planted path would score detections, so the
            # matched controls report how often each planted path is disputed without error.
            families[family]["disputed_target_paths"] = {
                path: sum(path in row["disputed_target_paths"] for row in family_rows)
                for path in TARGET_PATHS
            }
    return {
        "method": method,
        "execution_mode": mode,
        "scheduled": len(items),
        "families": families,
        "semantic_accuracy_claim": "excluded",
        "inference": "descriptive development controls; no independent-item intervals",
        "endpoint": "exact location of a non-accept challenge with a cited independent query; "
        "free-text reasoning correctness is not certified",
        "secondary_endpoint": "exact location of a non-accept challenge without the query rule",
        "paired_controls": "each base has one control and single section corruptions: "
        "three errors, in the wordings listed",
        "wordings": list(phrasings) if phrasings else ["original", "new"],
        "generalization": "new wording was never seen while revising the Reviewer prompt",
        "draws_per_finding": draws,
    }
