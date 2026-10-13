"""Quality gates, as pure functions of the frozen plan and the recorded results.

The runner uses them to decide what to pay for, and the summary uses the same functions to
explain every item that was not dispatched, so the two can never disagree.
"""

from __future__ import annotations

from typing import Any

from celltrace.evaluation.review.controls import CONTROL_CRITERIA

INVALID_STATUSES = frozenset(
    {"invalid_assessment", "unverified_provider_response", "unresolved_prior_attempt"}
)
Results = dict[str, dict[str, Any] | None]


def verdict(result: dict[str, Any] | None, criterion: str) -> str:
    if result is None or result.get("assessment") is None:
        return "unavailable"
    return str(result["assessment"][criterion]["verdict"])


def _entries(plan: dict[str, Any], model_id: str, stratum: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in plan["schedule"]
        if entry["model_id"] == model_id and entry["stratum"] == stratum
    ]


def invalid_streak(plan: dict[str, Any], results: Results, model_id: str) -> int:
    """Consecutive invalid replies at the end of this judge's recorded results."""
    streak = 0
    for entry in plan["schedule"]:
        result = results.get(entry["item_id"])
        if entry["model_id"] != model_id or result is None:
            continue
        streak = streak + 1 if result["status"] in INVALID_STATUSES else 0
    return streak


def qualification(plan: dict[str, Any], results: Results, model_id: str) -> dict[str, Any]:
    """Control agreement, validity and stability for one judge, against the frozen rule."""
    entries = _entries(plan, model_id, "control")
    by_key = {(entry["source_id"], entry["variant"]): entry["item_id"] for entry in entries}
    controls = plan["controls"]
    total = len(controls)
    recorded = sum(results.get(entry["item_id"]) is not None for entry in entries)
    valid = 0
    counts = {
        criterion: {"canonical_correct": 0, "repeat_consistent": 0, "order_consistent": 0}
        for criterion in CONTROL_CRITERIA
    }
    denominators = {
        criterion: sum(criterion in control["expected"] for control in controls)
        for criterion in CONTROL_CRITERIA
    }
    confusion: dict[str, dict[str, dict[str, int]]] = {
        criterion: {expected: {} for expected in ("true", "false", "unknown", "not_applicable")}
        for criterion in CONTROL_CRITERIA
    }
    for control in controls:
        canonical = results.get(by_key[(control["item_id"], "canonical")])
        valid += int(canonical is not None and canonical.get("assessment") is not None)
        for criterion in CONTROL_CRITERIA:
            if criterion not in control["expected"]:
                continue
            base = verdict(canonical, criterion)
            expected = control["expected"][criterion]
            observed = confusion[criterion][expected]
            observed[base] = observed.get(base, 0) + 1
            counts[criterion]["canonical_correct"] += int(base == control["expected"][criterion])
            for variant in ("repeat", "order"):
                other = verdict(results.get(by_key[(control["item_id"], variant)]), criterion)
                counts[criterion][variant + "_consistent"] += int(
                    base in ("true", "false", "unknown", "not_applicable") and base == other
                )
    rule = plan["settings"]["qualification"]
    all_valid = sum(
        result is not None and result.get("assessment") is not None
        for entry in entries
        for result in [results.get(entry["item_id"])]
    )
    qualified = (
        valid / total >= rule["minimum_valid_fraction"]
        and all_valid / len(entries) >= rule["minimum_valid_fraction"]
    ) and all(
        counts[criterion][metric] / denominators[criterion] >= rule[threshold]
        for criterion in CONTROL_CRITERIA
        for metric, threshold in (
            ("canonical_correct", "minimum_canonical_agreement"),
            ("repeat_consistent", "minimum_repeat_consistency"),
            ("order_consistent", "minimum_order_consistency"),
        )
    )
    return {
        "model_id": model_id,
        "controls": total,
        "control_items_recorded": recorded,
        "control_items_scheduled": len(entries),
        "canonical_valid": valid,
        "all_presentation_valid": all_valid,
        "criteria": counts,
        "criterion_denominators": denominators,
        "canonical_confusion": confusion,
        "reference_scope": "twenty construction assertions, not incident or human labels",
        "qualified": qualified,
    }


def dispatch_block(plan: dict[str, Any], results: Results, entry: dict[str, Any]) -> str | None:
    """Why this item must not be paid for now, or None when it may be dispatched."""
    model_id = entry["model_id"]
    if invalid_streak(plan, results, model_id) >= plan["settings"]["invalid_streak_limit"]:
        return "judge_invalid_streak"
    if entry["stratum"] == "control":
        return None
    controls = _entries(plan, model_id, "control")
    if any(results.get(control["item_id"]) is None for control in controls):
        return "judge_controls_incomplete"
    if not qualification(plan, results, model_id)["qualified"]:
        return "judge_not_qualified"
    return None
