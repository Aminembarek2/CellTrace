"""Qualification, panel verdicts, agreement, stability and spend for a review plan.

Only qualified judges form the panel. A panel verdict is a Boolean only when every panel judge
gave that same Boolean, otherwise it stays unresolved. Failed, invalid and undispatched items
remain in every denominator. The public summary holds aggregates only, never report text.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from celltrace.evaluation.review.gates import dispatch_block, qualification, verdict
from celltrace.evaluation.review.ledger import experiment_usage, load_results
from celltrace.evaluation.review.persistence import write_json_atomic
from celltrace.evaluation.review.plan import load_plan
from celltrace.evaluation.review.rubric import CRITERIA

BOOLEAN = ("true", "false")


def panel_verdict(verdicts: list[str]) -> str:
    if verdicts and all(value in BOOLEAN for value in verdicts) and len(set(verdicts)) == 1:
        return verdicts[0]
    return "unresolved"


def review_gate(
    plan: dict[str, Any], results: dict[str, dict[str, Any] | None], qualified: list[bool]
) -> dict[str, Any]:
    """Pass or fail for the workflow, from thresholds frozen in the plan.

    Every judge must pass its controls, every scheduled item must have a recorded outcome, and
    each judge must return valid assessments for at least the frozen valid fraction of reports.
    """
    minimum_valid = plan["settings"]["qualification"]["minimum_valid_fraction"]
    missing = sum(result is None for result in results.values())
    valid_fraction = {}
    for judge in plan["judges"]:
        reports = [
            results[entry["item_id"]]
            for entry in plan["schedule"]
            if entry["model_id"] == judge["model_id"] and entry["stratum"] == "report"
        ]
        valid = sum(result is not None and result["assessment"] is not None for result in reports)
        if reports:
            valid_fraction[judge["model_id"]] = valid / len(reports)
    if not all(qualified):
        status = "controls_failed"
    elif missing:
        status = "incomplete"
    elif any(fraction < minimum_valid for fraction in valid_fraction.values()):
        status = "too_many_invalid"
    else:
        status = "passed"
    return {
        "status": status,
        "items_without_outcome": missing,
        "report_valid_fraction": valid_fraction,
        "minimum_valid_fraction": minimum_valid,
    }


def _report_entries(plan: dict[str, Any], variant: str) -> dict[tuple[str, str], str]:
    return {
        (entry["source_id"], entry["model_id"]): entry["item_id"]
        for entry in plan["schedule"]
        if entry["stratum"] == "report" and entry["variant"] == variant
    }


def summarize_review(output: Path, public_output: Path) -> dict[str, Any]:
    if public_output.exists():
        raise FileExistsError(f"{public_output} exists. Preserve earlier summaries")
    plan = load_plan(output)
    results = load_results(output, plan)
    judges = [judge["model_id"] for judge in plan["judges"]]
    qualifications = [qualification(plan, results, judge) for judge in judges]
    panel = [record["model_id"] for record in qualifications if record["qualified"]]
    canonical = _report_entries(plan, "canonical")
    repeats = _report_entries(plan, "repeat")
    mapping = {
        entry["item_id"]: entry for entry in json.loads(Path(plan["mapping_path"]).read_text())
    }
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    pair_counts: dict[str, dict[str, Counter[str]]] = {}
    rows = []
    for source_id in sorted({source for source, _ in canonical}):
        arm, phase = mapping[source_id]["arm"], mapping[source_id]["phase"]
        group = groups.setdefault(
            (arm, phase),
            {
                "arm": arm,
                "phase": phase,
                "items": 0,
                "panel": {criterion: Counter() for criterion in CRITERIA},
            },
        )
        group["items"] += 1
        for criterion in CRITERIA:
            by_judge = {
                judge: verdict(results[canonical[(source_id, judge)]], criterion)
                for judge in judges
            }
            decided = panel_verdict([by_judge[judge] for judge in panel])
            group["panel"][criterion][decided] += 1
            for left, right in combinations(judges, 2):
                pair = pair_counts.setdefault(f"{left} | {right}", {})
                pair.setdefault(criterion, Counter())[f"{by_judge[left]}/{by_judge[right]}"] += 1
            rows.append(
                {"source_id": source_id, "arm": arm, "phase": phase, "criterion": criterion}
                | {"verdicts": by_judge, "panel": decided}
            )
    stability = []
    for judge in judges:
        for criterion in CRITERIA:
            pairs = [
                (
                    verdict(results[canonical[(source_id, judge)]], criterion),
                    verdict(results[repeated], criterion),
                )
                for (source_id, model_id), repeated in repeats.items()
                if model_id == judge
            ]
            stability.append(
                {
                    "model_id": judge,
                    "criterion": criterion,
                    "consistent": sum(
                        first in BOOLEAN and first == second for first, second in pairs
                    ),
                    "denominator": len(pairs),
                }
            )
    statuses: Counter[str] = Counter()
    for entry in plan["schedule"]:
        result = results[entry["item_id"]]
        if result is not None:
            statuses[f"{entry['model_id']} {result['status']}"] += 1
        else:
            reason = dispatch_block(plan, results, entry) or "not_reached"
            statuses[f"{entry['model_id']} not_dispatched:{reason}"] += 1
    summary = {
        "assessment_method": plan["assessment_method"],
        "scope": "judge_qualification_only"
        if plan["settings"]["controls_only"]
        else "saved_report_review",
        "protocol_version": plan["protocol_version"],
        "plan_hash": plan["plan_hash"],
        "judges": judges,
        "scheduled_items": len(plan["schedule"]),
        "status_counts": dict(sorted(statuses.items())),
        "qualification": qualifications,
        "panel_judges": panel,
        "panel_status": {0: "no_qualified_judge", 1: "single_qualified_judge"}.get(
            len(panel), "multiple_qualified_judges"
        ),
        "groups": [
            {**group, "panel": {key: dict(value) for key, value in group["panel"].items()}}
            for group in groups.values()
        ],
        "pairwise_verdicts": {
            pair: {criterion: dict(counts) for criterion, counts in by_criterion.items()}
            for pair, by_criterion in pair_counts.items()
        },
        "repeat_stability": stability,
        "gate": review_gate(plan, results, [record["qualified"] for record in qualifications]),
        "known_or_reserved_usd": experiment_usage(output, plan),
        "maximum_estimated_usd": plan["settings"]["maximum_estimated_usd"],
        "limitations": plan["limitations"],
    }
    public_output.mkdir(parents=True)
    write_json_atomic(public_output / "summary.json", summary)
    write_json_atomic(public_output / "synthetic_controls.json", plan["controls"])
    write_json_atomic(output / "panel_judgments.json", rows)
    return summary
