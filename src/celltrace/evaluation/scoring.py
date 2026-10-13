"""Score an experiment against the private manifest, after every run finished.

Deterministic metrics only: claim support, fabricated citations, numerical
validity, required fact coverage, observation pattern recovery, completion,
usage and review activity. Causal language and usefulness need the manual
annotation template this module writes. A run without claims gets NA for
factual support, never 100 percent.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from celltrace.canonical import file_sha256
from celltrace.clock import SystemClock
from celltrace.contracts.common import TimeWindow
from celltrace.contracts.evidence import RunContext
from celltrace.contracts.findings import InvestigationFindings
from celltrace.contracts.report import InvestigationReport
from celltrace.data.snapshot import open_snapshot
from celltrace.storage.run_store import ActionKind, RunStore
from celltrace.tools.repository import MeasurementRepository
from celltrace.verification.verifier import Verifier

FABRICATION_CODES = {"UNKNOWN_EVIDENCE", "EVIDENCE_WRONG_RUN", "EVIDENCE_WRONG_SNAPSHOT"}
NUMERICAL_CLAIM_TYPES = {"metric_statistic", "metric_change", "activity"}
BOOTSTRAP_RESAMPLES = 2000
MINIMUM_CLUSTERS_FOR_INTERVAL = 5


def _overlaps(first: dict[str, int], second: TimeWindow) -> bool:
    return first["start_ms"] < second.end_ms and second.start_ms < first["end_ms"]


def fact_kind_satisfied(
    kind: str, claims: list[dict[str, Any]], case: dict[str, Any], event: TimeWindow
) -> bool:
    expected = case["expected_facts"]
    for claim in claims:
        claim_type = claim["claim_type"]
        if (
            kind == "downlink_change"
            and claim_type == "metric_change"
            and claim["metric"] == "downlink_mbps"
        ):
            if not _overlaps(claim["event_window"], event):
                continue
            expected_drop = expected["downlink_relative_decrease"]
            if expected_drop is None or expected_drop < 0.5:
                return True
            reference, event_value = claim["reference_value"], claim["event_value"]
            if reference > 0 and (reference - event_value) / reference >= 0.5:
                return True
        if (
            kind == "radio_change"
            and claim_type == "metric_change"
            and claim["metric"] == "reported_rsrp_dbm"
            and _overlaps(claim["event_window"], event)
        ):
            return True
        if kind == "activity" and claim_type == "activity" and _overlaps(claim["window"], event):
            return True
        if kind == "transition" and claim_type == "transition":
            return True
    return False


def _post_hoc_checks(
    report: InvestigationReport, store: RunStore, case_dir: Path
) -> dict[str, Any]:
    """Scorer side verification for arm A. Nothing is fed back to the run."""
    assert report.final_findings is not None
    context = RunContext.model_validate(store.get_run(report.run_id)["context"])
    connection, _ = open_snapshot(case_dir / "snapshot")
    try:
        verifier = Verifier(context, store, MeasurementRepository(connection))
        result = verifier.verify(InvestigationFindings.model_validate(report.final_findings))
    finally:
        connection.close()
    return result.model_dump(mode="json")


def score_run(
    report: InvestigationReport, case: dict[str, Any], store: RunStore, case_dir: Path
) -> dict[str, Any]:
    final = report.final_findings
    claims = (
        [] if final is None else [claim.model_dump(mode="json") for claim in final.factual_claims]
    )
    final_findings_id = store.findings(report.run_id)[-1][0] if final is not None else None
    in_run = [
        entry
        for entry in store.verifications(report.run_id)
        if entry["findings_id"] == final_findings_id
    ]
    checked_post_hoc = False
    if final is None:
        checks: list[dict[str, Any]] = []
    elif report.arm.value == "investigator_raw" or not in_run:
        # Scorer side checks with no feedback to the run: arm A, or runs that
        # stopped before their final findings were verified.
        checks = _post_hoc_checks(report, store, case_dir)["claim_checks"]
        checked_post_hoc = True
    else:
        checks = in_run[-1]["result"]["claim_checks"]
    passed_ids = {check["claim_id"] for check in checks if check["passed"]}
    failed_codes = [check["code"] for check in checks if not check["passed"]]
    supported_claims = [claim for claim in claims if claim["claim_id"] in passed_ids]
    numerical = [claim for claim in claims if claim["claim_type"] in NUMERICAL_CLAIM_TYPES]
    numerical_passed = [claim for claim in numerical if claim["claim_id"] in passed_ids]
    if report.report_basis.value == "deterministic_fallback":
        supported_claims = [claim.model_dump(mode="json") for claim in report.verified_claims]
    candidate_event = TimeWindow.model_validate(
        json.loads((case_dir / "candidate.json").read_text())["event_window"]
    )
    required = case["required_fact_kinds"]
    satisfied = [
        kind
        for kind in required
        if fact_kind_satisfied(kind, supported_claims, case, candidate_event)
    ]
    patterns = set(report.observation_patterns)
    rule = case["expected_patterns"]
    recovered: bool | None
    if not rule["required"] and not rule["forbidden"]:
        recovered = None
    else:
        recovered = set(rule["required"]) <= patterns and not (set(rule["forbidden"]) & patterns)
    reviews = store.reviews(report.run_id)
    verifications = store.verifications(report.run_id)
    tool_actions = store.actions(report.run_id, ActionKind.tool_request)
    review_changes = review_change_counts(report.run_id, store, case_dir)
    return {
        **review_changes,
        "run_id": report.run_id,
        "model_id": report.model_id,
        "case_id": case["case_id"],
        "family": case["family"],
        "source_session_id": case["source_session_id"],
        "arm": report.arm.value,
        "execution_mode": report.execution_mode.value,
        "workflow_outcome": report.workflow_outcome.value,
        "fact_check_status": report.fact_check_status.value,
        "review_status": report.review_status.value,
        "cause_status": report.cause_status.value,
        "cause_status_appropriate": report.cause_status.value in case["expected_cause_status"],
        "report_basis": report.report_basis.value,
        "terminal_reason": report.terminal_reason,
        "claims_checked_post_hoc": checked_post_hoc,
        "checkable_claims": len(claims),
        "supported_claims": len([claim for claim in claims if claim["claim_id"] in passed_ids]),
        "factual_support": None
        if not claims
        else len(passed_ids & {c["claim_id"] for c in claims}) / len(claims),
        "fabricated_evidence_claims": sum(1 for code in failed_codes if code in FABRICATION_CODES),
        "numerical_claims": len(numerical),
        "numerical_validity": None if not numerical else len(numerical_passed) / len(numerical),
        "required_facts": len(required),
        "required_facts_recovered": len(satisfied),
        "fact_completeness": None if not required else len(satisfied) / len(required),
        "pattern_recovered": recovered,
        "reported_patterns": sorted(patterns),
        "first_verification_failed": bool(verifications) and not verifications[0]["passed"],
        "final_verification_passed": bool(verifications) and verifications[-1]["passed"],
        "review_rounds": len(reviews),
        "review_requested_evidence": any(
            r["review"]["decision"] == "request_more_evidence" for r in reviews
        ),
        "reviewer_independent_queries": sum(r["independent_query_count"] for r in reviews),
        "reviewer_disputed_claims": sum(len(r["review"]["disputed_claim_ids"]) for r in reviews),
        "tool_requests": report.usage.tool_requests,
        "tool_errors": sum(1 for action in tool_actions if action.error_code),
        "provider_attempts": report.usage.provider_attempts,
        "input_tokens": report.usage.input_tokens,
        "output_tokens": report.usage.output_tokens,
        "unknown_usage_attempts": report.usage.unknown_usage_attempts,
        "estimated_usd": report.usage.estimated_usd,
        "active_ms": report.usage.active_ms,
    }


def review_change_counts(run_id: str, store: RunStore, case_dir: Path) -> dict[str, int]:
    """Compare code checks before and after each review, without causal attribution."""
    counts = {
        "review_revision_pairs": 0,
        "review_retained_claims": 0,
        "review_initial_failed_claims": 0,
        "review_initial_passed_claims": 0,
        "review_corrected_claims": 0,
        "review_damaged_claims": 0,
        "review_dropped_claims": 0,
        "review_new_claims": 0,
        "review_new_failed_claims": 0,
    }
    reviews = store.reviews(run_id)
    if not reviews:
        return counts
    findings = store.findings(run_id)
    positions = {finding[0]: position for position, finding in enumerate(findings)}
    context = RunContext.model_validate(store.get_run(run_id)["context"])
    connection, _ = open_snapshot(case_dir / "snapshot")
    try:
        verifier = Verifier(context, store, MeasurementRepository(connection))
        for review in reviews:
            position = positions[review["findings_id"]]
            if position + 1 >= len(findings):
                continue
            before = verifier.verify(InvestigationFindings.model_validate(findings[position][2]))
            after = verifier.verify(InvestigationFindings.model_validate(findings[position + 1][2]))
            before_checks = {check.claim_id: check.passed for check in before.claim_checks}
            after_checks = {check.claim_id: check.passed for check in after.claim_checks}
            retained = before_checks.keys() & after_checks.keys()
            introduced = after_checks.keys() - before_checks.keys()
            counts["review_revision_pairs"] += 1
            counts["review_retained_claims"] += len(retained)
            counts["review_initial_failed_claims"] += sum(
                not before_checks[claim] for claim in retained
            )
            counts["review_initial_passed_claims"] += sum(
                before_checks[claim] for claim in retained
            )
            counts["review_corrected_claims"] += sum(
                not before_checks[claim] and after_checks[claim] for claim in retained
            )
            counts["review_damaged_claims"] += sum(
                before_checks[claim] and not after_checks[claim] for claim in retained
            )
            counts["review_dropped_claims"] += len(before_checks.keys() - after_checks.keys())
            counts["review_new_claims"] += len(introduced)
            counts["review_new_failed_claims"] += sum(
                not after_checks[claim] for claim in introduced
            )
    finally:
        connection.close()
    return counts


def _mean(values: list[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return None if not present else mean(present)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    table = []
    for group_key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        outcomes: dict[str, int] = defaultdict(int)
        for member in members:
            outcomes[member["workflow_outcome"]] += 1
        table.append(
            {
                **dict(zip(keys, group_key, strict=True)),
                "runs": len(members),
                "dispatched_runs": sum(member["provider_attempts"] > 0 for member in members),
                "unavailable": outcomes["unavailable"],
                "budget_prevented": outcomes["budget_prevented"],
                "crashed": outcomes["crashed"],
                "completed": outcomes["completed"],
                "partial": outcomes["partial"],
                "exhausted": outcomes["exhausted"],
                "failed": outcomes["failed"],
                "runs_with_claims": sum(1 for member in members if member["checkable_claims"]),
                "mean_factual_support": _mean([member["factual_support"] for member in members]),
                "fabricated_evidence_claims": sum(
                    member["fabricated_evidence_claims"] for member in members
                ),
                "mean_numerical_validity": _mean(
                    [member["numerical_validity"] for member in members]
                ),
                "mean_fact_completeness": _mean(
                    [member["fact_completeness"] for member in members]
                ),
                "pattern_recovery_rate": _mean(
                    [
                        None
                        if member["pattern_recovered"] is None
                        else int(member["pattern_recovered"])
                        for member in members
                    ]
                ),
                "pattern_scored_runs": sum(
                    1 for member in members if member["pattern_recovered"] is not None
                ),
                "corrected_after_rejection": sum(
                    1
                    for member in members
                    if member["first_verification_failed"] and member["final_verification_passed"]
                ),
                "review_change_counts": {
                    key: sum(member.get(key, 0) for member in members)
                    for key in (
                        "review_revision_pairs",
                        "review_retained_claims",
                        "review_initial_failed_claims",
                        "review_initial_passed_claims",
                        "review_corrected_claims",
                        "review_damaged_claims",
                        "review_dropped_claims",
                        "review_new_claims",
                        "review_new_failed_claims",
                    )
                },
                "mean_tool_requests": _mean([member["tool_requests"] for member in members]),
                "mean_provider_attempts": _mean(
                    [member["provider_attempts"] for member in members]
                ),
                "mean_input_tokens": _mean([member["input_tokens"] for member in members]),
                "mean_output_tokens": _mean([member["output_tokens"] for member in members]),
                "total_estimated_usd": None
                if any(member["estimated_usd"] is None for member in members)
                else sum(member["estimated_usd"] for member in members),
                "median_active_ms": None
                if not members
                else sorted(member["active_ms"] for member in members)[len(members) // 2],
            }
        )
    return table


def paired_differences(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Paired arm differences on case level metrics, bootstrapped over source sessions."""
    if len({row.get("model_id") for row in rows}) > 1:
        raise ValueError("paired architecture differences require a single model")
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("workflow_outcome") in ("unavailable", "crashed", "budget_prevented"):
            continue
        by_case[row["case_id"]][row["arm"]] = row
    results = {}
    for metric in ("fact_completeness", "pattern_recovered", "factual_support"):
        for first, second in (
            ("investigator_raw", "investigator_gated"),
            ("investigator_gated", "investigator_reviewer_gated"),
        ):
            differences: dict[str, list[float]] = defaultdict(list)
            for arms in by_case.values():
                if first not in arms or second not in arms:
                    continue
                left, right = arms[first][metric], arms[second][metric]
                if left is None or right is None:
                    continue
                differences[arms[first]["source_session_id"]].append(float(right) - float(left))
            pooled = [value for values in differences.values() for value in values]
            entry: dict[str, Any] = {
                "pairs": len(pooled),
                "source_session_clusters": len(differences),
                "mean_difference": None if not pooled else mean(pooled),
            }
            if len(differences) >= MINIMUM_CLUSTERS_FOR_INTERVAL and pooled:
                generator = random.Random(seed)
                clusters = list(differences)
                estimates = []
                for _ in range(BOOTSTRAP_RESAMPLES):
                    sample = [
                        value for _ in clusters for value in differences[generator.choice(clusters)]
                    ]
                    estimates.append(mean(sample))
                entry["cluster_bootstrap_95"] = [
                    _percentile(estimates, 0.025),
                    _percentile(estimates, 0.975),
                ]
            else:
                entry["cluster_bootstrap_95"] = None
                entry["note"] = "too few source sessions for an interval, descriptive only"
            results[f"{metric}:{second}_minus_{first}"] = entry
    return results


def tool_usage(stores: RunStore, run_ids: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {"requests": 0, "cache_hits": 0, "errors": 0, "executions": 0}
    )
    for run_id, model_id, arm in run_ids:
        for action in stores.actions(run_id, ActionKind.tool_request):
            bucket = counts[(model_id, arm, action.role, action.tool_name or "")]
            bucket["requests"] += 1
            bucket["cache_hits"] += int(action.cache_hit)
            bucket["errors"] += int(action.error_code is not None)
        for raw in stores.raw_actions(run_id):
            if raw["kind"] == "tool_request" and raw["physical_query"]:
                counts[(model_id, arm, raw["role"], raw["tool_name"] or "")]["executions"] += 1
    return [
        {"model_id": model_id, "arm": arm, "role": role, "tool": tool, **values}
        for (model_id, arm, role, tool), values in sorted(counts.items())
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        field_names = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def score_experiment(
    index_path: Path, manifest_path: Path, out_dir: Path, seed: int = 42
) -> dict[str, Any]:
    index = json.loads(index_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    cases_dir = manifest_path.parent.parent / "cases"
    store = RunStore(index_path.parent / "runs.sqlite3", SystemClock())
    rows: list[dict[str, Any]] = []
    unscored = []
    try:
        for entry in index["runs"]:
            case = cases[entry["case_id"]]
            stored = store.get_report(entry["run_id"]) if entry["status"] == "ran" else None
            if stored is None:
                unscored.append(entry)
                recorded_usage = store.usage(entry["run_id"])
                rows.append(
                    {
                        "run_id": entry["run_id"],
                        "model_id": entry.get("model_id") or index.get("model_id") or "unknown",
                        "case_id": entry["case_id"],
                        "family": case["family"],
                        "source_session_id": case["source_session_id"],
                        "arm": entry["arm"],
                        "workflow_outcome": entry["status"],
                        "checkable_claims": 0,
                        "factual_support": None,
                        "fabricated_evidence_claims": 0,
                        "numerical_validity": None,
                        "fact_completeness": 0.0 if case["required_fact_kinds"] else None,
                        "pattern_recovered": None,
                        "first_verification_failed": False,
                        "final_verification_passed": False,
                        "tool_requests": recorded_usage.tool_requests,
                        "provider_attempts": recorded_usage.provider_attempts,
                        "input_tokens": recorded_usage.input_tokens,
                        "output_tokens": recorded_usage.output_tokens,
                        "unknown_usage_attempts": recorded_usage.unknown_usage_attempts,
                        "estimated_usd": recorded_usage.estimated_usd,
                        "active_ms": recorded_usage.active_ms,
                    }
                )
                continue
            report = InvestigationReport.model_validate(stored)
            rows.append(score_run(report, case, store, cases_dir / case["case_id"]))
        usage = tool_usage(
            store,
            [(row["run_id"], row["model_id"] or "unknown", row["arm"]) for row in rows],
        )
    finally:
        store.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["model_id"] = row["model_id"] or "unknown"
    by_arm = aggregate(rows, ("model_id", "arm"))
    by_family = aggregate(rows, ("model_id", "arm", "family"))
    model_ids = sorted({row["model_id"] for row in rows})
    paired_by_model = {
        model_id: paired_differences([row for row in rows if row["model_id"] == model_id], seed)
        for model_id in model_ids
    }
    paired = paired_by_model[model_ids[0]] if len(model_ids) == 1 else {}
    modes = sorted({row.get("execution_mode", "unknown") for row in rows})
    label = (
        "OFFLINE HARNESS CHECK with the deterministic template policy. These numbers describe "
        "the scripted baseline and the harness, not model quality."
        if index["mode"] == "offline"
        else "Live experiment. Inspect completed and unavailable counts before interpreting scores."
    )
    _write_csv(out_dir / "runs.csv", rows)
    _write_csv(out_dir / "by_arm.csv", by_arm)
    _write_csv(out_dir / "by_family.csv", by_family)
    _write_csv(out_dir / "tool_usage.csv", usage)
    annotation = [
        {
            "run_id": row["run_id"],
            "model_id": row["model_id"],
            "case_id": row["case_id"],
            "arm": row["arm"],
            "unsupported_causal_claim": "",
            "missed_alternative": "",
            "useful_next_measurement": "",
            "annotator": "",
            "notes": "",
        }
        for row in rows
    ]
    _write_csv(out_dir / "manual_annotation_template.csv", annotation)
    summary = {
        "label": label,
        "experiment_id": index["experiment_id"],
        "mode": index["mode"],
        "execution_modes": modes,
        "manifest_sha256": file_sha256(manifest_path),
        "freeze": index["freeze"],
        "scoring_version": "model_separated_2",
        "scheduled_runs": len(rows),
        "attempted_runs": sum(row["provider_attempts"] > 0 for row in rows),
        "completed_runs": sum(row["workflow_outcome"] == "completed" for row in rows),
        "unscored_runs": len(unscored),
        "source_sessions": len({row["source_session_id"] for row in rows}),
        "by_arm": by_arm,
        "paired_differences": paired,
        "paired_differences_by_model": paired_by_model,
        "notes": [
            "Constructed variants are observation patterns with a known construction, "
            "not physical causes.",
            "Cases built from one base window are correlated. "
            "Source sessions are the bootstrap unit.",
            "cause_status is constrained by the schema, so its appropriateness "
            "is not evidence of reasoning.",
            "Scores cover code checked claims and observation labels. Causal language and "
            "usefulness are not scored here.",
            "Review corrections and damage compare retained claim IDs in the immediate revision "
            "after a review. Initial failed and passed retained claims are the respective "
            "denominators. New failed claims are reported over all new claims, and dropped "
            "claims are counted separately. These changes do not establish reviewer causation.",
        ],
    }
    (out_dir / "score.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    (out_dir / "summary.md").write_text(render_summary(summary))
    return summary


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_summary(summary: dict[str, Any]) -> str:
    columns = [
        "model_id",
        "arm",
        "runs",
        "completed",
        "partial",
        "exhausted",
        "failed",
        "unavailable",
        "runs_with_claims",
        "mean_factual_support",
        "fabricated_evidence_claims",
        "mean_fact_completeness",
        "pattern_recovery_rate",
        "corrected_after_rejection",
        "mean_provider_attempts",
        "mean_input_tokens",
        "total_estimated_usd",
    ]
    lines = [
        f"# Evaluation {summary['experiment_id']}",
        "",
        f"> {summary['label']}",
        "",
        f"Scheduled runs {summary['scheduled_runs']}, dispatched {summary['attempted_runs']}, "
        f"completed {summary['completed_runs']}, unscored {summary['unscored_runs']}, "
        f"source sessions {summary['source_sessions']}.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary["by_arm"]:
        lines.append("| " + " | ".join(_format(row.get(column)) for column in columns) + " |")
    lines += ["", "## Paired differences", ""]
    for model_id, differences in summary["paired_differences_by_model"].items():
        for name, entry in differences.items():
            interval = entry["cluster_bootstrap_95"]
            interval_text = "NA" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
            lines.append(
                f"- {model_id} {name}: mean {_format(entry['mean_difference'])} "
                f"over {entry['pairs']} pairs from {entry['source_session_clusters']} sessions, "
                f"interval {interval_text}"
            )
    lines += ["", "## Notes", ""] + [f"- {note}" for note in summary["notes"]]
    return "\n".join(lines) + "\n"
