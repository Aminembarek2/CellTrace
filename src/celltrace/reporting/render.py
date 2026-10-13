"""Deterministic Markdown rendering of a validated report.

Every sentence in the verified facts section comes from a typed claim and a
fixed template. Rounding happens only here, after verification. Agent prose
appears only under the clearly labeled unverified commentary heading.
"""

from __future__ import annotations

from typing import Any

from celltrace.canonical import canonical_json
from celltrace.contracts.findings import (
    ActivityClaim,
    AvailabilityClaim,
    MetricChangeClaim,
    MetricStatisticClaim,
    TransitionClaim,
)
from celltrace.contracts.report import InvestigationReport

BASIS_LABEL = {"native": "logged samples", "bin_1s": "one second bin means"}


def _number(value: float | None, unit: str) -> str:
    if value is None:
        return "not available"
    digits = 1 if unit in ("dB", "dBm") else 3
    return f"{value:.{digits}f} {unit}"


def _window(window: Any) -> str:
    return f"[{window.start_ms}, {window.end_ms}) ms"


def claim_sentence(claim: Any) -> str:
    if isinstance(claim, MetricChangeClaim):
        unit = claim.unit.value
        reference = _number(claim.reference_value, unit)
        event = _number(claim.event_value, unit)
        sentence = (
            f"{claim.statistic.value.capitalize()} {claim.metric.value} "
            f"({BASIS_LABEL[claim.basis.value]}) was {reference} "
            f"in {_window(claim.reference_window)} and {event} "
            f"in {_window(claim.event_window)}, a change of {_number(claim.delta, unit)}"
        )
        if claim.relative_decrease is not None:
            sentence += f" ({claim.relative_decrease * 100:.1f} percent decrease)"
        return sentence + "."
    if isinstance(claim, MetricStatisticClaim):
        return (
            f"{claim.statistic.value.capitalize()} {claim.metric.value} "
            f"({BASIS_LABEL[claim.basis.value]}) in {_window(claim.window)} was "
            f"{_number(claim.value, claim.unit.value)} from {claim.valid_count} valid and "
            f"{claim.missing_count} missing values."
        )
    if isinstance(claim, ActivityClaim):
        return (
            f"The {claim.state.value} share of application download state "
            f"({BASIS_LABEL[claim.basis.value]}) in {_window(claim.window)} was "
            f"{claim.fraction * 100:.1f} percent of {claim.denominator}."
        )
    if isinstance(claim, TransitionClaim):
        return (
            f"Reported serving cell identity changed from {claim.earlier_cell_id} at "
            f"{claim.earlier_time_ms} ms to {claim.later_cell_id} at {claim.later_time_ms} ms "
            f"(gap {claim.gap_ms} ms, {claim.transition_type.value}, basis "
            f"{claim.identity_basis.value}). This is a transition proxy, not a confirmed handover."
        )
    assert isinstance(claim, AvailabilityClaim)
    return (
        f"{claim.metric.value} in {_window(claim.window)} ({BASIS_LABEL[claim.basis.value]}) had "
        f"{claim.valid_count} valid and {claim.missing_count} missing values."
    )


def render_markdown(report: InvestigationReport) -> str:
    declared_scope = (
        report.final_findings.assessment_scope or "unspecified" if report.final_findings else "none"
    )
    lines = [
        f"# Investigation report {report.run_id}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Execution mode | {report.execution_mode.value} |",
        f"| Arm | {report.arm.value} |",
        f"| Model | {report.model_id or 'none'} |",
        f"| Event | {report.event_id} |",
        f"| Snapshot | {report.snapshot_hash[:16]} |",
        f"| Workflow outcome | {report.workflow_outcome.value} |",
        f"| Fact check status | {report.fact_check_status.value} |",
        f"| Review status | {report.review_status.value} |",
        f"| Cause status | {report.cause_status.value} |",
        f"| Report basis | {report.report_basis.value} |",
        f"| Declared assessment scope | {declared_scope} |",
        f"| Terminal reason | {report.terminal_reason} |",
        "",
    ]
    if report.execution_mode.value == "scripted_fixture":
        lines += [
            "> Scripted fixture run. Decisions were replayed by a test policy, not made by a "
            "model. This shows orchestration behavior only.",
            "",
        ]
    lines += ["## Verified facts", ""]
    if report.verified_claims:
        for claim in report.verified_claims:
            evidence = ", ".join(claim.evidence_ids)
            lines.append(f"- {claim_sentence(claim)} Evidence: {evidence}.")
    else:
        lines.append("- None.")
    if report.unverified_claims:
        lines += ["", "## Unverified claims (not code checked)", ""]
        for claim in report.unverified_claims:
            lines.append(
                f"- {claim_sentence(claim)} Evidence cited: {', '.join(claim.evidence_ids)}."
            )
    if report.rejected_claims:
        lines += ["", "## Claims rejected by code verification", ""]
        for entry in report.rejected_claims:
            check = entry["check"]
            lines.append(
                f"- {entry['claim']['claim_id']}: {check['code']} {check['message']} "
                f"(expected {canonical_json(check['expected'])}, "
                f"got {canonical_json(check['actual'])})."
            )
    if report.verification_history:
        lines += ["", "## Verification history", ""]
        for entry in report.verification_history:
            failed = ", ".join(
                f"{check['claim_id']} {check['code']}" for check in entry["failed_checks"]
            )
            outcome = "passed" if entry["passed"] else f"failed ({failed})"
            lines.append(f"- Findings revision {entry['findings_revision']}: {outcome}")
    lines += ["", "## Checked observation patterns", ""]
    lines += [f"- {pattern}" for pattern in report.verified_observation_patterns] or [
        "- None checked."
    ]
    unchecked_patterns = report.unverified_observation_patterns or [
        pattern
        for pattern in report.observation_patterns
        if pattern not in report.verified_observation_patterns
    ]
    if unchecked_patterns:
        lines += ["", "## Unverified observation labels from the investigator", ""]
        lines += [f"- {pattern}" for pattern in unchecked_patterns]
    rejected_patterns = [check for check in report.observation_pattern_checks if not check.passed]
    if rejected_patterns:
        lines += ["", "## Observation labels rejected by code", ""]
        lines += [
            f"- {check.pattern.value}: {check.code} {check.message}" for check in rejected_patterns
        ]
    lines += ["", "## Qualified hypotheses", ""]
    if report.qualified_hypotheses:
        for hypothesis in report.qualified_hypotheses:
            lines.append(
                f"- {hypothesis.hypothesis_id.value}: {hypothesis.status.value}. "
                f"Would be discriminated by: {hypothesis.discriminating_measurement}"
            )
    else:
        lines.append("- None.")
    lines += ["", "## Missing evidence", ""]
    lines += [f"- {item}" for item in report.missing_evidence] or ["- None listed."]
    lines += ["", "## Next measurements", ""]
    lines += [f"- {item}" for item in report.next_measurements] or ["- None listed."]
    if report.review is not None:
        lines += [
            "",
            "## Review",
            "",
            f"- Decision: {report.review.decision.value}",
            f"- Disputed claims: {', '.join(report.review.disputed_claim_ids) or 'none'}",
            "- Independent evidence: "
            f"{', '.join(report.review.independent_evidence_ids) or 'none'}",
            f"- Limitation: {report.review.review_limitation or 'none'}",
        ]
    usage = report.usage
    cost_text = (
        "not applicable" if usage.estimated_usd is None else f"{usage.estimated_usd:.4f} USD"
    )
    lines += [
        "",
        "## Usage",
        "",
        f"- Tool requests charged: {usage.tool_requests} (cache hits {usage.tool_cache_hits}, "
        f"executed tool queries {usage.physical_queries})",
        f"- Verification queries by code: {usage.verification_queries}",
        f"- Provider attempts: {usage.provider_attempts} "
        f"(unknown usage {usage.unknown_usage_attempts})",
        f"- Tokens: input {usage.input_tokens}, output {usage.output_tokens}",
        f"- Estimated cost: {cost_text}",
        f"- Active time: {usage.active_ms} ms, review rounds {usage.review_rounds}, "
        f"revisions {usage.revision_rounds}",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in report.limitations]
    lines += ["", "## Evidence manifest", ""]
    for manifest_entry in report.evidence_manifest:
        lines.append(
            f"- {manifest_entry.evidence_id} {manifest_entry.tool_name} "
            f"sha256 {manifest_entry.content_sha256[:16]}"
        )
    if not report.evidence_manifest:
        lines.append("- None.")
    if report.unverified_commentary:
        lines += [
            "",
            "## Unverified commentary from the investigator",
            "",
            "> " + report.unverified_commentary.replace("\n", "\n> "),
        ]
    return "\n".join(lines) + "\n"
