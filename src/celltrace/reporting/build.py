"""Assemble the final report from stored, verified records.

No new facts are created here. Verified claims are the claims whose code
checks passed. Arm A reports its claims as unverified because the gate is
deliberately not run in that arm.
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from celltrace.contracts.common import Arm, ExecutionMode
from celltrace.contracts.evidence import RunContext
from celltrace.contracts.findings import (
    CauseStatus,
    FactualClaim,
    InvestigationFindings,
    ReviewDecision,
    ReviewResult,
)
from celltrace.contracts.report import (
    EvidenceManifestEntry,
    FactCheckStatus,
    InvestigationReport,
    ReportBasis,
    ReviewStatus,
    UsageSummary,
    VerificationResult,
    WorkflowOutcome,
)
from celltrace.reporting.fallback import claims_from_results
from celltrace.storage.run_store import ActionKind, RunStore
from celltrace.verification.verifier import Verifier

DATASET_LIMITATIONS = (
    "Facts apply only to their cited windows. Whole-session interpretation is unsupported; "
    "free text remains unverified even when structured scope and numerical checks pass.",
    "Downlink and uplink are logged application traffic, not available network capacity.",
    "The bitrate unit (kbps) and speed unit are inferred from magnitude, not confirmed by the "
    "dataset paper, which could not be retrieved during the audit.",
    "Timestamps are local wall clock with unknown timezone and whole second resolution.",
    "A reported serving cell identity change is a transition proxy, not a confirmed handover.",
    "No load, scheduling, interference, transport or server measurements exist in this trace, "
    "so a physical cause cannot be established.",
)


def build_report(
    store: RunStore,
    context: RunContext,
    arm: Arm,
    execution_mode: ExecutionMode,
    model_id: str | None,
    terminal_reason: str,
    verifier: Verifier | None = None,
) -> InvestigationReport:
    run_id = context.run_id
    findings_rows = store.findings(run_id)
    initial = InvestigationFindings.model_validate(findings_rows[0][2]) if findings_rows else None
    final = InvestigationFindings.model_validate(findings_rows[-1][2]) if findings_rows else None
    final_id = findings_rows[-1][0] if findings_rows else None

    verification = None
    if final_id is not None:
        matching = [
            entry for entry in store.verifications(run_id) if entry["findings_id"] == final_id
        ]
        if matching:
            verification = VerificationResult.model_validate(matching[-1]["result"])

    reviews = store.reviews(run_id)
    final_reviews = [entry for entry in reviews if entry["findings_id"] == final_id]
    last_review = ReviewResult.model_validate(reviews[-1]["review"]) if reviews else None

    limitations = list(DATASET_LIMITATIONS)
    if final is not None and final.assessment_scope is None:
        limitations.append("Assessment scope was omitted; no explicit scope declaration checked.")
    verified_claims: list[Any] = []
    unverified_claims: list[Any] = []
    rejected: list[dict[str, Any]] = []
    hypotheses: list[Any] = []

    if final is None:
        fact_check = (
            FactCheckStatus.not_run if arm == Arm.investigator_raw else FactCheckStatus.incomplete
        )
        basis = ReportBasis.no_findings
    elif arm == Arm.investigator_raw:
        fact_check = FactCheckStatus.not_run
        basis = ReportBasis.raw_unverified_findings
        unverified_claims = list(final.factual_claims)
        hypotheses = list(final.hypotheses)
        limitations.append("Arm A: claims were not code verified during the run.")
    elif verification is None:
        fact_check = FactCheckStatus.incomplete
        basis = ReportBasis.agent_findings
        unverified_claims = list(final.factual_claims)
        limitations.append("Verification did not run before the run stopped.")
    else:
        basis = ReportBasis.agent_findings
        # Policy and pattern identifiers are separate namespaces from arbitrary factual
        # claim IDs. A scope failure must not relabel a numerically valid claim as rejected.
        failed = {check.claim_id: check for check in verification.claim_checks if not check.passed}
        failed_policy_ids = {check.claim_id for check in verification.policy_errors}
        for claim in final.factual_claims:
            if claim.claim_id in failed:
                rejected.append(
                    {
                        "claim": claim.model_dump(mode="json"),
                        "check": failed[claim.claim_id].model_dump(mode="json"),
                    }
                )
            else:
                verified_claims.append(claim)
        failed_ids = set(failed)
        for hypothesis in final.hypotheses:
            if (
                f"hypothesis:{hypothesis.hypothesis_id.value}" in failed_policy_ids
                or set(hypothesis.supporting_claim_ids + hypothesis.contradicting_claim_ids)
                & failed_ids
            ):
                limitations.append(
                    f"Hypothesis {hypothesis.hypothesis_id.value} omitted because it cites a "
                    "failed claim or failed a policy check."
                )
            else:
                hypotheses.append(hypothesis)
        fact_check = FactCheckStatus.passed if verification.passed else FactCheckStatus.failed
        for policy in verification.policy_errors:
            limitations.append(f"Policy check failed: {policy.claim_id}: {policy.message}")

    if (
        terminal_reason
        and not verified_claims
        and arm != Arm.investigator_raw
        and verifier is not None
    ):
        fallback = _fallback_claims(store, run_id, verifier)
        if fallback:
            verified_claims = fallback
            basis = ReportBasis.deterministic_fallback
            limitations.append(
                "Deterministic fallback: facts were built by code from this run's evidence "
                "records and verified, because the agent did not deliver verified findings."
            )

    limitations += chronology_warnings(hypotheses, verified_claims or unverified_claims, context)

    failed_tools = [
        action.error_code
        for action in store.actions(run_id, ActionKind.tool_request)
        if action.error_code is not None
    ]
    if failed_tools:
        counts = {code: failed_tools.count(code) for code in sorted(set(failed_tools))}
        limitations.append(f"Tool requests that returned errors: {counts}.")

    if arm != Arm.investigator_reviewer_gated or not reviews:
        review_status = ReviewStatus.not_run
        reviewer_actions = [action for action in store.actions(run_id) if action.role == "reviewer"]
        if reviewer_actions:
            # not_run means no review was recorded, not that the Reviewer never started.
            tool_requests = sum(
                action.kind == ActionKind.tool_request for action in reviewer_actions
            )
            limitations.append(
                f"The Reviewer started ({tool_requests} tool requests, "
                f"{len(reviewer_actions) - tool_requests} other actions) but recorded no "
                "review before the run stopped."
            )
    elif final_reviews and final_reviews[-1]["independent_query_count"] > 0:
        review_status = ReviewStatus.independently_checked
    else:
        review_status = ReviewStatus.limited
        if not final_reviews:
            limitations.append("The final revision was not re-reviewed within the round limit.")
        elif final_reviews[-1]["review"].get("review_limitation"):
            limitations.append(
                f"Review limitation: {final_reviews[-1]['review']['review_limitation']}"
            )

    outcome = _workflow_outcome(
        arm, terminal_reason, final, fact_check, review_status, last_review, verified_claims
    )
    if terminal_reason:
        limitations.append(f"Run stopped: {terminal_reason}.")

    pattern_checks = (
        verification.pattern_checks
        if verification is not None and arm != Arm.investigator_raw
        else []
    )
    checked_patterns = {check.pattern.value: check for check in pattern_checks}
    proposed_patterns = [pattern.value for pattern in final.observation_patterns] if final else []
    verified_patterns = [
        pattern
        for pattern in proposed_patterns
        if pattern in checked_patterns and checked_patterns[pattern].passed
    ]
    unverified_patterns = [
        pattern for pattern in proposed_patterns if pattern not in checked_patterns
    ]
    accepted_patterns = [
        pattern
        for pattern in proposed_patterns
        if pattern not in checked_patterns or checked_patterns[pattern].passed
    ]
    if basis == ReportBasis.deterministic_fallback:
        verified_patterns = []
        accepted_patterns = []
        unverified_patterns = proposed_patterns
    if unverified_patterns:
        limitations.append(
            "Some observation labels are unverified model interpretations. Numerical claim "
            "checks do not establish those labels or a physical cause."
        )

    cited = {
        evidence_id
        for claim in (verified_claims or unverified_claims)
        for evidence_id in claim.evidence_ids
    }
    manifest = []
    for evidence_id in sorted(cited):
        record = store.get_evidence(evidence_id)
        if record is None or record["run_id"] != run_id:
            continue
        manifest.append(
            EvidenceManifestEntry(
                evidence_id=evidence_id,
                tool_name=record["tool_name"],
                content_sha256=record["content_sha256"],
                effective_bounds=record["effective_bounds"],
            )
        )

    revision_by_findings = {row[0]: row[1] for row in findings_rows}
    history = []
    for entry in store.verifications(run_id):
        result = VerificationResult.model_validate(entry["result"])
        history.append(
            {
                "findings_revision": revision_by_findings.get(entry["findings_id"]),
                "passed": result.passed,
                "failed_checks": [
                    {
                        "claim_id": check.claim_id,
                        "code": None if check.code is None else check.code.value,
                    }
                    for check in result.failed_checks
                ],
            }
        )

    usage = store.usage(run_id)
    revisions = sum(
        1 for event in store.run_events(run_id) if event["event_type"] == "revision_requested"
    )
    return InvestigationReport(
        run_id=run_id,
        case_id=context.case_id,
        event_id=context.candidate.event_id,
        arm=arm,
        execution_mode=execution_mode,
        model_id=model_id,
        snapshot_hash=context.snapshot_hash,
        workflow_outcome=outcome,
        fact_check_status=fact_check,
        review_status=review_status,
        cause_status=final.cause_status if final is not None else CauseStatus.not_assessed,
        report_basis=basis,
        terminal_reason=terminal_reason or "completed_normally",
        verified_claims=verified_claims,
        unverified_claims=unverified_claims,
        rejected_claims=rejected,
        verification_history=history,
        qualified_hypotheses=hypotheses,
        missing_evidence=list(final.missing_evidence) if final else [],
        next_measurements=list(final.next_measurements) if final else [],
        observation_patterns=accepted_patterns,
        verified_observation_patterns=verified_patterns,
        unverified_observation_patterns=unverified_patterns,
        observation_pattern_checks=pattern_checks,
        eligibility=final.eligibility.model_dump(mode="json") if final else None,
        evidence_manifest=manifest,
        usage=UsageSummary(
            tool_requests=usage.tool_requests,
            tool_cache_hits=usage.tool_cache_hits,
            physical_queries=usage.physical_queries,
            verification_queries=usage.verification_queries,
            provider_attempts=usage.provider_attempts,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            unknown_usage_attempts=usage.unknown_usage_attempts,
            estimated_usd=round(usage.estimated_usd, 6)
            if execution_mode == ExecutionMode.live
            else None,
            active_ms=usage.active_ms,
            review_rounds=len(reviews),
            revision_rounds=revisions,
        ),
        limitations=limitations,
        review=last_review,
        unverified_commentary=final.decision_summary if final else None,
        initial_findings=initial,
        final_findings=final,
    )


def _workflow_outcome(
    arm: Arm,
    terminal_reason: str,
    final: InvestigationFindings | None,
    fact_check: FactCheckStatus,
    review_status: ReviewStatus,
    last_review: ReviewResult | None,
    verified_claims: list[Any],
) -> WorkflowOutcome:
    if terminal_reason.startswith("budget_exhausted"):
        return WorkflowOutcome.exhausted
    if terminal_reason:
        return WorkflowOutcome.partial if verified_claims else WorkflowOutcome.failed
    if final is None:
        return WorkflowOutcome.failed
    if arm == Arm.investigator_raw:
        return WorkflowOutcome.completed
    review_ok = arm == Arm.investigator_gated or (
        review_status == ReviewStatus.independently_checked
        and last_review is not None
        and last_review.decision != ReviewDecision.request_more_evidence
    )
    if fact_check == FactCheckStatus.passed and review_ok:
        return WorkflowOutcome.completed
    return WorkflowOutcome.partial if verified_claims else WorkflowOutcome.failed


def _fallback_claims(store: RunStore, run_id: str, verifier: Verifier) -> list[Any]:
    bundle = claims_from_results(store.run_evidence(run_id))
    claim_adapter: TypeAdapter[Any] = TypeAdapter(FactualClaim)
    verified = []
    for payload in bundle.claims:
        claim = claim_adapter.validate_python(payload)
        if verifier.check_claim(claim).passed:
            verified.append(claim)
    return verified


def chronology_warnings(hypotheses: list[Any], claims: list[Any], context: RunContext) -> list[str]:
    """Flag consistent hypotheses that lean on a transition seen only after the drop began.

    An identity change observed after the event onset cannot explain the onset.
    This is a warning in the report, not a failed check, because the hypothesis
    may still be meant as a consequence or a coincidence.
    """
    onset = context.candidate.event_window.start_ms
    late_transitions = {
        claim.claim_id
        for claim in claims
        if claim.claim_type == "transition" and claim.earlier_time_ms >= onset
    }
    warnings = []
    for hypothesis in hypotheses:
        cited = set(hypothesis.supporting_claim_ids) & late_transitions
        if hypothesis.status.value == "consistent" and cited:
            warnings.append(
                f"Chronology warning: hypothesis {hypothesis.hypothesis_id.value} cites "
                f"{sorted(cited)}, a cell identity change first observed at or after the event "
                f"onset ({onset} ms). It cannot explain the onset."
            )
    return warnings
