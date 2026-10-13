"""Verification results and the final investigation report."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from celltrace.contracts.common import (
    CONTRACT_SCHEMA_VERSION,
    Arm,
    ContractModel,
    ExecutionMode,
)
from celltrace.contracts.findings import (
    CauseStatus,
    FactualClaim,
    Hypothesis,
    InvestigationFindings,
    ObservationPattern,
    ReviewResult,
)


class VerificationCode(StrEnum):
    unknown_evidence = "UNKNOWN_EVIDENCE"
    evidence_wrong_run = "EVIDENCE_WRONG_RUN"
    evidence_wrong_snapshot = "EVIDENCE_WRONG_SNAPSHOT"
    evidence_wrong_dictionary = "EVIDENCE_WRONG_DICTIONARY"
    evidence_content_altered = "EVIDENCE_CONTENT_ALTERED"
    evidence_does_not_support = "EVIDENCE_DOES_NOT_SUPPORT"
    out_of_scope = "OUT_OF_SCOPE"
    value_mismatch = "VALUE_MISMATCH"
    count_mismatch = "COUNT_MISMATCH"
    unit_mismatch = "UNIT_MISMATCH"
    metric_not_applicable = "METRIC_NOT_APPLICABLE"
    transition_not_found = "TRANSITION_NOT_FOUND"
    chronology_invalid = "CHRONOLOGY_INVALID"
    relative_change_not_allowed = "RELATIVE_CHANGE_NOT_ALLOWED"
    causal_status_violation = "CAUSAL_STATUS_VIOLATION"
    review_incomplete = "REVIEW_INCOMPLETE"
    recomputation_failed = "RECOMPUTATION_FAILED"
    observation_pattern_unsupported = "OBSERVATION_PATTERN_UNSUPPORTED"
    observation_pattern_conflict = "OBSERVATION_PATTERN_CONFLICT"
    exclusion_not_observable = "EXCLUSION_NOT_OBSERVABLE"


class ClaimCheck(ContractModel):
    claim_id: str
    passed: bool
    code: VerificationCode | None = None
    message: str = ""
    expected: Any = None
    actual: Any = None


class ObservationPatternCheck(ClaimCheck):
    pattern: ObservationPattern
    supporting_claim_ids: list[str] = Field(default_factory=list, max_length=8)


class VerificationResult(ContractModel):
    schema_version: int = CONTRACT_SCHEMA_VERSION
    run_id: str
    passed: bool
    claim_checks: list[ClaimCheck]
    policy_errors: list[ClaimCheck] = Field(default_factory=list)
    pattern_checks: list[ObservationPatternCheck] = Field(default_factory=list)
    verification_queries: int = 0

    @property
    def failed_checks(self) -> list[ClaimCheck]:
        return [
            check
            for check in self.claim_checks + self.policy_errors + self.pattern_checks
            if not check.passed
        ]


class WorkflowOutcome(StrEnum):
    completed = "completed"
    partial = "partial"
    exhausted = "exhausted"
    failed = "failed"


class FactCheckStatus(StrEnum):
    passed = "passed"
    failed = "failed"
    incomplete = "incomplete"
    not_run = "not_run"


class ReviewStatus(StrEnum):
    independently_checked = "independently_checked"
    limited = "limited"
    not_run = "not_run"


class ReportBasis(StrEnum):
    agent_findings = "agent_findings"
    raw_unverified_findings = "raw_unverified_findings"
    deterministic_fallback = "deterministic_fallback"
    no_findings = "no_findings"


class UsageSummary(ContractModel):
    tool_requests: int
    tool_cache_hits: int
    physical_queries: int
    verification_queries: int
    provider_attempts: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    unknown_usage_attempts: int
    estimated_usd: float | None
    active_ms: int
    review_rounds: int
    revision_rounds: int


class EvidenceManifestEntry(ContractModel):
    evidence_id: str
    tool_name: str
    content_sha256: str
    effective_bounds: list[dict[str, int]]


class InvestigationReport(ContractModel):
    schema_version: int = CONTRACT_SCHEMA_VERSION
    run_id: str
    case_id: str
    event_id: str
    arm: Arm
    execution_mode: ExecutionMode
    model_id: str | None
    snapshot_hash: str
    workflow_outcome: WorkflowOutcome
    fact_check_status: FactCheckStatus
    review_status: ReviewStatus
    cause_status: CauseStatus
    report_basis: ReportBasis
    terminal_reason: str
    verified_claims: list[FactualClaim]
    unverified_claims: list[FactualClaim] = Field(default_factory=list)
    rejected_claims: list[dict[str, Any]] = Field(default_factory=list)
    verification_history: list[dict[str, Any]] = Field(default_factory=list)
    qualified_hypotheses: list[Hypothesis]
    missing_evidence: list[str]
    next_measurements: list[str]
    observation_patterns: list[str]
    verified_observation_patterns: list[str] = Field(default_factory=list)
    unverified_observation_patterns: list[str] = Field(default_factory=list)
    observation_pattern_checks: list[ObservationPatternCheck] = Field(default_factory=list)
    eligibility: dict[str, Any] | None
    evidence_manifest: list[EvidenceManifestEntry]
    usage: UsageSummary
    limitations: list[str]
    review: ReviewResult | None = None
    unverified_commentary: str | None = None
    initial_findings: InvestigationFindings | None = None
    final_findings: InvestigationFindings | None = None
