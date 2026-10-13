"""Typed factual claims, hypotheses, investigator findings and reviewer results."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from celltrace.contracts.common import (
    CONTRACT_SCHEMA_VERSION,
    AggregationBasis,
    ContractModel,
    DownloadState,
    IdentityBasis,
    MetricName,
    Statistic,
    TimeWindow,
    TransitionType,
    Unit,
)

EvidenceIds = Annotated[list[str], Field(min_length=1, max_length=8)]
ShortText = Annotated[str, Field(max_length=400)]


class MetricStatisticClaim(ContractModel):
    claim_type: Literal["metric_statistic"] = "metric_statistic"
    claim_id: str = Field(min_length=1, max_length=64)
    metric: MetricName
    statistic: Statistic
    window: TimeWindow
    basis: AggregationBasis
    value: float | None
    unit: Unit
    valid_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    evidence_ids: EvidenceIds


class MetricChangeClaim(ContractModel):
    claim_type: Literal["metric_change"] = "metric_change"
    claim_id: str = Field(min_length=1, max_length=64)
    metric: MetricName
    statistic: Statistic
    basis: AggregationBasis
    reference_window: TimeWindow
    event_window: TimeWindow
    reference_value: float
    event_value: float
    delta: float = Field(description="event_value minus reference_value")
    relative_decrease: float | None = Field(
        default=None, description="(reference - event) / reference, rates only"
    )
    unit: Unit
    evidence_ids: EvidenceIds


class ActivityClaim(ContractModel):
    claim_type: Literal["activity"] = "activity"
    claim_id: str = Field(min_length=1, max_length=64)
    window: TimeWindow
    basis: AggregationBasis
    state: DownloadState
    fraction: float = Field(ge=0, le=1)
    denominator: int = Field(ge=0)
    evidence_ids: EvidenceIds


class TransitionClaim(ContractModel):
    claim_type: Literal["transition"] = "transition"
    claim_id: str = Field(min_length=1, max_length=64)
    transition_type: TransitionType
    earlier_cell_id: str
    later_cell_id: str
    earlier_time_ms: int = Field(ge=0)
    later_time_ms: int = Field(ge=0)
    gap_ms: int = Field(ge=0)
    identity_basis: IdentityBasis
    evidence_ids: EvidenceIds

    @model_validator(mode="after")
    def check_chronology(self) -> TransitionClaim:
        if self.later_time_ms < self.earlier_time_ms:
            raise ValueError("later_time_ms precedes earlier_time_ms")
        return self


class AvailabilityClaim(ContractModel):
    claim_type: Literal["availability"] = "availability"
    claim_id: str = Field(min_length=1, max_length=64)
    metric: MetricName
    window: TimeWindow
    basis: AggregationBasis
    valid_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    evidence_ids: EvidenceIds


FactualClaim = Annotated[
    MetricStatisticClaim | MetricChangeClaim | ActivityClaim | TransitionClaim | AvailabilityClaim,
    Field(discriminator="claim_type"),
]


class EligibilityStatus(StrEnum):
    eligible = "eligible"
    ineligible = "ineligible"
    unknown = "unknown"


class Eligibility(ContractModel):
    status: EligibilityStatus
    supporting_claim_ids: list[str] = Field(default_factory=list, max_length=8)
    reason: ShortText


class ObservationPattern(StrEnum):
    active_rate_drop = "active_rate_drop"
    idle_or_inactive_traffic = "idle_or_inactive_traffic"
    reported_radio_degradation = "reported_radio_degradation"
    stable_reported_radio = "stable_reported_radio"
    qualified_cell_transition = "qualified_cell_transition"
    insufficient_measurement_coverage = "insufficient_measurement_coverage"
    no_clear_symptom = "no_clear_symptom"


class HypothesisName(StrEnum):
    radio_degradation = "radio_degradation"
    shared_resource_load = "shared_resource_load"
    application_demand = "application_demand"
    transport_or_server = "transport_or_server"
    mobility_related = "mobility_related"
    measurement_artifact = "measurement_artifact"


class HypothesisStatus(StrEnum):
    consistent = "consistent"
    contradicted = "contradicted"
    not_testable = "not_testable"


class Hypothesis(ContractModel):
    hypothesis_id: HypothesisName
    status: HypothesisStatus
    supporting_claim_ids: list[str] = Field(default_factory=list, max_length=8)
    contradicting_claim_ids: list[str] = Field(default_factory=list, max_length=8)
    assumptions: list[ShortText] = Field(default_factory=list, max_length=4)
    discriminating_measurement: ShortText


class CauseStatus(StrEnum):
    unresolved = "unresolved"
    not_assessed = "not_assessed"


class InvestigationFindings(ContractModel):
    schema_version: int = CONTRACT_SCHEMA_VERSION
    event_id: str
    eligibility: Eligibility
    factual_claims: list[FactualClaim] = Field(max_length=16)
    observation_patterns: list[ObservationPattern] = Field(max_length=7)
    hypotheses: list[Hypothesis] = Field(max_length=6)
    missing_evidence: list[ShortText] = Field(max_length=8)
    next_measurements: list[ShortText] = Field(max_length=8)
    decision_summary: str = Field(max_length=1200)
    cause_status: CauseStatus
    assessment_scope: Literal["cited_windows", "whole_session"] | None = None

    @model_validator(mode="after")
    def check_claim_references(self) -> InvestigationFindings:
        claim_ids = [claim.claim_id for claim in self.factual_claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique within findings")
        known = set(claim_ids)
        referenced = set(self.eligibility.supporting_claim_ids)
        for hypothesis in self.hypotheses:
            referenced.update(hypothesis.supporting_claim_ids)
            referenced.update(hypothesis.contradicting_claim_ids)
        unknown = sorted(referenced - known)
        if unknown:
            raise ValueError(f"references to unknown claim ids: {unknown}")
        return self


class ReviewDecision(StrEnum):
    accept = "accept"
    request_more_evidence = "request_more_evidence"
    unresolved = "unresolved"


class ReviewResult(ContractModel):
    schema_version: int = CONTRACT_SCHEMA_VERSION
    decision: ReviewDecision
    disputed_claim_ids: list[str] = Field(default_factory=list, max_length=16)
    disputed_finding_paths: list[Annotated[str, Field(max_length=160)]] = Field(
        default_factory=list,
        max_length=16,
        description="JSON pointers to challenged findings sections.",
    )
    independent_evidence_ids: list[str] = Field(default_factory=list, max_length=16)
    alternative_checks: list[ShortText] = Field(default_factory=list, max_length=6)
    requested_queries: list[ShortText] = Field(default_factory=list, max_length=6)
    reason_summary: str = Field(max_length=1200)
    review_limitation: ShortText | None = None

    @field_validator("disputed_finding_paths")
    @classmethod
    def check_paths(cls, paths: list[str]) -> list[str]:
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate disputed finding paths")
        for path in paths:
            finding_path_tokens(path)
        return paths

    @model_validator(mode="after")
    def check_acceptance(self) -> ReviewResult:
        if self.decision == ReviewDecision.accept and self.disputed_finding_paths:
            raise ValueError("accept cannot also challenge a finding section")
        return self

    def validate_finding_paths(self, findings: dict[str, Any]) -> None:
        for path in self.disputed_finding_paths:
            finding_path_value(findings, path)


def finding_path_tokens(path: str) -> list[str]:
    """Decode a non-root RFC 6901 pointer; reject malformed escapes and empty tokens."""
    if not path.startswith("/"):
        raise ValueError("finding path must be a non-root JSON pointer")
    tokens = path[1:].split("/")
    for token in tokens:
        if not token or any(
            token[index] == "~" and (index + 1 == len(token) or token[index + 1] not in "01")
            for index in range(len(token))
        ):
            raise ValueError("malformed finding path")
    return [token.replace("~1", "/").replace("~0", "~") for token in tokens]


def finding_path_value(findings: dict[str, Any], path: str) -> Any:
    value: Any = findings
    for token in finding_path_tokens(path):
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif (
            isinstance(value, list)
            and token.isascii()
            and token.isdecimal()
            and str(int(token)) == token
            and int(token) < len(value)
        ):
            value = value[int(token)]
        else:
            raise ValueError("finding path does not resolve in the supplied findings")
    return value
