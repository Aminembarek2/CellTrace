"""Evidence records, tool errors and the immutable run context."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from celltrace.contracts.common import (
    CONTRACT_SCHEMA_VERSION,
    AggregationBasis,
    CapabilityStatus,
    ContractModel,
    ErrorCode,
    TimeWindow,
    ToolName,
)


class ToolError(ContractModel):
    code: ErrorCode
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class EvidenceRecord(ContractModel):
    schema_version: int = CONTRACT_SCHEMA_VERSION
    evidence_id: str
    run_id: str
    snapshot_hash: str
    dictionary_hash: str
    tool_name: ToolName
    tool_version: str
    query_template_hash: str
    validated_arguments: dict[str, Any]
    requested_bounds: list[TimeWindow]
    effective_bounds: list[TimeWindow]
    aggregation_basis: AggregationBasis | None
    units: dict[str, str]
    row_counts: dict[str, int]
    missing_counts: dict[str, int]
    quality_flags: list[str]
    result: dict[str, Any]
    content_sha256: str
    created_at: str
    cache_key: str
    cache_hit: bool = False


class Capability(ContractModel):
    status: CapabilityStatus
    reason: str


class CandidateEvent(ContractModel):
    """Derived detector output. Lives outside the measurement snapshot."""

    event_id: str
    session_id: str
    segment_id: int
    event_window: TimeWindow
    representative_reference_window: TimeWindow
    representative_event_window: TimeWindow
    detection_time_ms: int
    detector_name: str
    detector_version: str
    detector_config_hash: str
    reference_median_mbps: float
    event_median_mbps: float
    absolute_drop_mbps: float
    relative_drop: float
    flag_count: int
    first_confirmation_ms: int
    last_confirmation_ms: int
    continues_event_id: str | None = None


class RunContext(ContractModel):
    """Built by code. The model can read parts of it but never change it."""

    run_id: str
    case_id: str
    session_id: str
    snapshot_hash: str
    snapshot_path: str
    dictionary_hash: str
    segment_id: int
    allowed_window: TimeWindow
    requested_allowed_window: TimeWindow
    candidate: CandidateEvent
    analysis_mode: str = "retrospective"
    capabilities: dict[str, Capability]
    maximum_tool_window_ms: int
    maximum_result_bytes: int
