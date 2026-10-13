"""Construct the immutable run context for one candidate event."""

from __future__ import annotations

from celltrace.config import RuntimeSection
from celltrace.contracts.common import CapabilityStatus, TimeWindow
from celltrace.contracts.evidence import CandidateEvent, Capability, RunContext
from celltrace.data.dictionary import dictionary_hash
from celltrace.tools.repository import MeasurementRepository


class ContextError(ValueError):
    pass


def build_run_context(
    run_id: str,
    case_id: str,
    candidate: CandidateEvent,
    snapshot_hash: str,
    snapshot_path: str,
    repository: MeasurementRepository,
    runtime: RuntimeSection,
) -> RunContext:
    metadata = repository.metadata()
    snapshot_dictionary = metadata.get("dictionary_hash")
    if snapshot_dictionary != dictionary_hash():
        raise ContextError(
            "snapshot dictionary hash differs from the running code, re-ingest or pin the code"
        )
    segment = repository.segment(candidate.session_id, candidate.segment_id)
    if segment is None:
        raise ContextError("candidate segment is not present in the snapshot")
    requested = TimeWindow(
        start_ms=max(candidate.event_window.start_ms - runtime.evidence_radius_before_ms, 0),
        end_ms=candidate.event_window.end_ms + runtime.evidence_radius_after_ms,
    )
    effective_start = max(requested.start_ms, segment[0])
    effective_end = min(requested.end_ms, segment[1])
    if effective_end <= effective_start:
        raise ContextError("allowed window is empty after clipping to the segment")
    capabilities = {
        name: Capability(status=CapabilityStatus(entry["status"]), reason=entry["reason"])
        for name, entry in repository.capabilities().items()
    }
    return RunContext(
        run_id=run_id,
        case_id=case_id,
        session_id=candidate.session_id,
        snapshot_hash=snapshot_hash,
        snapshot_path=snapshot_path,
        dictionary_hash=dictionary_hash(),
        segment_id=candidate.segment_id,
        allowed_window=TimeWindow(start_ms=effective_start, end_ms=effective_end),
        requested_allowed_window=requested,
        candidate=candidate,
        capabilities=capabilities,
        maximum_tool_window_ms=runtime.maximum_tool_window_ms,
        maximum_result_bytes=runtime.maximum_result_bytes,
    )
