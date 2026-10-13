"""Tool dispatch: validation, case scope, execution and evidence creation.

The dispatcher never receives SQL or paths from a model. It checks every
argument against the immutable ``RunContext`` before touching the snapshot,
and a failure returns a ``ToolError`` instead of a partial result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from celltrace.analysis.transitions import CellObservation, qualify_transitions
from celltrace.canonical import canonical_json, content_hash
from celltrace.clock import Clock
from celltrace.contracts.common import (
    RELATIVE_CHANGE_METRICS,
    AggregationBasis,
    ErrorCode,
    MetricName,
    TimeWindow,
    ToolName,
)
from celltrace.contracts.evidence import EvidenceRecord, RunContext, ToolError
from celltrace.contracts.tools import (
    TOOL_ARGUMENT_MODELS,
    CompareWindowsArguments,
    GetActivityStateArguments,
    GetCellTransitionsArguments,
    GetEvidenceArguments,
    GetMetricDefinitionArguments,
    GetSessionContextArguments,
    SummarizeWindowArguments,
)
from celltrace.data.dictionary import METRIC_DICTIONARY
from celltrace.data.metric_applicability import APPLICABILITY_VERSION, REFERENCE_SIGNAL_METRICS
from celltrace.storage.run_store import RunStore
from celltrace.tools.repository import QUERY_TEMPLATE_HASHES, MeasurementRepository

TOOL_VERSION = "tools-2"
MAXIMUM_TRANSITIONS = 30
MINIMUM_REFERENCE_COVERAGE_FOR_RELATIVE = 0.5

METRIC_CAPABILITY: dict[MetricName, str] = {
    MetricName.downlink_mbps: "rate_summaries",
    MetricName.uplink_mbps: "rate_summaries",
    MetricName.reported_rsrp_dbm: "radio_comparison",
    MetricName.reported_rsrq_db: "radio_comparison",
    MetricName.reported_snr_db: "radio_comparison",
    MetricName.reported_cqi: "radio_comparison",
    MetricName.reported_rssi_dbm: "radio_comparison",
    MetricName.reported_nrx_rsrp_dbm: "neighbour_metrics",
    MetricName.reported_nrx_rsrq_db: "neighbour_metrics",
    MetricName.speed_kph: "mobility",
}


@dataclass(frozen=True)
class ToolOutcome:
    tool_name: str
    evidence: EvidenceRecord | None = None
    retrieved: dict[str, Any] | None = None
    error: ToolError | None = None
    cache_hit: bool = False
    physical_query: bool = False

    def model_payload(self) -> dict[str, Any]:
        """What the model sees as the tool result."""
        if self.error is not None:
            return {"error": self.error.model_dump(mode="json")}
        if self.retrieved is not None:
            return {"evidence_record": self.retrieved}
        assert self.evidence is not None
        return {
            "evidence_id": self.evidence.evidence_id,
            "tool_name": self.evidence.tool_name.value,
            "effective_bounds": [
                bound.model_dump(mode="json") for bound in self.evidence.effective_bounds
            ],
            "aggregation_basis": self.evidence.aggregation_basis,
            "units": self.evidence.units,
            "quality_flags": self.evidence.quality_flags,
            "result": self.evidence.result,
        }


def _error(code: ErrorCode, message: str, **details: Any) -> ToolError:
    return ToolError(code=code, message=message, retryable=False, details=details)


class ToolDispatcher:
    def __init__(
        self,
        context: RunContext,
        repository: MeasurementRepository,
        run_store: RunStore,
        clock: Clock,
    ):
        self.context = context
        self.repository = repository
        self.run_store = run_store
        self.clock = clock

    # Public entry point

    def dispatch(self, tool_name: str, raw_arguments: dict[str, Any]) -> ToolOutcome:
        try:
            tool = ToolName(tool_name)
        except ValueError:
            return ToolOutcome(
                tool_name=tool_name,
                error=_error(ErrorCode.unknown_tool, f"unknown tool {tool_name!r}"),
            )
        try:
            arguments = TOOL_ARGUMENT_MODELS[tool].model_validate(raw_arguments)
        except ValidationError as validation_error:
            problems = [
                {"location": list(map(str, problem["loc"])), "message": problem["msg"]}
                for problem in validation_error.errors()
            ]
            return ToolOutcome(
                tool_name=tool.value,
                error=_error(
                    ErrorCode.invalid_argument, "arguments failed validation", problems=problems
                ),
            )

        if tool == ToolName.get_evidence:
            assert isinstance(arguments, GetEvidenceArguments)
            return self._get_evidence(arguments)

        scope_error = self._check_scope(arguments)
        if scope_error is not None:
            return ToolOutcome(tool_name=tool.value, error=scope_error)

        validated = arguments.model_dump(mode="json")
        cache_key = content_hash(
            {
                "snapshot_hash": self.context.snapshot_hash,
                "tool": tool.value,
                "tool_version": TOOL_VERSION,
                "arguments": validated,
                "scope": {
                    "session_id": self.context.session_id,
                    "allowed_window": self.context.allowed_window.model_dump(),
                },
            }
        )
        cached = self.run_store.evidence_by_cache_key(self.context.run_id, cache_key)
        if cached is not None:
            return ToolOutcome(
                tool_name=tool.value,
                evidence=EvidenceRecord.model_validate(cached),
                cache_hit=True,
            )

        try:
            payload = self._execute(tool, arguments)
        except ToolExecutionError as failure:
            return ToolOutcome(tool_name=tool.value, error=failure.error, physical_query=True)

        evidence = self._make_evidence(tool, validated, cache_key, payload)
        size = len(canonical_json(evidence.model_dump(mode="json")).encode())
        if size > self.context.maximum_result_bytes:
            return ToolOutcome(
                tool_name=tool.value,
                error=_error(
                    ErrorCode.result_limit,
                    "result exceeds the configured size limit, request a narrower window",
                    result_bytes=size,
                    limit=self.context.maximum_result_bytes,
                ),
                physical_query=True,
            )
        return ToolOutcome(tool_name=tool.value, evidence=evidence, physical_query=True)

    # Scope

    def _windows(self, arguments: Any) -> list[TimeWindow]:
        if isinstance(arguments, CompareWindowsArguments):
            return [arguments.reference_window, arguments.event_window]
        if isinstance(
            arguments,
            SummarizeWindowArguments | GetCellTransitionsArguments | GetActivityStateArguments,
        ):
            return [arguments.window]
        return []

    def _check_scope(self, arguments: Any) -> ToolError | None:
        if isinstance(arguments, GetSessionContextArguments):
            if arguments.session_id not in (None, self.context.session_id):
                return _error(ErrorCode.out_of_scope, "only this run's session is available")
            return None
        allowed = self.context.allowed_window
        for window in self._windows(arguments):
            if not allowed.contains(window):
                return _error(
                    ErrorCode.out_of_scope,
                    "window is outside the allowed bounds for this run",
                    requested=window.model_dump(),
                    allowed=allowed.model_dump(),
                )
            if window.length_ms > self.context.maximum_tool_window_ms:
                return _error(
                    ErrorCode.out_of_scope,
                    "window is longer than the maximum tool window",
                    requested_ms=window.length_ms,
                    maximum_ms=self.context.maximum_tool_window_ms,
                )
        for metric in getattr(arguments, "metrics", []) or []:
            capability = METRIC_CAPABILITY.get(metric)
            if capability is None:
                continue
            status = self.context.capabilities.get(capability)
            if status is not None and status.status.value == "unavailable":
                return _error(
                    ErrorCode.unsupported_metric,
                    f"{metric.value} is unavailable for this source",
                    capability=capability,
                    reason=status.reason,
                )
        return None

    # Execution

    def _execute(self, tool: ToolName, arguments: Any) -> dict[str, Any]:
        handlers: dict[ToolName, Callable[[Any], dict[str, Any]]] = {
            ToolName.get_session_context: self._session_context,
            ToolName.summarize_window: self._summarize_window,
            ToolName.compare_windows: self._compare_windows,
            ToolName.get_cell_transitions: self._cell_transitions,
            ToolName.get_activity_state: self._activity_state,
            ToolName.get_metric_definition: self._metric_definition,
        }
        try:
            return handlers[tool](arguments)
        except ToolExecutionError:
            raise
        except Exception as failure:  # snapshot or driver failure
            raise ToolExecutionError(
                ToolError(
                    code=ErrorCode.store_unavailable,
                    message=f"measurement store failed: {type(failure).__name__}",
                    retryable=True,
                )
            ) from failure

    def _partial_flags(self, metrics: list[MetricName]) -> list[str]:
        flags = []
        for metric in metrics:
            capability = METRIC_CAPABILITY.get(metric)
            status = self.context.capabilities.get(capability or "")
            if status is not None and status.status.value == "partial":
                flags.append(f"capability_partial:{capability}")
        audit_status = {
            metric.value: METRIC_DICTIONARY[metric].audit_status
            for metric in metrics
            if METRIC_DICTIONARY[metric].audit_status != "verified"
        }
        flags += [
            f"definition_{status}:{metric}" for metric, status in sorted(audit_status.items())
        ]
        return sorted(set(flags))

    def _session_context(self, arguments: GetSessionContextArguments) -> dict[str, Any]:
        session = self.repository.session(self.context.session_id)
        if session is None:
            raise ToolExecutionError(_error(ErrorCode.insufficient_data, "session not in snapshot"))
        candidate = self.context.candidate
        allowed = self.context.allowed_window
        availability = {}
        for metric in MetricName:
            summary = self.repository.summarize(
                self.context.session_id, allowed, metric, AggregationBasis.native
            )
            availability[metric.value] = {
                "valid_count": summary.valid_count,
                "missing_count": summary.missing_count,
                "unit": summary.unit,
            }
            if summary.applicability is not None:
                availability[metric.value]["measurement_applicability"] = (
                    summary.applicability.as_dict()
                )
        return {
            "run_mode": self.context.analysis_mode,
            "session_id": self.context.session_id,
            "segment_id": self.context.segment_id,
            "allowed_window": allowed.model_dump(),
            "allowed_window_was_clipped": allowed != self.context.requested_allowed_window,
            "maximum_tool_window_ms": self.context.maximum_tool_window_ms,
            "application_cohort": session["application_cohort"],
            "mobility_cohort": session["mobility_cohort"],
            "timestamp_basis": session["timestamp_basis"],
            "native_cadence_ms": 1000,
            "same_second_row_count": session["same_second_row_count"],
            "segment_count": session["segment_count"],
            "candidate": {
                "event_id": candidate.event_id,
                "event_window": candidate.event_window.model_dump(),
                "representative_reference_window": (
                    candidate.representative_reference_window.model_dump()
                ),
                "representative_event_window": candidate.representative_event_window.model_dump(),
                "detection_time_ms": candidate.detection_time_ms,
                "detector": f"{candidate.detector_name} {candidate.detector_version}",
            },
            "capabilities": {
                name: capability.model_dump(mode="json")
                for name, capability in sorted(self.context.capabilities.items())
            },
            "metric_availability_in_allowed_window": availability,
        }

    def _summarize_window(self, arguments: SummarizeWindowArguments) -> dict[str, Any]:
        summaries = {
            metric.value: self.repository.summarize(
                self.context.session_id, arguments.window, metric, arguments.basis
            ).as_dict()
            for metric in arguments.metrics
        }
        return {"window": arguments.window.model_dump(), "summaries": summaries}

    def _compare_windows(self, arguments: CompareWindowsArguments) -> dict[str, Any]:
        comparisons = {}
        for metric in arguments.metrics:
            reference = self.repository.summarize(
                self.context.session_id, arguments.reference_window, metric, arguments.basis
            )
            event = self.repository.summarize(
                self.context.session_id, arguments.event_window, metric, arguments.basis
            )
            comparisons[metric.value] = compare_summaries(
                metric, reference.as_dict(), event.as_dict(), arguments.statistic.value
            )
        return {
            "reference_window": arguments.reference_window.model_dump(),
            "event_window": arguments.event_window.model_dump(),
            "statistic": arguments.statistic.value,
            "comparisons": comparisons,
        }

    def _cell_transitions(self, arguments: GetCellTransitionsArguments) -> dict[str, Any]:
        result = transitions_in_window(
            self.repository, self.context.session_id, arguments.window, self.context.allowed_window
        )
        if len(result["transitions"]) > MAXIMUM_TRANSITIONS:
            result["transitions"] = result["transitions"][:MAXIMUM_TRANSITIONS]
            result["truncated"] = True
        return result

    def _activity_state(self, arguments: GetActivityStateArguments) -> dict[str, Any]:
        result = self.repository.activity(
            self.context.session_id, arguments.window, arguments.basis
        )
        result["window"] = arguments.window.model_dump()
        return result

    def _metric_definition(self, arguments: GetMetricDefinitionArguments) -> dict[str, Any]:
        entry = METRIC_DICTIONARY[arguments.metric]
        return {
            "metric": entry.metric,
            "raw_field": entry.raw_field,
            "raw_unit": entry.raw_unit,
            "canonical_unit": entry.canonical_unit,
            "conversion": entry.conversion,
            "technology_applicability": (
                "LTE and 5G rows only; measurement source on other modes is undocumented"
                if arguments.metric in REFERENCE_SIGNAL_METRICS
                else entry.technology_applicability
            ),
            "runtime_applicability_version": APPLICABILITY_VERSION,
            "missing_sentinels": list(entry.missing_sentinels),
            "suspected_fill_values": list(entry.suspected_fill_values),
            "documented_range": None
            if entry.documented_range is None
            else list(entry.documented_range),
            "definition_source": entry.definition_source,
            "audit_status": entry.audit_status,
            "limitations": entry.limitations,
            "note": (
                "A definition lookup does not show that a measurement occurred. Populated values "
                "on unsupported rows remain raw logger observations and cannot establish "
                "radio facts."
            ),
        }

    def _get_evidence(self, arguments: GetEvidenceArguments) -> ToolOutcome:
        record = self.run_store.get_evidence(arguments.evidence_id)
        if record is None or record.get("run_id") != self.context.run_id:
            # Same error for unknown ids and other runs so nothing leaks about other cases.
            return ToolOutcome(
                tool_name=ToolName.get_evidence.value,
                error=_error(ErrorCode.out_of_scope, "no evidence with that id in this run"),
            )
        return ToolOutcome(tool_name=ToolName.get_evidence.value, retrieved=record)

    # Evidence

    def _make_evidence(
        self,
        tool: ToolName,
        validated_arguments: dict[str, Any],
        cache_key: str,
        payload: dict[str, Any],
    ) -> EvidenceRecord:
        windows = self._windows(TOOL_ARGUMENT_MODELS[tool].model_validate(validated_arguments))
        basis = validated_arguments.get("basis")
        metrics = [MetricName(metric) for metric in validated_arguments.get("metrics", [])]
        units, row_counts, missing_counts = summary_counts(payload)
        flags = self._partial_flags(metrics)
        if payload.get("truncated") or payload.get("state_changes_truncated"):
            flags.append("result_truncated")
        if payload.get("left_context_unavailable"):
            flags.append("left_context_unavailable")
        body: dict[str, Any] = {
            "schema_version": 1,
            "evidence_id": f"ev_{content_hash([self.context.run_id, cache_key])[:16]}",
            "run_id": self.context.run_id,
            "snapshot_hash": self.context.snapshot_hash,
            "dictionary_hash": self.context.dictionary_hash,
            "tool_name": tool.value,
            "tool_version": TOOL_VERSION,
            "query_template_hash": content_hash(QUERY_TEMPLATE_HASHES),
            "validated_arguments": validated_arguments,
            "requested_bounds": [window.model_dump() for window in windows],
            "effective_bounds": [window.model_dump() for window in windows],
            "aggregation_basis": basis,
            "units": units,
            "row_counts": row_counts,
            "missing_counts": missing_counts,
            "quality_flags": sorted(set(flags)),
            "result": payload,
            "cache_key": cache_key,
        }
        body["content_sha256"] = evidence_content_hash(body)
        body["created_at"] = self.clock.utc_now_iso()
        body["cache_hit"] = False
        return EvidenceRecord.model_validate(body)


class ToolExecutionError(Exception):
    def __init__(self, error: ToolError):
        super().__init__(error.message)
        self.error = error


HASH_EXCLUDED_FIELDS = ("content_sha256", "created_at", "cache_hit")


def evidence_content_hash(record: dict[str, Any]) -> str:
    return content_hash(
        {key: value for key, value in record.items() if key not in HASH_EXCLUDED_FIELDS}
    )


def compare_summaries(
    metric: MetricName, reference: dict[str, Any], event: dict[str, Any], statistic: str
) -> dict[str, Any]:
    reference_value = reference[statistic]
    event_value = event[statistic]
    delta = None
    if reference_value is not None and event_value is not None:
        delta = event_value - reference_value
    relative_decrease = None
    relative_reason = "not_a_rate_metric"
    if metric in RELATIVE_CHANGE_METRICS:
        coverage = reference["coverage"] or 0.0
        if reference_value is None or event_value is None:
            relative_reason = "missing_value"
        elif reference_value <= 0:
            relative_reason = "reference_not_positive"
        elif coverage < MINIMUM_REFERENCE_COVERAGE_FOR_RELATIVE:
            relative_reason = "reference_insufficiently_observed"
        else:
            relative_decrease = (reference_value - event_value) / reference_value
            relative_reason = "computed"
    return {
        "unit": reference["unit"],
        "reference": reference,
        "event": event,
        "reference_value": reference_value,
        "event_value": event_value,
        "delta_event_minus_reference": delta,
        "relative_decrease": relative_decrease,
        "relative_decrease_status": relative_reason,
    }


def summary_counts(
    payload: dict[str, Any],
) -> tuple[dict[str, str], dict[str, int], dict[str, int]]:
    units: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    summaries = payload.get("summaries")
    if summaries is None and "comparisons" in payload:
        summaries = {}
        for metric, comparison in payload["comparisons"].items():
            summaries[f"{metric}:reference"] = comparison["reference"]
            summaries[f"{metric}:event"] = comparison["event"]
    for name, summary in (summaries or {}).items():
        units[name.split(":")[0]] = summary["unit"]
        row_counts[name] = summary["row_count"]
        missing_counts[name] = summary["missing_count"]
    if "counts" in payload:
        row_counts["activity_denominator"] = payload["denominator"]
    return units, row_counts, missing_counts


def transitions_in_window(
    repository: MeasurementRepository,
    session_id: str,
    window: TimeWindow,
    allowed_window: TimeWindow,
) -> dict[str, Any]:
    """Qualified transitions whose later observation falls inside ``window``.

    The last valid observation before the window is used as left context only
    when it lies inside the run's allowed window.
    """
    rows, left = repository.cell_observations(session_id, window, allowed_window)
    observations = [
        CellObservation(
            elapsed_ms=int(row[0]),
            segment_id=int(row[1]),
            serving_cell_id=row[2],
            cell_identity_basis=row[3],
            network_mode=row[4],
        )
        for row in ([left] if left is not None else []) + rows
    ]
    transitions = [
        transition
        for transition in qualify_transitions(observations)
        if window.start_ms <= transition.later_time_ms < window.end_ms
    ]
    valid_in_window = sum(1 for row in rows if row[2] is not None)
    return {
        "window": window.model_dump(),
        "left_context_unavailable": left is None,
        "left_context_time_ms": None if left is None else int(left[0]),
        "observations_in_window": len(rows),
        "valid_identity_observations_in_window": valid_in_window,
        "transitions": [transition_payload(transition) for transition in transitions],
        "truncated": False,
        "note": (
            "A reported identity change is a transition proxy. It does not show that a "
            "handover completed or caused any rate change."
        ),
    }


def transition_payload(transition: Any) -> dict[str, Any]:
    return {
        "transition_type": transition.transition_type.value,
        "earlier_cell_id": transition.earlier_cell_id,
        "later_cell_id": transition.later_cell_id,
        "earlier_time_ms": transition.earlier_time_ms,
        "later_time_ms": transition.later_time_ms,
        "gap_ms": transition.gap_ms,
        "identity_basis": transition.identity_basis.value,
        "earlier_network_mode": transition.earlier_network_mode,
        "later_network_mode": transition.later_network_mode,
        "uncertainty": list(transition.uncertainty),
    }
