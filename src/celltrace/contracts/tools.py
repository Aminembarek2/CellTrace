"""Arguments a model may send to each measurement tool."""

from __future__ import annotations

from pydantic import Field, model_validator

from celltrace.contracts.common import (
    AggregationBasis,
    ContractModel,
    MetricName,
    Statistic,
    TimeWindow,
    ToolName,
)

MAXIMUM_METRICS_PER_REQUEST = 6


class GetSessionContextArguments(ContractModel):
    session_id: str | None = Field(
        default=None, description="Optional. Must match the session of this run if given."
    )


class SummarizeWindowArguments(ContractModel):
    start_ms: int = Field(ge=0, description="Inclusive window start in session elapsed ms.")
    end_ms: int = Field(ge=0, description="Exclusive window end in session elapsed ms.")
    metrics: list[MetricName] = Field(min_length=1, max_length=MAXIMUM_METRICS_PER_REQUEST)
    basis: AggregationBasis = Field(
        description="native summarizes logged samples, bin_1s summarizes one second bin means."
    )

    @model_validator(mode="after")
    def check_window(self) -> SummarizeWindowArguments:
        TimeWindow(start_ms=self.start_ms, end_ms=self.end_ms)
        return self

    @property
    def window(self) -> TimeWindow:
        return TimeWindow(start_ms=self.start_ms, end_ms=self.end_ms)


class CompareWindowsArguments(ContractModel):
    reference_start_ms: int = Field(ge=0)
    reference_end_ms: int = Field(ge=0)
    event_start_ms: int = Field(ge=0)
    event_end_ms: int = Field(ge=0)
    metrics: list[MetricName] = Field(min_length=1, max_length=MAXIMUM_METRICS_PER_REQUEST)
    basis: AggregationBasis
    statistic: Statistic = Field(description="Statistic compared between the two windows.")

    @model_validator(mode="after")
    def check_windows(self) -> CompareWindowsArguments:
        TimeWindow(start_ms=self.reference_start_ms, end_ms=self.reference_end_ms)
        TimeWindow(start_ms=self.event_start_ms, end_ms=self.event_end_ms)
        return self

    @property
    def reference_window(self) -> TimeWindow:
        return TimeWindow(start_ms=self.reference_start_ms, end_ms=self.reference_end_ms)

    @property
    def event_window(self) -> TimeWindow:
        return TimeWindow(start_ms=self.event_start_ms, end_ms=self.event_end_ms)


class GetCellTransitionsArguments(ContractModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def check_window(self) -> GetCellTransitionsArguments:
        TimeWindow(start_ms=self.start_ms, end_ms=self.end_ms)
        return self

    @property
    def window(self) -> TimeWindow:
        return TimeWindow(start_ms=self.start_ms, end_ms=self.end_ms)


class GetActivityStateArguments(ContractModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    basis: AggregationBasis

    @model_validator(mode="after")
    def check_window(self) -> GetActivityStateArguments:
        TimeWindow(start_ms=self.start_ms, end_ms=self.end_ms)
        return self

    @property
    def window(self) -> TimeWindow:
        return TimeWindow(start_ms=self.start_ms, end_ms=self.end_ms)


class GetMetricDefinitionArguments(ContractModel):
    metric: MetricName


class GetEvidenceArguments(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=64)


TOOL_ARGUMENT_MODELS: dict[ToolName, type[ContractModel]] = {
    ToolName.get_session_context: GetSessionContextArguments,
    ToolName.summarize_window: SummarizeWindowArguments,
    ToolName.compare_windows: CompareWindowsArguments,
    ToolName.get_cell_transitions: GetCellTransitionsArguments,
    ToolName.get_activity_state: GetActivityStateArguments,
    ToolName.get_metric_definition: GetMetricDefinitionArguments,
    ToolName.get_evidence: GetEvidenceArguments,
}

TOOL_DESCRIPTIONS: dict[ToolName, str] = {
    ToolName.get_session_context: (
        "Allowed time bounds for this run, session cohorts, sampling cadence and gap audit, "
        "and which measurements are supported, partial or unavailable."
    ),
    ToolName.summarize_window: (
        "Count, valid and missing counts, coverage, units and min/max/mean/median/p10/p90 "
        "for metrics inside one half open window. Missing values are never zero."
    ),
    ToolName.compare_windows: (
        "Summaries of a reference and an event window plus the signed event minus reference "
        "difference. Relative change is given only for rates with a sufficiently observed, "
        "nonzero reference. dB and dBm metrics use additive differences."
    ),
    ToolName.get_cell_transitions: (
        "Qualified changes of reported serving cell identity inside a window, with both "
        "observation times, the gap and the identity basis. A change is a proxy, not proof "
        "of a completed handover."
    ),
    ToolName.get_activity_state: (
        "Counts and fractions of active, idle and unknown application download state with "
        "denominators, plus state change times."
    ),
    ToolName.get_metric_definition: (
        "Recorded definition of a metric: raw field, raw unit, conversion, audit status and "
        "known limitations. This does not prove a measurement occurred."
    ),
    ToolName.get_evidence: "Return an evidence record created earlier in this same run.",
}
