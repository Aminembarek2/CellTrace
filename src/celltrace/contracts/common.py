"""Enums and small value types shared by every contract."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_SCHEMA_VERSION = 1


class ContractModel(BaseModel):
    """Strict base: unknown fields and NaN or infinite floats are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MetricName(StrEnum):
    downlink_mbps = "downlink_mbps"
    uplink_mbps = "uplink_mbps"
    reported_rsrp_dbm = "reported_rsrp_dbm"
    reported_rsrq_db = "reported_rsrq_db"
    reported_snr_db = "reported_snr_db"
    reported_cqi = "reported_cqi"
    reported_rssi_dbm = "reported_rssi_dbm"
    reported_nrx_rsrp_dbm = "reported_nrx_rsrp_dbm"
    reported_nrx_rsrq_db = "reported_nrx_rsrq_db"
    speed_kph = "speed_kph"
    ping_average_ms = "ping_average_ms"


class Unit(StrEnum):
    megabits_per_second = "Mbps"
    decibel_milliwatt = "dBm"
    decibel = "dB"
    dimensionless_index = "index"
    kilometres_per_hour = "km/h"
    milliseconds = "ms"
    fraction = "fraction"


METRIC_UNITS: dict[MetricName, Unit] = {
    MetricName.downlink_mbps: Unit.megabits_per_second,
    MetricName.uplink_mbps: Unit.megabits_per_second,
    MetricName.reported_rsrp_dbm: Unit.decibel_milliwatt,
    MetricName.reported_rsrq_db: Unit.decibel,
    MetricName.reported_snr_db: Unit.decibel,
    MetricName.reported_cqi: Unit.dimensionless_index,
    MetricName.reported_rssi_dbm: Unit.decibel_milliwatt,
    MetricName.reported_nrx_rsrp_dbm: Unit.decibel_milliwatt,
    MetricName.reported_nrx_rsrq_db: Unit.decibel,
    MetricName.speed_kph: Unit.kilometres_per_hour,
    MetricName.ping_average_ms: Unit.milliseconds,
}

# Only rates get a relative change. Logarithmic quantities use additive differences.
RELATIVE_CHANGE_METRICS = frozenset({MetricName.downlink_mbps, MetricName.uplink_mbps})


class AggregationBasis(StrEnum):
    native = "native"
    bin_1s = "bin_1s"


class Statistic(StrEnum):
    mean = "mean"
    median = "median"
    minimum = "min"
    maximum = "max"
    p10 = "p10"
    p90 = "p90"


class DownloadState(StrEnum):
    active = "active"
    idle = "idle"
    unknown = "unknown"


class TransitionType(StrEnum):
    serving_cell_change = "serving_cell_change"
    reported_identity_change = "reported_identity_change"
    technology_change = "technology_change"


class IdentityBasis(StrEnum):
    eutran_cell_identity_layout = "eutran_cell_identity_layout"
    utran_cell_identity_layout = "utran_cell_identity_layout"
    unknown = "unknown"


class CapabilityStatus(StrEnum):
    supported = "supported"
    partial = "partial"
    unavailable = "unavailable"


class ToolName(StrEnum):
    get_session_context = "get_session_context"
    summarize_window = "summarize_window"
    compare_windows = "compare_windows"
    get_cell_transitions = "get_cell_transitions"
    get_activity_state = "get_activity_state"
    get_metric_definition = "get_metric_definition"
    get_evidence = "get_evidence"


MEASUREMENT_TOOLS = frozenset(
    {
        ToolName.summarize_window,
        ToolName.compare_windows,
        ToolName.get_cell_transitions,
        ToolName.get_activity_state,
    }
)


class ErrorCode(StrEnum):
    invalid_argument = "INVALID_ARGUMENT"
    out_of_scope = "OUT_OF_SCOPE"
    unsupported_metric = "UNSUPPORTED_METRIC"
    insufficient_data = "INSUFFICIENT_DATA"
    timeout = "TIMEOUT"
    store_unavailable = "STORE_UNAVAILABLE"
    result_limit = "RESULT_LIMIT"
    budget_exhausted = "BUDGET_EXHAUSTED"
    unknown_tool = "UNKNOWN_TOOL"
    invalid_output = "INVALID_OUTPUT"


RETRYABLE_ERROR_CODES = frozenset({ErrorCode.timeout, ErrorCode.store_unavailable})


class Role(StrEnum):
    investigator = "investigator"
    reviewer = "reviewer"
    verifier = "verifier"
    system = "system"


class Arm(StrEnum):
    investigator_raw = "investigator_raw"
    investigator_gated = "investigator_gated"
    investigator_reviewer_gated = "investigator_reviewer_gated"


class ExecutionMode(StrEnum):
    scripted_fixture = "scripted_fixture"
    live = "live"


class TimeWindow(ContractModel):
    """Half open interval [start_ms, end_ms) in session elapsed milliseconds."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def check_order(self) -> TimeWindow:
        if self.end_ms <= self.start_ms:
            raise ValueError("window end_ms must be greater than start_ms (half open, nonempty)")
        return self

    @property
    def length_ms(self) -> int:
        return self.end_ms - self.start_ms

    def contains(self, other: TimeWindow) -> bool:
        return self.start_ms <= other.start_ms and other.end_ms <= self.end_ms
