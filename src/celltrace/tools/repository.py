"""Read only measurement queries over an identified DuckDB snapshot.

Column identifiers come only from the ``MetricName`` allowlist because
identifiers cannot be bound as parameters. Every value, including session ids
and bounds, is a bound parameter. No method accepts SQL text.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from celltrace.canonical import sha256_text
from celltrace.contracts.common import (
    METRIC_UNITS,
    AggregationBasis,
    DownloadState,
    MetricName,
    TimeWindow,
)
from celltrace.data.metric_applicability import (
    REFERENCE_SIGNAL_METRICS,
    MetricApplicability,
    reference_signal_supported,
)

NATIVE_CADENCE_MS = 1_000
MAXIMUM_STATE_CHANGES = 40

_SUMMARY_NATIVE_TEMPLATE = """
SELECT count(*) AS row_count,
       count({column}) AS valid_count,
       min({column}), max({column}), avg({column}), median({column}),
       quantile_cont({column}, 0.1), quantile_cont({column}, 0.9),
       count_if(list_contains(quality_flags, ?)) AS flagged_count
FROM samples
WHERE session_id = ? AND elapsed_ms >= ? AND elapsed_ms < ?
"""

_SUMMARY_BIN_TEMPLATE = """
SELECT count(*) AS row_count,
       count({column}) AS valid_count,
       min({column}), max({column}), avg({column}), median({column}),
       quantile_cont({column}, 0.1), quantile_cont({column}, 0.9),
       0 AS flagged_count
FROM bins_1s
WHERE session_id = ? AND bin_start_ms >= ? AND bin_end_ms <= ?
"""

_METRIC_APPLICABILITY = """
SELECT network_mode, count(*) FROM samples
WHERE session_id = ? AND elapsed_ms >= ? AND elapsed_ms < ?
GROUP BY network_mode
"""

_ACTIVITY_NATIVE = """
SELECT download_state, count(*) FROM samples
WHERE session_id = ? AND elapsed_ms >= ? AND elapsed_ms < ?
GROUP BY download_state
"""

_ACTIVITY_BINS = """
SELECT bin_start_ms, sample_count, active_count, idle_count, unknown_count FROM bins_1s
WHERE session_id = ? AND bin_start_ms >= ? AND bin_end_ms <= ?
ORDER BY bin_start_ms
"""

_STATE_SEQUENCE = """
SELECT elapsed_ms, download_state FROM samples
WHERE session_id = ? AND elapsed_ms >= ? AND elapsed_ms < ?
ORDER BY elapsed_ms, source_row_number
"""

_CELL_OBSERVATIONS = """
SELECT elapsed_ms, segment_id, serving_cell_id, cell_identity_basis, network_mode FROM samples
WHERE session_id = ? AND elapsed_ms >= ? AND elapsed_ms < ?
ORDER BY elapsed_ms, source_row_number
"""

_LEFT_CONTEXT = """
SELECT elapsed_ms, segment_id, serving_cell_id, cell_identity_basis, network_mode FROM samples
WHERE session_id = ? AND elapsed_ms >= ? AND elapsed_ms < ? AND serving_cell_id IS NOT NULL
ORDER BY elapsed_ms DESC, source_row_number DESC
LIMIT 1
"""

_DROP_WINDOW = """
SELECT count(*), count(downlink_mbps), median(downlink_mbps),
       coalesce(sum(sample_count), 0), coalesce(sum(active_count), 0),
       count_if(is_complete), count_if(available_at_ms <= ?), count_if(segment_id = ?)
FROM bins_1s
WHERE session_id = ? AND bin_start_ms >= ? AND bin_end_ms <= ?
"""

QUERY_TEMPLATE_HASHES: dict[str, str] = {
    name: sha256_text(template)
    for name, template in {
        "summary_native": _SUMMARY_NATIVE_TEMPLATE,
        "summary_bin_1s": _SUMMARY_BIN_TEMPLATE,
        "metric_applicability": _METRIC_APPLICABILITY,
        "activity_native": _ACTIVITY_NATIVE,
        "activity_bin_1s": _ACTIVITY_BINS,
        "state_sequence": _STATE_SEQUENCE,
        "cell_observations": _CELL_OBSERVATIONS + _LEFT_CONTEXT,
        "drop_window": _DROP_WINDOW,
    }.items()
}

_ALLOWED_COLUMNS = frozenset(metric.value for metric in MetricName)


def _column(metric: MetricName) -> str:
    column = MetricName(metric).value
    if column not in _ALLOWED_COLUMNS:  # defensive, enum construction already validates
        raise ValueError(f"metric {metric} is not in the column allowlist")
    return column


def expected_units(window: TimeWindow) -> int:
    """Whole one second bins (or nominal samples) that fit inside a half open window."""
    first_bin = -(-window.start_ms // NATIVE_CADENCE_MS)
    last_bin_exclusive = window.end_ms // NATIVE_CADENCE_MS
    return max(last_bin_exclusive - first_bin, 0)


def _round(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    unit: str
    basis: str
    row_count: int
    valid_count: int
    missing_count: int
    expected_count: int
    coverage: float | None
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    p10: float | None
    p90: float | None
    outside_documented_range_count: int
    applicability: MetricApplicability | None = None

    def statistic(self, name: str) -> float | None:
        mapping = {
            "mean": self.mean,
            "median": self.median,
            "min": self.minimum,
            "max": self.maximum,
            "p10": self.p10,
            "p90": self.p90,
        }
        return mapping[name]

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "metric": self.metric,
            "unit": self.unit,
            "basis": self.basis,
            "row_count": self.row_count,
            "valid_count": self.valid_count,
            "missing_count": self.missing_count,
            "expected_count": self.expected_count,
            "coverage": self.coverage,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "p10": self.p10,
            "p90": self.p90,
            "outside_documented_range_count": self.outside_documented_range_count,
        }
        if self.applicability is not None:
            result["measurement_applicability"] = self.applicability.as_dict()
        return result


@dataclass(frozen=True)
class DropWindowSupport:
    expected_bins: int
    observed_bins: int
    valid_bins: int
    complete_bins: int
    available_bins: int
    same_segment_bins: int
    sample_count: int
    active_count: int
    median_mbps: float | None

    @property
    def coverage(self) -> float | None:
        return self.valid_bins / self.expected_bins if self.expected_bins else None

    @property
    def active_fraction(self) -> float | None:
        return self.active_count / self.sample_count if self.sample_count else None


class MeasurementRepository:
    """Typed queries against one read only snapshot connection."""

    def __init__(self, connection: Any):
        self.connection = connection

    def drop_window(
        self, session_id: str, segment_id: int, window: TimeWindow, available_by_ms: int
    ) -> DropWindowSupport:
        """Recompute detector support without trusting the candidate's stored medians."""
        row = self.connection.execute(
            _DROP_WINDOW,
            [available_by_ms, segment_id, session_id, window.start_ms, window.end_ms],
        ).fetchone()
        return DropWindowSupport(
            expected_bins=expected_units(window),
            observed_bins=int(row[0]),
            valid_bins=int(row[1]),
            median_mbps=_round(row[2]),
            sample_count=int(row[3]),
            active_count=int(row[4]),
            complete_bins=int(row[5] or 0),
            available_bins=int(row[6] or 0),
            same_segment_bins=int(row[7] or 0),
        )

    def summarize(
        self,
        session_id: str,
        window: TimeWindow,
        metric: MetricName,
        basis: AggregationBasis,
    ) -> MetricSummary:
        column = _column(metric)
        if basis == AggregationBasis.native:
            sql = _SUMMARY_NATIVE_TEMPLATE.format(column=column)
            parameters: list[Any] = [
                f"outside_documented_range:{column}",
                session_id,
                window.start_ms,
                window.end_ms,
            ]
        else:
            sql = _SUMMARY_BIN_TEMPLATE.format(column=column)
            parameters = [session_id, window.start_ms, window.end_ms]
        row = self.connection.execute(sql, parameters).fetchone()
        applicability = None
        if metric in REFERENCE_SIGNAL_METRICS:
            # Check only rows contributing to the requested aggregation basis.
            start = window.start_ms
            end = window.end_ms
            if basis == AggregationBasis.bin_1s:
                start = -(-start // NATIVE_CADENCE_MS) * NATIVE_CADENCE_MS
                end = end // NATIVE_CADENCE_MS * NATIVE_CADENCE_MS
            modes = self.connection.execute(
                _METRIC_APPLICABILITY, [session_id, start, end]
            ).fetchall()
            applicability = MetricApplicability(
                sample_count=sum(int(count) for _, count in modes),
                unsupported_sample_count=sum(
                    int(count) for mode, count in modes if not reference_signal_supported([mode])
                ),
                observed_modes=tuple(sorted((mode for mode, _ in modes), key=lambda x: x or "")),
            )
        row_count, valid_count = int(row[0]), int(row[1])
        expected = expected_units(window)
        missing = row_count - valid_count
        coverage = None if expected == 0 else valid_count / expected
        return MetricSummary(
            metric=column,
            unit=METRIC_UNITS[metric].value,
            basis=basis.value,
            row_count=row_count,
            valid_count=valid_count,
            missing_count=missing,
            expected_count=expected,
            coverage=coverage,
            minimum=_round(row[2]),
            maximum=_round(row[3]),
            mean=_round(row[4]),
            median=_round(row[5]),
            p10=_round(row[6]),
            p90=_round(row[7]),
            outside_documented_range_count=int(row[8]),
            applicability=applicability,
        )

    def activity(
        self, session_id: str, window: TimeWindow, basis: AggregationBasis
    ) -> dict[str, Any]:
        if basis == AggregationBasis.native:
            rows = self.connection.execute(
                _ACTIVITY_NATIVE, [session_id, window.start_ms, window.end_ms]
            ).fetchall()
            counts = {state.value: 0 for state in DownloadState}
            for state, count in rows:
                counts[str(state)] = int(count)
            denominator = sum(counts.values())
            empty_units = 0
        else:
            rows = self.connection.execute(
                _ACTIVITY_BINS, [session_id, window.start_ms, window.end_ms]
            ).fetchall()
            counts = {state.value: 0 for state in DownloadState}
            empty_units = 0
            for _, sample_count, active, idle, unknown in rows:
                if sample_count == 0:
                    empty_units += 1
                    continue
                counts[majority_state(int(active), int(idle), int(unknown))] += 1
            denominator = sum(counts.values())
        fractions = {
            state: (None if denominator == 0 else count / denominator)
            for state, count in counts.items()
        }
        changes, truncated = self.state_changes(session_id, window)
        return {
            "basis": basis.value,
            "counts": counts,
            "fractions": fractions,
            "denominator": denominator,
            "units_without_samples": empty_units,
            "expected_units": expected_units(window),
            "state_changes": changes,
            "state_changes_truncated": truncated,
        }

    def state_changes(
        self, session_id: str, window: TimeWindow
    ) -> tuple[list[dict[str, Any]], bool]:
        rows = self.connection.execute(
            _STATE_SEQUENCE, [session_id, window.start_ms, window.end_ms]
        ).fetchall()
        changes: list[dict[str, Any]] = []
        for previous, current in pairwise(rows):
            if previous[1] != current[1]:
                changes.append(
                    {
                        "earlier_time_ms": int(previous[0]),
                        "later_time_ms": int(current[0]),
                        "from_state": previous[1],
                        "to_state": current[1],
                    }
                )
        return changes[:MAXIMUM_STATE_CHANGES], len(changes) > MAXIMUM_STATE_CHANGES

    def cell_observations(
        self, session_id: str, window: TimeWindow, context_window: TimeWindow
    ) -> tuple[list[tuple[Any, ...]], tuple[Any, ...] | None]:
        """Observations inside ``window`` and the last valid one before it within context."""
        rows = self.connection.execute(
            _CELL_OBSERVATIONS, [session_id, window.start_ms, window.end_ms]
        ).fetchall()
        left = None
        if context_window.start_ms < window.start_ms:
            left = self.connection.execute(
                _LEFT_CONTEXT, [session_id, context_window.start_ms, window.start_ms]
            ).fetchone()
        return [tuple(row) for row in rows], (None if left is None else tuple(left))

    def session(self, session_id: str) -> dict[str, Any] | None:
        cursor = self.connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?", [session_id]
        )
        row = cursor.fetchone()
        if row is None:
            return None
        names = [description[0] for description in cursor.description]
        return dict(zip(names, row, strict=True))

    def segment(self, session_id: str, segment_id: int) -> tuple[int, int] | None:
        row = self.connection.execute(
            "SELECT start_ms, end_ms FROM segments WHERE session_id = ? AND segment_id = ?",
            [session_id, segment_id],
        ).fetchone()
        return None if row is None else (int(row[0]), int(row[1]))

    def capabilities(self) -> dict[str, dict[str, str]]:
        rows = self.connection.execute(
            "SELECT capability, status, reason FROM capabilities ORDER BY capability"
        ).fetchall()
        return {row[0]: {"status": row[1], "reason": row[2]} for row in rows}

    def metadata(self) -> dict[str, str]:
        rows = self.connection.execute("SELECT key, value FROM snapshot_metadata").fetchall()
        return {str(row[0]): str(row[1]) for row in rows}


def majority_state(active: int, idle: int, unknown: int) -> str:
    """State holding most samples in a bin. Ties resolve to unknown."""
    counts = {"active": active, "idle": idle, "unknown": unknown}
    best = max(counts.values())
    winners = [state for state, count in counts.items() if count == best]
    return winners[0] if len(winners) == 1 else "unknown"
