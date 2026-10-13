"""Past only feature rows and five second targets for forecasting observed downlink.

One row per (session, segment, as_of_ms) on a bin boundary of a file download
session. Every predictor is computed from a declared window of whole bins
[as_of + start_offset, as_of + end_offset) inside the same segment, and the
support columns of each row are measured from the bins those windows actually
read. A window that reaches into the future therefore produces a latest
availability later than as_of_ms, which assert_past_only rejects.

The target is the mean of the five one second downlink bin means over
[as_of, as_of + 5000). It is measured application traffic, not network capacity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from celltrace.analysis.transitions import fetch_cell_observations, qualify_transitions
from celltrace.canonical import content_hash
from celltrace.contracts.common import TransitionType
from celltrace.data.raca import BIN_MS

FEATURE_VERSION = "prediction_features_v1"
APPLICATION_COHORT = "file_download"
DEFAULT_HORIZON_MS = 5_000
DEFAULT_MAXIMUM_HISTORY_MS = 30_000

HISTORY = "history"
CONTEXT = "context"
FEATURE_GROUP_NAMES = ("history_only", "radio_context_only", "combined")
CATEGORICAL_FEATURES = ("network_mode_last",)

IDENTITY_COLUMNS = (
    "session_id",
    "segment_id",
    "as_of_ms",
    "earliest_support_ms",
    "latest_available_ms",
    "target_start_ms",
    "target_end_ms",
)
METADATA_COLUMNS = (
    *IDENTITY_COLUMNS,
    "mobility_cohort",
    "target_mbps",
    "target_active_fraction",
    "transition_in_window",
)
FORBIDDEN_PREDICTOR_TOKENS = (
    "session",
    "segment_id",
    "cell_id",
    "as_of",
    "target",
    "latitude",
    "longitude",
)

EXCLUSION_STAGES = (
    "candidate_rows",
    "excluded_target_window_extends_beyond_segment",
    "excluded_target_bin_absent",
    "excluded_target_bin_incomplete",
    "excluded_target_bin_without_valid_downlink",
    "excluded_no_valid_downlink_in_history",
    "feature_rows",
)

RADIO_METRICS = ("reported_rsrp_dbm", "reported_rsrq_db", "reported_snr_db", "reported_cqi")
_BIN_METRICS = ("downlink_mbps", "uplink_mbps", *RADIO_METRICS, "speed_kph")
_STATISTICS = (
    "value",
    "last_valid",
    "last_valid_age_s",
    "mean",
    "median",
    "std",
    "sum",
    "missing_bins",
    "active_fraction",
)


class PastOnlyViolation(ValueError):
    """A feature row reads data that is not available at its issuance time."""


@dataclass(frozen=True)
class WindowFeature:
    """One predictor computed over bins [as_of + start, as_of + end) in bin units."""

    name: str
    source: str
    statistic: str
    start_offset_bins: int
    end_offset_bins: int
    group: str

    def __post_init__(self) -> None:
        if self.end_offset_bins <= self.start_offset_bins:
            raise ValueError(f"{self.name}: window end must follow its start")
        if self.statistic not in _STATISTICS:
            raise ValueError(f"{self.name}: unknown statistic {self.statistic}")
        if self.group not in (HISTORY, CONTEXT):
            raise ValueError(f"{self.name}: unknown group {self.group}")


@dataclass
class FeatureBuild:
    frame: pd.DataFrame
    exclusions: dict[str, int]
    exclusions_by_session: dict[str, dict[str, int]]
    specs: tuple[WindowFeature, ...]
    horizon_ms: int
    maximum_history_ms: int
    feature_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def specification_hash(self) -> str:
        return feature_specification_hash(self.specs, self.horizon_ms, self.maximum_history_ms)


def default_feature_specs(
    maximum_history_ms: int = DEFAULT_MAXIMUM_HISTORY_MS,
) -> tuple[WindowFeature, ...]:
    history_bins = maximum_history_ms // BIN_MS
    specs: list[WindowFeature] = []
    for lag in (1, 2, 3, 5, 10):
        specs.append(
            WindowFeature(
                f"downlink_lag_{lag}_mbps", "downlink_mbps", "value", -lag, 1 - lag, HISTORY
            )
        )
    specs.append(
        WindowFeature(
            "downlink_last_valid_mbps", "downlink_mbps", "last_valid", -history_bins, 0, HISTORY
        )
    )
    specs.append(
        WindowFeature(
            "downlink_last_valid_age_s",
            "downlink_mbps",
            "last_valid_age_s",
            -history_bins,
            0,
            HISTORY,
        )
    )
    for seconds in (5, 10, 30):
        bins = min(seconds, history_bins)
        for statistic in ("mean", "median", "std", "missing_bins"):
            name = (
                f"downlink_missing_bins_{seconds}s"
                if statistic == "missing_bins"
                else f"downlink_{statistic}_{seconds}s_mbps"
            )
            specs.append(WindowFeature(name, "downlink_mbps", statistic, -bins, 0, HISTORY))
    specs.append(
        WindowFeature("history_bins_available_30s", "in_segment", "sum", -history_bins, 0, HISTORY)
    )
    for metric in RADIO_METRICS:
        specs.append(WindowFeature(f"{metric}_lag_1", metric, "value", -1, 0, CONTEXT))
        specs.append(WindowFeature(f"{metric}_mean_5s", metric, "mean", -5, 0, CONTEXT))
        specs.append(WindowFeature(f"{metric}_mean_30s", metric, "mean", -history_bins, 0, CONTEXT))
        specs.append(
            WindowFeature(f"{metric}_missing_bins_5s", metric, "missing_bins", -5, 0, CONTEXT)
        )
    specs.extend(
        [
            WindowFeature("uplink_lag_1_mbps", "uplink_mbps", "value", -1, 0, CONTEXT),
            WindowFeature("uplink_mean_5s_mbps", "uplink_mbps", "mean", -5, 0, CONTEXT),
            WindowFeature("speed_lag_1_kph", "speed_kph", "value", -1, 0, CONTEXT),
            WindowFeature("speed_mean_10s_kph", "speed_kph", "mean", -10, 0, CONTEXT),
            WindowFeature("active_fraction_5s", "active_count", "active_fraction", -5, 0, CONTEXT),
            WindowFeature(
                "active_fraction_30s", "active_count", "active_fraction", -history_bins, 0, CONTEXT
            ),
            WindowFeature(
                "serving_cell_changes_30s",
                "serving_cell_changes",
                "sum",
                -history_bins,
                0,
                CONTEXT,
            ),
            WindowFeature(
                "identity_transitions_30s",
                "identity_transitions",
                "sum",
                -history_bins,
                0,
                CONTEXT,
            ),
            WindowFeature(
                "network_mode_last", "network_mode", "last_valid", -history_bins, 0, CONTEXT
            ),
        ]
    )
    return tuple(specs)


def feature_groups(specs: Sequence[WindowFeature]) -> dict[str, tuple[str, ...]]:
    return {
        "history_only": tuple(spec.name for spec in specs if spec.group == HISTORY),
        "radio_context_only": tuple(spec.name for spec in specs if spec.group == CONTEXT),
        "combined": tuple(spec.name for spec in specs),
    }


def feature_specification_hash(
    specs: Sequence[WindowFeature], horizon_ms: int, maximum_history_ms: int
) -> str:
    return content_hash(
        {
            "feature_version": FEATURE_VERSION,
            "horizon_ms": horizon_ms,
            "maximum_history_ms": maximum_history_ms,
            "specs": [
                [
                    spec.name,
                    spec.source,
                    spec.statistic,
                    spec.start_offset_bins,
                    spec.end_offset_bins,
                    spec.group,
                ]
                for spec in specs
            ],
        }
    )


def check_predictor_names(names: Sequence[str]) -> None:
    """Refuse identifiers, targets and location columns as predictors."""
    for name in names:
        lowered = name.lower()
        for token in FORBIDDEN_PREDICTOR_TOKENS:
            if token in lowered:
                raise PastOnlyViolation(f"{name} is not an allowed predictor ({token})")


def assert_past_only(frame: pd.DataFrame, horizon_ms: int | None = None) -> None:
    """Raise if any row reads data after as_of_ms or its target does not start at as_of_ms."""
    missing = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing:
        raise PastOnlyViolation(f"feature frame lacks support columns {missing}")
    for column in IDENTITY_COLUMNS[2:]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not bool(np.isfinite(values).all()):
            raise PastOnlyViolation(
                f"feature support column {column} has missing or nonfinite times"
            )
    late = frame["latest_available_ms"] > frame["as_of_ms"]
    if bool(late.any()):
        example = frame.loc[late, ["session_id", "as_of_ms", "latest_available_ms"]].iloc[0]
        raise PastOnlyViolation(
            f"{int(late.sum())} rows read data available after as_of_ms, first {example.to_dict()}"
        )
    shifted = frame["target_start_ms"] != frame["as_of_ms"]
    if bool(shifted.any()):
        raise PastOnlyViolation(f"{int(shifted.sum())} rows have a target not starting at as_of")
    early_support = frame["earliest_support_ms"] > frame["latest_available_ms"]
    if bool(early_support.any()):
        raise PastOnlyViolation("earliest support follows latest availability")
    if horizon_ms is not None:
        length = frame["target_end_ms"] - frame["target_start_ms"]
        if bool((length != horizon_ms).any()):
            raise PastOnlyViolation("target window length differs from the horizon")


# loading


def _load_sessions(
    connection: duckdb.DuckDBPyConnection, session_ids: Sequence[str] | None
) -> dict[str, str]:
    rows = connection.execute(
        "SELECT session_id, mobility_cohort FROM sessions "
        "WHERE application_cohort = ? ORDER BY session_id",
        [APPLICATION_COHORT],
    ).fetchall()
    cohorts = {str(row[0]): str(row[1]) for row in rows}
    if session_ids is None:
        return cohorts
    unknown = sorted(set(session_ids) - set(cohorts))
    if unknown:
        raise ValueError(f"sessions not in the {APPLICATION_COHORT} cohort: {unknown}")
    return {session_id: cohorts[session_id] for session_id in sorted(session_ids)}


def _load_bins(connection: duckdb.DuckDBPyConnection, session_id: str) -> pd.DataFrame:
    metric_columns = ", ".join(
        f"CASE WHEN {metric}_valid_count > 0 THEN {metric} END AS {metric}"
        for metric in _BIN_METRICS
    )
    frame = connection.execute(
        f"""
        SELECT segment_id, bin_start_ms, available_at_ms, sample_count, active_count,
               is_complete, {metric_columns}
        FROM bins_1s WHERE session_id = ? ORDER BY bin_start_ms
        """,
        [session_id],
    ).df()
    modes = connection.execute(
        """
        SELECT bin_start_ms, network_mode FROM (
            SELECT (elapsed_ms // ?) * ? AS bin_start_ms, network_mode,
                   row_number() OVER (
                       PARTITION BY (elapsed_ms // ?)
                       ORDER BY elapsed_ms DESC, source_row_number DESC
                   ) AS position
            FROM samples WHERE session_id = ? AND network_mode IS NOT NULL
        ) WHERE position = 1
        """,
        [BIN_MS, BIN_MS, BIN_MS, session_id],
    ).df()
    return frame.merge(modes, on="bin_start_ms", how="left")


def _transition_counts(
    connection: duckdb.DuckDBPyConnection, session_id: str
) -> tuple[dict[int, int], dict[int, int]]:
    """Per later observation bin: serving cell changes and all qualified transitions."""
    serving: dict[int, int] = {}
    identity: dict[int, int] = {}
    for transition in qualify_transitions(fetch_cell_observations(connection, session_id)):
        bin_start = (transition.later_time_ms // BIN_MS) * BIN_MS
        identity[bin_start] = identity.get(bin_start, 0) + 1
        if transition.transition_type == TransitionType.serving_cell_change:
            serving[bin_start] = serving.get(bin_start, 0) + 1
    return serving, identity


# window computation


def _align(rolled: pd.Series, end_offset_bins: int) -> pd.Series:
    """Move a trailing rolling result at bin j to the row whose window ends at j."""
    return rolled.shift(-(end_offset_bins - 1))


def _window_values(grid: pd.DataFrame, spec: WindowFeature) -> pd.Series:
    length = spec.end_offset_bins - spec.start_offset_bins
    values = grid[spec.source]
    if spec.statistic in ("last_valid", "last_valid_age_s"):
        if spec.statistic == "last_valid_age_s":
            positions = np.where(values.notna(), np.arange(len(values), dtype=float), np.nan)
            values = pd.Series(positions, index=grid.index)
        rolled = values.ffill(limit=length - 1) if length > 1 else values
        aligned = _align(rolled, spec.end_offset_bins)
        if spec.statistic == "last_valid_age_s":
            row_positions = pd.Series(np.arange(len(values), dtype=float), index=grid.index)
            return (row_positions - aligned) * (BIN_MS / 1000)
        return aligned
    window = values.rolling(length, min_periods=1)
    if spec.statistic in ("value", "mean"):
        rolled = window.mean()
    elif spec.statistic == "median":
        rolled = window.median()
    elif spec.statistic == "std":
        rolled = window.std()
    elif spec.statistic == "sum":
        rolled = window.sum()
    elif spec.statistic == "missing_bins":
        in_segment = grid["in_segment"].rolling(length, min_periods=1).sum()
        valid = values.notna().astype(float).rolling(length, min_periods=1).sum()
        rolled = in_segment - valid
    else:
        active = grid["active_count"].rolling(length, min_periods=1).sum()
        samples = grid["sample_count"].rolling(length, min_periods=1).sum()
        rolled = active / samples.where(samples > 0)
    return _align(rolled, spec.end_offset_bins)


def _segment_grid(
    bins: pd.DataFrame, serving: dict[int, int], identity: dict[int, int]
) -> pd.DataFrame:
    """Dense one second grid of one segment. Bins absent from bins_1s become empty rows."""
    first = int(bins["bin_start_ms"].min())
    last = int(bins["bin_start_ms"].max())
    starts = np.arange(first, last + BIN_MS, BIN_MS, dtype=np.int64)
    grid = bins.set_index("bin_start_ms").reindex(starts)
    grid.index.name = "bin_start_ms"
    grid = grid.reset_index()
    grid["present"] = grid["available_at_ms"].notna().astype(float)
    grid["in_segment"] = 1.0
    grid["complete"] = (grid["is_complete"].fillna(False).astype(bool)).astype(float)
    grid["sample_count"] = grid["sample_count"].fillna(0).astype(float)
    grid["active_count"] = grid["active_count"].fillna(0).astype(float)
    grid["serving_cell_changes"] = [float(serving.get(int(start), 0)) for start in starts]
    grid["identity_transitions"] = [float(identity.get(int(start), 0)) for start in starts]
    grid["bin_start_value"] = grid["bin_start_ms"].astype(float)
    return grid


def _segment_rows(
    grid: pd.DataFrame,
    specs: Sequence[WindowFeature],
    horizon_bins: int,
    history_bins: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    count = len(grid)
    rows = pd.DataFrame({"as_of_ms": grid["bin_start_ms"].astype(np.int64)})
    for spec in specs:
        rows[spec.name] = _window_values(grid, spec).to_numpy()

    support_start = min(min(spec.start_offset_bins for spec in specs), -history_bins)
    support_end = max(spec.end_offset_bins for spec in specs)
    support_length = support_end - support_start
    latest = _align(
        grid["available_at_ms"].rolling(support_length, min_periods=1).max(), support_end
    )
    earliest = _align(
        grid["bin_start_value"].rolling(support_length, min_periods=1).min(), support_end
    )
    rows["latest_available_ms"] = latest.to_numpy()
    rows["earliest_support_ms"] = earliest.to_numpy()

    def target_sum(column: pd.Series) -> np.ndarray:
        rolled = column.rolling(horizon_bins, min_periods=horizon_bins).sum()
        return np.asarray(_align(rolled, horizon_bins).to_numpy(), dtype=float)

    in_range = np.arange(count) + horizon_bins <= count
    present = target_sum(grid["present"])
    complete = target_sum(grid["complete"])
    valid = target_sum(grid["downlink_mbps"].notna().astype(float))
    downlink_sum = target_sum(grid["downlink_mbps"].fillna(0.0))
    active = target_sum(grid["active_count"])
    samples = target_sum(grid["sample_count"])
    rows["target_mbps"] = downlink_sum / horizon_bins
    with np.errstate(divide="ignore", invalid="ignore"):
        rows["target_active_fraction"] = np.where(samples > 0, active / samples, np.nan)
    context = grid["identity_transitions"]
    past_transitions = _align(context.rolling(history_bins, min_periods=1).sum(), 0)
    future_transitions = target_sum(context.fillna(0.0)) if count else np.array([])
    rows["transition_in_window"] = (past_transitions.fillna(0.0).to_numpy() > 0) | (
        np.nan_to_num(future_transitions) > 0
    )

    keep = np.ones(count, dtype=bool)
    counts = {"candidate_rows": count}
    steps = (
        ("excluded_target_window_extends_beyond_segment", in_range),
        ("excluded_target_bin_absent", present >= horizon_bins),
        ("excluded_target_bin_incomplete", complete >= horizon_bins),
        ("excluded_target_bin_without_valid_downlink", valid >= horizon_bins),
        ("excluded_no_valid_downlink_in_history", rows["downlink_last_valid_mbps"].notna()),
    )
    for stage, passes in steps:
        passes_array = np.asarray(passes, dtype=bool)
        counts[stage] = int((keep & ~passes_array).sum())
        keep &= passes_array
    counts["feature_rows"] = int(keep.sum())
    return rows.loc[keep].reset_index(drop=True), counts


def build_feature_frame(
    connection: duckdb.DuckDBPyConnection,
    specs: Sequence[WindowFeature] | None = None,
    horizon_ms: int = DEFAULT_HORIZON_MS,
    maximum_history_ms: int = DEFAULT_MAXIMUM_HISTORY_MS,
    session_ids: Sequence[str] | None = None,
) -> FeatureBuild:
    """Feature rows for every eligible bin boundary of the file download sessions."""
    if horizon_ms <= 0 or maximum_history_ms <= 0:
        raise ValueError("horizon and history must be positive")
    if horizon_ms % BIN_MS or maximum_history_ms % BIN_MS:
        raise ValueError("horizon and history must be whole bins")
    chosen_specs = tuple(specs) if specs is not None else default_feature_specs(maximum_history_ms)
    names = [spec.name for spec in chosen_specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate feature names")
    check_predictor_names(names)
    horizon_bins = horizon_ms // BIN_MS
    history_bins = maximum_history_ms // BIN_MS
    cohorts = _load_sessions(connection, session_ids)
    totals = dict.fromkeys(EXCLUSION_STAGES, 0)
    by_session: dict[str, dict[str, int]] = {}
    parts: list[pd.DataFrame] = []
    for session_id, mobility in cohorts.items():
        bins = _load_bins(connection, session_id)
        serving, identity = _transition_counts(connection, session_id)
        session_counts = dict.fromkeys(EXCLUSION_STAGES, 0)
        for segment_id, segment_bins in bins.groupby("segment_id", sort=True):
            grid = _segment_grid(segment_bins.drop(columns="segment_id"), serving, identity)
            rows, counts = _segment_rows(grid, chosen_specs, horizon_bins, history_bins)
            for stage, value in counts.items():
                session_counts[stage] += value
            rows.insert(0, "segment_id", int(str(segment_id)))
            rows.insert(0, "session_id", session_id)
            rows["mobility_cohort"] = mobility
            parts.append(rows)
        by_session[session_id] = session_counts
        for stage, value in session_counts.items():
            totals[stage] += value
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not frame.empty:
        frame["target_start_ms"] = frame["as_of_ms"]
        frame["target_end_ms"] = frame["as_of_ms"] + horizon_ms
        frame["latest_available_ms"] = frame["latest_available_ms"].astype(np.int64)
        frame["earliest_support_ms"] = frame["earliest_support_ms"].astype(np.int64)
        frame["transition_in_window"] = frame["transition_in_window"].astype(bool)
        ordered = [*METADATA_COLUMNS, *names]
        frame = frame[ordered].sort_values(["session_id", "as_of_ms"]).reset_index(drop=True)
    return FeatureBuild(
        frame=frame,
        exclusions=totals,
        exclusions_by_session=by_session,
        specs=chosen_specs,
        horizon_ms=horizon_ms,
        maximum_history_ms=maximum_history_ms,
        feature_groups=feature_groups(chosen_specs),
    )
