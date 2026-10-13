"""SQL backed exploration summaries and figures for one snapshot.

Figures plot native samples with lines broken at segment gaps and missing
values, keep incompatible units on separate axes, and align windows on
qualified serving cell changes together with a transition free control and a
deterministically chosen counterexample.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from celltrace.analysis.transitions import (
    QualifiedTransition,
    fetch_cell_observations,
    qualify_transitions,
)
from celltrace.contracts.common import MetricName, TransitionType
from celltrace.data.raca import BIN_MS
from celltrace.data.schema import METRIC_COLUMNS
from celltrace.data.snapshot import open_snapshot

SUMMARY_NAME = "explore_summary.json"
CAPABILITY_MATRIX_NAME = "capability_matrix.json"
TIMELINE_FIGURE = "rate_activity_timeline.png"
RADIO_FIGURE = "radio_rate_small_multiples.png"
TRANSITION_FIGURE = "transition_aligned_windows.png"
ALIGN_BEFORE_MS = 30_000
ALIGN_AFTER_MS = 30_000
COUNTEREXAMPLE_SIDE_MS = 10_000
COUNTEREXAMPLE_MAXIMUM_DECLINE = 0.1
COUNTEREXAMPLE_MINIMUM_REFERENCE_MBPS = 1.0
COUNTEREXAMPLE_MINIMUM_ACTIVE_FRACTION = 0.8
PRIMARY_COHORT = "file_download"
CONTROL_STEP_MS = 10_000
CONTROL_MINIMUM_COVERAGE = 0.8

RADIO_PANELS: tuple[tuple[str, str], ...] = (
    (MetricName.downlink_mbps.value, "downlink (Mbps)"),
    (MetricName.reported_rsrp_dbm.value, "reported RSRP (dBm)"),
    (MetricName.reported_rsrq_db.value, "reported RSRQ (dB)"),
    (MetricName.reported_snr_db.value, "reported SNR (dB)"),
    (MetricName.reported_cqi.value, "reported CQI (index)"),
)


@dataclass(frozen=True)
class AlignedWindow:
    session_id: str
    anchor_ms: int
    offsets_s: np.ndarray
    downlink: np.ndarray
    active_fraction: np.ndarray
    rsrp: np.ndarray


def _rows_as_dicts(
    connection: duckdb.DuckDBPyConnection, query: str, parameters: list[Any] | None = None
) -> list[dict[str, Any]]:
    cursor = connection.execute(query, parameters or [])
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    return value


def _session_summaries(connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    valid_columns = ",\n".join(
        f"count(samples.{metric.value}) AS {metric.value}_valid" for metric in METRIC_COLUMNS
    )
    rows = _rows_as_dicts(
        connection,
        f"""
        SELECT sessions.session_id, application_cohort, mobility_cohort, raw_row_count,
               accepted_row_count, quarantined_row_count, segment_count, duration_ms,
               same_second_row_count, reordered_row_count,
               count(samples.session_id) FILTER (WHERE download_state = 'active') AS active_rows,
               count(samples.session_id) FILTER (WHERE download_state = 'idle') AS idle_rows,
               count(samples.session_id) FILTER (WHERE download_state = 'unknown') AS unknown_rows,
               {valid_columns}
        FROM sessions LEFT JOIN samples USING (session_id)
        GROUP BY ALL ORDER BY sessions.session_id
        """,
    )
    cadence = _rows_as_dicts(
        connection,
        """
        SELECT session_id, interval_ms, count(*) AS interval_count FROM (
            SELECT session_id,
                   elapsed_ms - lag(elapsed_ms) OVER (
                       PARTITION BY session_id ORDER BY elapsed_ms, source_row_number
                   ) AS interval_ms
            FROM samples)
        WHERE interval_ms IS NOT NULL
        GROUP BY ALL ORDER BY session_id, interval_ms
        """,
    )
    cadence_by_session: dict[str, dict[str, int]] = {}
    for item in cadence:
        cadence_by_session.setdefault(item["session_id"], {})[str(item["interval_ms"])] = int(
            item["interval_count"]
        )
    summaries: list[dict[str, Any]] = []
    for row in rows:
        accepted = int(row["accepted_row_count"])
        summaries.append(
            {
                "session_id": row["session_id"],
                "application_cohort": row["application_cohort"],
                "mobility_cohort": row["mobility_cohort"],
                "raw_row_count": int(row["raw_row_count"]),
                "accepted_row_count": accepted,
                "quarantined_row_count": int(row["quarantined_row_count"]),
                "segment_count": int(row["segment_count"]),
                "duration_ms": int(row["duration_ms"]),
                "same_second_row_count": int(row["same_second_row_count"]),
                "reordered_row_count": int(row["reordered_row_count"]),
                "state_counts": {
                    "active": int(row["active_rows"]),
                    "idle": int(row["idle_rows"]),
                    "unknown": int(row["unknown_rows"]),
                },
                "missing_fraction": {
                    metric.value: (
                        round(1 - int(row[f"{metric.value}_valid"]) / accepted, 6)
                        if accepted
                        else None
                    )
                    for metric in METRIC_COLUMNS
                },
                "interval_counts_ms": cadence_by_session.get(row["session_id"], {}),
            }
        )
    return summaries


def _cohort_summaries(connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    cohorts = _rows_as_dicts(
        connection,
        """
        SELECT application_cohort, mobility_cohort, count(*) AS sessions,
               sum(raw_row_count) AS raw_rows, sum(accepted_row_count) AS accepted_rows,
               sum(duration_ms) AS duration_ms, sum(segment_count) AS segments
        FROM sessions GROUP BY ALL ORDER BY ALL
        """,
    )
    states = _rows_as_dicts(
        connection,
        """
        SELECT application_cohort, mobility_cohort, download_state, count(*) AS sample_count,
               count(downlink_mbps) AS downlink_valid,
               avg(downlink_mbps) AS downlink_mean_mbps,
               median(downlink_mbps) AS downlink_median_mbps,
               quantile_cont(downlink_mbps, 0.9) AS downlink_p90_mbps
        FROM samples JOIN sessions USING (session_id)
        GROUP BY ALL ORDER BY ALL
        """,
    )
    by_cohort: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for state in states:
        key = (state.pop("application_cohort"), state.pop("mobility_cohort"))
        by_cohort.setdefault(key, []).append(state)
    return [
        {
            **{
                key: (int(value) if isinstance(value, int | float) else value)
                for key, value in cohort.items()
            },
            "downlink_by_state": by_cohort.get(
                (cohort["application_cohort"], cohort["mobility_cohort"]), []
            ),
        }
        for cohort in cohorts
    ]


def _transitions(
    connection: duckdb.DuckDBPyConnection, sessions: list[dict[str, Any]]
) -> dict[str, list[QualifiedTransition]]:
    return {
        session["session_id"]: qualify_transitions(
            fetch_cell_observations(connection, session["session_id"])
        )
        for session in sessions
    }


def _session_series(
    connection: duckdb.DuckDBPyConnection, session_id: str
) -> dict[str, np.ndarray]:
    columns = ", ".join(name for name, _ in RADIO_PANELS)
    rows = connection.execute(
        f"""
        SELECT elapsed_ms, segment_id, download_state, {columns}
        FROM samples WHERE session_id = ? ORDER BY elapsed_ms, source_row_number
        """,
        [session_id],
    ).fetchall()
    series: dict[str, np.ndarray] = {
        "elapsed_s": np.array([row[0] / 1000 for row in rows], dtype=float),
        "segment_id": np.array([row[1] for row in rows], dtype=int),
        "state": np.array([row[2] for row in rows], dtype=object),
    }
    for position, (name, _) in enumerate(RADIO_PANELS, start=3):
        series[name] = np.array(
            [np.nan if row[position] is None else float(row[position]) for row in rows],
            dtype=float,
        )
    return series


def _segments(connection: duckdb.DuckDBPyConnection, session_id: str) -> list[tuple[int, int, int]]:
    rows = connection.execute(
        "SELECT segment_id, start_ms, end_ms FROM segments "
        "WHERE session_id = ? ORDER BY segment_id",
        [session_id],
    ).fetchall()
    return [(int(row[0]), int(row[1]), int(row[2])) for row in rows]


def _empty_bins(connection: duckdb.DuckDBPyConnection, session_id: str) -> np.ndarray:
    rows = connection.execute(
        "SELECT bin_start_ms FROM bins_1s WHERE session_id = ? AND sample_count = 0 "
        "ORDER BY bin_start_ms",
        [session_id],
    ).fetchall()
    return np.array([row[0] / 1000 for row in rows], dtype=float)


def _plot_by_segment(axis: Any, series: dict[str, np.ndarray], name: str, **style: Any) -> None:
    for segment_id in np.unique(series["segment_id"]):
        mask = series["segment_id"] == segment_id
        axis.plot(series["elapsed_s"][mask], series[name][mask], **style)
        style.pop("label", None)


def _shade_gaps(axes: list[Any], segments: list[tuple[int, int, int]]) -> int:
    gap_count = 0
    for (_, _, previous_end), (_, next_start, _) in pairwise(segments):
        gap_count += 1
        for position, axis in enumerate(axes):
            axis.axvspan(
                previous_end / 1000,
                next_start / 1000,
                color="tab:red",
                alpha=0.25,
                linewidth=0,
                label="segment gap above 3 s" if position == 0 and gap_count == 1 else None,
            )
            axis.axvline(previous_end / 1000, color="tab:red", linewidth=0.8, linestyle=":")
    return gap_count


def _choose_timeline_session(sessions: list[dict[str, Any]]) -> dict[str, Any] | None:
    download = [item for item in sessions if item["application_cohort"] == PRIMARY_COHORT]
    pool = download or sessions
    if not pool:
        return None
    return sorted(
        pool,
        key=lambda item: (-item["segment_count"], -item["accepted_row_count"], item["session_id"]),
    )[0]


def _timeline_figure(
    connection: duckdb.DuckDBPyConnection, session: dict[str, Any], path: Path
) -> dict[str, Any]:
    session_id = session["session_id"]
    series = _session_series(connection, session_id)
    segments = _segments(connection, session_id)
    empty = _empty_bins(connection, session_id)
    figure, (rate_axis, state_axis) = plt.subplots(
        2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    _plot_by_segment(
        rate_axis,
        series,
        MetricName.downlink_mbps.value,
        color="tab:blue",
        linewidth=0.7,
        marker=".",
        markersize=2,
        label="native downlink samples",
    )
    if empty.size:
        rate_axis.plot(
            empty + 0.5,
            np.zeros_like(empty),
            linestyle="none",
            marker="|",
            color="tab:orange",
            markersize=8,
            label="second without a sample (no line bridges it)",
        )
    gap_count = _shade_gaps([rate_axis, state_axis], segments)
    rate_axis.set_ylabel("downlink (Mbps)")
    rate_axis.set_title(
        f"Downlink rate and download state, session {session_id} "
        f"({session['application_cohort']}, {session['mobility_cohort']}), "
        f"{len(segments)} segment(s), {int(empty.size)} empty bin(s)"
    )
    rate_axis.legend(loc="upper right", fontsize=8)
    rate_axis.grid(alpha=0.3)
    state_codes = {"unknown": 0, "idle": 1, "active": 2}
    state_colors = {"unknown": "tab:gray", "idle": "tab:orange", "active": "tab:green"}
    for state, code in state_codes.items():
        mask = series["state"] == state
        if mask.any():
            state_axis.plot(
                series["elapsed_s"][mask],
                np.full(int(mask.sum()), code),
                linestyle="none",
                marker="s",
                markersize=2,
                color=state_colors[state],
                label=state,
            )
    state_axis.set_yticks(list(state_codes.values()), list(state_codes.keys()))
    state_axis.set_ylim(-0.6, 2.6)
    state_axis.set_ylabel("download state\n(category, no unit)")
    state_axis.set_xlabel("elapsed time since session start (s)")
    state_axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return {
        "file": path.name,
        "session_id": session_id,
        "selection": "file download session with the most segments, then the most accepted "
        "rows, then the smallest session_id",
        "segment_gaps_shown": gap_count,
        "empty_bins_shown": int(empty.size),
    }


def _radio_figure(
    connection: duckdb.DuckDBPyConnection,
    session: dict[str, Any],
    transitions: list[QualifiedTransition],
    path: Path,
) -> dict[str, Any]:
    session_id = session["session_id"]
    series = _session_series(connection, session_id)
    segments = _segments(connection, session_id)
    figure, axes = plt.subplots(len(RADIO_PANELS), 1, figsize=(13, 11), sharex=True)
    for axis, (name, label) in zip(axes, RADIO_PANELS, strict=True):
        _plot_by_segment(axis, series, name, color="tab:blue", linewidth=0.7)
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    _shade_gaps(list(axes), segments)
    changes = [
        item for item in transitions if item.transition_type is TransitionType.serving_cell_change
    ]
    others = [
        item
        for item in transitions
        if item.transition_type is not TransitionType.serving_cell_change
    ]
    for axis in axes:
        for position, item in enumerate(changes):
            axis.axvline(
                item.later_time_ms / 1000,
                color="black",
                linewidth=0.6,
                alpha=0.6,
                label="serving cell change (proxy)" if position == 0 else None,
            )
        for position, item in enumerate(others):
            axis.axvline(
                item.later_time_ms / 1000,
                color="tab:purple",
                linewidth=0.8,
                linestyle="--",
                label="technology or unverified identity change" if position == 0 else None,
            )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(
        f"Rate and reported radio values on separate axes, session {session_id}. "
        "Breaks in a line are missing values or gaps."
    )
    axes[-1].set_xlabel("elapsed time since session start (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return {
        "file": path.name,
        "session_id": session_id,
        "panels": [label for _, label in RADIO_PANELS],
        "serving_cell_changes_marked": len(changes),
        "other_transitions_marked": len(others),
    }


def _bin_lookup(
    connection: duckdb.DuckDBPyConnection, session_id: str
) -> dict[int, tuple[int, float | None, float | None, float | None]]:
    rows = connection.execute(
        """
        SELECT bin_start_ms, segment_id, downlink_mbps, active_fraction, reported_rsrp_dbm
        FROM bins_1s WHERE session_id = ? ORDER BY bin_start_ms
        """,
        [session_id],
    ).fetchall()
    return {int(row[0]): (int(row[1]), row[2], row[3], row[4]) for row in rows}


def _aligned_window(
    session_id: str,
    anchor_ms: int,
    bins: dict[int, tuple[int, float | None, float | None, float | None]],
) -> AlignedWindow | None:
    anchor_bin = (anchor_ms // BIN_MS) * BIN_MS
    if anchor_bin not in bins:
        return None
    segment_id = bins[anchor_bin][0]
    offsets = np.arange(-ALIGN_BEFORE_MS, ALIGN_AFTER_MS, BIN_MS)
    downlink, active, rsrp = (np.full(offsets.size, np.nan) for _ in range(3))
    for position, offset in enumerate(offsets):
        entry = bins.get(anchor_bin + int(offset))
        if entry is None or entry[0] != segment_id:
            continue
        for target, value in zip((downlink, active, rsrp), entry[1:], strict=True):
            if value is not None:
                target[position] = float(value)
    return AlignedWindow(session_id, anchor_ms, offsets / 1000, downlink, active, rsrp)


def _side_statistics(window: AlignedWindow, before: bool) -> tuple[float | None, float | None, int]:
    if before:
        mask = (window.offsets_s >= -COUNTEREXAMPLE_SIDE_MS / 1000) & (window.offsets_s < 0)
    else:
        mask = (window.offsets_s >= 0) & (window.offsets_s < COUNTEREXAMPLE_SIDE_MS / 1000)
    rates = window.downlink[mask]
    activity = window.active_fraction[mask]
    valid = rates[~np.isnan(rates)]
    if valid.size == 0:
        return None, None, 0
    active_valid = activity[~np.isnan(activity)]
    return (
        float(np.median(valid)),
        float(np.mean(active_valid)) if active_valid.size else None,
        int(valid.size),
    )


def _choose_counterexample(windows: list[AlignedWindow]) -> dict[str, Any] | None:
    expected = COUNTEREXAMPLE_SIDE_MS // BIN_MS
    options: list[tuple[float, str, int, dict[str, Any]]] = []
    for window in windows:
        before_median, before_active, before_count = _side_statistics(window, before=True)
        after_median, _, after_count = _side_statistics(window, before=False)
        if before_count < expected or after_count < expected:
            continue
        if before_median is None or after_median is None or before_active is None:
            continue
        if before_median < COUNTEREXAMPLE_MINIMUM_REFERENCE_MBPS:
            continue
        if before_active < COUNTEREXAMPLE_MINIMUM_ACTIVE_FRACTION:
            continue
        change = (after_median - before_median) / before_median
        if change < -COUNTEREXAMPLE_MAXIMUM_DECLINE:
            continue
        options.append(
            (
                abs(change),
                window.session_id,
                window.anchor_ms,
                {
                    "session_id": window.session_id,
                    "later_time_ms": window.anchor_ms,
                    "before_median_mbps": before_median,
                    "after_median_mbps": after_median,
                    "relative_change": change,
                },
            )
        )
    if not options:
        return None
    options.sort(key=lambda option: option[:3])
    return options[0][3]


def _control_windows(
    sessions: list[str],
    transitions: dict[str, list[QualifiedTransition]],
    bin_tables: dict[str, dict[int, tuple[int, float | None, float | None, float | None]]],
) -> list[AlignedWindow]:
    """Non overlapping windows with no qualified transition of any type inside them.

    Anchors move in CONTROL_STEP_MS steps. A window needs at least
    CONTROL_MINIMUM_COVERAGE of its bins with a downlink value in one segment.
    """
    expected = (ALIGN_BEFORE_MS + ALIGN_AFTER_MS) // BIN_MS
    controls: list[AlignedWindow] = []
    for session_id in sessions:
        bins = bin_tables[session_id]
        if not bins:
            continue
        times = [item.later_time_ms for item in transitions[session_id]]
        last = max(bins)
        anchor = ALIGN_BEFORE_MS
        while anchor + ALIGN_AFTER_MS <= last + BIN_MS:
            quiet = not any(
                anchor - ALIGN_BEFORE_MS <= time < anchor + ALIGN_AFTER_MS for time in times
            )
            window = _aligned_window(session_id, anchor, bins) if quiet else None
            if window is not None:
                valid = int(np.sum(~np.isnan(window.downlink)))
                if valid >= CONTROL_MINIMUM_COVERAGE * expected:
                    controls.append(window)
                    anchor += ALIGN_BEFORE_MS + ALIGN_AFTER_MS
                    continue
            anchor += CONTROL_STEP_MS
    return controls


def _transition_figure(
    connection: duckdb.DuckDBPyConnection,
    sessions: list[dict[str, Any]],
    transitions: dict[str, list[QualifiedTransition]],
    path: Path,
) -> dict[str, Any]:
    download_ids = [
        item["session_id"] for item in sessions if item["application_cohort"] == PRIMARY_COHORT
    ]
    bin_tables = {session_id: _bin_lookup(connection, session_id) for session_id in download_ids}
    windows: list[AlignedWindow] = []
    for session_id in download_ids:
        for item in transitions[session_id]:
            if item.transition_type is not TransitionType.serving_cell_change:
                continue
            window = _aligned_window(session_id, item.later_time_ms, bin_tables[session_id])
            if window is not None:
                windows.append(window)
    counterexample = _choose_counterexample(windows)
    controls = _control_windows(download_ids, transitions, bin_tables)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    panels = (
        ("downlink", "downlink bin mean (Mbps)"),
        ("active_fraction", "active fraction of samples\n(fraction, 0 to 1)"),
        ("rsrp", "reported RSRP bin mean (dBm)"),
    )
    for axis, (attribute, label) in zip(axes, panels, strict=True):
        for window in windows:
            axis.plot(
                window.offsets_s,
                getattr(window, attribute),
                color="tab:gray",
                alpha=0.15,
                linewidth=0.6,
            )
        if windows:
            stacked = np.vstack([getattr(window, attribute) for window in windows])
            counts = np.sum(~np.isnan(stacked), axis=0)
            with np.errstate(all="ignore"):
                median = np.array(
                    [
                        np.nanmedian(stacked[:, column]) if counts[column] else np.nan
                        for column in range(stacked.shape[1])
                    ]
                )
            axis.plot(
                windows[0].offsets_s,
                median,
                color="black",
                linewidth=2,
                label="median over serving cell changes",
            )
        if controls:
            control_stack = np.vstack([getattr(window, attribute) for window in controls])
            with np.errstate(all="ignore"):
                control_median = np.array(
                    [
                        np.nanmedian(control_stack[:, column])
                        if np.any(~np.isnan(control_stack[:, column]))
                        else np.nan
                        for column in range(control_stack.shape[1])
                    ]
                )
            axis.plot(
                controls[0].offsets_s,
                control_median,
                color="tab:green",
                linestyle="--",
                linewidth=1.5,
                label="median over transition free controls",
            )
        if counterexample is not None:
            match = next(
                window
                for window in windows
                if window.session_id == counterexample["session_id"]
                and window.anchor_ms == counterexample["later_time_ms"]
            )
            axis.plot(
                match.offsets_s,
                getattr(match, attribute),
                color="tab:red",
                linewidth=1.5,
                label="counterexample, no large rate drop",
            )
        axis.axvline(0, color="tab:blue", linewidth=1, linestyle=":")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(
        f"{len(windows)} serving cell changes (reported proxy, not confirmed handovers) in "
        f"{len({window.session_id for window in windows})} file download sessions.\n"
        "Windows from one session are correlated. Gaps and other segments are left blank."
    )
    axes[-1].set_xlabel("time relative to the later observation of the change (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return {
        "file": path.name,
        "aligned_serving_cell_changes": len(windows),
        "sessions_with_aligned_changes": len({window.session_id for window in windows}),
        "transition_free_controls": len(controls),
        "control_rule": (
            "non overlapping 60 s windows in file download sessions with no qualified "
            f"transition of any type inside, anchors stepped by {CONTROL_STEP_MS} ms, at least "
            f"{CONTROL_MINIMUM_COVERAGE:.0%} of bins with a downlink value in the anchor segment"
        ),
        "counterexample": counterexample,
        "counterexample_rule": (
            "serving cell change with complete 10 s windows on both sides, mean active fraction "
            f"at least {COUNTEREXAMPLE_MINIMUM_ACTIVE_FRACTION} and median at least "
            f"{COUNTEREXAMPLE_MINIMUM_REFERENCE_MBPS} Mbps before, a decline no larger than "
            f"{COUNTEREXAMPLE_MAXIMUM_DECLINE:.0%} after, smallest absolute relative change "
            "first, ties by session_id and time"
        ),
    }


def run_explore(snapshot_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Write explore_summary.json, capability_matrix.json and three PNG figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    connection, manifest = open_snapshot(snapshot_dir)
    try:
        sessions = _session_summaries(connection)
        cohorts = _cohort_summaries(connection)
        transitions = _transitions(connection, sessions)
        rejections = dict(
            connection.execute(
                "SELECT reason, count(*) FROM value_rejections GROUP BY reason ORDER BY reason"
            ).fetchall()
        )
        quarantine = dict(
            connection.execute(
                "SELECT reason, count(*) FROM quarantined_rows GROUP BY reason ORDER BY reason"
            ).fetchall()
        )
        capabilities = {
            str(row[0]): {"status": str(row[1]), "reason": str(row[2])}
            for row in connection.execute(
                "SELECT capability, status, reason FROM capabilities ORDER BY capability"
            ).fetchall()
        }
        figures: dict[str, Any] = {}
        chosen = _choose_timeline_session(sessions)
        if chosen is not None:
            figures["rate_activity_timeline"] = _timeline_figure(
                connection, chosen, out_dir / TIMELINE_FIGURE
            )
            figures["radio_rate_small_multiples"] = _radio_figure(
                connection, chosen, transitions[chosen["session_id"]], out_dir / RADIO_FIGURE
            )
        figures["transition_aligned_windows"] = _transition_figure(
            connection, sessions, transitions, out_dir / TRANSITION_FIGURE
        )
    finally:
        connection.close()

    transition_counts: dict[str, dict[str, int]] = {}
    cohort_of = {item["session_id"]: item["application_cohort"] for item in sessions}
    for session_id, items in transitions.items():
        counts = transition_counts.setdefault(cohort_of[session_id], {})
        for item in items:
            counts[item.transition_type.value] = counts.get(item.transition_type.value, 0) + 1
    summary = _clean(
        {
            "generated_by": "celltrace.analysis.explore",
            "snapshot_id": manifest["snapshot_id"],
            "snapshot_logical_hash": manifest["logical_hash"],
            "session_count": len(sessions),
            "sessions": sessions,
            "cohorts": cohorts,
            "value_rejections_by_reason": rejections,
            "quarantined_rows_by_reason": quarantine,
            "transitions_by_application_cohort": transition_counts,
            "figures": figures,
            "capability_matrix": CAPABILITY_MATRIX_NAME,
        }
    )
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out_dir / CAPABILITY_MATRIX_NAME).write_text(
        json.dumps(
            {"snapshot_id": manifest["snapshot_id"], "capabilities": capabilities},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dict(summary)
