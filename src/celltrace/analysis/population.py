"""Population analytics across every file download session.

Answers the four predeclared questions in docs/TECHNICAL_REPORT.md: where
drops concentrate, whether they cluster near cell changes, which drops simple
rules match, and what missing data hides. The detector, the
transition qualifier and the reported RSRP rule are reused unchanged. The
session is the unit of independence, so every interval resamples sessions.
"""

from __future__ import annotations

import bisect
import csv
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator

from celltrace.analysis import cells, chance, impact
from celltrace.analysis.bootstrap import (
    BOOTSTRAP_RESAMPLES,
    MINIMUM_SESSIONS_FOR_INTERVAL,
    session_bootstrap,
)
from celltrace.analysis.detector import (
    detect_candidates,
    detector_config_hash,
    evaluate_times,
    load_bins,
    load_segments,
)
from celltrace.analysis.transitions import fetch_cell_observations, qualify_transitions
from celltrace.config import DetectorSection, ObservationSection
from celltrace.contracts.common import AggregationBasis, MetricName, TimeWindow
from celltrace.contracts.evidence import CandidateEvent
from celltrace.data.metric_applicability import reference_signal_supported
from celltrace.data.raca import BIN_MS, technology_family
from celltrace.data.snapshot import open_snapshot
from celltrace.release import source_tree_hash
from celltrace.tools.repository import MeasurementRepository

PRIMARY_COHORT = "file_download"
NEARBY_TRANSITION_LOOKBACK_MS = 5_000
MISSINGNESS_FIELDS: tuple[MetricName, ...] = (
    MetricName.reported_rsrq_db,
    MetricName.reported_snr_db,
    MetricName.reported_cqi,
)
FIELD_LABELS = {
    MetricName.reported_rsrp_dbm: "RSRP",
    MetricName.reported_rsrq_db: "RSRQ",
    MetricName.reported_snr_db: "SNR",
    MetricName.reported_cqi: "CQI",
}
MOBILITY_ORDER = ("driving", "static")
TECHNOLOGY_ORDER = ("5G", "LTE", "3G", "mixed", "no samples")
TRIAGE_CLASSES: dict[str, tuple[str, str]] = {
    "cell_change": ("Cell change nearby", "mobility analysis, confirm with signalling logs"),
    "rsrp_fell": ("Reported RSRP fell more than the tolerance", "coverage analysis"),
    "unexplained": (
        "Neither rule matched the observations",
        "low rate filter, then Investigator queue",
    ),
    "not_assessable": ("Radio evidence missing or undefined", "data collection gap"),
}
MONITORING_REASONS: dict[str, tuple[str, ...]] = {
    "Monitored": (
        "not_flagged_relative_drop_below_minimum",
        "not_flagged_absolute_drop_below_minimum",
        "flagged_times",
    ),
    "Within 35 s of a recording start or gap": ("rejected_window_not_inside_one_segment",),
    "Incomplete or late bins": ("rejected_incomplete_bin", "rejected_not_yet_available"),
    "Too few one second rate values": (
        "rejected_reference_coverage_below_minimum",
        "rejected_event_coverage_below_minimum",
    ),
    "Not actively downloading": (
        "rejected_reference_active_fraction_below_minimum",
        "rejected_event_active_fraction_below_minimum",
    ),
    "Baseline below minimum rate": ("rejected_reference_below_minimum_mbps",),
}

SUMMARY_NAME = "population_summary.json"
REPORT_NAME = "population_report.md"
TRIAGE_NAME = "drop_triage.csv"
QUEUE_NAME = "investigation_queue.json"
FIGURES = {
    "chance": "q0_chance_checks.png",
    "q1": "q1_drop_rate_by_technology.png",
    "q2": "q2_cell_change_association.png",
    "q3": "q3_drop_triage.png",
    "q4_monitoring": "q4_monitoring_coverage.png",
    "q4_missing": "q4_missing_by_technology.png",
    "q5": "q5_service_levels.png",
    "q6": "q6_detector_against_user_symptom.png",
    "q7": "q7_cell_replication.png",
}

CHANCE_PERMUTATIONS = 199
CHANCE_SHIFTS = 1000
CHANCE_ALPHA = 0.05

# Reference palette, light mode, validated with the dataviz palette script.
SERIES_PRIMARY = "#2568b8"
SERIES_SECONDARY = "#ba4b20"
SERIES_TERTIARY = "#087e58"
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


# Pure rules


def window_technology(modes: set[str | None]) -> str:
    """5G or LTE when the window is uniform, 3G for UMTS family only, otherwise mixed."""
    if not modes:
        return "no samples"
    if modes == {"5G"}:
        return "5G"
    if modes == {"LTE"}:
        return "LTE"
    if all(technology_family(mode) == "umts" for mode in modes):
        return "3G"
    return "mixed"


def rsrp_defined(modes: set[str | None]) -> bool:
    """RSRP is an LTE and NR measurement, so 3G or unknown rows make it undefined."""
    return reference_signal_supported(modes)


def has_nearby_transition(transition_times: list[int], start_ms: int, end_ms: int) -> bool:
    """Whether a sorted transition time falls in the half open interval [start_ms, end_ms)."""
    index = bisect.bisect_left(transition_times, start_ms)
    return index < len(transition_times) and transition_times[index] < end_ms


def rsrp_status(delta_db: float, tolerance_db: float) -> str:
    """The verifier's rule: stable within the tolerance, fell below its negative."""
    if delta_db < -tolerance_db:
        return "fell"
    if delta_db > tolerance_db:
        return "rose"
    return "stable"


def triage_class(nearby_transition: bool, status: str) -> str:
    """Route a drop. A cell change is observable even when radio fields are not."""
    if nearby_transition:
        return "cell_change"
    if status == "fell":
        return "rsrp_fell"
    if status in ("stable", "rose"):
        return "unexplained"
    return "not_assessable"


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


# Per session scan


@dataclass
class SessionTotals:
    mobility: str
    outcome_seconds: Counter[str] = field(default_factory=Counter)
    monitored_seconds: Counter[str] = field(default_factory=Counter)
    # Keyed by (nearby cell change, flagged).
    transition_seconds: Counter[tuple[bool, bool]] = field(default_factory=Counter)
    # Keyed by (field, flagged, "valid" or "samples").
    field_counts: Counter[tuple[str, bool, str]] = field(default_factory=Counter)
    events: Counter[str] = field(default_factory=Counter)
    triage: Counter[str] = field(default_factory=Counter)
    # User impact layer. Keyed by (technology, "observed" or service level).
    impact_seconds: Counter[tuple[str, str]] = field(default_factory=Counter)
    # Keyed by (service level, minimum sustained seconds).
    episodes: Counter[tuple[str, int]] = field(default_factory=Counter)
    episode_seconds: Counter[tuple[str, int]] = field(default_factory=Counter)
    # Relative detector scored against the symptom level.
    symptom: Counter[str] = field(default_factory=Counter)
    detection_delays_ms: list[int] = field(default_factory=list)
    impact_triage: Counter[str] = field(default_factory=Counter)


def bin_modes(connection: duckdb.DuckDBPyConnection, session_id: str) -> dict[int, set[str | None]]:
    rows = connection.execute(
        """
        SELECT elapsed_ms - (elapsed_ms % ?) AS bin_start_ms, network_mode
        FROM samples WHERE session_id = ? GROUP BY ALL
        """,
        [BIN_MS, session_id],
    ).fetchall()
    modes: dict[int, set[str | None]] = {}
    for bin_start, mode in rows:
        modes.setdefault(int(bin_start), set()).add(mode)
    return modes


def bin_field_validity(
    connection: duckdb.DuckDBPyConnection, session_id: str
) -> dict[int, tuple[int, dict[str, int]]]:
    columns = ", ".join(f"{name.value}_valid_count" for name in MISSINGNESS_FIELDS)
    rows = connection.execute(
        f"SELECT bin_start_ms, sample_count, {columns} FROM bins_1s WHERE session_id = ?",
        [session_id],
    ).fetchall()
    return {
        int(row[0]): (
            int(row[1]),
            {name.value: int(row[2 + index]) for index, name in enumerate(MISSINGNESS_FIELDS)},
        )
        for row in rows
    }


def transition_times(connection: duckdb.DuckDBPyConnection, session_id: str) -> list[int]:
    observations = fetch_cell_observations(connection, session_id)
    return sorted(item.later_time_ms for item in qualify_transitions(observations))


def window_modes(modes: dict[int, set[str | None]], start_ms: int, end_ms: int) -> set[str | None]:
    found: set[str | None] = set()
    for bin_start in range(start_ms, end_ms, BIN_MS):
        found |= modes.get(bin_start, set())
    return found


def scan_session(
    connection: duckdb.DuckDBPyConnection,
    session_id: str,
    mobility: str,
    detector: DetectorSection,
) -> tuple[
    SessionTotals,
    dict[int, set[str | None]],
    list[int],
    dict[tuple[str, int], list[impact.Episode]],
]:
    """Exposure, cell change association, missingness and user impact for one session."""
    modes = bin_modes(connection, session_id)
    validity = bin_field_validity(connection, session_id)
    transitions = transition_times(connection, session_id)
    totals = SessionTotals(mobility)
    bins = load_bins(connection, session_id)
    segments = load_segments(connection, session_id)
    for decision_ms, rate in (series := impact.level_series(bins, segments, detector)):
        technology = window_technology(
            window_modes(modes, decision_ms - impact.SHORT_WINDOW_MS, decision_ms)
        )
        totals.impact_seconds[(technology, "observed")] += 1
        for level, threshold in impact.SERVICE_LEVELS:
            if rate < threshold:
                totals.impact_seconds[(technology, level)] += 1
    episodes = impact.session_episodes(series, detector.step_ms)
    for key, found in episodes.items():
        totals.episodes[key] += len(found)
        totals.episode_seconds[key] += sum(episode.seconds for episode in found)
    decisions = evaluate_times(session_id, bins, segments, detector)
    for decision in decisions:
        totals.outcome_seconds[decision.outcome] += 1
        if not decision.eligible:
            continue
        event_start = decision.confirmation_ms - detector.event_ms
        flagged = decision.flag is not None
        technology = window_technology(window_modes(modes, event_start, decision.confirmation_ms))
        totals.monitored_seconds[technology] += 1
        nearby = has_nearby_transition(
            transitions, event_start - NEARBY_TRANSITION_LOOKBACK_MS, decision.confirmation_ms
        )
        totals.transition_seconds[(nearby, flagged)] += 1
        for bin_start in range(event_start, decision.confirmation_ms, BIN_MS):
            sample_count, valid_counts = validity.get(bin_start, (0, {}))
            for name in MISSINGNESS_FIELDS:
                totals.field_counts[(name.value, flagged, "valid")] += valid_counts.get(
                    name.value, 0
                )
                totals.field_counts[(name.value, flagged, "samples")] += sample_count
    return totals, modes, transitions, episodes


# Drop event triage


def classify_event(
    candidate: CandidateEvent,
    modes: dict[int, set[str | None]],
    transitions: list[int],
    repository: MeasurementRepository,
    observation: ObservationSection,
) -> dict[str, Any]:
    reference = candidate.representative_reference_window
    event = candidate.representative_event_window
    reference_modes = window_modes(modes, reference.start_ms, reference.end_ms)
    event_modes = window_modes(modes, event.start_ms, event.end_ms)
    nearby = has_nearby_transition(
        transitions, event.start_ms - NEARBY_TRANSITION_LOOKBACK_MS, event.end_ms
    )
    status, delta, reason = rsrp_assessment(
        repository,
        candidate.session_id,
        (reference, event),
        reference_modes | event_modes,
        observation,
    )
    category = triage_class(nearby, status)
    return {
        "event_id": candidate.event_id,
        "session_id": candidate.session_id,
        "technology": window_technology(event_modes),
        "reference_start_ms": reference.start_ms,
        "event_start_ms": event.start_ms,
        "event_end_ms": event.end_ms,
        "reference_median_mbps": candidate.reference_median_mbps,
        "event_median_mbps": candidate.event_median_mbps,
        "absolute_drop_mbps": candidate.absolute_drop_mbps,
        "relative_drop": candidate.relative_drop,
        "nearby_cell_change": nearby,
        "rsrp_status": status,
        "rsrp_delta_db": delta,
        "rsrp_not_assessable_reason": reason,
        "triage_class": category,
        "next_step": TRIAGE_CLASSES[category][1],
    }


def rsrp_assessment(
    repository: MeasurementRepository,
    session_id: str,
    windows: tuple[TimeWindow, TimeWindow],
    modes: set[str | None],
    observation: ObservationSection,
) -> tuple[str, float | None, str | None]:
    if not rsrp_defined(modes):
        return "not_assessable", None, "rsrp_undefined_outside_lte_and_5g"
    medians = []
    for window in windows:
        binned = repository.summarize(
            session_id, window, MetricName.reported_rsrp_dbm, AggregationBasis.bin_1s
        )
        native = repository.summarize(
            session_id, window, MetricName.reported_rsrp_dbm, AggregationBasis.native
        )
        if binned.applicability is None or not binned.applicability.supported:
            return "not_assessable", None, "rsrp_undefined_outside_lte_and_5g"
        if (
            binned.coverage is None
            or binned.coverage < observation.minimum_radio_coverage
            or binned.median is None
        ):
            return "not_assessable", None, "rsrp_coverage_below_minimum"
        if native.outside_documented_range_count:
            return "not_assessable", None, "rsrp_outside_documented_range"
        medians.append(binned.median)
    delta = medians[1] - medians[0]
    return rsrp_status(delta, observation.rsrp_stability_tolerance_db), delta, None


# Aggregation


def estimate(
    rows: Sequence[tuple[float, ...]],
    statistic: Callable[[np.ndarray], np.ndarray],
    seed: int,
    denominator_columns: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    sums = np.asarray(rows, dtype=float).sum(axis=0) if rows else None
    value = None
    if sums is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            computed = float(statistic(sums[np.newaxis, :])[0])
        value = computed if np.isfinite(computed) else None
    contributors = [sum(row[column] > 0 for row in rows) for column in denominator_columns]
    return {
        "value": value,
        "interval_95": (
            session_bootstrap(rows, statistic, seed)
            if all(count >= MINIMUM_SESSIONS_FOR_INTERVAL for count in contributors)
            else None
        ),
        "sessions": len(rows),
        "denominator_contributing_sessions": contributors,
    }


def strata(totals: dict[str, SessionTotals]) -> list[tuple[str, str]]:
    """Every mobility and technology pair with monitored time, in a fixed reading order."""
    present = {
        (item.mobility, technology)
        for item in totals.values()
        for technology, seconds in item.monitored_seconds.items()
        if seconds > 0
    }

    def order(pair: tuple[str, str]) -> tuple[int, int, str]:
        mobility, technology = pair
        return (
            MOBILITY_ORDER.index(mobility) if mobility in MOBILITY_ORDER else len(MOBILITY_ORDER),
            TECHNOLOGY_ORDER.index(technology)
            if technology in TECHNOLOGY_ORDER
            else len(TECHNOLOGY_ORDER),
            mobility,
        )

    return sorted(present, key=order)


def q1_drop_rates(totals: dict[str, SessionTotals], seed: int) -> list[dict[str, Any]]:
    results = []
    for mobility, technology in strata(totals):
        rows = [
            (float(item.events[technology]), float(item.monitored_seconds[technology]))
            for item in totals.values()
            if item.mobility == mobility and item.monitored_seconds[technology] > 0
        ]
        events = sum(row[0] for row in rows)
        seconds = sum(row[1] for row in rows)
        results.append(
            {
                "mobility": mobility,
                "technology": technology,
                "events": int(events),
                "monitored_hours": seconds / 3600,
                "events_per_monitored_hour": estimate(
                    rows, lambda sums: sums[:, 0] / (sums[:, 1] / 3600), seed
                ),
            }
        )
    return results


def q2_cell_change(totals: dict[str, SessionTotals], seed: int) -> dict[str, Any]:
    rows = [
        (
            float(item.transition_seconds[(True, True)]),
            float(item.transition_seconds[(True, True)] + item.transition_seconds[(True, False)]),
            float(item.transition_seconds[(False, True)]),
            float(item.transition_seconds[(False, True)] + item.transition_seconds[(False, False)]),
        )
        for item in totals.values()
        if item.mobility == "driving" and sum(item.transition_seconds.values()) > 0
    ]
    near = sum(row[1] for row in rows)
    far = sum(row[3] for row in rows)
    return {
        "scope": "driving sessions, monitored seconds",
        "lookback_ms": NEARBY_TRANSITION_LOOKBACK_MS,
        "exposure_basis": "eligible one-second detector confirmation times",
        "proximity_definition": "transition in the event window or preceding lookback",
        "seconds_near_cell_change": int(near),
        "seconds_without_cell_change": int(far),
        "drop_share_near_cell_change": estimate(rows, lambda sums: sums[:, 0] / sums[:, 1], seed),
        "drop_share_without_cell_change": estimate(
            rows, lambda sums: sums[:, 2] / sums[:, 3], seed, (3,)
        ),
        "ratio": estimate(
            rows, lambda sums: (sums[:, 0] / sums[:, 1]) / (sums[:, 2] / sums[:, 3]), seed, (1, 3)
        ),
    }


def q3_triage(
    totals: dict[str, SessionTotals], triage_rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    contributing = [item for item in totals.values() if sum(item.triage.values()) > 0]
    classes = {}
    for name in TRIAGE_CLASSES:
        rows = [
            (float(item.triage[name]), float(sum(item.triage.values()))) for item in contributing
        ]
        classes[name] = {
            "label": TRIAGE_CLASSES[name][0],
            "next_step": TRIAGE_CLASSES[name][1],
            "events": sum(row["triage_class"] == name for row in triage_rows),
            "share": estimate(rows, lambda sums: sums[:, 0] / sums[:, 1], seed),
        }
    reasons = Counter(
        row["rsrp_not_assessable_reason"]
        for row in triage_rows
        if row["triage_class"] == "not_assessable"
    )
    return {
        "events": len(triage_rows),
        "classes": classes,
        "not_assessable_reasons": dict(sorted(reasons.items())),
        "cell_change_with_rsrp_fell": sum(
            row["triage_class"] == "cell_change" and row["rsrp_status"] == "fell"
            for row in triage_rows
        ),
        "events_by_technology": dict(
            sorted(Counter(row["technology"] for row in triage_rows).items())
        ),
    }


def q4_monitoring(totals: dict[str, SessionTotals]) -> dict[str, Any]:
    result = {}
    for mobility in dict.fromkeys([*MOBILITY_ORDER, *(item.mobility for item in totals.values())]):
        outcomes: Counter[str] = Counter()
        for item in totals.values():
            if item.mobility == mobility:
                outcomes.update(item.outcome_seconds)
        evaluated = sum(outcomes.values())
        result[mobility] = {
            "evaluated_seconds": evaluated,
            "sessions": sum(item.mobility == mobility for item in totals.values()),
            "reasons": {
                reason: {
                    "seconds": sum(outcomes[stage] for stage in stages),
                    "share": ratio(sum(outcomes[stage] for stage in stages), evaluated),
                }
                for reason, stages in MONITORING_REASONS.items()
            },
        }
    return result


def q4_missing(totals: dict[str, SessionTotals], seed: int) -> dict[str, Any]:
    result = {}
    for name in MISSINGNESS_FIELDS:
        rows = []
        for item in totals.values():
            drop_samples = item.field_counts[(name.value, True, "samples")]
            other_samples = item.field_counts[(name.value, False, "samples")]
            if drop_samples + other_samples == 0:
                continue
            rows.append(
                (
                    float(drop_samples - item.field_counts[(name.value, True, "valid")]),
                    float(drop_samples),
                    float(other_samples - item.field_counts[(name.value, False, "valid")]),
                    float(other_samples),
                )
            )
        result[FIELD_LABELS[name]] = {
            "metric": name.value,
            "drop_window_samples": int(sum(row[1] for row in rows)),
            "other_window_samples": int(sum(row[3] for row in rows)),
            "missing_share_in_drops": estimate(rows, lambda sums: sums[:, 0] / sums[:, 1], seed),
            "missing_share_elsewhere": estimate(
                rows, lambda sums: sums[:, 2] / sums[:, 3], seed, (3,)
            ),
            "difference": estimate(
                rows, lambda sums: sums[:, 0] / sums[:, 1] - sums[:, 2] / sums[:, 3], seed, (1, 3)
            ),
        }
    return result


def missing_by_technology(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Follow up added after the predeclared run: missing share per field and technology."""
    fields = (MetricName.reported_rsrp_dbm, *MISSINGNESS_FIELDS)
    columns = ", ".join(f"count({name.value})" for name in fields)
    rows = connection.execute(
        f"""
        SELECT network_mode, count(*), {columns}
        FROM samples JOIN sessions USING (session_id)
        WHERE application_cohort = ? GROUP BY network_mode
        """,
        [PRIMARY_COHORT],
    ).fetchall()
    samples: Counter[str] = Counter()
    present: Counter[tuple[str, str]] = Counter()
    for row in rows:
        technology = window_technology({row[0]})
        samples[technology] += int(row[1])
        for index, name in enumerate(fields):
            present[(technology, FIELD_LABELS[name])] += int(row[2 + index])
    return {
        technology: {
            "samples": samples[technology],
            "missing_share": {
                FIELD_LABELS[name]: 1 - present[(technology, FIELD_LABELS[name])] / count
                for name in fields
            },
        }
        for technology, count in sorted(
            samples.items(),
            key=lambda item: (
                TECHNOLOGY_ORDER.index(item[0])
                if item[0] in TECHNOLOGY_ORDER
                else len(TECHNOLOGY_ORDER),
                item[0],
            ),
        )
    }


def cadence(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """How irregular logging leaves one second bins empty or crowded."""
    row = connection.execute(
        """
        SELECT count(*), avg((sample_count = 0)::INTEGER), avg((sample_count >= 2)::INTEGER),
               count(*) FILTER (WHERE sample_count > 0 AND downlink_mbps_valid_count = 0)
        FROM bins_1s JOIN sessions USING (session_id) WHERE application_cohort = ?
        """,
        [PRIMARY_COHORT],
    ).fetchone()
    assert row is not None
    return {
        "bins": int(row[0]),
        "empty_bin_share": float(row[1] or 0),
        "multiple_sample_bin_share": float(row[2] or 0),
        "bins_with_samples_but_no_rate": int(row[3]),
    }


# User impact aggregation


def _stratum_key(pair: tuple[str, str]) -> tuple[int, int, str]:
    mobility, technology = pair
    if mobility == "all":
        return (len(MOBILITY_ORDER) + 1, 0, mobility)
    return (
        MOBILITY_ORDER.index(mobility) if mobility in MOBILITY_ORDER else len(MOBILITY_ORDER),
        TECHNOLOGY_ORDER.index(technology)
        if technology in TECHNOLOGY_ORDER
        else len(TECHNOLOGY_ORDER),
        technology,
    )


def _impact_count(item: SessionTotals, mobility: str, technology: str, key: str) -> int:
    if mobility == "all":
        return sum(seconds for (_, name), seconds in item.impact_seconds.items() if name == key)
    if item.mobility != mobility:
        return 0
    return item.impact_seconds[(technology, key)]


def q5_service_levels(totals: dict[str, SessionTotals], seed: int) -> list[dict[str, Any]]:
    """Share of observed active download time below each service level, by stratum."""
    present = {
        (item.mobility, technology)
        for item in totals.values()
        for (technology, key), seconds in item.impact_seconds.items()
        if key == "observed" and seconds > 0
    }
    results = []
    for mobility, technology in [*sorted(present, key=_stratum_key), ("all", "all")]:
        contributing = [
            item
            for item in totals.values()
            if _impact_count(item, mobility, technology, "observed") > 0
        ]
        observed = sum(
            _impact_count(item, mobility, technology, "observed") for item in contributing
        )
        levels = {}
        for level, _ in impact.SERVICE_LEVELS:
            rows = [
                (
                    float(_impact_count(item, mobility, technology, level)),
                    float(_impact_count(item, mobility, technology, "observed")),
                )
                for item in contributing
            ]
            levels[level] = estimate(rows, lambda sums: sums[:, 0] / sums[:, 1], seed)
        results.append(
            {
                "mobility": mobility,
                "technology": technology,
                "sessions": len(contributing),
                "observed_hours": observed / 3600,
                "levels": levels,
            }
        )
    return results


def q5_episodes(
    totals: dict[str, SessionTotals],
    durations: dict[str, list[int]],
    seed: int,
) -> dict[str, Any]:
    """Sustained episodes per level, their rate per observed hour and sensitivity."""
    observed = {
        session_id: sum(
            seconds for (_, key), seconds in item.impact_seconds.items() if key == "observed"
        )
        for session_id, item in totals.items()
    }
    result: dict[str, Any] = {}
    for level, _ in impact.SERVICE_LEVELS:
        sensitivity = {}
        for minimum in impact.SENSITIVITY_MINIMUM_SECONDS:
            rows = [
                (float(item.episodes[(level, minimum)]), float(observed[session_id]))
                for session_id, item in totals.items()
                if observed[session_id] > 0
            ]
            sensitivity[str(minimum)] = {
                "episodes": sum(item.episodes[(level, minimum)] for item in totals.values()),
                "seconds": sum(item.episode_seconds[(level, minimum)] for item in totals.values()),
                "per_observed_hour": estimate(
                    rows, lambda sums: sums[:, 0] / (sums[:, 1] / 3600), seed
                ),
            }
        lengths = sorted(durations.get(level, []))
        result[level] = {
            "label": impact.LEVEL_LABELS[level],
            "minimum_seconds": sensitivity,
            "primary_median_seconds": float(np.median(lengths)) if lengths else None,
            "primary_longest_seconds": lengths[-1] if lengths else None,
        }
    return result


def q6_detector_against_symptom(totals: dict[str, SessionTotals], seed: int) -> dict[str, Any]:
    """Overlap of relative drops with sustained time under 1.5 Mbps."""

    def rows(
        numerator: str, denominator: str, level: str = impact.SYMPTOM_LEVEL
    ) -> list[tuple[float, ...]]:
        return [
            (
                float(item.symptom[f"{level}:{numerator}"]),
                float(item.symptom[f"{level}:{denominator}"]),
            )
            for item in totals.values()
            if item.symptom[f"{level}:{denominator}"] > 0
        ]

    def share(sums: np.ndarray) -> np.ndarray:
        ratios: np.ndarray = sums[:, 0] / sums[:, 1]
        return ratios

    def total(name: str, level: str = impact.SYMPTOM_LEVEL) -> int:
        return sum(item.symptom[f"{level}:{name}"] for item in totals.values())

    delays = sorted(delay for item in totals.values() for delay in item.detection_delays_ms)
    missed_episodes = total("episodes") - total("episodes_detected")
    return {
        "symptom": impact.LEVEL_LABELS[impact.SYMPTOM_LEVEL],
        "minimum_seconds": impact.PRIMARY_MINIMUM_SECONDS,
        "candidates": total("candidates"),
        "candidates_overlapping_symptom": total("candidates_overlapping_symptom"),
        "precision": estimate(rows("candidates_overlapping_symptom", "candidates"), share, seed),
        "episodes": total("episodes"),
        "episodes_detected": total("episodes_detected"),
        "episodes_detected_share": estimate(rows("episodes_detected", "episodes"), share, seed),
        "symptom_seconds": total("episode_ms") / 1000,
        "symptom_seconds_covered": total("episode_ms_covered") / 1000,
        "recall_by_time": estimate(rows("episode_ms_covered", "episode_ms"), share, seed),
        "detection_delay_seconds": {
            "median": float(np.median(delays)) / 1000 if delays else None,
            # Linear interpolation, as numpy.percentile and the measurement tools use.
            "p90": float(np.percentile(delays, 90)) / 1000 if delays else None,
        },
        "missed_episodes": missed_episodes,
        "missed_episodes_next_step": "separate low rate investigation, outside the agent queue",
        "by_level": {
            level: {
                "label": impact.LEVEL_LABELS[level],
                "precision": estimate(
                    rows("candidates_overlapping_symptom", "candidates", level), share, seed
                ),
                "recall_by_time": estimate(
                    rows("episode_ms_covered", "episode_ms", level), share, seed
                ),
                "episodes": total("episodes", level),
                "episodes_detected": total("episodes_detected", level),
            }
            for level, _ in impact.SERVICE_LEVELS
        },
    }


# Figures


def _style(axis: Any, title: str, subtitle: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=12, color=INK["primary"], pad=22)
    axis.text(
        0,
        1.02,
        subtitle,
        transform=axis.transAxes,
        fontsize=9,
        color=INK["secondary"],
        va="bottom",
    )
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)
    axis.tick_params(colors=INK["secondary"], labelsize=9, length=0)


def _interval_bar(
    axis: Any,
    position: float,
    value: float | None,
    interval: list[float] | None,
    color: str,
    height: float = 0.42,
) -> None:
    if value is None:
        return
    axis.barh(position, value, height=height, color=color)
    if interval is not None:
        axis.plot(interval, [position, position], color=INK["primary"], linewidth=1.2)


def _save(figure: Any, path: Path) -> None:
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def figure_q1(rates: list[dict[str, Any]], path: Path, chance_rate: float | None = None) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 3.6))
    labels = []
    for position, item in enumerate(reversed(rates)):
        rate = item["events_per_monitored_hour"]
        _interval_bar(axis, position, rate["value"], rate["interval_95"], SERIES_PRIMARY)
        labels.append(f"{item['mobility'].capitalize()}, {item['technology']}")
        note = (
            f"{item['events']} drops, {item['monitored_hours']:.1f} h, {rate['sessions']} sessions"
        )
        if rate["interval_95"] is None:
            note += ", no interval"
        right = max(rate["interval_95"][1] if rate["interval_95"] else 0, rate["value"] or 0)
        axis.text(right, position, "  " + note, va="center", fontsize=8, color=INK["secondary"])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel(
        "Drop events per monitored hour (bar), 95% session bootstrap interval (line)",
        fontsize=9,
        color=INK["secondary"],
    )
    axis.set_xlim(0, axis.get_xlim()[1] * 1.45)
    defined = [
        item["events_per_monitored_hour"]["value"]
        for item in rates
        if item["events_per_monitored_hour"]["value"] is not None
    ]
    title = (
        f"Q1. The relative detector fires {min(defined):.0f} to {max(defined):.0f} times per hour"
        if defined
        else "Q1. No group has monitored time"
    )
    chance_note = (
        f"Rates shuffled within segment and activity state give about {chance_rate:.0f} "
        "candidates per eligible hour"
        if chance_rate is not None
        else "Groups are not balanced, so compare technologies within one mobility group"
    )
    _style(axis, title, chance_note)
    _save(figure, path)


def figure_q2(
    result: dict[str, Any], path: Path, association: dict[str, Any] | None = None
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 2.6))
    bars = (
        (
            "Cell change in or just before the window",
            result["drop_share_near_cell_change"],
            result["seconds_near_cell_change"],
            SERIES_PRIMARY,
        ),
        (
            "No cell change nearby",
            result["drop_share_without_cell_change"],
            result["seconds_without_cell_change"],
            INK["muted"],
        ),
    )
    for position, (_label, share, seconds, color) in enumerate(reversed(bars)):
        value = None if share["value"] is None else 100 * share["value"]
        interval = None if share["interval_95"] is None else [100 * x for x in share["interval_95"]]
        _interval_bar(axis, position, value, interval, color)
        right = max(interval[1] if interval else 0, value or 0)
        axis.text(
            right,
            position,
            f"  {value or 0:.1f}% of {seconds} monitored s",
            va="center",
            fontsize=8,
            color=INK["secondary"],
        )
    axis.set_yticks(range(2), [bar[0] for bar in reversed(bars)])
    axis.set_xlabel("Flagged eligible confirmation times (%)", fontsize=9, color=INK["secondary"])
    axis.set_xlim(0, axis.get_xlim()[1] * 1.5)
    ratio_value = result["ratio"]["value"]
    title = "Q2. Drops near cell changes"
    subtitle = "Driving sessions, association only"
    if ratio_value is not None:
        title = f"Q2. Drops are {ratio_value:.1f} times as common near a reported cell change"
    if association is not None and association["p_value"] is not None:
        subtitle = (
            f"Shift-null range {_range(association['null_interval_95'])}; raw p = "
            f"{_fmt(association['p_value'], '.3f')}. Association; Holm adjustment in Q0"
        )
    _style(axis, title, subtitle)
    _save(figure, path)


def figure_q3(result: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 3.0))
    names = list(TRIAGE_CLASSES)
    for position, name in enumerate(reversed(names)):
        item = result["classes"][name]
        share = item["share"]
        value = None if share["value"] is None else 100 * share["value"]
        interval = None if share["interval_95"] is None else [100 * x for x in share["interval_95"]]
        color = INK["muted"] if name == "not_assessable" else SERIES_PRIMARY
        _interval_bar(axis, position, value, interval, color)
        right = max(interval[1] if interval else 0, value or 0)
        axis.text(
            right,
            position,
            f"  {item['events']} drops, next: {item['next_step']}",
            va="center",
            fontsize=8,
            color=INK["secondary"],
        )
    axis.set_yticks(range(len(names)), [TRIAGE_CLASSES[name][0] for name in reversed(names)])
    axis.set_xlabel("Share of drop events (%)", fontsize=9, color=INK["secondary"], loc="left")
    # Room for the labels, but the share scale itself stops at 100.
    axis.set_xlim(0, 215)
    axis.set_xticks(range(0, 101, 20))
    axis.spines["bottom"].set_bounds(0, 100)
    unexplained_share = result["classes"]["unexplained"]["share"]["value"] or 0.0
    _style(
        axis,
        f"Q3. {100 * unexplained_share:.0f}% of drops match neither observation rule",
        f"{result['events']} drop events, one class each, in this precedence from top",
    )
    _save(figure, path)


def figure_q4_monitoring(result: dict[str, Any], path: Path) -> None:
    reasons = list(MONITORING_REASONS)
    mobilities = list(result)
    figure, panels = plt.subplots(
        1, len(mobilities), figsize=(4.75 * len(mobilities), 3.2), sharey=True, squeeze=False
    )
    axes = panels[0]
    for axis, mobility in zip(axes, mobilities, strict=True):
        item = result[mobility]
        for position, reason in enumerate(reversed(reasons)):
            share = item["reasons"][reason]["share"] or 0.0
            color = SERIES_PRIMARY if reason == "Monitored" else INK["muted"]
            axis.barh(position, 100 * share, height=0.42, color=color)
            axis.text(
                100 * share,
                position,
                f" {100 * share:.1f}%",
                va="center",
                fontsize=8,
                color=INK["secondary"],
            )
        axis.set_xlim(0, 115)
        axis.set_xlabel("Share of evaluated seconds (%)", fontsize=9, color=INK["secondary"])
        _style(
            axis,
            mobility.capitalize(),
            f"{item['sessions']} sessions, {item['evaluated_seconds']} s evaluated",
        )
    axes[0].set_yticks(range(len(reasons)), list(reversed(reasons)))
    monitored = [
        100 * (result[name]["reasons"]["Monitored"]["share"] or 0.0) for name in mobilities
    ]
    figure.suptitle(
        f"Q4a. The detector can monitor {min(monitored):.0f} to {max(monitored):.0f}% of download "
        "time, mostly losing seconds with too few rate values",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK["primary"],
    )
    _save(figure, path)


def figure_q4_missing(result: dict[str, Any], path: Path) -> None:
    """Technology by field grid of missing shares: a heatmap, one sequential hue."""
    technologies = list(result)
    fields = list(FIELD_LABELS.values())
    grid = np.array(
        [[100 * result[name]["missing_share"][field] for field in fields] for name in technologies]
    )
    figure, axis = plt.subplots(figsize=(7.0, 0.55 * len(technologies) + 1.9))
    colormap = LinearSegmentedColormap.from_list("missing", [SURFACE, "#86b6ef", "#104281"])
    axis.imshow(grid, cmap=colormap, vmin=0, vmax=100, aspect="auto")
    for row in range(len(technologies)):
        for column in range(len(fields)):
            value = grid[row, column]
            text_value = "0%" if value == 0 else ("<1%" if value < 1 else f"{value:.0f}%")
            axis.text(
                column,
                row,
                text_value,
                ha="center",
                va="center",
                fontsize=9,
                color=SURFACE if value >= 60 else INK["primary"],
            )
    axis.set_xticks(range(len(fields)), fields)
    axis.set_yticks(
        range(len(technologies)),
        [f"{name} ({result[name]['samples']} samples)" for name in technologies],
    )
    axis.tick_params(colors=INK["secondary"], labelsize=9, length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    worst = max(technologies, key=lambda name: float(np.mean(grid[technologies.index(name)])))
    gaps = [field for field in fields if result[worst]["missing_share"][field] >= 0.5]
    title = (
        f"Q4c. {worst} rows lack {', '.join(gaps)}"
        if gaps
        else "Q4c. Radio fields are nearly complete in every technology"
    )
    axis.set_title(title, loc="left", fontsize=12, color=INK["primary"], pad=22)
    axis.text(
        0,
        1.02,
        "Share of samples with the field missing. Follow up added after the predeclared run",
        transform=axis.transAxes,
        fontsize=9,
        color=INK["secondary"],
        va="bottom",
    )
    _save(figure, path)


CHANCE_ROWS = (
    ("flag_share", "Seconds flagged by the relative detector", "relative drops"),
    ("candidates_per_hour", "Relative drop candidates per hour", "relative drops"),
    ("sustained_candidates", "Groups with at least 10 flagged decisions", "relative drops"),
    ("low_rate_episodes", "Episodes of 10 s or more under 1.5 Mbps", "sustained low rate"),
)


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _range(interval: list[float] | None) -> str:
    return "n/a" if interval is None else f"{interval[0]:.2f} to {interval[1]:.2f}"


def chance_rows(checks: dict[str, Any]) -> list[dict[str, Any]]:
    """Maximum across detector nulls, then Holm across five displayed statistics."""
    rows = []
    for statistic, label, short in CHANCE_ROWS:
        nulls = {name: checks["detector"][name][statistic] for name in chance.NULLS}
        rows.append(
            {
                "statistic": statistic,
                "label": label,
                "short": short,
                "observed": next(iter(nulls.values()))["observed"],
                "nulls": nulls,
                "p_value": None
                if any(item["p_value"] is None for item in nulls.values())
                else max(item["p_value"] for item in nulls.values()),
            }
        )
    association = checks["cell_change"]
    rows.append(
        {
            "statistic": "cell_change_ratio",
            "label": "Drop share near versus away from cell changes",
            "short": "the cell change link",
            "observed": association["observed"],
            "nulls": {"circular_shift": association},
            "p_value": association["p_value"],
        }
    )
    adjusted = chance.holm_adjust([row["p_value"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        draws = min(item["draws"] for item in row["nulls"].values())
        row["p_value_holm"] = value
        row["minimum_holm_p"] = chance.minimum_holm_p(draws, len(rows))
        row["rejection_possible"] = row["minimum_holm_p"] < CHANCE_ALPHA
        row["beats_chance"] = value is not None and value < CHANCE_ALPHA
        row["verdict"] = chance_verdict(row)
    return rows


def chance_verdict(row: dict[str, Any]) -> str:
    if row["p_value_holm"] is None:
        return "undefined"
    if row["beats_chance"]:
        return "excess over the null"
    if not row["rejection_possible"]:
        return "cannot reject at this resolution"
    return "no rejection"


def _chance_parts(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    passing = list(dict.fromkeys(row["short"] for row in rows if row["beats_chance"]))
    failing = [
        short
        for short in dict.fromkeys(row["short"] for row in rows if not row["beats_chance"])
        if short not in passing
    ]
    return passing, failing


def chance_headline(rows: list[dict[str, Any]]) -> str:
    passing, failing = _chance_parts(rows)
    if passing and failing:
        beating = " and ".join(passing).capitalize()
        return f"{beating} exceed their nulls"
    if passing:
        return "All five statistics exceed their nulls"
    return "No statistic exceeds its nulls after adjustment"


NULL_MARKERS = {"shuffled_seconds": "o", "shuffled_5s_blocks": "s", "circular_shift": "D"}
NULL_LABELS = {
    "shuffled_seconds": "Seconds shuffled within segment and activity state",
    "shuffled_5s_blocks": "5 s blocks shuffled within segment and activity state",
    "circular_shift": "Cell change exposure shifted within eligible runs",
}
NULL_OFFSETS = {"shuffled_seconds": 0.12, "shuffled_5s_blocks": -0.12, "circular_shift": 0.0}
NULL_SHORT = {
    "shuffled_seconds": "seconds",
    "shuffled_5s_blocks": "5 s blocks",
    "circular_shift": "shift",
}


def figure_chance(rows: list[dict[str, Any]], path: Path) -> None:
    """Observed value divided by its chance expectation, one marker per null model."""
    figure, axis = plt.subplots(figsize=(9.6, 0.62 * len(rows) + 2.0))
    seen: list[str] = []
    labels = []
    ratios = [1.0]
    for position, row in enumerate(reversed(rows)):
        color = SERIES_PRIMARY if row["beats_chance"] else INK["muted"]
        for name, item in row["nulls"].items():
            if item["ratio_to_null_mean"] is None:
                continue
            ratios.append(item["ratio_to_null_mean"])
            axis.scatter(
                [item["ratio_to_null_mean"]],
                # Dodge the two rate nulls so neither marker hides the other.
                [position + NULL_OFFSETS[name]],
                marker=NULL_MARKERS[name],
                s=56,
                color=color,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=3,
            )
            if name not in seen:
                seen.append(name)
        labels.append(
            f"{row['label']}\n{row['verdict']}, Holm p = {_fmt(row['p_value_holm'], '.3f')}"
        )
    axis.axvline(1, color=INK["secondary"], linewidth=1)
    axis.set_xscale("log")
    lower, upper = min(0.8, min(ratios) / 1.25), max(2.0, max(ratios) * 1.25)
    if upper / lower > 8:
        lower = min(lower, 0.45)
    axis.set_xlim(lower, upper)
    candidates = (
        (0.5, 1, 2, 5, 10, 20, 50, 100) if upper / lower > 8 else (0.5, 0.8, 1, 1.25, 1.5, 2)
    )
    ticks = [tick for tick in candidates if lower <= tick <= upper]
    axis.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_yticks(range(len(rows)), labels)
    axis.set_ylim(-0.6, len(rows) - 0.4)
    axis.set_xlabel(
        "Observed divided by surrogate mean (1 means equal counts, not equivalence)",
        fontsize=9,
        color=INK["secondary"],
    )
    order = [name for name in NULL_MARKERS if name in seen]
    axis.legend(
        [
            Line2D(
                [],
                [],
                marker=NULL_MARKERS[name],
                linestyle="",
                color=INK["secondary"],
                markersize=7,
            )
            for name in order
        ],
        [NULL_LABELS[name] for name in order],
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0, -0.2),
        ncol=1,
    )
    title = f"Q0. {chance_headline(rows)}"
    subtitle = "Blue: Holm p < 0.05 across five statistics. Null assumptions remain unverified"
    _style(axis, title, subtitle)
    _save(figure, path)


# Ordinal ramp for service levels, darker is worse, validated with the dataviz script.
LEVEL_COLORS = {
    "stalled": "#104281",
    "below_basic_use": "#2a78d6",
    "below_hd_video": "#86b6ef",
    "at_least_hd_video": AXIS,
}


def figure_q5(levels: list[dict[str, Any]], path: Path) -> None:
    """One stacked bar per group: share of eligible decision seconds in each rate band."""
    figure, axis = plt.subplots(figsize=(9.6, 0.55 * len(levels) + 1.8))
    order = ["stalled", "below_basic_use", "below_hd_video", "at_least_hd_video"]
    labels = []
    for position, item in enumerate(reversed(levels)):
        values = {name: (item["levels"][name]["value"] or 0.0) for name, _ in impact.SERVICE_LEVELS}
        segments = {
            "stalled": values["stalled"],
            "below_basic_use": values["below_basic_use"] - values["stalled"],
            "below_hd_video": values["below_hd_video"] - values["below_basic_use"],
            "at_least_hd_video": 1.0 - values["below_hd_video"],
        }
        left = 0.0
        for name in order:
            width = 100 * max(segments[name], 0.0)
            axis.barh(
                position,
                width,
                left=left,
                height=0.45,
                color=LEVEL_COLORS[name],
                edgecolor=SURFACE,
                linewidth=2,
            )
            left += width
        axis.text(
            101,
            position,
            f" {100 * values['below_basic_use']:.1f}% under 1.5, "
            f"{100 * values['stalled']:.1f}% under 0.1 Mbps",
            va="center",
            fontsize=8,
            color=INK["secondary"],
        )
        name = (
            "All sessions"
            if item["mobility"] == "all"
            else (f"{item['mobility'].capitalize()}, {item['technology']}")
        )
        labels.append(f"{name} ({item['observed_hours']:.1f} h, {item['sessions']} sessions)")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlim(0, 160)
    axis.set_xticks(range(0, 101, 20))
    axis.spines["bottom"].set_bounds(0, 100)
    axis.set_xlabel(
        "Share of eligible decision seconds (%)",
        fontsize=9,
        color=INK["secondary"],
        loc="left",
    )
    axis.legend(
        [Patch(color=LEVEL_COLORS[name]) for name in order],
        [impact.LEVEL_LABELS[name] for name in order],
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0, -0.22),
    )
    groups = [item for item in levels if item["mobility"] != "all"]
    title = "Q5. How often is the delivered rate below fixed thresholds?"
    if groups:
        worst = max(groups, key=lambda item: item["levels"]["below_basic_use"]["value"] or 0.0)
        best = min(groups, key=lambda item: item["levels"]["below_basic_use"]["value"] or 1.0)
        title = (
            f"Q5. {worst['mobility'].capitalize()} on {worst['technology']} spends "
            f"{100 * (worst['levels']['below_basic_use']['value'] or 0):.0f}% of time under "
            f"1.5 Mbps, {best['mobility']} on {best['technology']} "
            f"{100 * (best['levels']['below_basic_use']['value'] or 0):.0f}%"
        )
    _style(
        axis,
        title,
        "Throughput thresholds borrowed from Opensignal. Trailing 5 s median, rate only",
    )
    _save(figure, path)


def figure_q6(result: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 2.8))
    bars = (
        (
            "Drops overlapping a\nlow rate episode",
            result["precision"],
            f"{result['candidates_overlapping_symptom']} of {result['candidates']} drops",
        ),
        (
            "Low rate time inside\na drop window",
            result["recall_by_time"],
            f"{result['symptom_seconds_covered']:.0f} of {result['symptom_seconds']:.0f} s, "
            f"{result['missed_episodes']} of {result['episodes']} episodes never flagged",
        ),
    )
    for position, (_label, share, note) in enumerate(reversed(bars)):
        value = None if share["value"] is None else 100 * share["value"]
        interval = None if share["interval_95"] is None else [100 * x for x in share["interval_95"]]
        _interval_bar(axis, position, value, interval, SERIES_PRIMARY)
        right = max(interval[1] if interval else 0, value or 0)
        axis.text(
            right,
            position,
            f"  {value or 0:.1f}%, {note}",
            va="center",
            fontsize=8,
            color=INK["secondary"],
        )
    axis.set_yticks(range(len(bars)), [bar[0] for bar in reversed(bars)])
    axis.set_xlim(0, 150)
    axis.set_xticks(range(0, 101, 20))
    axis.spines["bottom"].set_bounds(0, 100)
    axis.set_xlabel(
        "Percent, 95% session bootstrap interval (line)",
        fontsize=9,
        color=INK["secondary"],
        loc="left",
    )
    delay = result["detection_delay_seconds"]["median"]
    delay_text = "" if delay is None else f", median detection delay {delay:.0f} s"
    _style(
        axis,
        f"Q6. Only {100 * (result['precision']['value'] or 0):.0f}% of relative drops overlap "
        "sustained low rate",
        f"Low rate: {impact.PRIMARY_MINIMUM_SECONDS} s or more under 1.5 Mbps{delay_text}",
    )
    _save(figure, path)


# Report


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _interval(estimate_value: dict[str, Any], scale: float = 1.0, digits: int = 1) -> str:
    interval = estimate_value["interval_95"]
    if interval is None:
        return f"no interval, {estimate_value['sessions']} sessions"
    return f"{scale * interval[0]:.{digits}f} to {scale * interval[1]:.{digits}f}"


def _share_text(estimate_value: dict[str, Any]) -> str:
    interval = estimate_value["interval_95"]
    span = f" ({100 * interval[0]:.0f} to {100 * interval[1]:.0f})" if interval else ""
    return f"{100 * (estimate_value['value'] or 0):.0f}%{span}"


def render_summary(summary: dict[str, Any]) -> list[str]:
    """Descriptive findings and exploratory tests, with explicit null assumptions."""
    checks = summary["chance_checks"]
    rows = {row["statistic"]: row for row in chance_rows(checks)}
    episodes, sustained = rows["low_rate_episodes"], rows["sustained_candidates"]
    per_hour, association_row = rows["candidates_per_hour"], rows["cell_change_ratio"]
    association = checks["cell_change"]
    groups = [item for item in summary["q5_service_levels"] if item["mobility"] != "all"]
    worst = max(groups, key=lambda item: item["levels"]["below_basic_use"]["value"] or 0.0)
    best = min(groups, key=lambda item: item["levels"]["below_basic_use"]["value"] or 1.0)
    monitored = [
        100 * (item["reasons"]["Monitored"]["share"] or 0.0)
        for item in summary["q4_monitoring"].values()
    ]
    missing = summary["q4_missing_by_technology"]
    worst_technology = max(missing, key=lambda name: sum(missing[name]["missing_share"].values()))
    gaps = [
        field for field, share in missing[worst_technology]["missing_share"].items() if share >= 0.5
    ]
    queue = summary["queue"]
    blocks, seconds = "shuffled_5s_blocks", "shuffled_seconds"
    detector_rows = [rows["flag_share"], per_hour, sustained]
    precision = summary["q6_detector_vs_symptom"]["precision"]["value"]
    episode_heading = (
        "Low rate time comes in stretches."
        if episodes["beats_chance"]
        else f"Low rate clustering: {episodes['verdict']}."
    )
    association_heading = (
        "Flagged windows align with cell change exposure."
        if association_row["beats_chance"]
        else f"Cell change alignment: {association_row['verdict']}."
    )
    if all(row["beats_chance"] for row in detector_rows):
        detector_heading = "Relative detector counts exceed both rate nulls."
    elif any(row["beats_chance"] for row in detector_rows):
        detector_heading = "Some relative detector counts exceed both rate nulls."
    else:
        detector_heading = f"Relative detector counts: {per_hour['verdict']}."
    episode_meaning = (
        "Throughput is autocorrelated, so this shows persistence, not a cause."
        if episodes["beats_chance"]
        else "Non-rejection does not show that the episodes are random."
    )
    if per_hour["beats_chance"]:
        detector_meaning = (
            "The detector responds to dips that persist in time, not only to the spread of rates."
        )
    elif all(
        per_hour["observed"] < item["null_mean"]
        for item in per_hour["nulls"].values()
        if item["null_mean"] is not None
    ):
        detector_meaning = (
            "Shuffled rates trigger more candidates than the real series, so the count itself is "
            "no evidence of structure. Non-rejection does not prove noise either."
        )
    else:
        detector_meaning = "Non-rejection does not prove noise."
    support = {name: checks["detector"][name]["support"] for name in chance.NULLS}
    replication = summary["q7_cell_replication"]
    alignment = summary["q6b_detector_alignment"]
    ranking = replication["ranking"]
    cell_skill = replication["primary"]["skill_against_technology"]
    cell_heading = (
        "Slow cells stay slow across days."
        if replication["decision"] == "replicates beyond technology"
        else "Cell rates do not replicate across days beyond technology."
    )
    low_in_slowest = (
        ranking["low_samples_in_slowest"] / ranking["low_samples_all_visits"]
        if ranking["low_samples_all_visits"]
        else None
    )
    minimum_rate = min(row["minimum_holm_p"] for row in [*detector_rows, episodes])
    return [
        "## Summary",
        "",
        "**What this archive shows, each result checked against chance or on held out days.** "
        "Technology and mobility rates below are descriptive. This reused archive is "
        "exploratory.",
        "",
        f"- **{cell_heading}** Predicting a held out day's visits from the same cell's other "
        f"days has {_percent(cell_skill)} lower error than predicting from the technology "
        f"(permutation p = {_fmt(replication['primary']['p_value'], '.3f')}, "
        f"{replication['primary']['scored_visits']} visits, {replication['days']} days). "
        f"{day_pattern(replication, explain=False)} "
        f"The {len(ranking['slowest'])} slowest of {ranking['cells_on_two_or_more_days']} "
        f"cells seen on two or more days hold {_percent(low_in_slowest)} of the samples under "
        "1.5 Mbps. Next step: coverage, load and backhaul data for those cells (Q7).",
        f"- **{episode_heading}** {_fmt(episodes['observed'], '.0f')} episodes of 10 s or "
        f"more under 1.5 Mbps, against {_fmt(episodes['nulls'][seconds]['null_mean'], '.1f')} "
        f"with seconds shuffled and {_fmt(episodes['nulls'][blocks]['null_mean'], '.1f')} with "
        "5 s blocks shuffled within each segment and activity state "
        f"(Holm p = {_fmt(episodes['p_value_holm'], '.3f')}). {episode_meaning} "
        f"{worst['mobility'].capitalize()} on {worst['technology']} spends "
        f"{_share_text(worst['levels']['below_basic_use'])} of eligible time under 1.5 Mbps, "
        f"{best['mobility']} on {best['technology']} "
        f"{_share_text(best['levels']['below_basic_use'])}. Next step: coverage and delivered "
        "rate measurements for the worst group.",
        f"- **{association_heading}** Flagged windows are {_fmt(association['observed'], '.1f')} "
        "times as common when a reported cell change occurs in the 10 s before confirmation, "
        f"against {_range(association['null_interval_95'])} when the exposure is shifted "
        f"within eligible runs (Holm p = {_fmt(association_row['p_value_holm'], '.3f')}). The "
        "change can fall inside the event window, so its order relative to the drop is not "
        "established, and technology and route confounding are not adjusted. Next step: "
        "signalling logs.",
        f"- **{detector_heading}** It fires {_fmt(per_hour['observed'], '.0f')} times per "
        f"eligible detector hour, against {_fmt(per_hour['nulls'][seconds]['null_mean'], '.0f')} "
        f"with seconds shuffled and {_fmt(per_hour['nulls'][blocks]['null_mean'], '.0f')} with "
        f"5 s blocks shuffled (Holm p = {_fmt(per_hour['p_value_holm'], '.3f')}). "
        f"{detector_meaning} Against sustained low rate, {_percent(precision)} of drops "
        f"overlap it, and flags are {_fmt(alignment['observed'], '.2f')} times as common inside "
        f"it as outside (chance placement p = {_fmt(alignment['p_value'], '.3f')}). "
        f"{alignment_reading(alignment)} Candidates remain triage input.",
        f"- **Data collection gaps limit diagnosis.** The detector can monitor "
        f"{min(monitored):.0f} to {max(monitored):.0f}% of download time. {worst_technology} rows "
        f"lack {', '.join(gaps) if gaps else 'no field'}, so radio rules cannot be assessed there.",
        f"- **Investigation queue.** {queue['incidents_queued']} low rate episodes map to "
        f"{queue['unique_candidates_queued']} drops queued for the Investigator.",
        "",
        "## Q0. Conditional surrogate sensitivity checks",
        "",
        f"Null models: {checks['permutations']} draws for each rate null and "
        f"{checks['shifts']} circular shifts of cell change exposure within contiguous eligible "
        "decision runs. Rate values move only within one segment and activity state. Segments "
        "end at gaps above 3 s, so a missing bin inside a segment is timing jitter, and missing "
        "bins keep their positions. The 5 s null moves blocks of five consecutive valid values. "
        "In the median draw the seconds null moved "
        f"{_percent(support[seconds]['moved_share_median'])} of valid bins by a median of "
        f"{support[seconds]['median_displacement_seconds_median']:.0f} s and the block null "
        f"{_percent(support[blocks]['moved_share_median'])} by "
        f"{support[blocks]['median_displacement_seconds_median']:.0f} s. Detector p values take "
        f"the larger of the two rate nulls, and Holm adjusts all five statistics at alpha = "
        f"{CHANCE_ALPHA}. The smallest attainable Holm p is {minimum_rate:.3f} for a rate "
        f"statistic and {association_row['minimum_holm_p']:.3f} for the cell change link.",
        "",
        "Rejection under a rate null shows that values are ordered in time within a stratum, "
        "which autocorrelated throughput is expected to show. It is not a cause or evidence of "
        "detector validity. Non-rejection is not evidence of noise. Null quantiles are surrogate "
        "ranges, not confidence intervals. Raw draws, exceedance counts and displacement per "
        "draw are in population_summary.json. The null design was repaired after an earlier run "
        "of this archive, which is recorded in docs/TECHNICAL_REPORT.md.",
        "",
        "| Pattern | Observed | Null mean | Raw p per null | Holm p | Verdict |",
        "| --- | ---: | --- | --- | ---: | --- |",
        *(
            f"| {row['label']} | {_fmt(row['observed'], '.3g')} | "
            + ", ".join(
                f"{NULL_SHORT[name]} {_fmt(item['null_mean'], '.3g')}"
                for name, item in row["nulls"].items()
            )
            + " | "
            + ", ".join(
                f"{NULL_SHORT[name]} {_fmt(item['p_value'], '.3f')}"
                for name, item in row["nulls"].items()
            )
            + f" | {_fmt(row['p_value_holm'], '.3f')} | {row['verdict']} |"
            for row in chance_rows(checks)
        ),
        "",
    ]


def render_report(summary: dict[str, Any]) -> str:
    q1, q2, q3 = summary["q1_drop_rates"], summary["q2_cell_change"], summary["q3_triage"]
    monitoring, missing = summary["q4_monitoring"], summary["q4_missing"]
    defined = [item for item in q1 if item["events_per_monitored_hour"]["value"] is not None]
    if defined:
        highest = max(defined, key=lambda item: item["events_per_monitored_hour"]["value"])
        lowest = min(defined, key=lambda item: item["events_per_monitored_hour"]["value"])
        ranking = (
            f"The highest measured rate is {highest['mobility']} {highest['technology']}, "
            f"{highest['events_per_monitored_hour']['value']:.1f} drops per monitored hour, and "
            f"the lowest is {lowest['mobility']} {lowest['technology']}, "
            f"{lowest['events_per_monitored_hour']['value']:.1f}."
        )
    else:
        ranking = "No group has monitored time."
    total_hours = sum(item["monitored_hours"] for item in q1)
    overall = (
        f"Across {total_hours:.1f} eligible decision hours the detector groups "
        f"{q3['events'] / total_hours:.0f} candidate drops per hour, about one every "
        f"{3600 * total_hours / q3['events']:.0f} s, so follow up work needs ranking (Q3)."
        if total_hours and q3["events"]
        else ""
    )
    unexplained = q3["classes"]["unexplained"]
    not_assessable = q3["classes"]["not_assessable"]
    lines = [
        "# Population analytics report",
        "",
        f"Snapshot `{summary['snapshot_id']}`, {summary['sessions']} file download sessions, "
        f"{q3['events']} drop events from the frozen detector. Plan and limits: "
        "`docs/TECHNICAL_REPORT.md`. Intervals resample whole sessions. Descriptive only. "
        "These are detector candidates, not independently confirmed network faults. "
        "Exposure counts "
        "eligible one second confirmation times, not continuous user time.",
        "",
        *render_summary(summary),
        "## Q1. How often does the relative detector fire?",
        "",
        f"{overall} {ranking} Compare technologies within one mobility group and mobility "
        "within one technology, because the groups are not balanced.",
        "",
        "| Mobility | Technology | Sessions | Candidates | Eligible decision hours "
        "| Candidates per hour | 95% interval |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in q1:
        rate = item["events_per_monitored_hour"]
        value = "n/a" if rate["value"] is None else f"{rate['value']:.1f}"
        lines.append(
            f"| {item['mobility']} | {item['technology']} | {rate['sessions']} | "
            f"{item['events']} | "
            f"{item['monitored_hours']:.2f} | {value} | {_interval(rate)} |"
        )
    near, far, association = (
        q2["drop_share_near_cell_change"],
        q2["drop_share_without_cell_change"],
        q2["ratio"],
    )
    association_text = "n/a" if association["value"] is None else f"{association['value']:.1f}"
    window_start, window_end = q2["transition_window_relative_to_confirmation_ms"]
    lines += [
        "",
        "Use: these rates describe detector activity. Diagnostic validity needs independent "
        "labels; Q0 tests surrogate sensitivity. The marginal intervals do not test "
        "paired differences between groups, and the small archive does not establish a fleet "
        "ranking.",
        "",
        "## Q2. Are drops associated with cell changes?",
        "",
        "Transition proximity is relative to confirmation time: "
        f"`[{window_start}, {window_end})` ms, start included and end excluded. "
        "The outcome is a flagged detector decision, not occupancy of a grouped drop event.",
        "",
        f"In driving sessions, eligible confirmation times with a reported transition in the "
        f"event window or its preceding 5 s were flagged {_percent(near['value'])} of the time "
        f"({_interval(near, 100)}), against "
        f"{_percent(far['value'])} otherwise ({_interval(far, 100)}). Ratio "
        f"{association_text} ({_interval(association)}).",
        "",
        "| Group | Eligible confirmation times | Flagged share | 95% interval |",
        "| --- | ---: | ---: | --- |",
        f"| Transition in or just before event window | {q2['seconds_near_cell_change']} | "
        f"{_percent(near['value'])} | {_interval(near, 100)} |",
        f"| No cell change nearby | {q2['seconds_without_cell_change']} | "
        f"{_percent(far['value'])} | {_interval(far, 100)} |",
        "",
        "Use: a ratio clearly above one justifies a mobility analysis track with signalling "
        "logs. It is an association: a reported transition is not a confirmed handover, and "
        "the overlapping windows do not establish the order or cause of the changes.",
        "",
        "## Q3. Which observation rules match candidate drops?",
        "",
        f"{unexplained['events']} of {q3['events']} drops "
        f"({_percent(unexplained['share']['value'])}) match neither the nearby-transition nor "
        f"the RSRP-fall rule. The low rate queue filter is applied below. "
        f"{not_assessable['events']} cannot be assessed "
        "because radio evidence is missing or undefined.",
        "",
        "| Class, in precedence order | Drops | Share | 95% interval | Next step |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for item in q3["classes"].values():
        lines.append(
            f"| {item['label']} | {item['events']} | {_percent(item['share']['value'])} | "
            f"{_interval(item['share'], 100)} | {item['next_step']} |"
        )
    reasons = ", ".join(f"{key} {value}" for key, value in q3["not_assessable_reasons"].items())
    lines += [
        "",
        f"Not assessable reasons: {reasons or 'none'}. Cell change drops where RSRP also fell: "
        f"{q3['cell_change_with_rsrp_fell']}.",
        "",
        "Use: route rule matches for specialist follow up. They do not explain physical cause. "
        f"Of {summary['queue']['unexplained']} drops that match neither rule, "
        f"{summary['queue']['low_rate_unexplained']} touch a sustained stretch under 1.5 Mbps "
        f"(Q5). The queue assigns {summary['queue']['incidents_queued']} low rate episodes "
        f"to {summary['queue']['unique_candidates_queued']} distinct candidates. Each episode "
        "selects its earliest overlapping unexplained candidate; duplicate event IDs are "
        "dispatched once, most severe episode first, through "
        "`celltrace investigate-queue --queue <private>/investigation_queue.json`. These are "
        "queued proposals, not completed investigations. Other relative drops remain in the "
        "audit; absence of this low rate proxy does not establish absence of user impact.",
        "",
        "## Q4. What does missing data hide?",
        "",
        "| Mobility | Evaluated seconds | " + " | ".join(MONITORING_REASONS) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in MONITORING_REASONS) + " |",
    ]
    for mobility, item in monitoring.items():
        lines.append(
            f"| {mobility} | {item['evaluated_seconds']} | "
            + " | ".join(
                _percent(item["reasons"][reason]["share"]) for reason in MONITORING_REASONS
            )
            + " |"
        )
    timing = summary["cadence"]
    if timing["bins_with_samples_but_no_rate"] == 0:
        lines += [
            "",
            "In this snapshot, the rate is never missing inside a bin that has a "
            "sample, so the coverage "
            f"rejections come from empty bins: {_percent(timing['empty_bin_share'])} of one "
            f"second bins hold no sample while {_percent(timing['multiple_sample_bin_share'])} "
            "hold two or more. This is consistent with irregular timing; audit "
            "timestamps and collection "
            "behavior before choosing a logging change.",
        ]
    lines += [
        "",
        "Predeclared Q4b. Missing shares in event windows at flagged and other "
        "eligible confirmation "
        "times. These are window-weighted sample occurrences: overlapping windows count a sample "
        "more than once. Technology mix contributes to this descriptive comparison.",
        "",
        "| Field | Missing in flagged windows | Missing in other windows | "
        "Difference, 95% interval |",
        "| --- | ---: | ---: | --- |",
    ]
    for label, item in missing.items():
        lines.append(
            f"| {label} | {_percent(item['missing_share_in_drops']['value'])} | "
            f"{_percent(item['missing_share_elsewhere']['value'])} | "
            f"{_interval(item['difference'], 100)} |"
        )
    by_technology = summary["q4_missing_by_technology"]
    field_names = list(FIELD_LABELS.values())
    lines += [
        "",
        "Q4c, a follow up added after the predeclared run. Missing share per technology over "
        "all file download samples.",
        "",
        "| Technology | Samples | " + " | ".join(field_names) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in field_names) + " |",
    ]
    for technology, item in by_technology.items():
        lines.append(
            f"| {technology} | {item['samples']} | "
            + " | ".join(_percent(item["missing_share"][name]) for name in field_names)
            + " |"
        )
    if "3G" in by_technology:
        third = by_technology["3G"]["missing_share"]
        lines += [
            "",
            f"Data finding: on 3G rows SNR is missing in {_percent(third['SNR'])}, CQI in "
            f"{_percent(third['CQI'])} and RSRQ in {_percent(third['RSRQ'])} of samples, yet "
            f"RSRP is missing in only {_percent(third['RSRP'])}. RSRP is an LTE and NR "
            "measurement; its meaning on these 3G rows is undocumented. The shared "
            "applicability rule "
            "preserves populated values but rejects them as serving-radio RSRP evidence. "
            "Field presence in this table does not imply measurement validity.",
        ]
    lines += [
        "",
        "Use: missingness is strongly associated with technology in this archive. "
        "Do not assume it is "
        "random or impute undocumented measurements. Complete-case filtering can change technology "
        "mix. The detector cannot flag ineligible times; raw samples there may still support other "
        "analyses. Audit sampling cadence and technology-specific collection support.",
        "",
    ]
    lines += render_impact(summary)
    return "\n".join(embed_figures(lines))


SECTION_FIGURES = (
    ("## Q0.", ("chance",)),
    ("## Q1.", ("q1",)),
    ("## Q2.", ("q2",)),
    ("## Q3.", ("q3",)),
    ("## Q4.", ("q4_monitoring", "q4_missing")),
    ("## Q5.", ("q5",)),
    ("## Q6.", ("q6",)),
    ("## Q7.", ("q7",)),
)


def embed_figures(lines: list[str]) -> list[str]:
    """Each section shows its figure under the heading. The tables stay as the text twin."""
    result: list[str] = []
    for line in lines:
        result.append(line)
        for prefix, keys in SECTION_FIGURES:
            if line.startswith(prefix):
                result.append("")
                for key in keys:
                    result.append(f"![{line.removeprefix('## ')}]({FIGURES[key]})")
    return result


def render_impact(summary: dict[str, Any]) -> list[str]:
    levels, episodes = summary["q5_service_levels"], summary["q5_episodes"]
    detector = summary["q6_detector_vs_symptom"]
    overall = levels[-1]["levels"]
    lines = [
        "## Q5. How often is the delivered rate below fixed thresholds?",
        "",
        "Rate-only proxies use 1.5 and 5 Mbps, borrowed from historical Opensignal throughput "
        "thresholds. Its full Consistent Quality metric also uses latency, jitter and loss; "
        "this archive cannot compute it. Below 0.1 Mbps is a project near zero rate threshold, "
        "not an observed playback stall. Each eligible decision second is judged by its "
        "trailing 5 second median, with the detector's coverage/activity rules and no baseline. "
        "Across all sessions "
        f"{_percent(overall['below_basic_use']['value'])} "
        "of eligible decision seconds have a trailing median below 1.5 Mbps "
        f"({_interval(overall['below_basic_use'], 100)}) "
        f"and {_percent(overall['stalled']['value'])} are below 0.1 Mbps.",
        "",
        "| Mobility | Technology | Sessions | Observed hours | Under 0.1 Mbps | Under 1.5 Mbps "
        "| Under 5 Mbps |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in levels:
        cells = [
            f"{_percent(item['levels'][level]['value'])} ({_interval(item['levels'][level], 100)})"
            for level, _ in impact.SERVICE_LEVELS
        ]
        lines.append(
            f"| {item['mobility']} | {item['technology']} | {item['sessions']} | "
            f"{item['observed_hours']:.2f} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        f"Sustained episodes need at least {impact.PRIMARY_MINIMUM_SECONDS} consecutive seconds "
        "with a trailing median below a level. This is a project persistence filter, not "
        "SRE multiwindow burn-rate alerting. An episode represents [first decision minus "
        "1 s, last decision); its ten-second qualification is available only at the tenth "
        "decision. This is a retrospective description. Sensitivity at other durations:",
        "",
        "| Level | "
        + " | ".join(f"Episodes, {m} s or more" for m in impact.SENSITIVITY_MINIMUM_SECONDS)
        + " | Median length, primary |",
        "| --- | " + " | ".join("---:" for _ in impact.SENSITIVITY_MINIMUM_SECONDS) + " | ---: |",
    ]
    for item in episodes.values():
        counts = " | ".join(
            str(item["minimum_seconds"][str(m)]["episodes"])
            for m in impact.SENSITIVITY_MINIMUM_SECONDS
        )
        median = item["primary_median_seconds"]
        lines.append(
            f"| {item['label']} | {counts} | {'n/a' if median is None else f'{median} s'} |"
        )
    delay = detector["detection_delay_seconds"]
    lines += [
        "",
        "Limits: the kbps unit of the archive is inferred from magnitude, and absolute levels "
        "depend on it while relative drops do not. The archive has almost no latency under "
        "load, which Apple's networkQuality and the IETF Responsiveness draft treat as a core "
        "symptom, so that experience cannot be measured here and is a data collection gap.",
        "",
        "## Q6. How often do relative drops coincide with sustained low rate?",
        "",
        "Overlap between two rate rules, not detector accuracy. Low rate episode: "
        f"{detector['symptom']}, sustained {detector['minimum_seconds']} s or more.",
        "",
        "| Measure | Value | 95% interval | Counts |",
        "| --- | ---: | --- | --- |",
        f"| Drops overlapping a low rate episode | "
        f"{_percent(detector['precision']['value'])} | {_interval(detector['precision'], 100)} | "
        f"{detector['candidates_overlapping_symptom']} of {detector['candidates']} drops |",
        f"| Low rate time inside a drop window | "
        f"{_percent(detector['recall_by_time']['value'])} | "
        f"{_interval(detector['recall_by_time'], 100)} | "
        f"{detector['symptom_seconds_covered']:.0f} of {detector['symptom_seconds']:.0f} s |",
        f"| Low rate episodes touched by any drop | "
        f"{_percent(detector['episodes_detected_share']['value'])} | "
        f"{_interval(detector['episodes_detected_share'], 100)} | "
        f"{detector['episodes_detected']} of {detector['episodes']} episodes |",
        "",
        f"Median detection delay {_fmt(delay['median'], '.1f')} s, 90th percentile "
        f"{_fmt(delay['p90'], '.1f')} s "
        "(linear interpolation), from the "
        "start of an episode's decision intervals to the first overlapping confirmation. "
        f"{detector['missed_episodes']} low rate episodes never overlap a relative drop. "
        "The overlap count alone does not establish whether poor initial rate, insufficient "
        "reference coverage or another detector condition explains each miss. Their next "
        "step is a separate low rate investigation with demand, coverage and transport checks.",
        "",
        "Sensitivity across levels, same 10 s duration:",
        "",
        "| Low rate level | Drops overlapping | Time inside drop windows | Episodes touched |",
        "| --- | ---: | ---: | ---: |",
        *(
            f"| {item['label']} | {_percent(item['precision']['value'])} | "
            f"{_percent(item['recall_by_time']['value'])} | "
            f"{item['episodes_detected']} of {item['episodes']} |"
            for item in detector["by_level"].values()
        ),
        "",
        "These scores measure agreement between two rate rules, not accuracy against "
        "independently labeled user harm or physical incidents. Unmatched drops can still "
        "matter to higher-rate applications. Time coverage and episode recall have different "
        "denominators. The queue is a retrospective cost-control policy; its smaller size "
        "does not establish fewer false alarms, successful investigations or better recall.",
        "",
        *render_alignment(summary["q6b_detector_alignment"]),
    ]
    return lines + render_cells(summary["q7_cell_replication"])


def alignment_reading(alignment: dict[str, Any]) -> str:
    if alignment["p_value"] is None:
        return "The alignment is undefined on this archive."
    if alignment["p_value"] < CHANCE_ALPHA:
        return "Alerts concentrate on sustained low rate beyond chance placement."
    return "Alert placement carries no information about sustained low rate."


def render_alignment(alignment: dict[str, Any]) -> list[str]:
    counts = alignment.get("observed_counts")
    inside = (
        f" ({counts[0]:.0f} of {counts[1]:.0f} eligible decisions inside episodes flagged, "
        f"{counts[2]:.0f} of {counts[3]:.0f} outside)"
        if counts
        else ""
    )
    return [
        "Q6b, frozen on 12 October 2026 before it was computed: flagged decisions are "
        f"{_fmt(alignment['observed'], '.2f')} times as common inside sustained low rate "
        f"episodes as outside them{inside}, against {_range(alignment['null_interval_95'])} "
        "when the episode indicator is shifted within eligible runs "
        f"(p = {_fmt(alignment['p_value'], '.3f')}, {len(alignment['null_draws'])} shifts). "
        f"{alignment_reading(alignment)} This is its own test, outside the Q0 family.",
        "",
    ]


def _value_range(bounds: list[float]) -> str:
    return f"{bounds[0]:.1f} to {bounds[1]:.1f}"


def day_pattern(result: dict[str, Any], explain: bool = True) -> str:
    days = result["per_day"]
    worse = [item["fold"] for item in days if item["skill"] is not None and item["skill"] <= 0]
    if not days:
        return "No held out day could be scored."
    text = (
        f"The same cell beats its technology on {len(days) - len(worse)} of {len(days)} "
        "held out days."
    )
    if worse:
        listed = worse[0] if len(worse) == 1 else f"{', '.join(worse[:-1])} and {worse[-1]}"
        text += f" It does worse on {listed}."
        if explain:
            text += (
                " Cells can change over weeks, and this archive cannot tell why, so a ranking "
                "needs refreshing."
            )
    return text


def _factor(mae_log10: float | None) -> str:
    return "n/a" if mae_log10 is None else f"{10**mae_log10:.2f}"


def render_cells(result: dict[str, Any]) -> list[str]:
    primary = result["primary"]
    sensitivity = result["sensitivity"]
    ranking = result["ranking"]
    larger, by_session = (
        sensitivity["minimum_visit_samples_20"],
        sensitivity["leave_one_session_out"],
    )
    low = result["low_share_error"]
    correlation = result["rank_correlation"]
    low_in_slowest = (
        ranking["low_samples_in_slowest"] / ranking["low_samples_all_visits"]
        if ranking["low_samples_all_visits"]
        else None
    )
    samples_in_slowest = (
        ranking["samples_in_slowest"] / ranking["samples_all_visits"]
        if ranking["samples_all_visits"]
        else None
    )
    return [
        "## Q7. Does a serving cell keep its delivered rate across days?",
        "",
        "Frozen on 12 October 2026 before any per cell rate was computed. A visit is one "
        f"driving session on one serving cell with at least {result['minimum_visit_samples']} "
        "active download samples, valued by its median rate. Each held out day is predicted "
        f"from the other days only. {result['visits']} visits on {result['cells']} cells from "
        f"{result['sessions']} sessions over {result['days']} days. {primary['scored_visits']} "
        f"visits on {primary['scored_cells']} cells meet the same cell on another day and are "
        "scored. Errors are on log10 of the rate, so an error of 0.30 means a factor of 2.",
        "",
        "| Prediction from other days | Mean absolute error, log10 | Typical factor |",
        "| --- | ---: | ---: |",
        f"| Same cell | {_fmt(primary['mae_cell'], '.3f')} | {_factor(primary['mae_cell'])} |",
        f"| Same technology | {_fmt(primary['mae_technology'], '.3f')} | "
        f"{_factor(primary['mae_technology'])} |",
        f"| All visits | {_fmt(primary['mae_global'], '.3f')} | {_factor(primary['mae_global'])} |",
        "",
        f"Skill of the cell against its technology: "
        f"{_percent(primary['skill_against_technology'])}. With cell labels permuted within day "
        f"and technology, skill averages {_percent(primary['null_mean'])} (95% of "
        f"{primary['permutations']} permutations between "
        + (
            f"{_percent(primary['null_interval_95'][0])} and "
            f"{_percent(primary['null_interval_95'][1])}"
            if primary["null_interval_95"]
            else "n/a"
        )
        + f"), p = {_fmt(primary['p_value'], '.3f')}. Decision by the frozen rule: the cell "
        f"rate {result['decision']}."
        + (
            f" Technology fell back to all visits for {primary['technology_fallbacks']} visits."
            if primary["technology_fallbacks"]
            else ""
        ),
        "",
        "| Held out day | Visits | Error, same cell | Error, same technology | Skill |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            f"| {item['fold']} | {item['visits']} | {item['mae_cell']:.3f} | "
            f"{item['mae_technology']:.3f} | {_percent(item['skill'])} |"
            for item in result["per_day"]
        ),
        "",
        day_pattern(result),
        "",
        "Sensitivity, not decisions: with at least 20 samples per visit the skill is "
        f"{_percent(larger['skill_against_technology'])} (p = {_fmt(larger['p_value'], '.3f')}, "
        f"{larger['scored_visits']} visits), and leaving one session out instead of one day it "
        f"is {_percent(by_session['skill_against_technology'])} "
        f"(p = {_fmt(by_session['p_value'], '.3f')}). Rank correlation between the cell "
        f"prediction and the held out rate: 5G {_fmt(correlation['5G']['spearman'], '.2f')} "
        f"({correlation['5G']['visits']} visits), LTE "
        f"{_fmt(correlation['LTE']['spearman'], '.2f')} ({correlation['LTE']['visits']} "
        "visits). Share of samples under 1.5 Mbps, mean absolute error: same cell "
        f"{_fmt(low['mae_cell'], '.3f')}, same technology {_fmt(low['mae_technology'], '.3f')}.",
        "",
        f"Slowest of the {ranking['cells_on_two_or_more_days']} cells seen on two or more days. "
        "The ranking uses every day, so it is descriptive. The test above is what makes it "
        "worth acting on.",
        "",
        "| Cell | Technology | Days | Visits | Samples | Median visit rate (Mbps) "
        "| Visit medians, lowest to highest (Mbps) | Share under 1.5 Mbps "
        "| Median reported RSRP (dBm) |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        *(
            f"| {item['cell']} | {item['technology']} | {item['days']} | {item['visits']} | "
            f"{item['samples']} | {item['median_visit_mbps']:.1f} | "
            f"{_value_range(item['visit_median_range_mbps'])} | "
            f"{_percent(item['low_share'])} | {_fmt(item['median_rsrp_dbm'], '.0f')} |"
            for item in ranking["slowest"]
        ),
        "",
        "The range and RSRP columns were added after the frozen run and are descriptive. RSRP "
        "is undefined on 3G rows. Good signal with a low rate does not establish congestion, "
        "and weak signal does not establish a coverage hole.",
        "",
        f"These {len(ranking['slowest'])} cells hold {_percent(low_in_slowest)} of the "
        f"under 1.5 Mbps samples in all visits and {_percent(samples_in_slowest)} of all "
        "visit samples.",
        "",
        "Use: a cell that stays slow across days is a candidate for a coverage, load or "
        "backhaul check, which this archive cannot separate. What is measured is the cell's "
        "area as driven on these routes. Time of day is not controlled, and one operator's "
        "cells in one city do not make a network wide ranking.",
        "",
    ]


TECHNOLOGY_COLORS = {"5G": SERIES_PRIMARY, "LTE": SERIES_SECONDARY, "3G": SERIES_TERTIARY}


def q7_title(result: dict[str, Any]) -> str:
    skill_value = result["primary"]["skill_against_technology"]
    if result["decision"] == "replicates beyond technology":
        return (
            f"Q7. A cell's other days predict a new day with "
            f"{100 * skill_value:.0f}% less error than technology"
        )
    return "Q7. A cell's other days predict a new day no better than its technology"


def figure_q7(result: dict[str, Any], path: Path) -> None:
    """Held out visit rate against the same cell's other days, both relative to technology.

    Dividing by the technology prediction removes the 5G, LTE and 3G separation, which
    would otherwise make the cell look predictive by technology alone.
    """
    figure, axis = plt.subplots(figsize=(9.6, 6.4))
    rows = result["scored_rows"]
    correlation = result.get("rank_correlation", {})
    ratios = [
        abs(row[key] - row["technology_prediction"])
        for row in rows
        for key in ("actual", "cell_prediction")
    ]
    for technology, color in TECHNOLOGY_COLORS.items():
        members = [row for row in rows if row["technology"] == technology]
        if not members:
            continue
        axis.scatter(
            [10 ** (row["cell_prediction"] - row["technology_prediction"]) for row in members],
            [10 ** (row["actual"] - row["technology_prediction"]) for row in members],
            s=26,
            color=color,
            edgecolors=SURFACE,
            linewidths=0.6,
            label=f"{technology}, {len(members)} visits"
            + (
                f", rank correlation {_fmt(correlation[technology]['spearman'], '.2f')}"
                if technology in correlation
                else ""
            ),
            zorder=3,
        )
    if ratios:
        extent = 10 ** (max(ratios) + 0.2)
        limits = (1 / extent, extent)
        axis.plot(limits, limits, color=INK["secondary"], linewidth=1, zorder=2)
        axis.axhline(1, color=INK["muted"], linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
    axis.set_xscale("log")
    axis.set_yscale("log")
    for scale in (axis.xaxis, axis.yaxis):
        scale.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_xlabel(
        "Same cell's other days, ratio to the technology median",
        fontsize=9,
        color=INK["secondary"],
    )
    axis.set_ylabel(
        "Held out visit, ratio to the technology median", fontsize=9, color=INK["secondary"]
    )
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    primary = result["primary"]
    subtitle = (
        f"Leave one day out, {primary['scored_visits']} visits on {primary['scored_cells']} "
        f"cells, p = {_fmt(primary['p_value'], '.3f')}. Diagonal: the cell explains the "
        f"visit. Dashed: technology alone"
    )
    _style(axis, q7_title(result), subtitle)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines["left"].set_visible(True)
    axis.spines["left"].set_color(AXIS)
    _save(figure, path)


# Entry point


def run_population(
    snapshot_dir: Path,
    out_dir: Path,
    private_dir: Path,
    detector: DetectorSection,
    observation: ObservationSection,
    seed: int,
    chance_permutations: int = CHANCE_PERMUTATIONS,
    chance_shifts: int = CHANCE_SHIFTS,
) -> dict[str, Any]:
    """Write the aggregate report and figures publicly and per event rows privately."""
    public = out_dir.resolve()
    private = private_dir.resolve()
    if public == private or public in private.parents or private in public.parents:
        raise ValueError(
            "public and private output directories must be separate and non-overlapping"
        )
    targets = [
        out_dir / SUMMARY_NAME,
        out_dir / REPORT_NAME,
        *(out_dir / name for name in FIGURES.values()),
        private_dir / TRIAGE_NAME,
        private_dir / QUEUE_NAME,
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("preserve the previous population analysis outputs")
    connection, manifest = open_snapshot(snapshot_dir)
    try:
        sessions = connection.execute(
            "SELECT session_id, mobility_cohort FROM sessions WHERE application_cohort = ? "
            "ORDER BY session_id",
            [PRIMARY_COHORT],
        ).fetchall()
        totals: dict[str, SessionTotals] = {}
        context: dict[
            str,
            tuple[
                dict[int, set[str | None]], list[int], dict[tuple[str, int], list[impact.Episode]]
            ],
        ] = {}
        for session_id, mobility in sessions:
            session_totals, modes, transitions, episodes = scan_session(
                connection, session_id, str(mobility), detector
            )
            totals[session_id] = session_totals
            context[session_id] = (modes, transitions, episodes)
        chance_inputs = [
            chance.SessionInput(
                str(session_id),
                str(mobility),
                load_bins(connection, session_id),
                load_segments(connection, session_id),
                context[session_id][1],
            )
            for session_id, mobility in sessions
        ]
        chance_checks: dict[str, Any] = {
            "permutations": chance_permutations,
            "shifts": chance_shifts,
            "alpha": CHANCE_ALPHA,
            "seed": seed,
            "method": (
                "segment and activity stratified rate surrogates and eligible run exposure shifts"
            ),
            "multiple_testing": "maximum across detector nulls, then Holm across five statistics",
            "interpretation": "exploratory, exchangeability and shift invariance unverified",
            "detector": chance.detector_chance(chance_inputs, detector, chance_permutations, seed),
            "cell_change": chance.association_chance(
                chance_inputs, detector, NEARBY_TRANSITION_LOOKBACK_MS, chance_shifts, seed
            ),
        }
        detector_alignment = chance.low_rate_alignment_chance(
            chance_inputs, detector, chance_shifts, seed
        )
        cell_replication = cells.cell_replication(connection, PRIMARY_COHORT, seed)
        detection = detect_candidates(connection, detector, PRIMARY_COHORT)
        candidates = detection.candidates
        by_session: dict[str, list[CandidateEvent]] = {}
        for candidate in candidates:
            by_session.setdefault(candidate.session_id, []).append(candidate)
        for session_id, (_, _, episodes) in context.items():
            windows = [
                (c.event_window.start_ms, c.event_window.end_ms, c.first_confirmation_ms)
                for c in by_session.get(session_id, [])
            ]
            # Every level is scored for transparency. The symptom level is the primary result.
            for level, _ in impact.SERVICE_LEVELS:
                scored = impact.compare_with_detector(
                    windows,
                    episodes[(level, impact.PRIMARY_MINIMUM_SECONDS)],
                    [
                        (
                            flag["confirmation_ms"] - detector.event_ms,
                            flag["confirmation_ms"],
                            flag["confirmation_ms"],
                        )
                        for flag in detection.flags
                        if flag["session_id"] == session_id
                    ],
                )
                for name, value in scored.items():
                    if isinstance(value, list):
                        if level == impact.SYMPTOM_LEVEL:
                            totals[session_id].detection_delays_ms.extend(value)
                    else:
                        totals[session_id].symptom[f"{level}:{name}"] += value
        durations: dict[str, list[int]] = {}
        for _, _, episodes in context.values():
            for level, _ in impact.SERVICE_LEVELS:
                durations.setdefault(level, []).extend(
                    episode.seconds for episode in episodes[(level, impact.PRIMARY_MINIMUM_SECONDS)]
                )
        repository = MeasurementRepository(connection)
        triage_rows = []
        for candidate in candidates:
            modes, transitions, episodes = context[candidate.session_id]
            row = classify_event(candidate, modes, transitions, repository, observation)
            row["mobility"] = totals[candidate.session_id].mobility
            level, seconds, minimum, _ = impact.candidate_impact(
                (candidate.event_window.start_ms, candidate.event_window.end_ms),
                {
                    name: episodes[(name, impact.PRIMARY_MINIMUM_SECONDS)]
                    for name, _ in impact.SERVICE_LEVELS
                },
            )
            # One incident is one sustained episode of the symptom level. Several drops can
            # fall inside it, and only its earliest drop needs an investigation.
            symptom_start = impact.candidate_impact(
                (candidate.event_window.start_ms, candidate.event_window.end_ms),
                {
                    impact.SYMPTOM_LEVEL: episodes[
                        (impact.SYMPTOM_LEVEL, impact.PRIMARY_MINIMUM_SECONDS)
                    ]
                },
            )[3]
            row.update(
                impact_level=level,
                impact_seconds=seconds,
                impact_minimum_mbps=minimum,
                symptom_incident_start_ms=symptom_start,
                first_confirmation_ms=candidate.first_confirmation_ms,
                impact_window_start_ms=candidate.event_window.start_ms,
                impact_window_end_ms=candidate.event_window.end_ms,
            )
            triage_rows.append(row)
            totals[candidate.session_id].events[row["technology"]] += 1
            totals[candidate.session_id].triage[row["triage_class"]] += 1
            if level in impact.SYMPTOM_LEVELS:
                totals[candidate.session_id].impact_triage[row["triage_class"]] += 1
        missing_by_mode = missing_by_technology(connection)
        logging_cadence = cadence(connection)
    finally:
        connection.close()

    # Only drops that coincide with sustained low delivered rate reach the paid agent, and
    # each incident is investigated once, from its earliest drop.
    assignments = impact.incident_queue(
        triage_rows,
        {
            session_id: {
                level: episodes[(level, impact.PRIMARY_MINIMUM_SECONDS)]
                for level, _ in impact.SERVICE_LEVELS
            }
            for session_id, (_, _, episodes) in context.items()
        },
    )
    queue_ids = list(dict.fromkeys(str(row["event_id"]) for row in assignments))
    summary: dict[str, Any] = {
        "generated_by": "celltrace.analysis.population",
        "method_version": "population_descriptive_v9_interval_minimum_eight",
        "source_sha256": source_tree_hash(),
        "plan": "docs/TECHNICAL_REPORT.md",
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_logical_hash": manifest["logical_hash"],
        "detector_config_hash": detector_config_hash(detector),
        "observation_config": observation.model_dump(mode="json"),
        "sessions": len(totals),
        "bootstrap": {
            "unit": "session",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": seed,
            "minimum_sessions_for_interval": MINIMUM_SESSIONS_FOR_INTERVAL,
        },
        "q1_drop_rates": q1_drop_rates(totals, seed),
        "q2_cell_change": {
            **q2_cell_change(totals, seed),
            "transition_window_relative_to_confirmation_ms": [
                -detector.event_ms - NEARBY_TRANSITION_LOOKBACK_MS,
                0,
            ],
        },
        "q3_triage": q3_triage(totals, triage_rows, seed),
        "q4_monitoring": q4_monitoring(totals),
        "q4_missing": q4_missing(totals, seed),
        "q4_missing_weighting": "sample occurrences in overlapping eligible event "
        "windows; repeated samples count repeatedly",
        "q4_missing_by_technology": missing_by_mode,
        "q4_missing_by_technology_note": "follow up added after the predeclared run",
        "cadence": logging_cadence,
        "impact_definitions": {
            "plan": "docs/TECHNICAL_REPORT.md",
            "service_levels_mbps": dict(impact.SERVICE_LEVELS),
            "level_sources": "Historical Opensignal throughput components: 1.5 and 5 Mbps; "
            "other consistency components unavailable. Near zero 0.1 Mbps is a project choice",
            "short_window_ms": impact.SHORT_WINDOW_MS,
            "primary_minimum_seconds": impact.PRIMARY_MINIMUM_SECONDS,
            "sensitivity_minimum_seconds": list(impact.SENSITIVITY_MINIMUM_SECONDS),
            "symptom_level": impact.SYMPTOM_LEVEL,
            "unit_assumption": "kbps inferred from magnitude; absolute levels depend on it",
            "episode_interval": "[first low decision - 1000, last low decision)",
            "episode_confirmation": "minimum_seconds-th consecutive low decision",
            "interpretation": "retrospective rate-only proxy; no independent QoE labels",
        },
        "q5_service_levels": q5_service_levels(totals, seed),
        "q5_episodes": q5_episodes(totals, durations, seed),
        "q6_detector_vs_symptom": q6_detector_against_symptom(totals, seed),
        "queue": {
            "unexplained": sum(row["triage_class"] == "unexplained" for row in triage_rows),
            "low_rate_unexplained": sum(
                row["triage_class"] == "unexplained"
                and row["impact_level"] in impact.SYMPTOM_LEVELS
                for row in triage_rows
            ),
            "incidents_queued": len(assignments),
            "unique_candidates_queued": len(queue_ids),
            "low_rate_by_class": dict(
                sum((item.impact_triage for item in totals.values()), Counter())
            ),
        },
        "chance_checks": chance_checks,
        "q6b_detector_alignment": detector_alignment,
        "q7_cell_replication": cell_replication,
        "figures": FIGURES,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    figure_chance(chance_rows(chance_checks), out_dir / FIGURES["chance"])
    figure_q1(
        summary["q1_drop_rates"],
        out_dir / FIGURES["q1"],
        chance_checks["detector"]["shuffled_seconds"]["candidates_per_hour"]["null_mean"],
    )
    figure_q2(summary["q2_cell_change"], out_dir / FIGURES["q2"], chance_checks["cell_change"])
    figure_q3(summary["q3_triage"], out_dir / FIGURES["q3"])
    figure_q4_monitoring(summary["q4_monitoring"], out_dir / FIGURES["q4_monitoring"])
    figure_q4_missing(summary["q4_missing_by_technology"], out_dir / FIGURES["q4_missing"])
    figure_q5(summary["q5_service_levels"], out_dir / FIGURES["q5"])
    figure_q6(summary["q6_detector_vs_symptom"], out_dir / FIGURES["q6"])
    figure_q7(summary["q7_cell_replication"], out_dir / FIGURES["q7"])
    (out_dir / REPORT_NAME).write_text(render_report(summary), encoding="utf-8")
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (private_dir / TRIAGE_NAME).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(triage_rows[0]) if triage_rows else ["event_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(triage_rows)
    by_id = {candidate.event_id: candidate for candidate in candidates}
    (private_dir / QUEUE_NAME).write_text(
        json.dumps(
            {
                "generated_by": "celltrace.analysis.population",
                "snapshot_id": manifest["snapshot_id"],
                "detector_config_hash": detector_config_hash(detector),
                "selection": "triage class unexplained and touching a sustained stretch below "
                "1.5 Mbps, earliest candidate per episode; duplicate event IDs dispatched once. "
                "Rank whole episodes by sustained severity, duration and lowest median. "
                "Retrospective selection; no online latency or QoE guarantee",
                "incident_assignments": [
                    {
                        key: row[key]
                        for key in (
                            "event_id",
                            "session_id",
                            "symptom_incident_start_ms",
                            "symptom_incident_end_ms",
                            "symptom_confirmation_ms",
                            "impact_level",
                            "impact_seconds",
                            "impact_minimum_mbps",
                        )
                    }
                    for row in assignments
                ],
                "candidates": [by_id[event_id].model_dump(mode="json") for event_id in queue_ids],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
