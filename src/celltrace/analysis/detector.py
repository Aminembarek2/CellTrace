"""Transparent sustained relative drop detector over one second bins.

At each confirmation time t on a bin boundary the detector compares the median
bin downlink rate of the reference window [t - reference_ms - event_ms,
t - event_ms) with the event window [t - event_ms, t). Both windows must lie
inside one segment, use complete bins that are available by t, and meet the
coverage and activity minimums. A flag is a retrospective confirmation, so the
detection delay is at least event_ms by construction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from celltrace.canonical import canonical_json, content_hash
from celltrace.config import DetectorSection
from celltrace.contracts.common import TimeWindow
from celltrace.contracts.evidence import CandidateEvent
from celltrace.data.raca import BIN_MS
from celltrace.data.snapshot import open_snapshot

DETECTOR_VERSION = "sustained_relative_drop_v1"
CANDIDATES_FILE_NAME = "candidates.json"

ELIGIBILITY_STAGES: tuple[str, ...] = (
    "evaluated_times",
    "rejected_window_not_inside_one_segment",
    "after_segment_containment",
    "rejected_incomplete_bin",
    "rejected_not_yet_available",
    "after_completeness_and_availability",
    "rejected_reference_coverage_below_minimum",
    "rejected_event_coverage_below_minimum",
    "after_coverage",
    "rejected_reference_active_fraction_below_minimum",
    "rejected_event_active_fraction_below_minimum",
    "after_activity",
    "rejected_reference_below_minimum_mbps",
    "eligible_times",
    "not_flagged_relative_drop_below_minimum",
    "not_flagged_absolute_drop_below_minimum",
    "flagged_times",
)


@dataclass(frozen=True)
class BinView:
    segment_id: int
    bin_start_ms: int
    available_at_ms: int
    is_complete: bool
    sample_count: int
    active_count: int
    downlink_mbps: float | None
    downlink_valid_count: int


@dataclass(frozen=True)
class WindowStatistics:
    start_ms: int
    end_ms: int
    expected_bins: int
    valid_bins: int
    sample_count: int
    active_count: int
    median_mbps: float | None

    @property
    def coverage(self) -> float:
        return self.valid_bins / self.expected_bins

    @property
    def active_fraction(self) -> float | None:
        if self.sample_count == 0:
            return None
        return self.active_count / self.sample_count


@dataclass
class DetectionResult:
    candidates: list[CandidateEvent]
    flags: list[dict[str, Any]]
    eligibility_counts: dict[str, int]
    config_hash: str
    session_flag_counts: dict[str, int] = field(default_factory=dict)
    session_eligibility_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    candidate_details: list[dict[str, Any]] = field(default_factory=list)


def interpolated_median(values: Sequence[float]) -> float:
    """Median with linear interpolation, matching DuckDB median() for DOUBLE values."""
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("median of an empty sequence")
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def detector_config_hash(detector_config: DetectorSection) -> str:
    return content_hash(
        {
            "detector_version": DETECTOR_VERSION,
            "config": detector_config.model_dump(mode="json"),
        }
    )


def make_event_id(
    session_id: str, segment_id: int, first_confirmation_ms: int, config_hash: str
) -> str:
    payload = canonical_json([session_id, segment_id, first_confirmation_ms, config_hash])
    return "evt_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _window_statistics(bins: dict[int, BinView], start_ms: int, end_ms: int) -> WindowStatistics:
    members = [bins[start] for start in range(start_ms, end_ms, BIN_MS)]
    valid = [
        item.downlink_mbps
        for item in members
        if item.downlink_valid_count > 0 and item.downlink_mbps is not None
    ]
    return WindowStatistics(
        start_ms=start_ms,
        end_ms=end_ms,
        expected_bins=(end_ms - start_ms) // BIN_MS,
        valid_bins=len(valid),
        sample_count=sum(item.sample_count for item in members),
        active_count=sum(item.active_count for item in members),
        median_mbps=interpolated_median(valid) if valid else None,
    )


def load_bins(connection: duckdb.DuckDBPyConnection, session_id: str) -> dict[int, BinView]:
    rows = connection.execute(
        """
        SELECT segment_id, bin_start_ms, available_at_ms, is_complete, sample_count,
               active_count, downlink_mbps, downlink_mbps_valid_count
        FROM bins_1s WHERE session_id = ? ORDER BY bin_start_ms
        """,
        [session_id],
    ).fetchall()
    return {
        int(row[1]): BinView(
            segment_id=int(row[0]),
            bin_start_ms=int(row[1]),
            available_at_ms=int(row[2]),
            is_complete=bool(row[3]),
            sample_count=int(row[4]),
            active_count=int(row[5]),
            downlink_mbps=None if row[6] is None else float(row[6]),
            downlink_valid_count=int(row[7]),
        )
        for row in rows
    }


def load_segments(
    connection: duckdb.DuckDBPyConnection, session_id: str
) -> list[tuple[int, int, int]]:
    rows = connection.execute(
        "SELECT segment_id, start_ms, end_ms FROM segments "
        "WHERE session_id = ? ORDER BY segment_id",
        [session_id],
    ).fetchall()
    return [(int(row[0]), int(row[1]), int(row[2])) for row in rows]


def _containing_segment(
    segments: list[tuple[int, int, int]], bins: dict[int, BinView], start_ms: int, end_ms: int
) -> int | None:
    for segment_id, segment_start, segment_end in segments:
        if segment_start <= start_ms and end_ms <= segment_end:
            if all(
                start in bins and bins[start].segment_id == segment_id
                for start in range(start_ms, end_ms, BIN_MS)
            ):
                return segment_id
            return None
    return None


@dataclass(frozen=True)
class TimeOutcome:
    """The detector decision at one confirmation time: stages passed and where it stopped."""

    confirmation_ms: int
    passed: tuple[str, ...]
    outcome: str
    flag: dict[str, Any] | None = None

    @property
    def eligible(self) -> bool:
        return "eligible_times" in self.passed


def evaluate_times(
    session_id: str,
    bins: dict[int, BinView],
    segments: list[tuple[int, int, int]],
    config: DetectorSection,
) -> Iterator[TimeOutcome]:
    """Yield the decision at every confirmation time, the single source of eligibility."""
    if not segments:
        return
    session_end = max(segment_end for _, _, segment_end in segments)
    window_ms = config.reference_ms + config.event_ms
    for confirmation in range(config.step_ms, session_end + 1, config.step_ms):
        passed: list[str] = []
        reference_start = confirmation - window_ms
        event_start = confirmation - config.event_ms
        segment_id = (
            _containing_segment(segments, bins, reference_start, confirmation)
            if reference_start >= 0
            else None
        )
        if segment_id is None:
            yield TimeOutcome(confirmation, tuple(passed), "rejected_window_not_inside_one_segment")
            continue
        passed.append("after_segment_containment")
        window_bins = [bins[start] for start in range(reference_start, confirmation, BIN_MS)]
        if not all(item.is_complete for item in window_bins):
            yield TimeOutcome(confirmation, tuple(passed), "rejected_incomplete_bin")
            continue
        if max(item.available_at_ms for item in window_bins) > confirmation:
            yield TimeOutcome(confirmation, tuple(passed), "rejected_not_yet_available")
            continue
        passed.append("after_completeness_and_availability")
        reference = _window_statistics(bins, reference_start, event_start)
        event = _window_statistics(bins, event_start, confirmation)
        if reference.coverage < config.minimum_coverage:
            yield TimeOutcome(
                confirmation, tuple(passed), "rejected_reference_coverage_below_minimum"
            )
            continue
        if event.coverage < config.minimum_coverage:
            yield TimeOutcome(confirmation, tuple(passed), "rejected_event_coverage_below_minimum")
            continue
        passed.append("after_coverage")
        reference_active = reference.active_fraction
        event_active = event.active_fraction
        if reference_active is None or reference_active < config.minimum_active_fraction:
            yield TimeOutcome(
                confirmation, tuple(passed), "rejected_reference_active_fraction_below_minimum"
            )
            continue
        if event_active is None or event_active < config.minimum_active_fraction:
            yield TimeOutcome(
                confirmation, tuple(passed), "rejected_event_active_fraction_below_minimum"
            )
            continue
        passed.append("after_activity")
        assert reference.median_mbps is not None and event.median_mbps is not None
        if reference.median_mbps < config.minimum_reference_mbps:
            yield TimeOutcome(confirmation, tuple(passed), "rejected_reference_below_minimum_mbps")
            continue
        passed.append("eligible_times")
        absolute_drop = reference.median_mbps - event.median_mbps
        relative_drop = absolute_drop / reference.median_mbps
        if relative_drop < config.minimum_relative_drop:
            yield TimeOutcome(
                confirmation, tuple(passed), "not_flagged_relative_drop_below_minimum"
            )
            continue
        if absolute_drop < config.minimum_absolute_drop_mbps:
            yield TimeOutcome(
                confirmation, tuple(passed), "not_flagged_absolute_drop_below_minimum"
            )
            continue
        yield TimeOutcome(
            confirmation,
            tuple(passed),
            "flagged_times",
            {
                "session_id": session_id,
                "segment_id": segment_id,
                "confirmation_ms": confirmation,
                "reference_window": {"start_ms": reference.start_ms, "end_ms": reference.end_ms},
                "event_window": {"start_ms": event.start_ms, "end_ms": event.end_ms},
                "reference_median_mbps": reference.median_mbps,
                "event_median_mbps": event.median_mbps,
                "absolute_drop_mbps": absolute_drop,
                "relative_drop": relative_drop,
                "reference_valid_bins": reference.valid_bins,
                "reference_expected_bins": reference.expected_bins,
                "reference_coverage": reference.coverage,
                "event_valid_bins": event.valid_bins,
                "event_expected_bins": event.expected_bins,
                "event_coverage": event.coverage,
                "reference_active_count": reference.active_count,
                "reference_sample_count": reference.sample_count,
                "reference_active_fraction": reference_active,
                "event_active_count": event.active_count,
                "event_sample_count": event.sample_count,
                "event_active_fraction": event_active,
            },
        )


def short_window_rate(
    bins: dict[int, BinView],
    segments: list[tuple[int, int, int]],
    end_ms: int,
    window_ms: int,
    config: DetectorSection,
) -> float | None:
    """Median rate of [end_ms - window_ms, end_ms) under the detector's own window rules.

    The window must sit inside one segment, use complete bins available by end_ms and
    meet the coverage and activity minimums. No reference window is required, so a rate
    that is low from the start still counts. Returns None when the window is not usable.
    """
    start_ms = end_ms - window_ms
    if start_ms < 0 or _containing_segment(segments, bins, start_ms, end_ms) is None:
        return None
    members = [bins[start] for start in range(start_ms, end_ms, BIN_MS)]
    if not all(item.is_complete for item in members):
        return None
    if max(item.available_at_ms for item in members) > end_ms:
        return None
    window = _window_statistics(bins, start_ms, end_ms)
    active_fraction = window.active_fraction
    if window.coverage < config.minimum_coverage or active_fraction is None:
        return None
    if active_fraction < config.minimum_active_fraction:
        return None
    return window.median_mbps


def _evaluate_session(
    session_id: str,
    bins: dict[int, BinView],
    segments: list[tuple[int, int, int]],
    config: DetectorSection,
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for decision in evaluate_times(session_id, bins, segments, config):
        counts["evaluated_times"] += 1
        for stage in decision.passed:
            counts[stage] += 1
        counts[decision.outcome] += 1
        if decision.flag is not None:
            flags.append(decision.flag)
    return flags


def group_flags(
    flags: list[dict[str, Any]], config: DetectorSection
) -> list[tuple[list[dict[str, Any]], bool]]:
    """Group flags of one session and segment whose confirmations are within merge_gap_ms.

    Returns (group, continues_previous) pairs. A group that would exceed
    maximum_event_ms is closed and the next group continues it.
    """
    ordered = sorted(
        flags, key=lambda item: (item["session_id"], item["segment_id"], item["confirmation_ms"])
    )
    groups: list[tuple[list[dict[str, Any]], bool]] = []
    current: list[dict[str, Any]] = []
    continues = False
    for flag in ordered:
        if current:
            last = current[-1]
            same_scope = (
                flag["session_id"] == last["session_id"]
                and flag["segment_id"] == last["segment_id"]
            )
            within_gap = flag["confirmation_ms"] - last["confirmation_ms"] <= config.merge_gap_ms
            if same_scope and within_gap:
                group_start = current[0]["event_window"]["start_ms"]
                if flag["event_window"]["end_ms"] - group_start <= config.maximum_event_ms:
                    current.append(flag)
                    continue
                groups.append((current, continues))
                current, continues = [flag], True
                continue
            groups.append((current, continues))
        current, continues = [flag], False
    if current:
        groups.append((current, continues))
    return groups


def _candidate(
    group: list[dict[str, Any]],
    config: DetectorSection,
    config_hash: str,
    continues_event_id: str | None,
) -> CandidateEvent:
    representative = group[0]
    first = representative["confirmation_ms"]
    return CandidateEvent(
        event_id=make_event_id(
            representative["session_id"], representative["segment_id"], first, config_hash
        ),
        session_id=representative["session_id"],
        segment_id=representative["segment_id"],
        event_window=TimeWindow(
            start_ms=min(flag["event_window"]["start_ms"] for flag in group),
            end_ms=max(flag["event_window"]["end_ms"] for flag in group),
        ),
        representative_reference_window=TimeWindow(**representative["reference_window"]),
        representative_event_window=TimeWindow(**representative["event_window"]),
        detection_time_ms=first,
        detector_name=config.name,
        detector_version=DETECTOR_VERSION,
        detector_config_hash=config_hash,
        reference_median_mbps=representative["reference_median_mbps"],
        event_median_mbps=representative["event_median_mbps"],
        absolute_drop_mbps=representative["absolute_drop_mbps"],
        relative_drop=representative["relative_drop"],
        flag_count=len(group),
        first_confirmation_ms=first,
        last_confirmation_ms=group[-1]["confirmation_ms"],
        continues_event_id=continues_event_id,
    )


def _select_sessions(
    connection: duckdb.DuckDBPyConnection,
    application_cohort: str | None,
    session_ids: list[str] | None,
) -> list[str]:
    rows = connection.execute(
        "SELECT session_id, application_cohort FROM sessions ORDER BY session_id"
    ).fetchall()
    known = {str(row[0]): str(row[1]) for row in rows}
    if session_ids is not None:
        unknown = sorted(set(session_ids) - set(known))
        if unknown:
            raise ValueError(f"unknown session ids: {unknown}")
    selected = sorted(session_ids) if session_ids is not None else sorted(known)
    return [
        session_id
        for session_id in selected
        if application_cohort is None or known[session_id] == application_cohort
    ]


def detect_candidates(
    connection: duckdb.DuckDBPyConnection,
    detector_config: DetectorSection,
    application_cohort: str | None = "file_download",
    session_ids: list[str] | None = None,
) -> DetectionResult:
    """Evaluate every bin boundary of the selected sessions and group flags into candidates."""
    config_hash = detector_config_hash(detector_config)
    totals = dict.fromkeys(ELIGIBILITY_STAGES, 0)
    all_flags: list[dict[str, Any]] = []
    session_counts: dict[str, dict[str, int]] = {}
    flag_counts: dict[str, int] = {}
    selected = _select_sessions(connection, application_cohort, session_ids)
    for session_id in selected:
        counts = dict.fromkeys(ELIGIBILITY_STAGES, 0)
        flags = _evaluate_session(
            session_id,
            load_bins(connection, session_id),
            load_segments(connection, session_id),
            detector_config,
            counts,
        )
        session_counts[session_id] = counts
        flag_counts[session_id] = len(flags)
        for stage, value in counts.items():
            totals[stage] += value
        all_flags.extend(flags)

    candidates: list[CandidateEvent] = []
    details: list[dict[str, Any]] = []
    previous_event_id: str | None = None
    for group, continues in group_flags(all_flags, detector_config):
        candidate = _candidate(
            group, detector_config, config_hash, previous_event_id if continues else None
        )
        candidates.append(candidate)
        previous_event_id = candidate.event_id
        details.append(
            {
                "event_id": candidate.event_id,
                "detection_delay_ms": candidate.first_confirmation_ms
                - candidate.event_window.start_ms,
                "constituent_confirmation_ms": [flag["confirmation_ms"] for flag in group],
            }
        )
    totals["sessions_evaluated"] = len(selected)
    totals["candidates"] = len(candidates)
    return DetectionResult(
        candidates=candidates,
        flags=sorted(
            all_flags,
            key=lambda item: (item["session_id"], item["segment_id"], item["confirmation_ms"]),
        ),
        eligibility_counts=totals,
        config_hash=config_hash,
        session_flag_counts=flag_counts,
        session_eligibility_counts=session_counts,
        candidate_details=details,
    )


def run_detect(snapshot_dir: Path, out_dir: Path, detector_config: DetectorSection) -> Path:
    """Run the detector on file download sessions and write candidates.json."""
    connection, manifest = open_snapshot(snapshot_dir)
    try:
        result = detect_candidates(connection, detector_config)
    finally:
        connection.close()
    payload = {
        "detector_name": detector_config.name,
        "detector_version": DETECTOR_VERSION,
        "config": detector_config.model_dump(mode="json"),
        "config_hash": result.config_hash,
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_logical_hash": manifest["logical_hash"],
        "application_cohort": "file_download",
        "semantics": {
            "reference_window": "[t - reference_ms - event_ms, t - event_ms), half open",
            "event_window": "[t - event_ms, t), half open",
            "statistic": "median of one second bin mean downlink rates over bins with a valid "
            "value, linear interpolation for an even count",
            "eligibility": "both windows inside one segment, complete bins available by t, "
            "coverage and active fraction minimums in both windows, reference median minimum",
            "grouping": "flags of one session and segment within merge_gap_ms are grouped, a "
            "group longer than maximum_event_ms starts a continuation candidate",
            "representative": "earliest confirmation of the group supplies the representative "
            "windows and medians, no merged reference is computed",
            "detection_delay": "first_confirmation_ms minus event_window.start_ms, at least "
            "event_ms by construction",
        },
        "candidates": [candidate.model_dump(mode="json") for candidate in result.candidates],
        "candidate_details": result.candidate_details,
        "eligibility_counts": result.eligibility_counts,
        "eligibility_counts_by_session": result.session_eligibility_counts,
        "session_flag_counts": result.session_flag_counts,
        "flags": result.flags,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CANDIDATES_FILE_NAME
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path
