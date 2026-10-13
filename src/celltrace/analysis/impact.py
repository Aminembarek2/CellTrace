"""Retrospective low delivered-rate proxies over the relative drop detector.

Borrowed throughput thresholds do not measure application QoE or Opensignal's full
Consistent Quality metric. The persistence filter is a project choice, distinct
from SRE multiwindow burn-rate alerting. See docs/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from celltrace.analysis.detector import BinView, short_window_rate
from celltrace.config import DetectorSection

# Opensignal Consistent Quality: Core needs 1.5 Mbps download (browsing, SD video) and
# Excellent 5 Mbps (HD video). The stall level is a project choice meaning effectively
# no delivery, fifteen times below basic use. Ordered from most to least severe.
SERVICE_LEVELS: tuple[tuple[str, float], ...] = (
    ("stalled", 0.1),
    ("below_basic_use", 1.5),
    ("below_hd_video", 5.0),
)
LEVEL_LABELS = {
    "stalled": "Near zero delivered rate, under 0.1 Mbps",
    "below_basic_use": "Delivered rate under 1.5 Mbps",
    "below_hd_video": "Delivered rate under 5 Mbps",
    "at_least_hd_video": "Delivered rate at least 5 Mbps",
}
SEVERITY = {"stalled": 0, "below_basic_use": 1, "below_hd_video": 2, "none": 3}
SHORT_WINDOW_MS = 5_000
PRIMARY_MINIMUM_SECONDS = 10
SENSITIVITY_MINIMUM_SECONDS = (5, 10, 30)
# The symptom that admits a drop to the Investigator queue and that the detector is
# scored against. Stalled stretches are also below basic use.
SYMPTOM_LEVEL = "below_basic_use"
SYMPTOM_LEVELS = frozenset({"stalled", "below_basic_use"})


@dataclass(frozen=True)
class Episode:
    """Low decision intervals, distinct from the trailing windows supporting them.

    The indicator at decision t represents [t-1000, t). It is only available at t.
    This does not assert that every raw rate in that second was below the level.
    """

    level: str
    start_ms: int
    end_ms: int
    seconds: int
    minimum_mbps: float
    confirmation_ms: int | None = None


def level_series(
    bins: dict[int, BinView], segments: list[tuple[int, int, int]], detector: DetectorSection
) -> list[tuple[int, float]]:
    """(decision time, trailing 5 second median) for every usable active window."""
    if detector.step_ms != 1_000:
        raise ValueError("low rate exposure requires one-second decision steps")
    if not segments:
        return []
    session_end = max(end for _, _, end in segments)
    series = []
    for decision_ms in range(detector.step_ms, session_end + 1, detector.step_ms):
        rate = short_window_rate(bins, segments, decision_ms, SHORT_WINDOW_MS, detector)
        if rate is not None:
            series.append((decision_ms, rate))
    return series


def level_of(rate: float) -> str:
    for name, threshold in SERVICE_LEVELS:
        if rate < threshold:
            return name
    return "at_least_hd_video"


def find_episodes(
    series: Sequence[tuple[int, float]],
    level: str,
    minimum_seconds: int,
    step_ms: int,
) -> list[Episode]:
    """Runs of consecutive decision times below the level that last long enough.

    A gap in usable windows ends a run, so an episode never bridges missing data.
    """
    if step_ms != 1_000 or minimum_seconds < 1:
        raise ValueError("episodes require one-second steps and a positive duration")
    threshold = dict(SERVICE_LEVELS)[level]
    runs: list[list[tuple[int, float]]] = []
    for decision_ms, rate in series:
        if rate >= threshold:
            continue
        if runs and decision_ms - runs[-1][-1][0] == step_ms:
            runs[-1].append((decision_ms, rate))
        else:
            runs.append([(decision_ms, rate)])
    return [
        Episode(
            level=level,
            start_ms=run[0][0] - step_ms,
            end_ms=run[-1][0],
            seconds=len(run),
            minimum_mbps=min(rate for _, rate in run),
            confirmation_ms=run[minimum_seconds - 1][0],
        )
        for run in runs
        if len(run) >= minimum_seconds
    ]


def session_episodes(
    series: Sequence[tuple[int, float]], step_ms: int
) -> dict[tuple[str, int], list[Episode]]:
    """Episodes for every service level and every sensitivity duration."""
    return {
        (level, minimum): find_episodes(series, level, minimum, step_ms)
        for level, _ in SERVICE_LEVELS
        for minimum in SENSITIVITY_MINIMUM_SECONDS
    }


def overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def covered_ms(span: tuple[int, int], merged: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, min(span[1], end) - max(span[0], start)) for start, end in merged)


def compare_with_detector(
    candidates: Sequence[tuple[int, int, int]],
    episodes: Sequence[Episode],
    flags: Sequence[tuple[int, int, int]] | None = None,
) -> dict[str, int | list[int]]:
    """Score relative candidates against symptom episodes of one session.

    Candidates are (event start, event end, first confirmation). Precision counts
    candidates that overlap any episode, recall counts episode time inside candidate
    windows, and detection delay runs from episode start to the first overlapping
    confirmation. Group-level first confirmation is not a confirmation of every
    later constituent window: delay must use the constituent flags when grouped.
    """
    windows = [(start, end) for start, end, _ in candidates]
    merged = merge_intervals(windows)
    delays = []
    detected = 0
    for episode in episodes:
        span = (episode.start_ms, episode.end_ms)
        confirmations = [
            confirmation
            for start, end, confirmation in (flags if flags is not None else candidates)
            if overlaps((start, end), span)
        ]
        if confirmations:
            detected += 1
            delays.append(max(0, min(confirmations) - episode.start_ms))
    return {
        "candidates": len(candidates),
        "candidates_overlapping_symptom": sum(
            any(overlaps(window, (e.start_ms, e.end_ms)) for e in episodes) for window in windows
        ),
        "episodes": len(episodes),
        "episodes_detected": detected,
        "episode_ms": sum(episode.end_ms - episode.start_ms for episode in episodes),
        "episode_ms_covered": sum(
            covered_ms((episode.start_ms, episode.end_ms), merged) for episode in episodes
        ),
        "detection_delays_ms": delays,
    }


def candidate_impact(
    window: tuple[int, int], primary: dict[str, list[Episode]]
) -> tuple[str, int, float | None, int | None]:
    """Worst sustained level a candidate window touches, its seconds, lowest rate and the
    start of the first touching symptom episode, which identifies the incident."""
    for level, _ in SERVICE_LEVELS:
        touching = [e for e in primary.get(level, []) if overlaps(window, (e.start_ms, e.end_ms))]
        if touching:
            return (
                level,
                sum(episode.seconds for episode in touching),
                min(episode.minimum_mbps for episode in touching),
                min(episode.start_ms for episode in touching),
            )
    return "none", 0, None, None


def queue_order(row: dict[str, object]) -> tuple[int, int, float, str]:
    """Most severe level first, then longest time below it, then lowest rate."""
    minimum = row["impact_minimum_mbps"]
    return (
        SEVERITY[str(row["impact_level"])],
        -int(str(row["impact_seconds"])),
        float(str(minimum)) if minimum is not None else float("inf"),
        str(row["event_id"]),
    )


def incident_queue(
    rows: Sequence[dict[str, object]],
    episodes_by_session: Mapping[str, dict[str, list[Episode]]],
) -> list[dict[str, object]]:
    """Select an unexplained candidate for each low rate episode, rank the whole episode.

    One grouped candidate may cover two episodes. Return both assignments; callers
    deduplicate event IDs for dispatch. Selection is retrospective, including severity
    later in the episode, so these are not online decisions at first confirmation.
    """
    assignments = []
    for session_id, primary in sorted(episodes_by_session.items()):
        for episode in primary.get(SYMPTOM_LEVEL, []):
            span = (episode.start_ms, episode.end_ms)
            matching = [
                row
                for row in rows
                if row["session_id"] == session_id
                and row["triage_class"] == "unexplained"
                and overlaps(
                    (
                        int(str(row["impact_window_start_ms"])),
                        int(str(row["impact_window_end_ms"])),
                    ),
                    span,
                )
            ]
            if not matching:
                continue
            selected = min(
                matching,
                key=lambda row: (int(str(row["first_confirmation_ms"])), str(row["event_id"])),
            )
            level, seconds, minimum, _ = candidate_impact(span, primary)
            assignments.append(
                {
                    **selected,
                    "impact_level": level,
                    "impact_seconds": seconds,
                    "impact_minimum_mbps": minimum,
                    "symptom_incident_start_ms": episode.start_ms,
                    "symptom_incident_end_ms": episode.end_ms,
                    "symptom_confirmation_ms": episode.confirmation_ms,
                }
            )
    return sorted(assignments, key=queue_order)
