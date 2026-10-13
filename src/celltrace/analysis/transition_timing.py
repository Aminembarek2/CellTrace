"""Does the drop or the reported cell change come first?

Onset lags compare each detector candidate's representative event window start with the
nearest reported serving cell change. A circular shift of change times within contiguous
eligible decision runs gives the null. A peri change profile and pre and post windows of
1 s and 5 s follow handover studies (Hassan et al. 2022), at one second logging resolution.
A reported identity change is a transition proxy and may lag the radio handover.
Method: docs/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import duckdb
import numpy as np

from celltrace.analysis.detector import (
    BinView,
    detect_candidates,
    evaluate_times,
    load_bins,
    load_segments,
)
from celltrace.analysis.population import transition_times
from celltrace.config import DetectorSection

LAG_RADIUS_MS = 10_000
# Two sided 95% Student t quantiles, so the jackknife needs no undeclared dependency.
STUDENT_T_975 = (
    12.7062,
    4.3027,
    3.1824,
    2.7764,
    2.5706,
    2.4469,
    2.3646,
    2.3060,
    2.2622,
    2.2281,
    2.2010,
    2.1788,
    2.1604,
    2.1448,
    2.1314,
    2.1199,
    2.1098,
    2.1009,
    2.0930,
    2.0860,
    2.0796,
    2.0739,
    2.0687,
    2.0639,
    2.0595,
    2.0555,
    2.0518,
    2.0484,
    2.0452,
    2.0423,
)
PROFILE_SECONDS = 20
BASELINE_OFFSETS = range(-20, -10)
MINIMUM_BASELINE_BINS = 8
MINIMUM_BASELINE_MBPS = 0.5
# Ratios to the baseline are floored before logs, so a zero rate stays finite.
RATIO_FLOOR = 0.001
MINIMUM_ACTIVE_SHARE = 0.8


@dataclass(frozen=True)
class SessionTiming:
    session_id: str
    onsets: np.ndarray
    transitions: np.ndarray
    runs: tuple[tuple[int, int], ...]
    bins: dict[int, BinView]


def nearest_lags(
    onsets: np.ndarray, transitions: np.ndarray, radius: int = LAG_RADIUS_MS
) -> np.ndarray:
    """Change time minus onset for each onset with a change within the radius."""
    if len(transitions) == 0:
        return np.array([], dtype=np.int64)
    ordered = np.sort(transitions)
    lags = []
    for onset in onsets.tolist():
        index = int(np.searchsorted(ordered, onset))
        nearby = [
            ordered[position] for position in (index - 1, index) if 0 <= position < len(ordered)
        ]
        lag = min((int(time) - onset for time in nearby), key=lambda value: (abs(value), value))
        if abs(lag) <= radius:
            lags.append(lag)
    return np.array(lags, dtype=np.int64)


def share_after(lags: np.ndarray) -> float | None:
    """Share of changes after drop onset, exact ties counted as half."""
    if len(lags) == 0:
        return None
    return float((np.sum(lags > 0) + 0.5 * np.sum(lags == 0)) / len(lags))


def eligible_runs(times: Sequence[int]) -> tuple[tuple[int, int], ...]:
    runs: list[list[int]] = []
    for time in sorted(times):
        if runs and time - runs[-1][1] == 1000:
            runs[-1][1] = time
        else:
            runs.append([time, time])
    return tuple((start, end) for start, end in runs)


def shifted_transitions(
    transitions: np.ndarray, runs: Sequence[tuple[int, int]], generator: np.random.Generator
) -> np.ndarray:
    """Move changes inside each eligible run by one circular offset per run, others stay."""
    shifted = transitions.copy()
    for start, end in runs:
        span = end - start + 1000
        inside = (transitions >= start) & (transitions <= end)
        if not inside.any() or span <= 1000:
            continue
        offset = int(generator.integers(0, span // 1000)) * 1000
        shifted[inside] = start + (transitions[inside] - start + offset) % span
    return shifted


def order_test(sessions: Sequence[SessionTiming], draws: int, seed: int) -> dict[str, Any]:
    lags = np.concatenate([nearest_lags(item.onsets, item.transitions) for item in sessions])
    observed = share_after(lags)
    generator = np.random.default_rng(seed)
    null = []
    for _ in range(draws):
        null_lags = np.concatenate(
            [
                nearest_lags(
                    item.onsets, shifted_transitions(item.transitions, item.runs, generator)
                )
                for item in sessions
            ]
        )
        null.append(share_after(null_lags))
    finite = [value for value in null if value is not None]
    p_value = None
    if observed is not None and len(finite) == len(null) and finite:
        upper = (1 + sum(value >= observed for value in finite)) / (len(finite) + 1)
        lower = (1 + sum(value <= observed for value in finite)) / (len(finite) + 1)
        p_value = min(1.0, 2 * min(upper, lower))
    return {
        "candidates_with_nearby_change": len(lags),
        "share_change_after_onset": observed,
        "median_lag_ms": float(np.median(lags)) if len(lags) else None,
        "lag_quartiles_ms": [float(value) for value in np.quantile(lags, [0.25, 0.75])]
        if len(lags)
        else None,
        "null_mean": float(np.mean(finite)) if finite else None,
        "null_interval_95": [float(value) for value in np.quantile(finite, [0.025, 0.975])]
        if finite
        else None,
        "p_value_two_sided": p_value,
        "draws": draws,
        "seed": seed,
    }


def _window_ok(bins: dict[int, BinView], first: int, last: int) -> bool:
    members = [bins.get(start) for start in range(first, last + 1000, 1000)]
    if any(member is None for member in members):
        return False
    present = [member for member in members if member is not None]
    if len({member.segment_id for member in present}) != 1:
        return False
    samples = sum(member.sample_count for member in present)
    active = sum(member.active_count for member in present)
    return samples > 0 and active / samples >= MINIMUM_ACTIVE_SHARE


def _rate(view: BinView | None) -> float | None:
    if view is None or view.downlink_valid_count == 0 or view.downlink_mbps is None:
        return None
    return float(view.downlink_mbps)


def change_profiles(item: SessionTiming) -> list[dict[str, Any]]:
    """Rates around isolated changes, relative to the baseline 20 s to 10 s before."""
    ordered = np.sort(item.transitions)
    profiles = []
    for time in ordered.tolist():
        others = ordered[(ordered != time) & (np.abs(ordered - time) <= PROFILE_SECONDS * 1000)]
        if len(others):
            continue
        anchor = time - time % 1000
        first, last = anchor - PROFILE_SECONDS * 1000, anchor + PROFILE_SECONDS * 1000
        if not _window_ok(item.bins, first, last):
            continue
        baseline = [
            rate
            for offset in BASELINE_OFFSETS
            if (rate := _rate(item.bins.get(anchor + offset * 1000))) is not None
        ]
        if (
            len(baseline) < MINIMUM_BASELINE_BINS
            or float(np.median(baseline)) < MINIMUM_BASELINE_MBPS
        ):
            continue
        reference = float(np.median(baseline))
        ratios: dict[int, float | None] = {}
        for offset in range(-PROFILE_SECONDS, PROFILE_SECONDS + 1):
            rate = _rate(item.bins.get(anchor + offset * 1000))
            ratios[offset] = None if rate is None else rate / reference
        profiles.append({"session_id": item.session_id, "time_ms": int(time), "ratios": ratios})
    return profiles


def _window_median(ratios: dict[int, float | None], offsets: range) -> float | None:
    values = [value for offset in offsets if (value := ratios.get(offset)) is not None]
    return float(np.median(values)) if values else None


def _log_ratio(post: float | None, pre: float | None) -> float | None:
    if post is None or pre is None:
        return None
    return math.log10(max(post, RATIO_FLOOR)) - math.log10(max(pre, RATIO_FLOOR))


def t_quantile_975(degrees: int) -> float:
    """Table up to 30 degrees of freedom, a Cornish and Fisher expansion beyond."""
    if degrees < 1:
        raise ValueError("degrees of freedom must be positive")
    if degrees <= len(STUDENT_T_975):
        return STUDENT_T_975[degrees - 1]
    z = 1.959964
    return z + (z**3 + z) / (4 * degrees) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * degrees**2)


def jackknife_mean(values_by_session: dict[str, list[float]]) -> dict[str, Any]:
    """Mean over all values with a delete one session jackknife t interval."""
    pooled = [value for values in values_by_session.values() for value in values]
    sessions = [name for name, values in values_by_session.items() if values]
    if len(sessions) < 3:
        return {
            "mean": float(np.mean(pooled)) if pooled else None,
            "interval_95": None,
            "sessions": len(sessions),
        }
    leave_one = np.array(
        [
            np.mean(
                [value for name in sessions if name != left for value in values_by_session[name]]
            )
            for left in sessions
        ]
    )
    groups = len(sessions)
    variance = (groups - 1) / groups * float(np.sum((leave_one - leave_one.mean()) ** 2))
    half = t_quantile_975(groups - 1) * math.sqrt(variance)
    mean = float(np.mean(pooled))
    return {"mean": mean, "interval_95": [mean - half, mean + half], "sessions": groups}


def profile_summary(profiles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_offset = {
        offset: [float(p["ratios"][offset]) for p in profiles if p["ratios"][offset] is not None]
        for offset in range(-PROFILE_SECONDS, PROFILE_SECONDS + 1)
    }
    contrasts: dict[str, dict[str, list[float]]] = {"one_second": {}, "five_seconds": {}}
    for profile in profiles:
        ratios = profile["ratios"]
        pairs = {
            "one_second": (ratios.get(1), ratios.get(-1)),
            "five_seconds": (
                _window_median(ratios, range(1, 6)),
                _window_median(ratios, range(-5, 0)),
            ),
        }
        for name, (post, pre) in pairs.items():
            value = _log_ratio(post, pre)
            if value is not None:
                contrasts[name].setdefault(profile["session_id"], []).append(value)
    return {
        "changes": len(profiles),
        "sessions": len({p["session_id"] for p in profiles}),
        "median_ratio_by_offset": {
            str(offset): float(np.median(values)) if values else None
            for offset, values in by_offset.items()
        },
        "log10_post_over_pre": {name: jackknife_mean(values) for name, values in contrasts.items()},
    }


def load_sessions(
    connection: duckdb.DuckDBPyConnection, detector: DetectorSection, mobility: str = "driving"
) -> list[SessionTiming]:
    session_ids = [
        row[0]
        for row in connection.execute(
            "SELECT session_id FROM sessions WHERE application_cohort = 'file_download'"
            " AND mobility_cohort = ? ORDER BY session_id",
            [mobility],
        ).fetchall()
    ]
    candidates = detect_candidates(connection, detector, session_ids=session_ids).candidates
    sessions = []
    for session_id in session_ids:
        bins = load_bins(connection, session_id)
        segments = load_segments(connection, session_id)
        eligible = [
            decision.confirmation_ms
            for decision in evaluate_times(session_id, bins, segments, detector)
            if decision.eligible
        ]
        sessions.append(
            SessionTiming(
                session_id=session_id,
                onsets=np.array(
                    [
                        candidate.representative_event_window.start_ms
                        for candidate in candidates
                        if candidate.session_id == session_id
                    ],
                    dtype=np.int64,
                ),
                transitions=np.array(transition_times(connection, session_id), dtype=np.int64),
                runs=eligible_runs(eligible),
                bins=bins,
            )
        )
    return sessions
