"""Exploratory sensitivity checks against explicitly conditional surrogate nulls.

Rates are permuted within one segment and activity state, one value or five consecutive
valid values at a time. Segments end at gaps above SEGMENT_GAP_THRESHOLD_MS, so an empty
bin inside a segment is timing jitter. Missing and incomplete bins keep their positions.
Cell change exposure is circularly shifted within contiguous eligible decision runs. These
nulls require exchangeability or shift invariance. Neither establishes diagnostic
validity, equivalence, causality or freedom from confounding.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from celltrace.analysis import impact
from celltrace.analysis.detector import BinView, evaluate_times, group_flags
from celltrace.config import DetectorSection

BLOCK_SECONDS = 5
SUSTAINED_FLAGS = impact.PRIMARY_MINIMUM_SECONDS
STATISTICS = (
    "flag_share",
    "candidates_per_hour",
    "sustained_candidates",
    "low_rate_episodes",
)
NULLS = {"shuffled_seconds": 1, "shuffled_5s_blocks": BLOCK_SECONDS}


@dataclasses.dataclass(frozen=True)
class SessionInput:
    session_id: str
    mobility: str
    bins: dict[int, BinView]
    segments: list[tuple[int, int, int]]
    transitions: list[int]


def _stratum(view: BinView) -> tuple[int, bool]:
    return view.segment_id, view.active_count == view.sample_count


def permutation_plan(
    bins: Mapping[int, BinView], generator: np.random.Generator, block_bins: int
) -> list[tuple[int, int]]:
    """Target and source bin starts for one draw, permuting blocks of valid values.

    A block holds consecutive valid values of one segment and activity state and may span
    an empty bin. The tail shorter than a block keeps its positions.
    """
    if block_bins < 1:
        raise ValueError("block_bins must be positive")
    strata: dict[tuple[int, bool], list[int]] = {}
    for start, view in sorted(bins.items()):
        if view.downlink_mbps is not None and view.is_complete:
            strata.setdefault(_stratum(view), []).append(start)
    plan: list[tuple[int, int]] = []
    for starts in strata.values():
        full_length = len(starts) // block_bins * block_bins
        order = generator.permutation(full_length // block_bins)
        sources = [
            starts[block * block_bins + offset] for block in order for offset in range(block_bins)
        ]
        plan.extend(zip(starts[:full_length], sources, strict=True))
    return plan


def permuted_rates(
    bins: Mapping[int, BinView], generator: np.random.Generator, block_bins: int
) -> dict[int, BinView]:
    """Rates moved by one permutation draw. Every other bin field stays in place."""
    return _apply_plan(bins, permutation_plan(bins, generator, block_bins))


def _apply_plan(bins: Mapping[int, BinView], plan: Sequence[tuple[int, int]]) -> dict[int, BinView]:
    result = dict(bins)
    for target, source in plan:
        result[target] = dataclasses.replace(bins[target], downlink_mbps=bins[source].downlink_mbps)
    return result


def minimum_holm_p(draws: int, family_size: int) -> float:
    """Smallest Holm adjusted p a test with this many draws can reach when ranked first."""
    return family_size / (draws + 1)


def detector_counts(
    session: SessionInput, bins: Mapping[int, BinView], detector: DetectorSection
) -> np.ndarray:
    """Eligible seconds, flags, candidates, sustained candidates and low rate episodes."""
    eligible = 0
    flags = []
    for decision in evaluate_times(session.session_id, dict(bins), session.segments, detector):
        eligible += decision.eligible
        if decision.flag is not None:
            flags.append(decision.flag)
    groups = group_flags(flags, detector)
    series = impact.level_series(dict(bins), session.segments, detector)
    episodes = impact.find_episodes(
        series, impact.SYMPTOM_LEVEL, impact.PRIMARY_MINIMUM_SECONDS, detector.step_ms
    )
    return np.array(
        [
            eligible,
            len(flags),
            len(groups),
            sum(len(group) >= SUSTAINED_FLAGS for group, _ in groups),
            len(episodes),
        ],
        dtype=float,
    )


def _total(arrays: Sequence[np.ndarray]) -> np.ndarray:
    total: np.ndarray = np.sum(np.stack(arrays), axis=0)
    return total


def _statistics(totals: np.ndarray) -> dict[str, float]:
    eligible, flags, candidates, sustained, episodes = totals
    hours = eligible / 3600
    return {
        "flag_share": float(flags / eligible) if eligible else float("nan"),
        "candidates_per_hour": float(candidates / hours) if hours else float("nan"),
        "sustained_candidates": float(sustained),
        "low_rate_episodes": float(episodes),
    }


def p_value(observed: float, null: Sequence[float] | np.ndarray) -> float:
    """One sided permutation p value, never zero."""
    draws = np.asarray(null, dtype=float)
    if (
        draws.ndim != 1
        or len(draws) == 0
        or not np.isfinite(observed)
        or not np.isfinite(draws).all()
    ):
        raise ValueError("p-values require a finite observed statistic and nonempty finite draws")
    return float((1 + np.sum(draws >= observed)) / (len(draws) + 1))


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjustment, retaining undefined tests in the fixed family."""
    result: list[float | None] = [None] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if value is not None]
    if any(not np.isfinite(value) or not 0 <= value <= 1 for _, value in valid):
        raise ValueError("p-values must be finite and in [0, 1]")
    previous = 0.0
    for rank, (index, value) in enumerate(sorted(valid, key=lambda pair: pair[1])):
        previous = max(previous, min(1.0, (len(p_values) - rank) * value))
        result[index] = previous
    return result


def _summary(observed: float | None, null: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Observed value against its null draws. Undefined values become None, never NaN."""
    requested = np.asarray(null, dtype=float)
    draws = requested[np.isfinite(requested)]
    observed = None if observed is None or not np.isfinite(observed) else float(observed)
    audit = {
        "draws": len(requested),
        "finite_draws": len(draws),
        "undefined_draws": len(requested) - len(draws),
        "null_draws": [float(value) if np.isfinite(value) else None for value in requested],
        "exceedances": None if observed is None else int(np.sum(draws >= observed)),
        "p_value_status": "undefined_observed"
        if observed is None
        else (
            "no_draws"
            if len(requested) == 0
            else ("undefined_null_draws" if len(draws) != len(requested) else "complete")
        ),
    }
    if observed is None or len(draws) == 0:
        return {
            "observed": observed,
            "null_mean": None,
            "null_interval_95": None,
            "ratio_to_null_mean": None,
            "p_value": None,
            **audit,
        }
    mean = float(draws.mean())
    return {
        "observed": observed,
        "null_mean": mean,
        "null_interval_95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "ratio_to_null_mean": float(observed / mean) if mean > 0 else None,
        "p_value": p_value(observed, draws) if len(draws) == len(requested) else None,
        **audit,
    }


def detector_chance(
    sessions: Sequence[SessionInput], detector: DetectorSection, permutations: int, seed: int
) -> dict[str, Any]:
    if not sessions or permutations < 1 or detector.step_ms != 1_000:
        raise ValueError("chance checks require sessions, positive draws and a one-second step")
    observed = _statistics(
        _total([detector_counts(item, item.bins, detector) for item in sessions])
    )
    valid_bins = sum(
        view.downlink_mbps is not None and view.is_complete
        for item in sessions
        for view in item.bins.values()
    )
    result: dict[str, Any] = {}
    for null_name, block_bins in NULLS.items():
        generator = np.random.default_rng([seed, block_bins])
        draws = []
        moved_shares = []
        median_displacements = []
        for _ in range(permutations):
            counts = []
            displacements: list[float] = []
            for item in sessions:
                plan = permutation_plan(item.bins, generator, block_bins)
                displacements += [abs(target - source) / 1_000 for target, source in plan]
                counts.append(detector_counts(item, _apply_plan(item.bins, plan), detector))
            draws.append(_statistics(_total(counts)))
            moved = [seconds for seconds in displacements if seconds > 0]
            moved_shares.append(len(moved) / valid_bins if valid_bins else 0.0)
            median_displacements.append(float(np.median(moved)) if moved else 0.0)
        result[null_name] = {
            name: _summary(observed[name], [draw[name] for draw in draws]) for name in STATISTICS
        }
        result[null_name]["support"] = {
            "valid_bins": valid_bins,
            "moved_share_per_draw": moved_shares,
            "median_displacement_seconds_per_draw": median_displacements,
            "moved_share_median": float(np.median(moved_shares)),
            "median_displacement_seconds_median": float(np.median(median_displacements)),
        }
    return result


def _near_far_counts(
    confirmations: np.ndarray, flagged: np.ndarray, transitions: np.ndarray, window_ms: int
) -> np.ndarray:
    """Flagged and total decisions with and without a transition in the preceding window."""
    nearby = _near_mask(confirmations, transitions, window_ms)
    return _exposure_counts(nearby, flagged)


def _near_mask(confirmations: np.ndarray, transitions: np.ndarray, window_ms: int) -> np.ndarray:
    times = np.sort(transitions)
    index = np.searchsorted(times, confirmations - window_ms, side="left")
    clipped = np.minimum(index, max(len(times) - 1, 0))
    nearby = (
        (index < len(times)) & (times[clipped] < confirmations)
        if len(times)
        else (np.zeros(len(confirmations), dtype=bool))
    )
    return nearby


def _exposure_counts(nearby: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.sum(nearby & flagged),
            np.sum(nearby),
            np.sum(~nearby & flagged),
            np.sum(~nearby),
        ],
        dtype=float,
    )


def _ratio(counts: np.ndarray) -> float:
    if counts[1] == 0 or counts[3] == 0 or counts[2] == 0:
        return float("nan")
    return float((counts[0] / counts[1]) / (counts[2] / counts[3]))


def _eligible_runs(
    item: SessionInput,
    detector: DetectorSection,
    exposure_of: Callable[[np.ndarray], np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flags and exposure for each contiguous run of eligible decisions in one segment."""
    decisions = [
        (decision.confirmation_ms, decision.flag is not None)
        for decision in evaluate_times(item.session_id, item.bins, item.segments, detector)
        if decision.eligible
    ]
    confirmations = np.array([time for time, _ in decisions], dtype=np.int64)
    flagged = np.array([flag for _, flag in decisions], dtype=bool)
    exposure = exposure_of(confirmations)
    # Segment boundaries and gaps in eligibility delimit the shift-invariance assumption.
    segment_ids = np.array(
        [
            next((segment for segment, start, end in item.segments if start < time <= end), -1)
            for time in confirmations
        ]
    )
    boundaries = (
        np.flatnonzero((np.diff(confirmations) != detector.step_ms) | (np.diff(segment_ids) != 0))
        + 1
    )
    return [
        (flags, near)
        for times, flags, near in zip(
            np.split(confirmations, boundaries),
            np.split(flagged, boundaries),
            np.split(exposure, boundaries),
            strict=True,
        )
        if len(times)
    ]


def _shift_test(
    inputs: Sequence[tuple[np.ndarray, np.ndarray]],
    shifts: int,
    generator: np.random.Generator,
    statistic: str,
    null: str,
) -> dict[str, Any]:
    if not inputs:
        return {
            "statistic": statistic,
            "null": "no eligible decisions",
            "eligible_run_lengths": [],
            **_summary(None, []),
        }
    counts = _total([_exposure_counts(near, flags) for flags, near in inputs])
    draws = []
    for _ in range(shifts):
        total = np.zeros(4)
        for flagged, near in inputs:
            offset = int(generator.integers(0, len(near)))
            total += _exposure_counts(np.roll(near, offset), flagged)
        draws.append(_ratio(total))
    return {
        "statistic": statistic,
        "null": null,
        "eligible_run_lengths": [len(near) for _, near in inputs],
        "observed_counts": counts.tolist(),
        **_summary(_ratio(counts), draws),
    }


def association_chance(
    sessions: Sequence[SessionInput],
    detector: DetectorSection,
    lookback_ms: int,
    shifts: int,
    seed: int,
) -> dict[str, Any]:
    """Alignment sensitivity, preserving exposure and eligible support within runs."""
    if shifts < 1 or lookback_ms < 0 or detector.step_ms != 1_000:
        raise ValueError("alignment checks require positive draws and a one-second step")
    window_ms = detector.event_ms + lookback_ms
    inputs = [
        run
        for item in sessions
        if item.mobility == "driving" and item.segments
        for run in _eligible_runs(
            item,
            detector,
            functools.partial(
                _near_mask, transitions=np.asarray(item.transitions), window_ms=window_ms
            ),
        )
    ]
    return _shift_test(
        inputs,
        shifts,
        np.random.default_rng([seed, 2]),
        "drop share near a cell change divided by drop share away from one",
        "cell change exposure circularly shifted within contiguous eligible decision runs",
    )


def inside_episodes(confirmations: np.ndarray, episodes: Sequence[impact.Episode]) -> np.ndarray:
    """Decisions whose confirmation time lies in an episode, start excluded, end included."""
    inside = np.zeros(len(confirmations), dtype=bool)
    for episode in episodes:
        inside |= (confirmations > episode.start_ms) & (confirmations <= episode.end_ms)
    return inside


def low_rate_alignment_chance(
    sessions: Sequence[SessionInput], detector: DetectorSection, shifts: int, seed: int
) -> dict[str, Any]:
    """Do relative drop flags concentrate inside sustained low rate episodes?"""
    if shifts < 1 or detector.step_ms != 1_000:
        raise ValueError("alignment checks require positive draws and a one-second step")
    inputs = []
    for item in sessions:
        if not item.segments:
            continue
        series = impact.level_series(item.bins, item.segments, detector)
        episodes = impact.find_episodes(
            series, impact.SYMPTOM_LEVEL, impact.PRIMARY_MINIMUM_SECONDS, detector.step_ms
        )
        inputs += _eligible_runs(
            item, detector, functools.partial(inside_episodes, episodes=episodes)
        )
    return _shift_test(
        inputs,
        shifts,
        np.random.default_rng([seed, 3]),
        "flag rate inside sustained low rate episodes divided by flag rate outside them",
        "episode indicator circularly shifted within contiguous eligible decision runs",
    )
