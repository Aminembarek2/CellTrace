"""Is low rate more persistent than the rate's own autocorrelation predicts?

Iterative amplitude adjusted Fourier transform (IAAFT) surrogates keep a series' amplitude
distribution and power spectrum, so their null is a stationary linear Gaussian process seen
through a static monotone measurement function (Schreiber and Schmitz 2000, sections 4.1 and
4.3). Rates move only within one segment and activity state, and empty bins keep their place,
as in the repaired chance checks. Method: docs/TECHNICAL_REPORT.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import duckdb
import numpy as np

from celltrace.analysis import impact
from celltrace.analysis.detector import BinView, load_bins, load_segments
from celltrace.config import DetectorSection

LOW_RATE_MBPS = 1.5
MINIMUM_EPISODE_SECONDS = 10
MINIMUM_STRATUM_VALUES = 32
MAXIMUM_ITERATIONS = 200
WINDOW_BINS = impact.SHORT_WINDOW_MS // 1000


@dataclass(frozen=True)
class SessionSeries:
    """One session's valid one second rates and the bins behind each usable decision."""

    session_id: str
    rates: np.ndarray
    strata: tuple[np.ndarray, ...]
    decision_times: np.ndarray
    decision_bins: np.ndarray


def iaaft(values: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """A surrogate with the same values and, approximately, the same Fourier amplitudes."""
    if len(values) < 3:
        return values.copy()
    ordered = np.sort(values)
    amplitudes = np.abs(np.fft.rfft(values))
    surrogate = generator.permutation(values)
    for _ in range(MAXIMUM_ITERATIONS):
        phases = np.angle(np.fft.rfft(surrogate))
        filtered = np.fft.irfft(amplitudes * np.exp(1j * phases), n=len(values))
        adjusted = ordered[np.argsort(np.argsort(filtered, kind="stable"), kind="stable")]
        if np.array_equal(adjusted, surrogate):
            break
        surrogate = adjusted
    result: np.ndarray = surrogate
    return result


def gaussian_iaaft(values: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """IAAFT on rank Gaussian scores, values put back by the surrogate's rank order.

    Matching the spectrum of the Gaussianized series targets the null of a linear Gaussian
    process seen through a monotone function. On skewed rates, raw IAAFT loses dependence in
    the low tail (docs/TECHNICAL_REPORT.md).
    """
    if len(values) < 3:
        return values.copy()
    normal = NormalDist()
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    scores = np.array([normal.inv_cdf((rank + 0.5) / len(values)) for rank in ranks.tolist()])
    order = np.argsort(np.argsort(iaaft(scores, generator), kind="stable"), kind="stable")
    result: np.ndarray = np.sort(values)[order]
    return result


def surrogate_rates(
    series: SessionSeries, generator: np.random.Generator, gaussianize: bool = True
) -> np.ndarray:
    method = gaussian_iaaft if gaussianize else iaaft
    rates = series.rates.copy()
    for members in series.strata:
        if len(members) >= MINIMUM_STRATUM_VALUES:
            rates[members] = method(series.rates[members], generator)
    return rates


def decision_medians(rates: np.ndarray, decision_bins: np.ndarray) -> np.ndarray:
    values = np.where(decision_bins >= 0, rates[np.maximum(decision_bins, 0)], np.nan)
    medians: np.ndarray = np.nanmedian(values, axis=1)
    return medians


def episode_statistics(decision_times: np.ndarray, medians: np.ndarray) -> tuple[int, int, int]:
    """Episodes of ten consecutive low decisions or more, low decisions inside them, all low."""
    episodes = inside = 0
    run = 0
    previous: int | None = None
    for time, median in zip(decision_times.tolist(), medians.tolist(), strict=True):
        continues = previous is not None and time - previous == 1000 and run > 0
        if median < LOW_RATE_MBPS and continues:
            run += 1
        else:
            if run >= MINIMUM_EPISODE_SECONDS:
                episodes += 1
                inside += run
            run = 1 if median < LOW_RATE_MBPS else 0
        previous = time
    if run >= MINIMUM_EPISODE_SECONDS:
        episodes += 1
        inside += run
    low = int(np.sum(medians < LOW_RATE_MBPS))
    return episodes, inside, low


def archive_statistics(
    sessions: Sequence[SessionSeries], rates: Sequence[np.ndarray]
) -> dict[str, float | None]:
    episodes = inside = low = 0
    for series, values in zip(sessions, rates, strict=True):
        found, contained, total = episode_statistics(
            series.decision_times, decision_medians(values, series.decision_bins)
        )
        episodes, inside, low = episodes + found, inside + contained, low + total
    return {"episodes": float(episodes), "long_share": inside / low if low else None}


def series_from_bins(
    session_id: str,
    bins: dict[int, BinView],
    segments: list[tuple[int, int, int]],
    detector: DetectorSection,
) -> SessionSeries:
    """Usable trailing 5 s decisions from the shared rule, and the strata of valid bins."""
    starts = sorted(bins)
    position = {start: index for index, start in enumerate(starts)}
    rates = np.array(
        [
            np.nan if bins[start].downlink_mbps is None else bins[start].downlink_mbps
            for start in starts
        ]
    )
    valid = np.array(
        [
            bins[start].downlink_valid_count > 0 and bins[start].downlink_mbps is not None
            for start in starts
        ]
    )
    strata: dict[tuple[int, bool], list[int]] = {}
    for start in starts:
        view = bins[start]
        if view.downlink_mbps is not None and view.is_complete:
            key = (view.segment_id, view.active_count == view.sample_count)
            strata.setdefault(key, []).append(position[start])
    times, matrix = [], []
    for decision_ms, _ in impact.level_series(bins, segments, detector):
        members = [
            position[start]
            for start in range(decision_ms - impact.SHORT_WINDOW_MS, decision_ms, 1000)
        ]
        times.append(decision_ms)
        matrix.append([index if valid[index] else -1 for index in members])
    return SessionSeries(
        session_id=session_id,
        rates=rates,
        strata=tuple(np.array(members, dtype=np.int64) for members in strata.values()),
        decision_times=np.array(times, dtype=np.int64),
        decision_bins=np.array(matrix, dtype=np.int64).reshape(-1, WINDOW_BINS),
    )


def load_archive(
    connection: duckdb.DuckDBPyConnection, detector: DetectorSection, cohort: str = "file_download"
) -> list[SessionSeries]:
    session_ids = [
        row[0]
        for row in connection.execute(
            "SELECT session_id FROM sessions WHERE application_cohort = ? ORDER BY session_id",
            [cohort],
        ).fetchall()
    ]
    return [
        series_from_bins(
            session_id,
            load_bins(connection, session_id),
            load_segments(connection, session_id),
            detector,
        )
        for session_id in session_ids
    ]


def holm(p_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def surrogate_test(
    sessions: Sequence[SessionSeries], draws: int, seed: int, gaussianize: bool = True
) -> dict[str, Any]:
    """Observed statistics against IAAFT surrogates, one sided for excess, plus one p values."""
    observed = archive_statistics(sessions, [series.rates for series in sessions])
    generator = np.random.default_rng(seed)
    null: dict[str, list[float | None]] = {"episodes": [], "long_share": []}
    for _ in range(draws):
        values = archive_statistics(
            sessions, [surrogate_rates(series, generator, gaussianize) for series in sessions]
        )
        for name in null:
            null[name].append(values[name])
    raw = {}
    for name, draws_for_name in null.items():
        finite = [value for value in draws_for_name if value is not None]
        value = observed[name]
        raw[name] = (
            None
            if value is None or len(finite) != len(draws_for_name)
            else (1 + sum(draw >= value for draw in finite)) / (len(finite) + 1)
        )
    defined = [name for name, value in raw.items() if value is not None]
    adjusted = dict(zip(defined, holm([float(raw[name] or 0) for name in defined]), strict=True))
    return {
        "observed": observed,
        "null_mean": {
            name: float(np.mean([v for v in values if v is not None]))
            if any(v is not None for v in values)
            else None
            for name, values in null.items()
        },
        "null_interval_95": {
            name: [
                float(q) for q in np.quantile([v for v in values if v is not None], [0.025, 0.975])
            ]
            if any(v is not None for v in values)
            else None
            for name, values in null.items()
        },
        "p_value": raw,
        "holm_p_value": adjusted,
        "draws": draws,
        "seed": seed,
        "method": "gaussianized_iaaft" if gaussianize else "iaaft",
        "null_draws": null,
    }


def synthetic_archive(
    generator: np.random.Generator,
    sessions: int = 8,
    seconds: int = 900,
    coefficient: float = 0.88,
    planted_outages: int = 0,
) -> list[SessionSeries]:
    """Autoregressive log rates with one empty bin in seven, optionally with near zero stretches."""
    archive = []
    for index in range(sessions):
        noise = generator.normal(0, 0.45 * np.sqrt(1 - coefficient**2), seconds)
        log_rate = np.empty(seconds)
        log_rate[0] = generator.normal(0, 0.45)
        for step in range(1, seconds):
            log_rate[step] = coefficient * log_rate[step - 1] + noise[step]
        rates = 10 ** (1.0 + log_rate)
        for _ in range(planted_outages):
            length = int(generator.integers(20, 41))
            start = int(generator.integers(0, seconds - length))
            rates[start : start + length] = 0.05
        valid = generator.random(seconds) >= 1 / 7
        rates = np.where(valid, rates, np.nan)
        times, matrix = [], []
        for end in range(WINDOW_BINS, seconds + 1):
            members = list(range(end - WINDOW_BINS, end))
            if valid[members].sum() >= 4:
                times.append(end * 1000)
                matrix.append([member if valid[member] else -1 for member in members])
        archive.append(
            SessionSeries(
                session_id=f"synthetic_{index}",
                rates=rates,
                strata=(np.flatnonzero(valid),),
                decision_times=np.array(times, dtype=np.int64),
                decision_bins=np.array(matrix, dtype=np.int64).reshape(-1, WINDOW_BINS),
            )
        )
    return archive
