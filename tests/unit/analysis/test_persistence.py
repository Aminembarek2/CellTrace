"""IAAFT surrogates keep values and spectrum, episodes are counted exactly, controls behave."""

from __future__ import annotations

import numpy as np

from celltrace.analysis import impact, persistence
from celltrace.analysis.detector import BinView
from celltrace.config import DetectorSection


def test_iaaft_keeps_every_value_and_approximately_the_spectrum() -> None:
    generator = np.random.default_rng(3)
    values = np.cumsum(generator.normal(size=512)) + 50
    surrogate = persistence.iaaft(values, generator)
    assert np.array_equal(np.sort(surrogate), np.sort(values))
    assert not np.array_equal(surrogate, values)
    original = np.abs(np.fft.rfft(values))
    kept = np.abs(np.fft.rfft(surrogate))
    assert np.linalg.norm(kept - original) / np.linalg.norm(original) < 0.05


def test_episode_statistics_count_runs_and_the_share_inside_them() -> None:
    times = np.arange(1, 31) * 1000
    medians = np.full(30, 10.0)
    medians[2:14] = 1.0  # 12 low seconds, one episode
    medians[20:25] = 1.0  # 5 low seconds, too short
    assert persistence.episode_statistics(times, medians) == (1, 12, 17)
    gapped = times.copy()
    gapped[8:] += 5000  # a gap splits the first stretch into 6 and 6
    assert persistence.episode_statistics(gapped, medians) == (0, 0, 17)


def test_surrogates_leave_empty_bins_and_short_strata_in_place() -> None:
    generator = np.random.default_rng(5)
    rates = np.arange(100, dtype=float)
    rates[[3, 50]] = np.nan
    long_stratum = np.array([index for index in range(80) if index not in (3, 50)])
    short_stratum = np.arange(80, 100)
    series = persistence.SessionSeries(
        "s",
        rates,
        (long_stratum, short_stratum),
        np.array([], dtype=np.int64),
        np.empty((0, 5), dtype=np.int64),
    )
    moved = persistence.surrogate_rates(series, generator)
    assert np.isnan(moved[[3, 50]]).all()
    assert np.array_equal(moved[short_stratum], rates[short_stratum])
    assert np.array_equal(np.sort(moved[long_stratum]), np.sort(rates[long_stratum]))


def test_decision_medians_match_the_shared_trailing_rule() -> None:
    bins = {}
    for start in range(0, 40_000, 1000):
        rate = None if start % 7000 == 3000 else float(1 + (start // 1000) % 9)
        bins[start] = BinView(0, start, start + 1000, True, 1, 1, rate, 0 if rate is None else 1)
    detector = DetectorSection()
    series = persistence.series_from_bins("s", bins, [(0, 0, 40_000)], detector)
    expected = impact.level_series(bins, [(0, 0, 40_000)], detector)
    assert series.decision_times.tolist() == [time for time, _ in expected]
    medians = persistence.decision_medians(series.rates, series.decision_bins)
    assert np.allclose(medians, [value for _, value in expected])


def test_gaussianized_surrogate_keeps_values_and_log_scale_dependence() -> None:
    generator = np.random.default_rng(13)
    log_rate = np.empty(600)
    log_rate[0] = 0.0
    for step in range(1, 600):
        log_rate[step] = 0.88 * log_rate[step - 1] + generator.normal(0, 0.2)
    values = 10 ** (1 + log_rate)
    surrogate = persistence.gaussian_iaaft(values, generator)
    assert np.array_equal(np.sort(surrogate), np.sort(values))

    def lag_one(series: np.ndarray) -> float:
        logged = np.log10(series)
        return float(np.corrcoef(logged[:-1], logged[1:])[0, 1])

    assert abs(lag_one(surrogate) - lag_one(values)) < 0.05


def test_planted_outages_reject_and_a_linear_process_does_not() -> None:
    generator = np.random.default_rng(11)
    planted = persistence.synthetic_archive(generator, sessions=4, planted_outages=2)
    linear = persistence.synthetic_archive(generator, sessions=4, planted_outages=0)
    assert persistence.surrogate_test(planted, 39, 1)["p_value"]["long_share"] <= 0.05
    assert persistence.surrogate_test(linear, 39, 1)["p_value"]["long_share"] > 0.05
