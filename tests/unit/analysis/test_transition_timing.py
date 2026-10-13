"""Onset lags, the shift null and peri change profiles on constructed data."""

from __future__ import annotations

import numpy as np

from celltrace.analysis import transition_timing as timing
from celltrace.analysis.detector import BinView


def test_nearest_lag_uses_the_closest_change_inside_the_radius() -> None:
    onsets = np.array([10_000, 50_000, 90_000])
    changes = np.array([12_000, 47_000, 53_000, 130_000])
    # 50 s has changes 3 s before and after: the tie keeps the earlier one.
    assert timing.nearest_lags(onsets, changes).tolist() == [2000, -3000]
    assert timing.share_after(np.array([1000, 0, -1000, 2000])) == 0.625
    assert timing.share_after(np.array([], dtype=np.int64)) is None


def test_eligible_runs_and_shifts_stay_inside_their_run() -> None:
    runs = timing.eligible_runs([1000, 2000, 3000, 10_000, 11_000])
    assert runs == ((1000, 3000), (10_000, 11_000))
    changes = np.array([2000, 6000, 11_000])
    generator = np.random.default_rng(1)
    for _ in range(50):
        moved = timing.shifted_transitions(changes, runs, generator)
        assert 1000 <= moved[0] <= 3000 and moved[1] == 6000 and 10_000 <= moved[2] <= 11_000


def _session(onsets: list[int], changes: list[int]) -> timing.SessionTiming:
    # Each onset sits in its own 30 s eligible run, like the archive's short runs.
    runs = tuple((onset - 10_000, onset + 19_000) for onset in onsets)
    return timing.SessionTiming(
        "s", np.array(onsets, dtype=np.int64), np.array(changes, dtype=np.int64), runs, {}
    )


def test_changes_planted_after_onset_are_detected_and_random_ones_are_not() -> None:
    onsets = list(range(20_000, 1_980_000, 40_000))
    after = _session(onsets, [onset + 3000 for onset in onsets])
    result = timing.order_test([after], 199, 7)
    assert result["share_change_after_onset"] == 1.0
    assert result["p_value_two_sided"] <= 0.05
    generator = np.random.default_rng(4)
    offsets = generator.integers(-9, 10, len(onsets)) * 1000
    random_order = _session(
        onsets, [onset + int(offset) for onset, offset in zip(onsets, offsets, strict=True)]
    )
    assert timing.order_test([random_order], 199, 7)["p_value_two_sided"] > 0.05


def test_profile_shows_a_drop_that_follows_the_change() -> None:
    bins = {}
    for start in range(0, 60_000, 1000):
        rate = 20.0 if start < 31_000 else 2.0
        bins[start] = BinView(0, start, start + 1000, True, 1, 1, rate, 1)
    item = timing.SessionTiming("s", np.array([], dtype=np.int64), np.array([30_400]), (), bins)
    profiles = timing.change_profiles(item)
    assert len(profiles) == 1
    summary = timing.profile_summary(profiles)
    assert summary["median_ratio_by_offset"]["-5"] == 1.0
    assert summary["median_ratio_by_offset"]["5"] == 0.1
    assert summary["log10_post_over_pre"]["one_second"]["mean"] == -1.0


def test_t_quantiles_match_scipy() -> None:
    from scipy.stats import t as student_t

    for degrees in (1, 2, 4, 7, 15, 20, 30, 31, 45, 120):
        assert abs(timing.t_quantile_975(degrees) - student_t.ppf(0.975, degrees)) < 2e-4
