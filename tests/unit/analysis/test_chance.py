"""Null models detect planted structure and stay quiet when there is none."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from celltrace.analysis import chance
from celltrace.analysis.detector import BinView
from celltrace.config import DetectorSection


def bin_view(start_ms: int, rate: float | None, segment_id: int = 0) -> BinView:
    return BinView(
        segment_id=segment_id,
        bin_start_ms=start_ms,
        available_at_ms=start_ms + 1_000,
        is_complete=True,
        sample_count=1,
        active_count=1,
        downlink_mbps=rate,
        downlink_valid_count=0 if rate is None else 1,
    )


def session(rates: list[float | None], mobility: str = "driving", **kwargs: Any) -> Any:
    bins = {1_000 * index: bin_view(1_000 * index, rate) for index, rate in enumerate(rates)}
    return chance.SessionInput(
        session_id=kwargs.get("session_id", "s"),
        mobility=mobility,
        bins=bins,
        segments=[(0, 0, 1_000 * len(rates))],
        transitions=kwargs.get("transitions", []),
    )


def test_permutation_moves_only_rate_values_inside_their_segment() -> None:
    rates = [float(value) for value in range(1, 41)]
    bins = {1_000 * index: bin_view(1_000 * index, rate) for index, rate in enumerate(rates)}
    bins[5_000] = bin_view(5_000, None)
    shuffled = chance.permuted_rates(bins, np.random.default_rng(1), 1)
    assert shuffled[5_000].downlink_mbps is None
    before = Counter(view.downlink_mbps for view in bins.values())
    assert Counter(view.downlink_mbps for view in shuffled.values()) == before
    assert [view.downlink_mbps for view in shuffled.values()] != [
        view.downlink_mbps for view in bins.values()
    ]
    for start, view in shuffled.items():
        assert (view.segment_id, view.active_count, view.is_complete) == (0, 1, True)
        assert view.bin_start_ms == start


def test_block_permutation_keeps_runs_of_five_together() -> None:
    bins = {1_000 * index: bin_view(1_000 * index, float(index)) for index in range(30)}
    shuffled = chance.permuted_rates(bins, np.random.default_rng(3), 5)
    values = [shuffled[1_000 * index].downlink_mbps for index in range(30)]
    blocks = [values[index : index + 5] for index in range(0, 30, 5)]
    for block in blocks:
        assert block == [block[0] + offset for offset in range(5)] and block[0] % 5 == 0


def test_blocks_span_jitter_gaps_and_missing_bins_keep_their_positions() -> None:
    bins = {1_000 * index: bin_view(1_000 * index, float(index)) for index in range(40)}
    del bins[12_000]
    bins[26_000] = bin_view(26_000, None)
    shuffled = chance.permuted_rates(bins, np.random.default_rng(7), 5)
    valid = [start for start in sorted(bins) if bins[start].downlink_mbps is not None]
    original_blocks = [
        [bins[start].downlink_mbps for start in valid[index : index + 5]]
        for index in range(0, 35, 5)
    ]
    moved_blocks = [
        [shuffled[start].downlink_mbps for start in valid[index : index + 5]]
        for index in range(0, 35, 5)
    ]
    assert sorted(moved_blocks) == sorted(original_blocks)
    assert moved_blocks != original_blocks
    for start in valid[35:]:
        assert shuffled[start] == bins[start]
    assert shuffled[26_000].downlink_mbps is None and 12_000 not in shuffled


def test_rates_never_cross_segments() -> None:
    bins = {
        1_000 * index: bin_view(1_000 * index, float(index), segment_id=index // 20)
        for index in range(40)
    }
    for block_bins in chance.NULLS.values():
        shuffled = chance.permuted_rates(bins, np.random.default_rng(4), block_bins)
        for view in shuffled.values():
            assert view.downlink_mbps is not None
            assert view.downlink_mbps // 20 == view.segment_id


def test_rates_cannot_move_between_activity_states_or_into_incomplete_bins() -> None:
    bins = {1_000 * i: bin_view(1_000 * i, float(i)) for i in range(20)}
    for i in range(10, 20):
        bins[1_000 * i] = dataclasses.replace(bins[1_000 * i], active_count=0)
    bins[19_000] = dataclasses.replace(bins[19_000], is_complete=False)
    shuffled = chance.permuted_rates(bins, np.random.default_rng(2), 1)
    assert all(shuffled[1_000 * i].downlink_mbps < 10 for i in range(10))
    assert all(shuffled[1_000 * i].downlink_mbps >= 10 for i in range(10, 19))
    assert shuffled[19_000] == bins[19_000]


def test_p_value_counts_the_observed_draw_and_is_never_zero() -> None:
    assert chance.p_value(10.0, [1.0, 2.0, 3.0]) == pytest.approx(1 / 4)
    assert chance.p_value(0.0, [1.0, 2.0, 3.0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "observed,draws",
    [(float("nan"), [1.0]), (1.0, []), (1.0, [float("inf")]), (1.0, [float("nan")])],
)
def test_p_value_refuses_undefined_inputs(observed: float, draws: list[float]) -> None:
    with pytest.raises(ValueError):
        chance.p_value(observed, draws)


def test_undefined_draw_is_retained_and_withholds_inference() -> None:
    result = chance._summary(2.0, [1.0, float("nan"), 3.0])
    assert result["null_draws"] == [1.0, None, 3.0]
    assert result["draws"] == 3 and result["finite_draws"] == 2
    assert result["undefined_draws"] == 1 and result["exceedances"] == 1
    assert result["p_value"] is None and result["p_value_status"] == "undefined_null_draws"
    json.dumps(result, allow_nan=False)


def test_holm_retains_undefined_tests_in_the_family() -> None:
    assert chance.holm_adjust([1.0, 1.0, 1.0, 1 / 51, 1 / 1001]) == pytest.approx(
        [1.0, 1.0, 1.0, 4 / 51, 5 / 1001]
    )
    assert chance.holm_adjust([0.01, None, 0.02]) == [0.03, None, 0.04]


def test_fifty_draws_cannot_reject_in_a_family_of_five() -> None:
    # The earlier design: no detector row could fall below alpha whatever the data.
    assert chance.minimum_holm_p(50, 5) > 0.05
    assert chance.minimum_holm_p(199, 5) == pytest.approx(0.025)


def test_planted_sustained_low_rate_beats_the_shuffled_null() -> None:
    rates = [12.0] * 60 + [0.5] * 20 + [12.0] * 60
    result = chance.detector_chance([session(rates)], DetectorSection(), 39, 7)
    episodes = result["shuffled_seconds"]["low_rate_episodes"]
    assert episodes["observed"] == 1
    assert episodes["null_mean"] < 0.2 and episodes["p_value"] <= 0.05


def jittered_session(rates: list[float]) -> Any:
    """One empty bin in seven, like the timing jitter of the archive."""
    bins = {}
    for index, rate in enumerate(rates):
        start = 1_000 * index
        empty = index % 7 == 3
        bins[start] = dataclasses.replace(
            bin_view(start, None if empty else rate),
            sample_count=0 if empty else 1,
            active_count=0 if empty else 1,
        )
    return chance.SessionInput("s", "driving", bins, [(0, 0, 1_000 * len(rates))], [])


def test_nulls_mix_values_despite_jitter_gaps() -> None:
    rates = list(12.0 * np.exp(np.random.default_rng(5).normal(0, 0.15, 300)))
    result = chance.detector_chance([jittered_session(rates)], DetectorSection(), 3, 7)
    for null in chance.NULLS:
        support = result[null]["support"]
        assert support["moved_share_median"] > 0.9
        assert support["median_displacement_seconds_median"] > 30
        assert len(support["moved_share_per_draw"]) == 3


def test_planted_low_stretch_with_jitter_gaps_beats_both_nulls() -> None:
    rates = list(12.0 * np.exp(np.random.default_rng(5).normal(0, 0.15, 600)))
    rates[300:312] = [0.5] * 12
    result = chance.detector_chance([jittered_session(rates)], DetectorSection(), 39, 7)
    for null in chance.NULLS:
        episodes = result[null]["low_rate_episodes"]
        assert episodes["observed"] == 1 and episodes["p_value"] <= 0.05


def test_unstructured_rates_with_jitter_gaps_do_not_beat_the_nulls() -> None:
    # One fixed realization: its p values need not be large, only not below alpha here.
    rates = list(12.0 * np.exp(np.random.default_rng(6).normal(0, 0.6, 600)))
    result = chance.detector_chance([jittered_session(rates)], DetectorSection(), 39, 7)
    for null in chance.NULLS:
        for statistic in chance.STATISTICS:
            assert result[null][statistic]["p_value"] > 0.05


def decisions_with(flagged_times: set[int], count: int = 2_000) -> list[Any]:
    return [
        SimpleNamespace(
            confirmation_ms=1_000 * index,
            eligible=True,
            flag={"confirmation_ms": 1_000 * index} if index in flagged_times else None,
        )
        for index in range(count)
    ]


@pytest.mark.parametrize(
    "linked,expect_beats",
    [(True, True), (False, False)],
)
def test_association_detects_a_planted_link_and_not_an_unrelated_one(
    monkeypatch: pytest.MonkeyPatch, linked: bool, expect_beats: bool
) -> None:
    transitions = [1_000 * time for time in range(37, 2_000, 97)]
    if linked:
        flagged = {time // 1_000 + offset for time in transitions for offset in range(1, 4)}
    else:
        flagged = set()
    flagged |= set(range(5, 2_000, 41))
    monkeypatch.setattr(chance, "evaluate_times", lambda *args: decisions_with(flagged))
    result = chance.association_chance(
        [session([1.0] * 2_000, transitions=transitions)], DetectorSection(), 5_000, 200, 11
    )
    assert (result["p_value"] < 0.05) is expect_beats
    if linked:
        assert result["observed"] > result["null_interval_95"][1]


def test_no_driving_session_gives_undefined_values_not_nan() -> None:
    result = chance.association_chance(
        [session([1.0] * 60, mobility="static")], DetectorSection(), 5_000, 50, 1
    )
    assert result["observed"] is None and result["p_value"] is None
    json.dumps(result, allow_nan=False)


def test_exposure_shifts_preserve_support_across_eligibility_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions = decisions_with({1, 2, 50, 51}, count=60)
    for decision in decisions[10:50]:
        decision.eligible = False
    monkeypatch.setattr(chance, "evaluate_times", lambda *args: decisions)
    result = chance.association_chance(
        [session([1.0] * 60, transitions=[0, 50_000])], DetectorSection(), 5_000, 20, 3
    )
    assert sum(result["eligible_run_lengths"]) == 20
    assert max(result["eligible_run_lengths"]) <= 10
    assert sum(result["observed_counts"][1::2]) == 20
    assert len(result["null_draws"]) == 20


def test_near_exposure_does_not_require_transition_before_event_onset() -> None:
    # The five-second event ends at 100 s; a change at 98 s is inside the event.
    counts = chance._near_far_counts(
        np.array([100_000]), np.array([True]), np.array([98_000]), 10_000
    )
    assert counts.tolist() == [1, 1, 0, 0]


def test_episode_exposure_excludes_the_start_and_includes_the_end() -> None:
    episode = chance.impact.Episode("below_basic_use", 10_000, 20_000, 10, 0.5)
    inside = chance.inside_episodes(np.array([10_000, 11_000, 20_000, 21_000]), [episode])
    assert inside.tolist() == [False, True, True, False]


@pytest.mark.parametrize("linked,expect_beats", [(True, True), (False, False)])
def test_low_rate_alignment_finds_flags_planted_inside_episodes(
    monkeypatch: pytest.MonkeyPatch, linked: bool, expect_beats: bool
) -> None:
    rates = [12.0] * 600
    for start in (100, 300, 500):
        rates[start : start + 20] = [0.5] * 20
    item = session(rates)
    episodes = chance.impact.find_episodes(
        chance.impact.level_series(item.bins, item.segments, DetectorSection()),
        chance.impact.SYMPTOM_LEVEL,
        chance.impact.PRIMARY_MINIMUM_SECONDS,
        1_000,
    )
    assert len(episodes) == 3
    times = np.arange(1, 601) * 1_000
    inside = chance.inside_episodes(times, episodes)
    flagged = {int(time // 1_000) for time in times[inside]} if linked else set()
    flagged |= set(range(7, 600, 37))
    monkeypatch.setattr(chance, "evaluate_times", lambda *args: decisions_with(flagged, 601))
    result = chance.low_rate_alignment_chance([item], DetectorSection(), 200, 11)
    assert (result["p_value"] < 0.05) is expect_beats
