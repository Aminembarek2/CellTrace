"""User impact layer: service levels, sustained episodes and scoring the relative detector."""

from __future__ import annotations

import pytest

from celltrace.analysis import impact
from celltrace.analysis.detector import BinView, short_window_rate
from celltrace.config import DetectorSection


@pytest.mark.parametrize(
    "rate,expected",
    [
        (0.0, "stalled"),
        (0.0999, "stalled"),
        (0.1, "below_basic_use"),
        (1.4999, "below_basic_use"),
        (1.5, "below_hd_video"),
        (4.9999, "below_hd_video"),
        (5.0, "at_least_hd_video"),
    ],
)
def test_service_levels_are_strict_published_thresholds(rate: float, expected: str) -> None:
    assert impact.level_of(rate) == expected


def series(rates: list[float], start_ms: int = 5_000) -> list[tuple[int, float]]:
    return [(start_ms + 1_000 * index, rate) for index, rate in enumerate(rates)]


def test_episode_needs_the_minimum_sustained_duration() -> None:
    nine = series([1.0] * 9)
    ten = series([1.0] * 10)
    assert impact.find_episodes(nine, "below_basic_use", 10, 1_000) == []
    [episode] = impact.find_episodes(ten, "below_basic_use", 10, 1_000)
    assert episode.seconds == 10
    # Ten decisions represent ten seconds, not the union of their trailing windows.
    assert (episode.start_ms, episode.end_ms) == (4_000, 14_000)
    assert episode.end_ms - episode.start_ms == episode.seconds * 1_000
    assert episode.confirmation_ms == 14_000


def test_earlier_rate_drop_is_not_a_low_rate_overlap() -> None:
    [episode] = impact.find_episodes(series([1.0] * 10), "below_basic_use", 10, 1_000)
    scores = impact.compare_with_detector([(0, 4_000, 4_000)], [episode])
    assert scores["candidates_overlapping_symptom"] == 0
    assert scores["episode_ms_covered"] == 0


def test_short_recovery_keeps_episode_denominators_disjoint() -> None:
    episodes = impact.find_episodes(
        series([1.0] * 10 + [2.0] + [1.0] * 10), "below_basic_use", 10, 1_000
    )
    assert len(episodes) == 2
    assert episodes[0].end_ms <= episodes[1].start_ms
    scores = impact.compare_with_detector([(0, 40_000, 40_000)], episodes)
    assert scores["episode_ms"] == scores["episode_ms_covered"] == 20_000


def test_decisions_are_not_mislabeled_as_seconds() -> None:
    with pytest.raises(ValueError, match="one-second"):
        impact.find_episodes(series([1.0] * 10), "below_basic_use", 10, 2_000)
    with pytest.raises(ValueError, match="one-second"):
        impact.level_series({}, [], DetectorSection(step_ms=2_000))


def test_episode_never_bridges_a_gap_in_usable_windows() -> None:
    left = series([1.0] * 6)
    right = series([1.0] * 6, start_ms=13_000)
    assert impact.find_episodes(left + right, "below_basic_use", 10, 1_000) == []
    assert len(impact.find_episodes(left + right, "below_basic_use", 5, 1_000)) == 2


def test_a_rate_above_the_level_ends_the_run() -> None:
    rates = [1.0] * 5 + [2.0] + [1.0] * 5
    assert impact.find_episodes(series(rates), "below_basic_use", 10, 1_000) == []
    assert len(impact.find_episodes(series(rates), "below_hd_video", 10, 1_000)) == 1


def test_detector_scoring_counts_precision_recall_and_delay() -> None:
    episode = impact.Episode("below_basic_use", 10_000, 30_000, 16, 0.5)
    candidates = [
        (12_000, 17_000, 17_000),  # overlaps, confirmed 7 s after the episode starts
        (40_000, 45_000, 45_000),  # a relative dip with no user symptom
    ]
    result = impact.compare_with_detector(candidates, [episode])
    assert result["candidates_overlapping_symptom"] == 1
    assert result["episodes_detected"] == 1
    assert result["episode_ms"] == 20_000
    assert result["episode_ms_covered"] == 5_000
    assert result["detection_delays_ms"] == [7_000]
    missed = impact.compare_with_detector([], [episode])
    assert missed["episodes_detected"] == 0 and missed["episode_ms_covered"] == 0


def test_delay_uses_first_overlapping_flag_not_group_onset() -> None:
    episode = impact.Episode("below_basic_use", 20_000, 40_000, 20, 1.0)
    result = impact.compare_with_detector(
        [(0, 30_000, 5_000)],
        [episode],
        [(0, 5_000, 5_000), (16_000, 21_000, 21_000), (25_000, 30_000, 30_000)],
    )
    assert result["detection_delays_ms"] == [1_000]


def test_candidate_takes_the_worst_level_it_touches_and_queue_orders_by_severity() -> None:
    primary = {
        "stalled": [impact.Episode("stalled", 20_000, 32_000, 8, 0.0)],
        "below_basic_use": [impact.Episode("below_basic_use", 10_000, 40_000, 26, 0.0)],
        "below_hd_video": [],
    }
    assert impact.candidate_impact((25_000, 30_000), primary) == ("stalled", 8, 0.0, 20_000)
    assert impact.candidate_impact((11_000, 15_000), primary) == (
        "below_basic_use",
        26,
        0.0,
        10_000,
    )
    assert impact.candidate_impact((50_000, 55_000), primary) == ("none", 0, None, None)
    rows: list[dict[str, object]] = [
        {
            "event_id": "b",
            "impact_level": "below_basic_use",
            "impact_seconds": 40,
            "impact_minimum_mbps": 1.0,
        },
        {
            "event_id": "a",
            "impact_level": "stalled",
            "impact_seconds": 10,
            "impact_minimum_mbps": 0.0,
        },
        {
            "event_id": "c",
            "impact_level": "below_basic_use",
            "impact_seconds": 60,
            "impact_minimum_mbps": 1.2,
        },
    ]
    assert [row["event_id"] for row in sorted(rows, key=impact.queue_order)] == ["a", "c", "b"]


def bin_view(start_ms: int, rate: float | None, active: bool = True) -> BinView:
    return BinView(
        segment_id=0,
        bin_start_ms=start_ms,
        available_at_ms=start_ms + 1_000,
        is_complete=True,
        sample_count=1,
        active_count=1 if active else 0,
        downlink_mbps=rate,
        downlink_valid_count=0 if rate is None else 1,
    )


def test_short_window_rate_needs_no_baseline_but_keeps_activity_and_coverage_rules() -> None:
    config = DetectorSection()
    segments = [(0, 0, 10_000)]
    chronic = {start: bin_view(start, 0.6) for start in range(0, 10_000, 1_000)}
    # A rate that was low from the start still counts, unlike the relative detector.
    assert short_window_rate(chronic, segments, 10_000, 5_000, config) == 0.6
    idle = {start: bin_view(start, 0.0, active=start < 6_000) for start in range(0, 10_000, 1_000)}
    assert short_window_rate(idle, segments, 10_000, 5_000, config) is None
    sparse = {
        start: bin_view(start, None if start >= 7_000 else 3.0) for start in range(0, 10_000, 1_000)
    }
    assert short_window_rate(sparse, segments, 10_000, 5_000, config) is None
    assert short_window_rate(chronic, segments, 4_000, 5_000, config) is None


def test_queue_tracks_each_episode_and_ranks_later_sustained_severity() -> None:
    episodes = {
        "session_a": {
            "below_basic_use": [
                impact.Episode("below_basic_use", 10_000, 30_000, 20, 0.0, 20_000),
                impact.Episode("below_basic_use", 31_000, 45_000, 14, 1.0, 41_000),
            ],
            "stalled": [impact.Episode("stalled", 20_000, 30_000, 10, 0.0, 30_000)],
        }
    }
    rows: list[dict[str, object]] = [
        {
            "session_id": "session_a",
            "event_id": "event_a",
            "triage_class": "unexplained",
            "impact_window_start_ms": 11_000,
            "impact_window_end_ms": 40_000,
            "first_confirmation_ms": 15_000,
        },
        {
            "session_id": "session_a",
            "event_id": "event_b",
            "triage_class": "unexplained",
            "impact_window_start_ms": 20_000,
            "impact_window_end_ms": 30_000,
            "first_confirmation_ms": 25_000,
        },
    ]
    assignments = impact.incident_queue(rows, episodes)
    assert [row["event_id"] for row in assignments] == ["event_a", "event_a"]
    assert assignments[0]["impact_level"] == "stalled"
    assert assignments[0]["symptom_confirmation_ms"] == 20_000
    assert assignments[1]["symptom_incident_start_ms"] == 31_000
    # Two episodes may choose one grouped candidate; dispatch it only once.
    assert len({row["event_id"] for row in assignments}) == 1
