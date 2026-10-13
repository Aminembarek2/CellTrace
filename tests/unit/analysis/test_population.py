"""Population analytics: rule boundaries, session bootstrap and a fixture run with known answers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from celltrace.analysis import population
from celltrace.cli import load_candidate
from celltrace.config import DetectorSection, ObservationSection
from celltrace.data.fixture import (
    NEAR_ZERO_REFERENCE_MEMBER,
    RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
    STABLE_RADIO_DROP_MEMBER,
    STEADY_MEMBER,
)


@pytest.mark.parametrize(
    "modes,expected",
    [
        ({"5G"}, "5G"),
        ({"LTE"}, "LTE"),
        ({"HSPA+", "UMTS"}, "3G"),
        ({"5G", "LTE"}, "mixed"),
        ({"LTE", None}, "mixed"),
        (set(), "no samples"),
    ],
)
def test_window_technology_labels_only_uniform_windows(
    modes: set[str | None], expected: str
) -> None:
    assert population.window_technology(modes) == expected


def test_rsrp_is_only_defined_on_lte_and_5g_rows() -> None:
    assert population.rsrp_defined({"5G", "LTE"})
    assert not population.rsrp_defined({"LTE", "HSPA+"})
    assert not population.rsrp_defined({"LTE", None})
    assert not population.rsrp_defined(set())


def test_nearby_transition_uses_a_half_open_interval() -> None:
    times = [10_000, 50_000]
    assert population.has_nearby_transition(times, 10_000, 20_000)
    assert not population.has_nearby_transition(times, 0, 10_000)
    assert not population.has_nearby_transition(times, 10_001, 50_000)


def test_episode_and_delay_medians_interpolate_even_counts() -> None:
    totals = population.SessionTotals(mobility="driving")
    totals.impact_seconds[("LTE", "observed")] = 100
    totals.detection_delays_ms = [10_000, 20_000]
    episodes = population.q5_episodes({"s": totals}, {"below_basic_use": [10, 20]}, 42)
    assert episodes["below_basic_use"]["primary_median_seconds"] == 15.0
    detector = population.q6_detector_against_symptom({"s": totals}, 42)
    assert detector["detection_delay_seconds"]["median"] == 15.0


@pytest.mark.parametrize(
    "delta,expected", [(-3.0, "stable"), (-3.01, "fell"), (3.0, "stable"), (3.01, "rose")]
)
def test_rsrp_status_matches_the_verifier_tolerance(delta: float, expected: str) -> None:
    assert population.rsrp_status(delta, 3.0) == expected


@pytest.mark.parametrize(
    "nearby,status,expected",
    [
        (True, "fell", "cell_change"),
        (True, "not_assessable", "cell_change"),
        (False, "fell", "rsrp_fell"),
        (False, "stable", "unexplained"),
        (False, "rose", "unexplained"),
        (False, "not_assessable", "not_assessable"),
    ],
)
def test_triage_precedence(nearby: bool, status: str, expected: str) -> None:
    assert population.triage_class(nearby, status) == expected


def test_bootstrap_needs_eight_sessions_and_is_deterministic() -> None:
    # Raised from five after a coverage check on this archive (subsampling coverage check).
    def share(sums: np.ndarray) -> np.ndarray:
        return sums[:, 0] / sums[:, 1]

    assert population.session_bootstrap([(1.0, 2.0)] * 7, share, 42) is None
    constant = population.session_bootstrap([(1.0, 4.0)] * 8, share, 42)
    assert constant == pytest.approx([0.25, 0.25])
    varied = [(float(index), 10.0) for index in range(8)]
    first = population.session_bootstrap(varied, share, 42)
    assert first == population.session_bootstrap(varied, share, 42)
    assert first is not None and first[0] < 0.35 < first[1]


def test_bootstrap_refuses_when_most_resamples_are_undefined() -> None:
    rows = [(1.0, 1.0)] + [(0.0, 0.0)] * 9

    def share(sums: np.ndarray) -> np.ndarray:
        return sums[:, 0] / sums[:, 1]

    assert population.session_bootstrap(rows, share, 42) is None


@pytest.fixture(scope="module")
def population_run(
    fixture_snapshot_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict[str, object], Path, Path]:
    root = tmp_path_factory.mktemp("population")
    public, private = root / "public", root / "private"
    summary = population.run_population(
        fixture_snapshot_dir, public, private, DetectorSection(), ObservationSection(), 42
    )
    return summary, public, private


def triage_rows(private: Path) -> list[dict[str, str]]:
    with (private / population.TRIAGE_NAME).open() as handle:
        return list(csv.DictReader(handle))


def test_fixture_drops_route_to_the_expected_classes(
    population_run: tuple[dict[str, object], Path, Path], fixture_session_ids: dict[str, str]
) -> None:
    _, _, private = population_run
    by_session = {row["session_id"]: row for row in triage_rows(private)}
    stable = by_session[fixture_session_ids[STABLE_RADIO_DROP_MEMBER]]
    assert stable["triage_class"] == "unexplained"
    assert stable["rsrp_status"] == "stable"
    changed = by_session[fixture_session_ids[RADIO_DROP_WITH_CELL_CHANGE_MEMBER]]
    assert changed["triage_class"] == "cell_change"
    assert changed["rsrp_status"] == "fell"
    assert float(changed["rsrp_delta_db"]) == pytest.approx(-32.0)
    assert fixture_session_ids[STEADY_MEMBER] not in by_session


def test_queue_holds_only_unexplained_drops_with_a_user_symptom(
    population_run: tuple[dict[str, object], Path, Path],
) -> None:
    _, _, private = population_run
    queue_path = private / population.QUEUE_NAME
    queue = json.loads(queue_path.read_text())
    eligible = [
        row
        for row in triage_rows(private)
        if row["triage_class"] == "unexplained"
        and row["impact_level"] in {"stalled", "below_basic_use"}
    ]
    queued = {entry["event_id"] for entry in queue["candidates"]}
    # One queued drop per symptom incident, drawn only from eligible drops.
    assert queued <= {row["event_id"] for row in eligible}
    incidents = {(row["session_id"], row["symptom_incident_start_ms"]) for row in eligible}
    assert len(queued) == len(incidents)
    expected = queued
    for event_id in expected:
        assert load_candidate(event_id, queue_path).event_id == event_id


def test_a_dip_that_still_supports_basic_use_stays_out_of_the_queue(
    population_run: tuple[dict[str, object], Path, Path], fixture_session_ids: dict[str, str]
) -> None:
    _, _, private = population_run
    by_session = {row["session_id"]: row for row in triage_rows(private)}
    # 20 Mbps falling to 3 Mbps for 30 s: a real relative drop, but above basic use.
    stable = by_session[fixture_session_ids[STABLE_RADIO_DROP_MEMBER]]
    assert stable["triage_class"] == "unexplained"
    assert stable["impact_level"] == "below_hd_video"
    queue = json.loads((private / population.QUEUE_NAME).read_text())
    assert stable["event_id"] not in {entry["event_id"] for entry in queue["candidates"]}


def test_chronically_poor_service_is_seen_even_without_a_relative_drop(
    population_run: tuple[dict[str, object], Path, Path], fixture_session_ids: dict[str, str]
) -> None:
    summary, _, private = population_run
    # 0.6 Mbps then 0.05 Mbps: the relative detector rejects the low baseline, so no drop.
    chronic = fixture_session_ids[NEAR_ZERO_REFERENCE_MEMBER]
    assert chronic not in {row["session_id"] for row in triage_rows(private)}
    detector = summary["q6_detector_vs_symptom"]
    assert isinstance(detector, dict)
    assert detector["episodes"] >= 1 and detector["missed_episodes"] >= 1
    stalled = summary["q5_episodes"]
    assert isinstance(stalled, dict)
    assert stalled["stalled"]["minimum_seconds"]["10"]["episodes"] >= 1


def test_summary_counts_are_internally_consistent(
    population_run: tuple[dict[str, object], Path, Path],
) -> None:
    summary, public, _ = population_run
    triage = summary["q3_triage"]
    assert isinstance(triage, dict)
    assert sum(item["events"] for item in triage["classes"].values()) == triage["events"]
    rates = summary["q1_drop_rates"]
    assert isinstance(rates, list)
    assert sum(item["events"] for item in rates) == triage["events"]
    monitoring = summary["q4_monitoring"]
    assert isinstance(monitoring, dict)
    for item in monitoring.values():
        if item["evaluated_seconds"]:
            shares = [reason["share"] for reason in item["reasons"].values()]
            assert sum(shares) == pytest.approx(1.0)
    for name in population.FIGURES.values():
        assert (public / name).stat().st_size > 0
    assert (
        "Q3. Which observation rules match candidate drops?"
        in (public / population.REPORT_NAME).read_text()
    )


def test_existing_outputs_are_never_overwritten(
    population_run: tuple[dict[str, object], Path, Path], fixture_snapshot_dir: Path
) -> None:
    _, public, private = population_run
    with pytest.raises(FileExistsError):
        population.run_population(
            fixture_snapshot_dir, public, private, DetectorSection(), ObservationSection(), 42
        )


def test_cadence_and_missing_by_technology_report_bounded_shares(
    population_run: tuple[dict[str, object], Path, Path],
) -> None:
    summary, public, _ = population_run
    timing = summary["cadence"]
    assert isinstance(timing, dict)
    # The fixture's 3 s and 2 s intervals leave empty bins inside one segment.
    assert 0 < timing["empty_bin_share"] < 1
    by_technology = summary["q4_missing_by_technology"]
    assert isinstance(by_technology, dict)
    assert {"LTE", "3G"} <= set(by_technology)
    for item in by_technology.values():
        assert set(item["missing_share"]) == {"RSRP", "RSRQ", "SNR", "CQI"}
        assert all(0 <= share <= 1 for share in item["missing_share"].values())
    assert (
        "Q4c, a follow up added after the predeclared run"
        in (public / population.REPORT_NAME).read_text()
    )


def test_intervals_require_enough_exposed_sessions_in_each_denominator() -> None:
    rows = [(1.0, 10.0, 1.0, 10.0)] * 3 + [(0.0, 0.0, 1.0, 10.0)] * 17
    result = population.estimate(rows, lambda sums: sums[:, 0] / sums[:, 1], 42)
    assert result["value"] == 0.1
    assert result["interval_95"] is None
    assert result["denominator_contributing_sessions"] == [3]


@pytest.mark.parametrize("filename", [population.REPORT_NAME, *population.FIGURES.values()])
def test_partial_report_outputs_are_preserved(
    tmp_path: Path,
    fixture_snapshot_dir: Path,
    filename: str,
) -> None:
    public = tmp_path / "public"
    public.mkdir()
    existing = public / filename
    existing.write_text("previous output")
    with pytest.raises(FileExistsError):
        population.run_population(
            fixture_snapshot_dir,
            public,
            tmp_path / "private",
            DetectorSection(),
            ObservationSection(),
            42,
        )
    assert existing.read_text() == "previous output"


def test_private_output_cannot_be_nested_in_the_public_output(
    tmp_path: Path,
    fixture_snapshot_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="non-overlapping"):
        population.run_population(
            fixture_snapshot_dir,
            tmp_path,
            tmp_path / "private",
            DetectorSection(),
            ObservationSection(),
            42,
        )


def test_report_labels_the_actual_overlapping_window_estimands(
    population_run: tuple[dict[str, object], Path, Path],
) -> None:
    summary, public, _ = population_run
    text = (public / population.REPORT_NAME).read_text()
    assert "[-10000, 0)" in text
    assert "within 5 s after" not in text
    assert "window-weighted sample occurrences" in text
    assert "do not test paired differences" in text
    assert (
        summary["q2_cell_change"]["exposure_basis"]
        == "eligible one-second detector confirmation times"
    )


def test_unknown_mobility_exposure_is_not_silently_omitted() -> None:
    totals = population.SessionTotals("walking")
    totals.outcome_seconds["eligible_not_flagged"] = 3
    result = population.q4_monitoring({"session": totals})
    assert result["walking"]["evaluated_seconds"] == 3


def test_report_leads_with_findings_chance_checks_and_embedded_figures(
    population_run: tuple[dict[str, object], Path, Path],
) -> None:
    summary, public, _ = population_run
    report = (public / population.REPORT_NAME).read_text()
    headings = [line for line in report.splitlines() if line.startswith("## ")]
    assert headings[:2] == ["## Summary", "## Q0. Conditional surrogate sensitivity checks"]
    for name in population.FIGURES.values():
        assert f"]({name})" in report and (public / name).is_file()
    checks = summary["chance_checks"]
    assert isinstance(checks, dict) and set(checks["detector"]) == set(population.chance.NULLS)
