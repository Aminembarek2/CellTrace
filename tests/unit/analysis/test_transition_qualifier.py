"""Qualified transitions: no transition from null, across segments or at the first row."""

from __future__ import annotations

import duckdb

from celltrace.analysis.transitions import (
    CellObservation,
    fetch_cell_observations,
    qualify_transitions,
)
from celltrace.contracts.common import IdentityBasis, TransitionType
from celltrace.data import fixture

EUTRAN = "eutran_cell_identity_layout"
UTRAN = "utran_cell_identity_layout"
UNKNOWN = "unknown"


def observation(
    elapsed_ms: int,
    cell: str | None,
    basis: str = EUTRAN,
    mode: str | None = "LTE",
    segment_id: int = 0,
) -> CellObservation:
    return CellObservation(elapsed_ms, segment_id, cell, basis if cell else UNKNOWN, mode)


def test_first_observation_and_repeats_are_not_transitions() -> None:
    assert qualify_transitions([]) == []
    assert qualify_transitions([observation(0, "100")]) == []
    assert qualify_transitions([observation(0, "100"), observation(1000, "100")]) == []


def test_null_rows_are_skipped_and_never_start_a_transition() -> None:
    assert qualify_transitions([observation(0, None), observation(1000, "100")]) == []
    assert (
        qualify_transitions(
            [observation(0, "100"), observation(1000, None), observation(2000, "100")]
        )
        == []
    )
    transitions = qualify_transitions(
        [observation(0, "100"), observation(1000, None), observation(2000, "200")]
    )
    assert len(transitions) == 1
    assert transitions[0].earlier_time_ms == 0 and transitions[0].gap_ms == 2000
    assert "null_identity_between_observations" in transitions[0].uncertainty


def test_no_transition_across_segments() -> None:
    observations = [observation(0, "100"), observation(9000, "200", segment_id=1)]
    assert qualify_transitions(observations) == []
    observations = [
        observation(0, "100"),
        observation(9000, None, segment_id=1),
        observation(10000, "200", segment_id=1),
    ]
    assert qualify_transitions(observations) == []


def test_serving_cell_change_with_same_layout() -> None:
    [transition] = qualify_transitions([observation(0, "100"), observation(1000, "200")])
    assert transition.transition_type is TransitionType.serving_cell_change
    assert transition.identity_basis is IdentityBasis.eutran_cell_identity_layout
    assert (transition.earlier_cell_id, transition.later_cell_id) == ("100", "200")
    assert (transition.earlier_time_ms, transition.later_time_ms, transition.gap_ms) == (
        0,
        1000,
        1000,
    )
    assert transition.uncertainty == ["identity_scope_unverified"]


def test_five_g_and_lte_share_a_family() -> None:
    [transition] = qualify_transitions(
        [observation(0, "100", mode="5G"), observation(1000, "200", mode="LTE")]
    )
    assert transition.transition_type is TransitionType.serving_cell_change
    assert "network_mode_changed_within_family" in transition.uncertainty


def test_technology_change_by_family_and_by_layout() -> None:
    [by_family] = qualify_transitions(
        [observation(0, "100"), observation(1000, "300", basis=UTRAN, mode="HSPA+")]
    )
    assert by_family.transition_type is TransitionType.technology_change
    assert by_family.identity_basis is IdentityBasis.unknown
    [by_layout] = qualify_transitions(
        [observation(0, "100"), observation(1000, "300", basis=UTRAN, mode="LTE")]
    )
    assert by_layout.transition_type is TransitionType.technology_change
    [gsm] = qualify_transitions(
        [
            observation(0, "100", basis=UNKNOWN, mode="UMTS"),
            observation(1000, "4660", basis=UNKNOWN, mode="EDGE"),
        ]
    )
    assert gsm.transition_type is TransitionType.technology_change
    assert "identity_basis_unknown" in gsm.uncertainty


def test_reported_identity_change_with_unknown_layout() -> None:
    [transition] = qualify_transitions(
        [
            observation(0, "4660", basis=UNKNOWN, mode="GPRS"),
            observation(1000, "4661", basis=UNKNOWN, mode="GPRS"),
        ]
    )
    assert transition.transition_type is TransitionType.reported_identity_change
    assert transition.identity_basis is IdentityBasis.unknown
    [unknown_mode] = qualify_transitions(
        [observation(0, "100", mode=None), observation(1000, "200", mode="LTE")]
    )
    assert unknown_mode.transition_type is TransitionType.serving_cell_change
    assert "network_mode_unknown" in unknown_mode.uncertainty


def test_gap_uncertainty_inside_one_segment() -> None:
    [transition] = qualify_transitions(
        [
            observation(0, "100"),
            observation(1000, None),
            observation(2000, None),
            observation(3000, None),
            observation(4000, "200"),
        ]
    )
    assert transition.gap_ms == 4000
    assert "gap_exceeds_cadence" in transition.uncertainty


def test_fixture_sessions(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    def transitions_for(member: str) -> list[tuple[str, int, int]]:
        observations = fetch_cell_observations(fixture_connection, fixture_session_ids[member])
        return [
            (item.transition_type.value, item.earlier_time_ms, item.later_time_ms)
            for item in qualify_transitions(observations)
        ]

    assert transitions_for(fixture.GAP_MEMBER) == []
    assert transitions_for(fixture.STEADY_MEMBER) == [("serving_cell_change", 29000, 30000)]
    assert transitions_for(fixture.RADIO_DROP_WITH_CELL_CHANGE_MEMBER) == [
        ("serving_cell_change", 39000, 40000)
    ]
    assert transitions_for(fixture.ORDER_AND_TECHNOLOGY_MEMBER) == [
        ("technology_change", 19000, 20000),
        ("technology_change", 29000, 30000),
        ("reported_identity_change", 34000, 35000),
        ("technology_change", 39000, 45000),
        ("technology_change", 54000, 55000),
    ]
    bounded = fetch_cell_observations(
        fixture_connection, fixture_session_ids[fixture.STEADY_MEMBER], 30000, 31000
    )
    assert [item.elapsed_ms for item in bounded] == [30000]
