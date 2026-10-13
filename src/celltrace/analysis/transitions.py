"""Qualified serving cell transitions from reported cell identity observations.

A change of reported identity is a transition proxy. It does not show that a
handover procedure ran, completed or affected the rate. The qualifier only
compares consecutive known identities inside one segment and labels how much
the identity comparison can be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from celltrace.contracts.common import IdentityBasis, TransitionType
from celltrace.data.raca import technology_family
from celltrace.data.schema import SEGMENT_GAP_THRESHOLD_MS

UNCERTAINTY_GAP = "gap_exceeds_cadence"
UNCERTAINTY_SCOPE = "identity_scope_unverified"
UNCERTAINTY_NULL_BETWEEN = "null_identity_between_observations"
UNCERTAINTY_BASIS_UNKNOWN = "identity_basis_unknown"
UNCERTAINTY_MODE_UNKNOWN = "network_mode_unknown"
UNCERTAINTY_MODE_CHANGED = "network_mode_changed_within_family"


@dataclass(frozen=True)
class CellObservation:
    elapsed_ms: int
    segment_id: int
    serving_cell_id: str | None
    cell_identity_basis: str
    network_mode: str | None


@dataclass(frozen=True)
class QualifiedTransition:
    transition_type: TransitionType
    earlier_cell_id: str
    later_cell_id: str
    earlier_time_ms: int
    later_time_ms: int
    gap_ms: int
    identity_basis: IdentityBasis
    earlier_network_mode: str | None
    later_network_mode: str | None
    uncertainty: list[str] = field(default_factory=list)


def _basis(value: str) -> IdentityBasis:
    try:
        return IdentityBasis(value)
    except ValueError:
        return IdentityBasis.unknown


def _classify(earlier: CellObservation, later: CellObservation) -> tuple[TransitionType, list[str]]:
    """Order of rules: known technology family change, unknown basis, basis change, same basis."""
    earlier_basis = _basis(earlier.cell_identity_basis)
    later_basis = _basis(later.cell_identity_basis)
    earlier_family = technology_family(earlier.network_mode)
    later_family = technology_family(later.network_mode)
    notes: list[str] = []
    if "unknown" in (earlier_family, later_family):
        notes.append(UNCERTAINTY_MODE_UNKNOWN)
    basis_unknown = IdentityBasis.unknown in (earlier_basis, later_basis)
    if basis_unknown:
        notes.append(UNCERTAINTY_BASIS_UNKNOWN)
    if "unknown" not in (earlier_family, later_family) and earlier_family != later_family:
        return TransitionType.technology_change, notes
    if basis_unknown:
        return TransitionType.reported_identity_change, notes
    if earlier_basis != later_basis:
        return TransitionType.technology_change, notes
    if earlier.network_mode != later.network_mode and UNCERTAINTY_MODE_UNKNOWN not in notes:
        notes.append(UNCERTAINTY_MODE_CHANGED)
    notes.append(UNCERTAINTY_SCOPE)
    return TransitionType.serving_cell_change, notes


def qualify_transitions(observations: list[CellObservation]) -> list[QualifiedTransition]:
    """Transitions between consecutive known identities within one segment.

    Observations must already be ordered by (elapsed_ms, source order). Rows
    with a null identity are skipped, so a change from null to a known
    identity is never a transition, and the first known observation of each
    segment has no predecessor.
    """
    transitions: list[QualifiedTransition] = []
    previous: CellObservation | None = None
    nulls_since_previous = False
    for observation in observations:
        if previous is not None and observation.segment_id != previous.segment_id:
            previous = None
            nulls_since_previous = False
        if observation.serving_cell_id is None:
            nulls_since_previous = previous is not None
            continue
        if previous is not None and previous.serving_cell_id != observation.serving_cell_id:
            assert previous.serving_cell_id is not None
            transition_type, notes = _classify(previous, observation)
            gap_ms = observation.elapsed_ms - previous.elapsed_ms
            uncertainty: list[str] = []
            if gap_ms > SEGMENT_GAP_THRESHOLD_MS:
                uncertainty.append(UNCERTAINTY_GAP)
            if nulls_since_previous:
                uncertainty.append(UNCERTAINTY_NULL_BETWEEN)
            uncertainty.extend(notes)
            earlier_basis = _basis(previous.cell_identity_basis)
            later_basis = _basis(observation.cell_identity_basis)
            transitions.append(
                QualifiedTransition(
                    transition_type=transition_type,
                    earlier_cell_id=previous.serving_cell_id,
                    later_cell_id=observation.serving_cell_id,
                    earlier_time_ms=previous.elapsed_ms,
                    later_time_ms=observation.elapsed_ms,
                    gap_ms=gap_ms,
                    identity_basis=(
                        later_basis if earlier_basis == later_basis else IdentityBasis.unknown
                    ),
                    earlier_network_mode=previous.network_mode,
                    later_network_mode=observation.network_mode,
                    uncertainty=uncertainty,
                )
            )
        previous = observation
        nulls_since_previous = False
    return transitions


def fetch_cell_observations(
    connection: duckdb.DuckDBPyConnection,
    session_id: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[CellObservation]:
    """Observations of one session in (elapsed_ms, source_row_number) order.

    Optional bounds form the half open interval [start_ms, end_ms).
    """
    lower = start_ms if start_ms is not None else -1
    upper = end_ms if end_ms is not None else 2**62
    rows = connection.execute(
        """
        SELECT elapsed_ms, segment_id, serving_cell_id, cell_identity_basis, network_mode
        FROM samples
        WHERE session_id = ? AND elapsed_ms >= ? AND elapsed_ms < ?
        ORDER BY elapsed_ms, source_row_number
        """,
        [session_id, lower, upper],
    ).fetchall()
    return [
        CellObservation(
            elapsed_ms=int(row[0]),
            segment_id=int(row[1]),
            serving_cell_id=row[2],
            cell_identity_basis=str(row[3]),
            network_mode=row[4],
        )
        for row in rows
    ]
