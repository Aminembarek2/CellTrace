"""Cell replication: planted cell effects are found, absent ones are not, days never leak."""

from __future__ import annotations

from collections import Counter

import duckdb
import numpy as np
import pytest

from celltrace.analysis import cells


def visit(
    day: str, cell: str, rate: float, technology: str = "LTE", session: str = ""
) -> cells.Visit:
    return cells.Visit(
        session_id=session or f"{day}-{cell}",
        day=day,
        cell=cell,
        technology=technology,
        samples=20,
        low_samples=20 if rate < cells.LOW_RATE_MBPS else 0,
        median_mbps=rate,
    )


def planted(effect: bool, seed: int) -> list[cells.Visit]:
    generator = np.random.default_rng(seed)
    cell_rates = {f"c{index}": 10 ** generator.uniform(0, 2) for index in range(12)}
    visits = []
    for day in ("d1", "d2", "d3", "d4"):
        for cell, rate in cell_rates.items():
            level = rate if effect else 10 ** generator.uniform(0, 2)
            visits.append(visit(day, cell, level * 10 ** generator.normal(0, 0.05)))
    return visits


def by_day(item: cells.Visit) -> str:
    return item.day


def test_a_stable_cell_effect_replicates_across_days() -> None:
    result = cells.replication_test(planted(True, 1), by_day, 99, 5)
    assert result["skill_against_technology"] > 0.5
    assert result["p_value"] <= 0.05


def test_permutation_p_value_is_never_below_one_over_draws_plus_one() -> None:
    # Expert audit mutation check: dropping the plus one left every test passing.
    result = cells.replication_test(planted(True, 1), by_day, 99, 5)
    assert max(result["null_draws"]) < result["skill_against_technology"]
    assert result["p_value"] == pytest.approx(1 / 100)


def test_no_cell_effect_does_not_replicate() -> None:
    result = cells.replication_test(planted(False, 2), by_day, 99, 5)
    assert result["skill_against_technology"] < 0
    assert result["p_value"] > 0.05


def test_a_held_out_day_is_predicted_from_other_days_only() -> None:
    visits = [
        visit("d1", "a", 100.0),
        visit("d1", "a", 0.2, session="second"),
        visit("d2", "a", 10.0),
        visit("d3", "a", 30.0),
        visit("d2", "b", 1.0),
    ]
    rows = cells.predictions(visits, by_day)
    held = [row for row in rows if row["fold"] == "d1"]
    assert len(held) == 2
    # Same day visits of cell a (100 and 0.2 Mbps) must not enter the prediction.
    expected = np.median([np.log10(10.0), np.log10(30.0)])
    for row in held:
        assert row["cell_prediction"] == pytest.approx(expected)


def test_unseen_cells_are_not_scored_and_technology_falls_back_to_all_visits() -> None:
    visits = [visit("d1", "a", 5.0, "3G"), visit("d2", "a", 5.0, "LTE"), visit("d2", "z", 5.0)]
    rows = cells.predictions(visits, by_day)
    assert {(row["fold"], row["cell"]) for row in rows} == {("d1", "a"), ("d2", "a")}
    assert [row["technology_fallback"] for row in rows if row["fold"] == "d1"] == [True]


def test_permutation_keeps_label_counts_within_day_and_technology() -> None:
    visits = [*planted(True, 3), visit("d1", "g", 3.0, "5G"), visit("d1", "h", 4.0, "5G")]
    shuffled = cells.permuted_cells(visits, np.random.default_rng(4))
    for key in {(item.day, item.technology) for item in visits}:
        before = Counter(item.cell for item in visits if (item.day, item.technology) == key)
        after = Counter(item.cell for item in shuffled if (item.day, item.technology) == key)
        assert before == after
    assert [item.cell for item in shuffled] != [item.cell for item in visits]
    assert [item.median_mbps for item in shuffled] == [item.median_mbps for item in visits]


def test_visits_come_from_active_driving_samples_with_a_known_cell() -> None:
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE sessions (session_id VARCHAR, application_cohort VARCHAR, "
        "mobility_cohort VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE samples (session_id VARCHAR, local_timestamp TIMESTAMP, "
        "serving_cell_id VARCHAR, cell_identity_basis VARCHAR, network_mode VARCHAR, "
        "download_state VARCHAR, downlink_mbps DOUBLE, reported_rsrp_dbm DOUBLE)"
    )
    connection.execute(
        "INSERT INTO sessions VALUES ('drive', 'file_download', 'driving'), "
        "('parked', 'file_download', 'static')"
    )
    rows = [("drive", "a", "eutran_cell_identity_layout", "LTE", "active", 2.0, -80.0)] * 12
    rows += [("drive", "a", "eutran_cell_identity_layout", "LTE", "idle", 0.0, -120.0)] * 5
    rows += [("drive", "b", "unknown", "LTE", "active", 9.0, -90.0)] * 12
    rows += [("drive", "c", "eutran_cell_identity_layout", "HSPA+", "active", 1.0, -70.0)] * 9
    rows += [("parked", "a", "eutran_cell_identity_layout", "LTE", "active", 50.0, -60.0)] * 12
    connection.executemany(
        "INSERT INTO samples VALUES (?, TIMESTAMP '2020-01-16 10:00:00', ?, ?, ?, ?, ?, ?)", rows
    )
    visits = cells.load_visits(connection, "file_download", 10)
    # Idle samples, the unknown identity, the short 3G visit and the static session drop out.
    assert visits == [
        cells.Visit(
            "drive",
            "2020-01-16",
            "a",
            "LTE",
            samples=12,
            low_samples=0,
            median_mbps=2.0,
            median_rsrp_dbm=-80.0,
        )
    ]


def test_an_archive_without_repeated_cells_gives_undefined_results() -> None:
    result = cells.replication_test([visit("d1", "a", 5.0)], by_day, 9, 1)
    assert result["scored_visits"] == 0 and result["skill_against_technology"] is None
    assert result["p_value"] is None
    assert cells.ranked_cells([])["slowest"] == []


def test_ranking_shows_how_far_a_cell_moved_and_leaves_3g_rsrp_undefined() -> None:
    visits = [
        visit("d1", "a", 0.0),
        visit("d2", "a", 7.0),
        visit("d1", "g", 0.5, "3G"),
        visit("d2", "g", 0.7, "3G"),
    ]
    visits[0] = cells.Visit(**{**visits[0].__dict__, "median_rsrp_dbm": -75.0})
    ranking = {item["cell"]: item for item in cells.ranked_cells(visits)["slowest"]}
    assert ranking["a"]["visit_median_range_mbps"] == [0.0, 7.0]
    assert ranking["a"]["median_rsrp_dbm"] == -75.0
    assert ranking["g"]["median_rsrp_dbm"] is None


def test_proxy_identity_is_opt_in_and_drops_keys_shared_by_two_families() -> None:
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE sessions (session_id VARCHAR, application_cohort VARCHAR, "
        "mobility_cohort VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE samples (session_id VARCHAR, local_timestamp TIMESTAMP, "
        "serving_cell_id VARCHAR, cell_identity_basis VARCHAR, network_mode VARCHAR, "
        "download_state VARCHAR, downlink_mbps DOUBLE, reported_rsrp_dbm DOUBLE)"
    )
    connection.execute("INSERT INTO sessions VALUES ('drive', 'file_download', 'car')")
    rows = [("LTE:-8.40,51.90:2", "LTE")] * 12 + [("3G:-8.40,51.90:2", "HSPA+")] * 12
    rows += [("LTE:-8.50,51.80:3", "LTE")] * 12
    connection.executemany(
        "INSERT INTO samples VALUES ('drive', TIMESTAMP '2018-01-16 10:00:00', ?, 'unknown', "
        "?, 'active', 5.0, -90.0)",
        rows,
    )
    assert cells.load_visits(connection, "file_download", 10, ("car",)) == []
    accepted = cells.load_visits(connection, "file_download", 10, ("car",), True)
    assert [visit.cell for visit in accepted] == ["LTE:-8.50,51.80:3"]
