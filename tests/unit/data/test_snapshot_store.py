"""Immutable snapshot: bins, availability, hashes, integrity and read only access."""

from __future__ import annotations

import json
import shutil
import stat
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import duckdb
import pytest

from celltrace.data import fixture
from celltrace.data.raca import RACA_HEADER
from celltrace.data.schema import (
    CAPABILITY_NAMES,
    SNAPSHOT_FILE_NAME,
    SNAPSHOT_MANIFEST_NAME,
    TABLE_ORDER_KEYS,
)
from celltrace.data.snapshot import (
    SnapshotIntegrityError,
    load_manifest,
    open_snapshot,
    run_ingest,
    verify_logical_hash,
)


def test_bins_are_available_at_their_end_and_only_the_final_bin_is_incomplete(
    fixture_connection: duckdb.DuckDBPyConnection,
) -> None:
    rows = fixture_connection.execute(
        "SELECT session_id, bin_start_ms, bin_end_ms, available_at_ms, is_complete "
        "FROM bins_1s ORDER BY session_id, bin_start_ms"
    ).fetchall()
    by_session: dict[str, list[tuple[int, int, int, bool]]] = defaultdict(list)
    for session_id, start, end, available, complete in rows:
        assert end == start + 1000
        assert available == end
        by_session[session_id].append((start, end, available, complete))
    for bins in by_session.values():
        assert [entry[3] for entry in bins] == [True] * (len(bins) - 1) + [False]


def test_empty_bins_inside_a_segment(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    rows = fixture_connection.execute(
        "SELECT bin_start_ms, segment_id, sample_count, active_fraction, downlink_mbps, "
        "downlink_mbps_valid_count FROM bins_1s WHERE session_id = ? AND sample_count = 0 "
        "ORDER BY bin_start_ms",
        [fixture_session_ids[fixture.MISSING_VALUES_MEMBER]],
    ).fetchall()
    assert rows == [
        (21000, 0, 0, None, None, 0),
        (22000, 0, 0, None, None, 0),
        (31000, 0, 0, None, None, 0),
    ]


def test_no_bins_inside_a_segment_gap(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    rows = fixture_connection.execute(
        "SELECT segment_id, min(bin_start_ms), max(bin_end_ms), count(*) FROM bins_1s "
        "WHERE session_id = ? GROUP BY segment_id ORDER BY segment_id",
        [fixture_session_ids[fixture.GAP_MEMBER]],
    ).fetchall()
    assert rows == [(0, 0, 20000, 20), (1, 30000, 50000, 20)]


def test_bin_means_match_an_independent_calculation(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    session_id = fixture_session_ids[fixture.MISSING_VALUES_MEMBER]
    samples = fixture_connection.execute(
        "SELECT elapsed_ms, downlink_mbps, reported_rsrp_dbm, download_state FROM samples "
        "WHERE session_id = ?",
        [session_id],
    ).fetchall()
    downlink: dict[int, list[float]] = defaultdict(list)
    rsrp: dict[int, list[float]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    for elapsed, rate, power, _ in samples:
        counts[elapsed] += 1
        if rate is not None:
            downlink[elapsed].append(rate)
        if power is not None:
            rsrp[elapsed].append(power)
    bins = fixture_connection.execute(
        "SELECT bin_start_ms, sample_count, downlink_mbps, downlink_mbps_valid_count, "
        "reported_rsrp_dbm, active_fraction FROM bins_1s WHERE session_id = ?",
        [session_id],
    ).fetchall()
    for start, sample_count, rate_mean, rate_valid, rsrp_mean, active in bins:
        assert sample_count == counts.get(start, 0)
        assert rate_valid == len(downlink.get(start, []))
        expected_rate = statistics.fmean(downlink[start]) if downlink.get(start) else None
        expected_rsrp = statistics.fmean(rsrp[start]) if rsrp.get(start) else None
        assert rate_mean == pytest.approx(expected_rate)
        assert rsrp_mean == pytest.approx(expected_rsrp)
        assert active == (1.0 if sample_count else None)
    by_start = {row[0]: row for row in bins}
    assert by_start[12000][1:4] == (2, pytest.approx(10.5), 2)
    assert by_start[12000][4] == pytest.approx(-86.0)
    assert by_start[9000][2] == 0.0
    assert by_start[10000][2:4] == (None, 0)


def test_logical_hash_is_stable_across_ingests(fixture_archive_dir: Path, tmp_path: Path) -> None:
    first = run_ingest(fixture_archive_dir, tmp_path / "first")
    second = run_ingest(fixture_archive_dir, tmp_path / "second")
    assert first.name == second.name
    first_manifest = load_manifest(first)
    second_manifest = load_manifest(second)
    assert first_manifest["logical_hash"] == second_manifest["logical_hash"]
    assert first_manifest["tables"] == second_manifest["tables"]
    database_mtime = (first / SNAPSHOT_FILE_NAME).stat().st_mtime_ns
    again = run_ingest(fixture_archive_dir, tmp_path / "first")
    assert again == first
    assert (first / SNAPSHOT_FILE_NAME).stat().st_mtime_ns == database_mtime
    leftovers = [path.name for path in (tmp_path / "first").iterdir() if path != first]
    assert leftovers == []
    assert verify_logical_hash(first)


def test_open_snapshot_rejects_a_modified_file(fixture_snapshot_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / fixture_snapshot_dir.name
    shutil.copytree(fixture_snapshot_dir, copy)
    connection, _ = open_snapshot(copy)
    connection.close()
    database = copy / SNAPSHOT_FILE_NAME
    database.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with database.open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(SnapshotIntegrityError):
        open_snapshot(copy)


def test_manifest_identity_mismatch_is_rejected(fixture_snapshot_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / fixture_snapshot_dir.name
    shutil.copytree(fixture_snapshot_dir, copy)
    manifest_path = copy / SNAPSHOT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["logical_hash"] = "f" * 64
    manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SnapshotIntegrityError):
        open_snapshot(copy)


def test_read_only_connection_rejects_writes_and_file_access(
    fixture_connection: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    with pytest.raises(duckdb.Error):
        fixture_connection.execute("DELETE FROM samples")
    with pytest.raises(duckdb.Error):
        fixture_connection.execute("CREATE TABLE extra (value INTEGER)")
    with pytest.raises(duckdb.Error):
        fixture_connection.execute(f"COPY (SELECT 1) TO '{tmp_path / 'leak.csv'}'")
    with pytest.raises(duckdb.Error):
        fixture_connection.execute("SET enable_external_access = true")
    assert not (tmp_path / "leak.csv").exists()


def test_snapshot_files_are_read_only_on_disk(fixture_snapshot_dir: Path) -> None:
    for name in (SNAPSHOT_FILE_NAME, SNAPSHOT_MANIFEST_NAME):
        mode = (fixture_snapshot_dir / name).stat().st_mode
        assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def test_manifest_records_checks_hashes_and_sources(fixture_snapshot_dir: Path) -> None:
    manifest = load_manifest(fixture_snapshot_dir)
    assert manifest["snapshot_id"] == fixture_snapshot_dir.name
    assert manifest["snapshot_id"] == "snap_" + manifest["logical_hash"][:16]
    assert set(manifest["tables"]) == set(TABLE_ORDER_KEYS)
    checks = {check["name"]: check for check in manifest["sql_checks"]}
    for required in (
        "nonnegative_rates",
        "nonnegative_elapsed_time",
        "elapsed_order_and_segments",
        "source_row_reconciliation",
        "bin_sample_totals",
        "unique_key_samples",
        "unique_key_bins_1s",
    ):
        assert checks[required]["passed"] and checks[required]["violations"] == 0
    members = manifest["sources"][0]["members"]
    assert sorted(member["member_path"] for member in members) == sorted(fixture.FIXTURE_SESSIONS)
    assert all(len(member["sha256"]) == 64 for member in members)
    for key in ("mapping_version", "normalization_version", "dictionary_hash", "created_at"):
        assert manifest[key]


def test_metadata_provenance_and_capabilities(
    fixture_connection: duckdb.DuckDBPyConnection,
) -> None:
    metadata = dict(
        fixture_connection.execute("SELECT key, value FROM snapshot_metadata").fetchall()
    )
    for key in (
        "snapshot_format_version",
        "source_id",
        "mapping_version",
        "normalization_version",
        "dictionary_version",
        "dictionary_hash",
        "segment_gap_threshold_ms",
        "timestamp_basis",
    ):
        assert metadata[key]
    assert metadata["source_id"] == fixture.FIXTURE_SOURCE_ID
    assert metadata["segment_gap_threshold_ms"] == "3000"
    provenance = fixture_connection.execute(
        "SELECT canonical_metric, raw_field FROM metric_provenance ORDER BY canonical_metric"
    ).fetchall()
    assert ("downlink_mbps", "DL_bitrate") in provenance and len(provenance) == 11
    capabilities = dict(
        fixture_connection.execute("SELECT capability, status FROM capabilities").fetchall()
    )
    assert set(capabilities) == set(CAPABILITY_NAMES)
    assert set(capabilities.values()) <= {"supported", "partial", "unavailable"}
    assert capabilities["neighbour_metrics"] == "unavailable"
    assert capabilities["serving_cell_transitions"] == "partial"


def test_source_counts_reconcile(fixture_connection: duckdb.DuckDBPyConnection) -> None:
    totals = fixture_connection.execute(
        "SELECT sum(raw_row_count), sum(accepted_row_count), sum(quarantined_row_count) "
        "FROM sessions"
    ).fetchone()
    samples = fixture_connection.execute("SELECT count(*) FROM samples").fetchone()
    quarantined = fixture_connection.execute("SELECT count(*) FROM quarantined_rows").fetchone()
    assert totals is not None and samples is not None and quarantined is not None
    assert totals[0] == totals[1] + totals[2]
    assert (totals[1], totals[2]) == (samples[0], quarantined[0])


def test_failed_ingest_leaves_no_snapshot(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    header = ",".join(reversed(RACA_HEADER))
    with zipfile.ZipFile(raw / "bad.zip", "w") as archive:
        archive.writestr("5G-production-dataset/Download/Static/B_2030.csv", header + "\n")
    out_dir = tmp_path / "processed"
    with pytest.raises(ValueError):
        run_ingest(raw, out_dir)
    assert list(out_dir.iterdir()) == []


def test_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_ingest(tmp_path, tmp_path / "processed")
