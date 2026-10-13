"""Raca adapter: archive safety, header checks, value mapping, time order and segments."""

from __future__ import annotations

import re
import zipfile
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from celltrace.data import fixture
from celltrace.data.raca import (
    RACA_HEADER,
    ArchiveMember,
    ArchiveSafetyError,
    RawRow,
    RawSession,
    UnexpectedHeaderError,
    cell_identity,
    convert_metric,
    inspect_archive,
    map_state,
    normalize_session,
    parse_member,
)
from celltrace.data.schema import SEGMENT_GAP_THRESHOLD_MS


def _member(
    path: str = "5G-production-dataset/Download/Static/X_2030.01.01_00.00.00.csv",
) -> ArchiveMember:
    return ArchiveMember(
        archive_path=Path("unused.zip"),
        member_path=path,
        uncompressed_bytes=0,
        compressed_bytes=0,
        sha256="0" * 64,
        application_cohort="file_download",
        mobility_cohort="static",
    )


def _sample_line(timestamp: str, downlink: str = "1000", state: str = "D") -> str:
    values = {name: "-" for name in RACA_HEADER}
    values.update(
        {
            "Timestamp": timestamp,
            "Speed": "0",
            "Operatorname": "Z",
            "CellID": "7",
            "NetworkMode": "LTE",
            "RSRP": "-90",
            "DL_bitrate": downlink,
            "UL_bitrate": "10",
            "State": state,
            "CELLHEX": "0C",
            "NODEHEX": "1A2B",
            "LACHEX": "00AA",
            "RAWCELLID": str(0x1A2B * 256 + 0x0C),
        }
    )
    return ",".join(values[name] for name in RACA_HEADER)


def _raw_session(lines: list[str]) -> RawSession:
    content = ("\n".join([",".join(RACA_HEADER), *lines]) + "\n").encode()
    return parse_member(_member(), content, "unit_test_source")


def _zip_with(tmp_path: Path, entries: dict[str, bytes]) -> Path:
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


@pytest.mark.parametrize(
    "header",
    [
        [RACA_HEADER[0], RACA_HEADER[2], RACA_HEADER[1], *RACA_HEADER[3:]],
        [*RACA_HEADER, "Extra"],
        ["Time", *RACA_HEADER[1:]],
        list(RACA_HEADER[:-1]),
    ],
    ids=["swapped", "extra_column", "renamed", "missing_column"],
)
def test_unknown_header_raises(header: list[str]) -> None:
    content = (",".join(header) + "\n" + _sample_line("2030.01.01_00.00.00") + "\n").encode()
    with pytest.raises(UnexpectedHeaderError):
        parse_member(_member(), content, "unit_test_source")


def test_semicolon_delimited_header_raises() -> None:
    content = (";".join(RACA_HEADER) + "\n").encode()
    with pytest.raises(UnexpectedHeaderError):
        parse_member(_member(), content, "unit_test_source")


@pytest.mark.parametrize("name", ["../escape.csv", "/absolute/path.csv", "folder\\windows.csv"])
def test_unsafe_member_names_are_rejected(tmp_path: Path, name: str) -> None:
    archive = _zip_with(tmp_path, {name: b"x"})
    with pytest.raises(ArchiveSafetyError):
        inspect_archive(archive)


def test_uncompressed_size_ceiling(tmp_path: Path) -> None:
    archive = _zip_with(tmp_path, {"5G-production-dataset/Download/Static/a.csv": b"x" * 100})
    with pytest.raises(ArchiveSafetyError):
        inspect_archive(archive, maximum_uncompressed_bytes=50)


def test_metadata_entries_are_skipped(fixture_archive_dir: Path) -> None:
    inventory = inspect_archive(fixture_archive_dir / fixture.FIXTURE_ARCHIVE_NAME)
    reasons = {entry.reason for entry in inventory.skipped}
    assert {"directory", "macos_resource_metadata", "finder_metadata"} <= reasons
    assert [member.member_path for member in inventory.members] == sorted(fixture.FIXTURE_SESSIONS)
    cohorts = {member.member_path: member.application_cohort for member in inventory.members}
    assert cohorts[fixture.STREAMING_STATE_TEXT_MEMBER] == "video_netflix"
    assert cohorts[fixture.STEADY_MEMBER] == "file_download"


def test_kbps_conversion_and_zero_versus_null() -> None:
    assert convert_metric("downlink_mbps", "20000") == (20.0, None, False)
    assert convert_metric("downlink_mbps", "0") == (0.0, None, False)
    assert convert_metric("downlink_mbps", "-") == (None, None, False)
    assert convert_metric("uplink_mbps", "532905") == (532.905, None, False)
    assert convert_metric("downlink_mbps", "-5")[1] == "negative_rate"
    assert convert_metric("reported_rsrp_dbm", "nan")[1] == "unparseable_number"
    assert convert_metric("reported_rsrp_dbm", "")[1] == "empty_value"


def test_sentinels_and_fill_values_become_null() -> None:
    assert convert_metric("reported_rssi_dbm", "2147483647") == (
        None,
        "android_unavailable_sentinel",
        False,
    )
    assert convert_metric("reported_nrx_rsrq_db", "2147483647.0")[1] == (
        "android_unavailable_sentinel"
    )
    assert convert_metric("reported_rsrp_dbm", "-200") == (None, "suspected_fill_value", False)
    assert convert_metric("reported_nrx_rsrq_db", "-204.0")[1] == "suspected_fill_value"
    assert convert_metric("reported_rssi_dbm", "-45") == (-45.0, None, True)


def test_snapshot_rejections_and_flags_for_missing_values(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    session_id = fixture_session_ids[fixture.MISSING_VALUES_MEMBER]
    rejections = set(
        fixture_connection.execute(
            """
            SELECT samples.elapsed_ms, raw_field, raw_value, reason
            FROM value_rejections JOIN samples USING (session_id, source_row_number)
            WHERE session_id = ?
            """,
            [session_id],
        ).fetchall()
    )
    assert rejections == {
        (6000, "RSSI", "2147483647", "android_unavailable_sentinel"),
        (6000, "CQI", "2147483647", "android_unavailable_sentinel"),
        (7000, "RSRP", "-200", "suspected_fill_value"),
        (15000, "NRxRSRP", "-205", "suspected_fill_value"),
        (15000, "NRxRSRQ", "-204.0", "suspected_fill_value"),
    }
    rows = {
        int(row[0]): row[1:]
        for row in fixture_connection.execute(
            """
            SELECT elapsed_ms, downlink_mbps, reported_rsrp_dbm, reported_rssi_dbm,
                   reported_cqi, reported_rsrq_db, quality_flags, ping_average_ms
            FROM samples WHERE session_id = ? AND elapsed_ms <> 12000
            """,
            [session_id],
        ).fetchall()
    }
    assert rows[5000][3] is None and rows[5000][4] is None
    assert rows[6000][2] is None and rows[6000][3] is None
    assert rows[7000][1] is None
    assert rows[8000][2] == -45.0
    assert "outside_documented_range:reported_rssi_dbm" in rows[8000][5]
    assert rows[9000][0] == 0.0
    assert rows[10000][0] is None
    assert rows[15000][6] == 35.0
    assert rows[0][0] == 10.0


def test_same_second_rows_are_kept_in_source_order(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    rows = fixture_connection.execute(
        """
        SELECT source_row_number, downlink_mbps, quality_flags FROM samples
        WHERE session_id = ? AND elapsed_ms = 12000 ORDER BY source_row_number
        """,
        [fixture_session_ids[fixture.MISSING_VALUES_MEMBER]],
    ).fetchall()
    assert [row[1] for row in rows] == [9.0, 12.0]
    assert rows[0][0] + 1 == rows[1][0]
    assert all("same_second" in row[2] for row in rows)


def test_reorder_regression_and_quarantine(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    session_id = fixture_session_ids[fixture.ORDER_AND_TECHNOLOGY_MEMBER]
    quarantined = fixture_connection.execute(
        "SELECT source_row_number, reason FROM quarantined_rows WHERE session_id = ? "
        "ORDER BY source_row_number",
        [session_id],
    ).fetchall()
    assert quarantined == [
        (14, "timestamp_regression"),
        (15, "unparseable_timestamp"),
        (17, "duplicate_row"),
        (18, "malformed_field_count"),
    ]
    ordered = fixture_connection.execute(
        "SELECT source_row_number, elapsed_ms, quality_flags FROM samples WHERE session_id = ? "
        "AND elapsed_ms BETWEEN 9000 AND 12000 ORDER BY elapsed_ms, source_row_number",
        [session_id],
    ).fetchall()
    assert [(row[0], row[1]) for row in ordered] == [
        (10, 9000),
        (12, 10000),
        (11, 11000),
        (13, 12000),
    ]
    assert "reordered" in ordered[1][2]
    session = fixture_connection.execute(
        "SELECT raw_row_count, accepted_row_count, quarantined_row_count, reordered_row_count "
        "FROM sessions WHERE session_id = ?",
        [session_id],
    ).fetchone()
    assert session == (69, 65, 4, 1)


def test_backward_step_threshold_boundary() -> None:
    lines = [
        _sample_line("2030.01.01_00.00.00"),
        _sample_line("2030.01.01_00.00.05"),
        _sample_line("2030.01.01_00.00.02", downlink="2000"),
        _sample_line("2030.01.01_00.00.01", downlink="3000"),
        _sample_line("2030.01.01_00.00.06"),
    ]
    session = normalize_session(_raw_session(lines))
    assert SEGMENT_GAP_THRESHOLD_MS == 3000
    assert [row.reason for row in session.quarantined] == ["timestamp_regression"]
    assert session.quarantined[0].source_row_number == 4
    assert session.reordered_rows == [(3, 3000)]
    assert [sample.source_row_number for sample in session.samples] == [1, 3, 2, 5]
    assert [sample.elapsed_ms for sample in session.samples] == [0, 2000, 5000, 6000]


def test_elapsed_time_is_naive_local_and_whole_seconds(
    fixture_connection: duckdb.DuckDBPyConnection,
) -> None:
    rows = fixture_connection.execute(
        "SELECT raw_timestamp, local_timestamp, elapsed_ms FROM samples"
    ).fetchall()
    assert all(row[2] % 1000 == 0 for row in rows)
    for raw_timestamp, local_timestamp, _ in rows[:20]:
        assert isinstance(local_timestamp, datetime) and local_timestamp.tzinfo is None
        assert local_timestamp == datetime.strptime(raw_timestamp, "%Y.%m.%d_%H.%M.%S")
    origins = fixture_connection.execute(
        "SELECT min(elapsed_ms) FROM samples GROUP BY session_id"
    ).fetchall()
    assert {row[0] for row in origins} == {0}


def test_segment_split_only_above_three_seconds(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    gap_segments = fixture_connection.execute(
        "SELECT segment_id, start_ms, end_ms, sample_count FROM segments WHERE session_id = ? "
        "ORDER BY segment_id",
        [fixture_session_ids[fixture.GAP_MEMBER]],
    ).fetchall()
    assert gap_segments == [(0, 0, 20000, 20), (1, 30000, 50000, 20)]
    missing_segments = fixture_connection.execute(
        "SELECT count(*) FROM segments WHERE session_id = ?",
        [fixture_session_ids[fixture.MISSING_VALUES_MEMBER]],
    ).fetchone()
    assert missing_segments == (1,)


def test_cell_identity_layouts() -> None:
    eutran = str(0x1A2B * 256 + 0x0C)
    utran = str(0x00C1 * 65536 + 0x1F2E)
    assert cell_identity(eutran, "1A2B", "0C") == (eutran, "eutran_cell_identity_layout")
    assert cell_identity(utran, "00C1", "1F2E") == (utran, "utran_cell_identity_layout")
    assert cell_identity("4660", "-", "-") == ("4660", "unknown")
    assert cell_identity("0012345", "1A2B", "0C") == ("0012345", "unknown")
    assert cell_identity("0", "1A2B", "0C") == (None, "unknown")
    assert cell_identity("-", "1A2B", "0C") == (None, "unknown")
    assert cell_identity("", "1A2B", "0C") == (None, "unknown")
    assert cell_identity("12a", "1A2B", "0C") == (None, "unknown")


def test_state_mapping_preserves_instruction_like_text(
    fixture_connection: duckdb.DuckDBPyConnection, fixture_session_ids: dict[str, str]
) -> None:
    assert map_state("D") == "active" and map_state("I") == "idle"
    assert map_state("V") == map_state("VD") == map_state(None) == "unknown"
    states = dict(
        fixture_connection.execute(
            "SELECT raw_state, any_value(download_state) FROM samples WHERE session_id = ? "
            "GROUP BY raw_state",
            [fixture_session_ids[fixture.STREAMING_STATE_TEXT_MEMBER]],
        ).fetchall()
    )
    assert states == {
        "D": "active",
        "I": "idle",
        "V": "unknown",
        "VD": "unknown",
        fixture.INSTRUCTION_TEXT: "unknown",
    }
    modes = fixture_connection.execute(
        "SELECT network_mode FROM samples WHERE session_id = ? AND elapsed_ms = 52000",
        [fixture_session_ids[fixture.STABLE_RADIO_DROP_MEMBER]],
    ).fetchone()
    assert modes == (fixture.NETWORK_MODE_INSTRUCTION_TEXT,)


def test_session_ids_are_opaque(fixture_session_ids: dict[str, str]) -> None:
    for member, session_id in fixture_session_ids.items():
        assert re.fullmatch(r"ses_[0-9a-f]{12}", session_id)
        assert "2030" not in session_id and Path(member).stem not in session_id
    assert len(set(fixture_session_ids.values())) == len(fixture_session_ids)


def test_raw_row_lines_are_kept_for_quarantine() -> None:
    raw = _raw_session([_sample_line("2030.01.01_00.00.00"), "broken,row"])
    assert raw.rows[1] == RawRow(2, "broken,row", ("broken", "row"))
    session = normalize_session(raw)
    assert [(row.source_row_number, row.raw_line, row.reason) for row in session.quarantined] == [
        (2, "broken,row", "malformed_field_count")
    ]
