"""The 2018 LTE adapter: operator split, field mapping, proxy cell identity, snapshot build."""

from __future__ import annotations

import zipfile
from pathlib import Path

import duckdb
import pytest

from celltrace.data import raca, raca_lte
from celltrace.data.snapshot import load_manifest, run_ingest


def row(second: int, cell: str = "2", operator: str = "A", mode: str = "LTE") -> str:
    values = {
        "Timestamp": f"2018.01.16_10.00.{second:02d}",
        "Longitude": "-8.47",
        "Latitude": "51.89",
        "Speed": "30",
        "Operatorname": operator,
        "CellID": cell,
        "NetworkMode": mode,
        "RSRP": "-90",
        "RSRQ": "-10",
        "SNR": "5",
        "CQI": "9",
        "RSSI": "-60",
        "DL_bitrate": "8000",
        "UL_bitrate": "50",
        "State": "D",
        "NRxRSRP": "-",
        "NRxRSRQ": "-",
        "ServingCell_Lon": "-8.46",
        "ServingCell_Lat": "51.90",
        "ServingCell_Distance": "300",
    }
    return ",".join(values[name] for name in raca_lte.LTE_HEADER)


def archive(directory: Path) -> Path:
    path = directory / "LTE_Dataset.zip"
    header = ",".join(raca_lte.LTE_HEADER)
    car_a = [header] + [row(second) for second in range(30)]
    car_a[5] = row(4, operator="27202", cell="9")
    car_a[6] = row(5, cell="0")
    car_b = [header] + [row(second, cell="3", operator="B") for second in range(30)]
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("Dataset/car/A_2018.01.16_10.00.00.csv", "\n".join(car_a) + "\n")
        handle.writestr("Dataset/car/B_2018.01.16_10.00.00.csv", "\n".join(car_b) + "\n")
        handle.writestr("Dataset/readme.txt", "notes")
    return path


def test_inventory_keeps_one_operator_and_names_why_others_were_skipped(tmp_path: Path) -> None:
    inventory = raca_lte.inspect_archive(archive(tmp_path), "A")
    assert [member.member_path for member in inventory.members] == [
        "Dataset/car/A_2018.01.16_10.00.00.csv"
    ]
    assert inventory.members[0].mobility_cohort == "car"
    reasons = {entry.entry_name: entry.reason for entry in inventory.skipped}
    assert reasons["Dataset/car/B_2018.01.16_10.00.00.csv"] == "other_operator"
    assert reasons["Dataset/readme.txt"] == "outside_dataset_layout"


def test_rows_are_placed_in_the_5g_layout_with_absent_fields_missing() -> None:
    values = tuple(row(0).split(","))
    mapped = raca_lte.to_archive_layout(values)
    index = raca.FIELD_INDEX
    assert len(mapped) == len(raca.RACA_HEADER)
    assert mapped[index["CellID"]] == "2" and mapped[index["RAWCELLID"]] == "2"
    assert mapped[index["DL_bitrate"]] == "8000"
    assert mapped[index["PINGAVG"]] == raca.DASH_MISSING
    assert mapped[index["NODEHEX"]] == raca.DASH_MISSING
    assert raca_lte.to_archive_layout(values[:5]) == values[:5]


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (("2", "LTE", "A", "-8.46", "51.90", "A"), "LTE:-8.46,51.90:2"),
        (("2", "HSPA+", "A", "-8.46", "51.90", "A"), "3G:-8.46,51.90:2"),
        (("2", "LTE", "27202", "-8.46", "51.90", "A"), None),
        (("0", "LTE", "A", "-8.46", "51.90", "A"), None),
        (("2", "LTE", "A", "-", "51.90", "A"), None),
        (("2", "NR", "A", "-8.46", "51.90", "A"), None),
        ((None, "LTE", "A", "-8.46", "51.90", "A"), None),
    ],
)
def test_proxy_cell_identity(arguments: tuple[str | None, ...], expected: str | None) -> None:
    assert raca_lte.proxy_cell(*arguments) == expected  # type: ignore[arg-type]


def test_snapshot_builds_with_proxy_cells_and_unknown_identity_basis(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    archive(raw)
    snapshot = run_ingest(
        raw,
        tmp_path / "snapshots",
        sources=[raca_lte.source_archive(raw, "A")],
        session_iterator=raca_lte.iter_sessions,
    )
    manifest = load_manifest(snapshot)
    assert manifest["source_ids"] == ["raca_irish_lte_2018_operator_a"]
    assert all(check["passed"] for check in manifest["sql_checks"])
    connection = duckdb.connect(str(snapshot / "measurements.duckdb"), read_only=True)
    cells = connection.execute(
        "SELECT serving_cell_id, cell_identity_basis, count(*) FROM samples "
        "GROUP BY 1, 2 ORDER BY 1 NULLS LAST"
    ).fetchall()
    cohorts = connection.execute(
        "SELECT application_cohort, mobility_cohort FROM sessions"
    ).fetchall()
    connection.close()
    assert cohorts == [("file_download", "car")]
    # Two rows lose their identity: another network's cell and a zero cell number.
    assert cells == [("LTE:-8.46,51.90:2", "unknown", 28), (None, "unknown", 2)]
