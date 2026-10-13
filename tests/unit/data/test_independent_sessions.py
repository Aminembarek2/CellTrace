"""Native decimal time, missing/zero, duplicate and schema admission boundaries."""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from celltrace.data.independent_sessions import (
    RATE_FIELD,
    TIME_FIELD,
    audit_file,
    decimal_observation,
)


def write_rows(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([TIME_FIELD, RATE_FIELD])
        writer.writerows(rows)


def test_decimal_precision_zero_missing_duplicates_and_order_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "native.csv"
    write_rows(
        path, [("1659975385.456287", "0"), ("1659975385.456287", ""), ("1659975385.456288", "12.3")]
    )
    result = audit_file(path)
    assert result["rows"] == 3 and result["rate_missing"] == 1 and result["rate_zero"] == 1
    assert result["duplicate_timestamp_rows"] == 1 and result["backward_steps"] == 0
    assert result["minimum_epoch_seconds"] == "1659975385.456287"
    assert result["maximum_epoch_seconds"] == "1659975385.456288"
    assert decimal_observation("1659975385.456287") == Decimal("1659975385.456287")
    write_rows(path, [("2", "3"), ("1", "4")])
    assert audit_file(path)["backward_steps"] == 1


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "bad"])
def test_invalid_native_numbers_fail_closed(value: str) -> None:
    with pytest.raises(ValueError):
        decimal_observation(value)


def test_missing_schema_and_malformed_csv_cannot_be_admitted(tmp_path: Path) -> None:
    path = tmp_path / "native.csv"
    path.write_text("time,rate\n1,2\n")
    with pytest.raises(ValueError, match="lacks"):
        audit_file(path)
    path.write_text(f"{TIME_FIELD},{RATE_FIELD}\n1,2,extra\n")
    with pytest.raises(ValueError, match="malformed"):
        audit_file(path)
