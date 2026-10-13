"""Audit outputs on the synthetic fixture: provenance, mapping, aggregates, no raw rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from celltrace.data import fixture
from celltrace.data.audit import run_audit
from celltrace.data.raca import RACA_HEADER, RAW_FIELDS_NOT_INGESTED


@pytest.fixture(scope="module")
def audit_outputs(
    fixture_archive_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    out_dir = tmp_path_factory.mktemp("fixture_audit")
    summary = run_audit(fixture_archive_dir, out_dir)
    return {
        "summary": summary,
        "out_dir": out_dir,
        "data_manifest": json.loads((out_dir / "data_manifest.json").read_text()),
        "schema_mapping": json.loads((out_dir / "schema_mapping.json").read_text()),
        "audit": json.loads((out_dir / "audit.json").read_text()),
    }


def test_summary_counts(audit_outputs: dict[str, Any]) -> None:
    summary = audit_outputs["summary"]
    assert summary["csv_member_count"] == len(fixture.FIXTURE_SESSIONS) == 10
    assert summary["raw_row_count"] == summary["accepted_row_count"] + 4
    assert summary["quarantined_row_count"] == 4
    assert summary["file_download_sessions"] == 9
    # Usable needs a 35 s segment: the gap session (two 20 s segments) and the 10 s session fail.
    assert summary["usable_download_sessions"] == 7
    assert summary["archive_hashes_match_records"] == [True]


def test_audit_aggregates(audit_outputs: dict[str, Any]) -> None:
    audit = audit_outputs["audit"]
    assert audit["header"]["files_matching"] == 10
    assert audit["encodings"] == {"utf-8": 10}
    assert audit["delimiters"] == {",": 10}
    assert audit["quarantine_reasons"] == {
        "duplicate_row": 1,
        "malformed_field_count": 1,
        "timestamp_regression": 1,
        "unparseable_timestamp": 1,
    }
    assert audit["value_rejection_reasons"] == {
        "android_unavailable_sentinel": 2,
        "suspected_fill_value": 3,
    }
    assert audit["totals"]["same_second_rows"] == 2
    assert audit["totals"]["reordered_rows"] == 1
    assert audit["totals"]["regression_rows"] == 1
    assert audit["totals"]["segments"] == 11
    assert audit["outside_documented_range"]["reported_rssi_dbm"]["count"] == 1
    assert audit["suspected_fill_values_by_field"]["RSRP"] == {"-200": 1}
    assert audit["state_distribution"]["mapped_by_application"]["video_netflix"]["unknown"] == 13
    assert audit["cohort_counts"]["file_download"]["driving"]["sessions"] == 5
    assert audit["cohort_counts"]["file_download"]["static"]["sessions"] == 4
    gprs = audit["identity_basis_by_network_mode"]["GPRS"]
    assert gprs == {"unknown": 10}
    order_file = next(
        item
        for item in audit["files"]
        if item["member_path"] == fixture.ORDER_AND_TECHNOLOGY_MEMBER
    )
    assert order_file["reordered_rows"] == [{"source_row_number": 12, "backward_ms": 1000}]
    assert order_file["regression_rows"] == [{"source_row_number": 14, "backward_ms": 10000}]
    assert order_file["raw_interval_counts_ms"]["-1000"] == 1


def test_schema_mapping_covers_every_raw_field(audit_outputs: dict[str, Any]) -> None:
    mapping = audit_outputs["schema_mapping"]
    assert [entry["raw_field"] for entry in mapping["fields"]] == list(RACA_HEADER)
    not_ingested = {
        entry["raw_field"] for entry in mapping["fields"] if entry.get("canonical_field") is None
    }
    assert not_ingested == set(RAW_FIELDS_NOT_INGESTED)
    assert set(mapping["not_ingested"].values()) == {"not ingested, preserved in raw archive"}
    downlink = next(entry for entry in mapping["fields"] if entry["raw_field"] == "DL_bitrate")
    assert downlink["canonical_field"] == "downlink_mbps" and downlink["divisor"] == 1000.0
    assert mapping["segmentation"]["gap_threshold_ms"] == 3000
    assert mapping["state_mapping"]["mapping"] == {"D": "active", "I": "idle"}


def test_data_manifest_provenance(audit_outputs: dict[str, Any]) -> None:
    manifest = audit_outputs["data_manifest"]
    archive = manifest["archives"][0]
    assert archive["source_id"] == fixture.FIXTURE_SOURCE_ID
    assert archive["synthetic"] is True
    assert archive["sha256_matches_record"] is True
    assert archive["csv_member_count"] == 10
    assert archive["skipped_entries"]["macos_resource_metadata"] == 1
    assert all(len(member["sha256"]) == 64 for member in archive["members"])
    assert all(member["session_id"].startswith("ses_") for member in archive["members"])


def test_public_outputs_contain_no_raw_rows(audit_outputs: dict[str, Any]) -> None:
    text = "".join(
        (audit_outputs["out_dir"] / name).read_text()
        for name in ("audit.json", "data_manifest.json", "schema_mapping.json")
    )
    for member, content in fixture.fixture_member_bytes().items():
        lines = content.decode().splitlines()[1:]
        assert not any(line in text for line in lines), member
    assert "2030.01.09 10:00:13" not in text
