"""Audit, ingest and detection on the downloaded Raca archive.

Skipped when the archive is absent. The expected counts were recorded by the
raw audit on 2026-10-08 for repository commit 306d92e.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from celltrace.analysis.detector import detect_candidates
from celltrace.config import DetectorSection
from celltrace.data.audit import run_audit
from celltrace.data.snapshot import load_manifest, open_snapshot, run_ingest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE = RAW_DIR / "raca_5g" / "5G-production-dataset.zip"
ARCHIVE_SHA256 = "abc729d696b5e0ba34a9c6ed35a851f2868d2b5071be5efb8a49691c23bb1a9b"
REGRESSION_MEMBER = (
    "5G-production-dataset/Netflix/Driving/Season3-StrangerThings/B_2019.11.27_07.29.47.csv"
)

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.skipif(not ARCHIVE.is_file(), reason="Raca archive not downloaded"),
]


@pytest.fixture(scope="module")
def real_audit(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out_dir = tmp_path_factory.mktemp("real_audit")
    summary = run_audit(RAW_DIR, out_dir)
    return {
        "summary": summary,
        "audit": json.loads((out_dir / "audit.json").read_text()),
        "manifest": json.loads((out_dir / "data_manifest.json").read_text()),
    }


@pytest.fixture(scope="module")
def real_snapshot(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return run_ingest(RAW_DIR, tmp_path_factory.mktemp("real_processed"))


def test_archive_provenance(real_audit: dict[str, Any]) -> None:
    [archive] = [
        item
        for item in real_audit["manifest"]["archives"]
        if item["archive_relative_path"] == "raca_5g/5G-production-dataset.zip"
    ]
    assert archive["sha256"] == ARCHIVE_SHA256
    assert archive["sha256_matches_record"] is True
    assert archive["entry_count"] == 206
    assert archive["total_uncompressed_bytes"] == 23_350_189
    assert archive["csv_member_count"] == 83
    assert archive["repository_commit"] == "306d92eba81cb4f1195d8b8eba625666a02a10ec"


def test_audit_counts(real_audit: dict[str, Any]) -> None:
    audit = real_audit["audit"]
    assert audit["file_count"] == 83
    assert audit["header"]["files_matching"] == 83
    assert audit["encodings"] == {"utf-8": 83}
    assert audit["totals"]["raw_rows"] == 188_711
    assert audit["totals"]["accepted_rows"] == 188_711
    assert audit["totals"]["quarantined_rows"] == 0
    assert audit["interval_counts_ms"]["source_order"] == {
        "-1000": 1,
        "0": 7966,
        "1000": 161_295,
        "2000": 18_851,
        "3000": 510,
        "4000": 4,
        "5000": 1,
    }
    assert audit["totals"]["backward_steps_source_order"] == 1
    assert audit["totals"]["regression_rows"] == 0
    regression_file = next(
        item for item in audit["files"] if item["member_path"] == REGRESSION_MEMBER
    )
    assert regression_file["reordered_rows"][0] == {"source_row_number": 917, "backward_ms": 1000}
    assert audit["totals"]["segments"] == 88
    download = audit["cohort_counts"]["file_download"]
    assert download["driving"]["sessions"] == 16
    assert download["static"]["sessions"] == 5
    assert audit["file_download_sessions"] == 21
    assert audit["usable_download_sessions"]["count"] >= 5


def test_snapshot_checks_and_counts(real_snapshot: Path) -> None:
    manifest = load_manifest(real_snapshot)
    assert all(check["passed"] for check in manifest["sql_checks"])
    assert manifest["tables"]["samples"]["row_count"] == 188_711
    assert manifest["tables"]["sessions"]["row_count"] == 83
    assert manifest["tables"]["segments"]["row_count"] == 88


def test_detector_runs_deterministically(real_snapshot: Path) -> None:
    connection, _ = open_snapshot(real_snapshot)
    try:
        first = detect_candidates(connection, DetectorSection())
        second = detect_candidates(connection, DetectorSection())
    finally:
        connection.close()
    assert [item.model_dump() for item in first.candidates] == [
        item.model_dump() for item in second.candidates
    ]
    assert first.eligibility_counts["sessions_evaluated"] == 21
    for details in first.candidate_details:
        assert details["detection_delay_ms"] >= DetectorSection().event_ms
