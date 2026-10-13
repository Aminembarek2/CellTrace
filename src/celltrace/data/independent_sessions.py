"""Readonly admission of pinned PAM native observations, without inventing capabilities."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from celltrace.canonical import content_hash, file_sha256
from celltrace.data.snapshot import open_snapshot

RATE_FIELD = "Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]"
TIME_FIELD = "TIME_STAMP"
FILES = {
    "atnt_d1_dl.csv": "drive_1",
    "atnt_day_1_d2_dl.csv": "drive_2",
    "atnt_day_1_d3_dl.csv": "drive_3",
}
MISSING = frozenset(("", "nan", "NaN"))
ADMISSION_VERSION = "pam_native_observations_1"


def decimal_observation(value: str, missing: bool = False) -> Decimal | None:
    if missing and value in MISSING:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation as invalid:
        raise ValueError("invalid decimal observation") from invalid
    if not number.is_finite() or number < 0:
        raise ValueError("nonfinite or negative observation")
    return number


def audit_file(path: Path) -> dict[str, Any]:
    timestamps: CounterLike = {}
    minimum = maximum = previous = None
    rows = missing = zeros = backward = 0
    fingerprint = hashlib.sha256()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {TIME_FIELD, RATE_FIELD} <= set(reader.fieldnames):
            raise ValueError("native observation source lacks documented timestamp/rate columns")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("duplicate source headers")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("malformed CSV row")
            stamp = decimal_observation(row[TIME_FIELD])
            rate = decimal_observation(row[RATE_FIELD], missing=True)
            assert stamp is not None
            rows += 1
            missing += rate is None
            zeros += rate == 0
            backward += previous is not None and stamp < previous
            minimum = stamp if minimum is None else min(minimum, stamp)
            maximum = stamp if maximum is None else max(maximum, stamp)
            timestamps[stamp] = timestamps.get(stamp, 0) + 1
            previous = stamp
            # No premature millisecond rounding, coordinates or derived event labels.
            fingerprint.update(
                (
                    str(stamp.normalize())
                    + ":"
                    + ("null" if rate is None else str(rate.normalize()))
                    + "\n"
                ).encode()
            )
    if rows == 0 or minimum == maximum:
        raise ValueError("empty or zero-duration native observation session")
    return {
        "rows": rows,
        "rate_valid": rows - missing,
        "rate_missing": missing,
        "rate_zero": zeros,
        "duplicate_timestamp_rows": sum(count - 1 for count in timestamps.values()),
        "backward_steps": backward,
        "minimum_epoch_seconds": str(minimum),
        "maximum_epoch_seconds": str(maximum),
        "observation_fingerprint": fingerprint.hexdigest(),
        "timestamp_precision": "exact decimal epoch seconds; not rounded to milliseconds",
        "activity": "unknown",
        "global_serving_cell_identity": "unavailable; PCI is not a global identity",
        "original_header": reader.fieldnames,
    }


CounterLike = dict[Decimal, int]


def admit(source_record: Path, prior_snapshot: Path, prior_raw: Path, out: Path) -> dict[str, Any]:
    record = json.loads(source_record.read_text())
    if record.get("source_id") != "pam2025_nuwins" or record.get("license") != "CC0-1.0":
        raise ValueError("unsupported source or undocumented local-use rights")
    if not re.fullmatch("[a-f0-9]{40}", record.get("commit", "")):
        raise ValueError("source requires a pinned revision")
    license_path = source_record.parent / "LICENSE"
    if "CC0" not in license_path.read_text():
        raise ValueError("CC0 license evidence missing")
    entries = record["files"]
    if {Path(entry["file"]).name for entry in entries} != set(FILES) or len(entries) != 3:
        raise ValueError("source must contain the three declared campaign exports exactly once")
    archives = sorted(prior_raw.rglob("*.zip"))
    if not archives:
        raise ValueError("prior archive is required to verify member non-overlap")
    member_hashes = set()
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                digest = hashlib.sha256()
                with archive.open(member) as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                member_hashes.add(digest.hexdigest())
    connection, manifest = open_snapshot(prior_snapshot)
    try:
        old_times = connection.execute(
            "SELECT min(raw_timestamp), max(raw_timestamp) FROM samples"
        ).fetchone()
    finally:
        connection.close()
    assert old_times is not None
    # The prior clock has unknown timezone. Non-overlapping calendar years are sufficient
    # here; do not invent a UTC conversion for the Irish local timestamps.
    old_years = {int(str(value)[:4]) for value in old_times}
    from datetime import UTC, datetime

    sessions = []
    inputs = {
        str(source_record.resolve()): file_sha256(source_record),
        str(license_path.resolve()): file_sha256(license_path),
    }
    for entry in entries:
        path = Path(entry["file"])
        if path.resolve().parent != source_record.resolve().parent:
            raise ValueError("source file must stay in its recorded directory")
        if file_sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("acquired source changed")
        if f"/{record['commit']}/" not in entry["url"] or not entry["url"].startswith(
            "https://media.githubusercontent.com/media/NUWiNS/pam2025-multi-carrier-dataset/"
        ):
            raise ValueError("source URL is not pinned to the author repository")
        if entry["sha256"] in member_hashes:
            raise ValueError("new session overlaps a prior archive member")
        audit = audit_file(path)
        years = {
            datetime.fromtimestamp(int(Decimal(audit[key])), UTC).year
            for key in ("minimum_epoch_seconds", "maximum_epoch_seconds")
        }
        if min(years) <= max(old_years) and min(old_years) <= max(years):
            raise ValueError("acquisition calendar ranges overlap; deeper overlap audit required")
        inputs[str(path.resolve())] = entry["sha256"]
        sessions.append(
            {
                "session_id": "pam_" + content_hash([record["source_id"], entry["sha256"]])[:16],
                "acquisition_group": FILES[path.name],
                "source_sha256": entry["sha256"],
                "source_file": str(path.resolve()),
                "prior_member_hash_overlap": False,
                "prior_acquisition_year_overlap": False,
                **audit,
            }
        )
    result = {
        "method": ADMISSION_VERSION,
        "source_id": record["source_id"],
        "commit": record["commit"],
        "license": record["license"],
        "inputs": inputs,
        "prior_snapshot_logical_hash": manifest["logical_hash"],
        "prior_archives": {str(path.resolve()): file_sha256(path) for path in archives},
        "prior_calendar_range": list(old_times),
        "sessions": sessions,
        "independence_unit": (
            "whole acquisition campaign; operators/days within a campaign stay grouped"
        ),
        "drive_1_reuse": "PAM Drive 1 reuses IMC 2023; never split these releases as independent",
        "admission": "acquired_nonoverlapping_campaigns_for_future_native_observation_profile",
        "existing_active_download_benchmark": "ineligible: activity state unavailable",
        "dispatch_gate": (
            "blocked until source-bound normalization, windows and factual grading are frozen"
        ),
        "model_outcomes_observed": 0,
        "pretraining_exposure": "unknown",
        "limitations": (
            "Non-overlap against this project's archive does not prove statistical independence. "
            "No rate imputation, activity inference, derived handover labels or new held-out "
            "performance claim."
        ),
    }
    result["admission_sha256"] = content_hash(result)
    out.mkdir(parents=True, exist_ok=False)
    (out / "admission.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def register(subparsers: Any) -> None:
    import argparse

    def command(arguments: argparse.Namespace) -> int:
        result = admit(
            arguments.source_record.resolve(),
            arguments.prior_snapshot.resolve(),
            arguments.prior_raw.resolve(),
            arguments.out.resolve(),
        )
        print(
            json.dumps(
                {
                    "admission": str(arguments.out / "admission.json"),
                    "campaigns": len(result["sessions"]),
                    "status": result["admission"],
                    "heldout_dispatch": result["dispatch_gate"],
                },
                indent=2,
            )
        )
        return 0

    parser = subparsers.add_parser(
        "admit-sessions", help="audit independent PAM campaigns without model calls"
    )
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--prior-snapshot", type=Path, required=True)
    parser.add_argument("--prior-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=command)
