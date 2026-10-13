"""Raw archive audit: provenance manifest, schema mapping and aggregate statistics.

The audit runs the same parser and normalization as ingestion, so every count
it reports describes exactly what the snapshot will contain. Public outputs
hold aggregates and row locations only, never raw rows.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from celltrace.config import DetectorSection
from celltrace.contracts.common import MetricName
from celltrace.data.dictionary import (
    APPLICATION_FOLDERS,
    COHORT_MAPPING_NOTE,
    DASH_SENTINEL_SOURCE,
    DICTIONARY_VERSION,
    METRIC_DICTIONARY,
    MOBILITY_FOLDERS,
    STATE_MAPPING,
    STATE_MAPPING_NOTE,
    TIMESTAMP_BASIS,
    TIMESTAMP_FORMAT,
    dictionary_hash,
)
from celltrace.data.raca import (
    BIN_MS,
    MAXIMUM_ARCHIVE_UNCOMPRESSED_BYTES,
    NOT_INGESTED_NOTE,
    RACA_HEADER,
    RAW_FIELDS_NOT_INGESTED,
    NormalizedSession,
    SourceArchive,
    find_source_archives,
    iter_sessions,
    make_session_id,
)
from celltrace.data.schema import (
    MAPPING_VERSION,
    METRIC_COLUMNS,
    NORMALIZATION_VERSION,
    SEGMENT_GAP_THRESHOLD_MS,
)

DATA_MANIFEST_NAME = "data_manifest.json"
SCHEMA_MAPPING_NAME = "schema_mapping.json"
AUDIT_NAME = "audit.json"
USABLE_MINIMUM_VALID_FRACTION = 0.8
MAXIMUM_LISTED_LOCATIONS = 20
MAXIMUM_LISTED_CODES = 20
MAXIMUM_CODE_LENGTH = 80

VALUE_REJECTION_REASONS: dict[str, str] = {
    "android_unavailable_sentinel": "2147483647 (Android CellInfo.UNAVAILABLE) becomes null",
    "suspected_fill_value": "a value listed as a suspected fill value in the dictionary "
    "becomes null",
    "empty_value": "an empty field becomes null",
    "unparseable_number": "a field that is not a plain decimal number becomes null",
    "nonfinite_number": "a nan or infinite literal becomes null",
    "negative_rate": "a negative bitrate is physically impossible and becomes null",
}
QUARANTINE_REASONS: dict[str, str] = {
    "malformed_field_count": "the line does not have exactly 26 fields",
    "unparseable_timestamp": f"the timestamp does not match {TIMESTAMP_FORMAT}",
    "duplicate_row": "the full row is identical to an earlier accepted row of the session",
    "timestamp_regression": f"the timestamp steps back more than {SEGMENT_GAP_THRESHOLD_MS} ms "
    "behind the latest accepted timestamp",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _code_key(code: str | None) -> str:
    if code is None:
        return "<null>"
    if len(code) > MAXIMUM_CODE_LENGTH:
        return code[:MAXIMUM_CODE_LENGTH] + "<truncated>"
    return code


def _capped(counter: Counter[str]) -> dict[str, int]:
    return dict(counter.most_common(MAXIMUM_LISTED_CODES))


def _extend_extremes(extremes: dict[str, list[float]], metric: str, value: float) -> None:
    current = extremes.setdefault(metric, [value, value])
    current[0] = min(current[0], value)
    current[1] = max(current[1], value)


def _interval_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


class _AuditAccumulator:
    """Aggregates per session statistics without keeping raw rows."""

    def __init__(self) -> None:
        detector = DetectorSection()
        self.minimum_window_ms = detector.reference_ms + detector.event_ms
        self.files: list[dict[str, Any]] = []
        self.encodings: Counter[str] = Counter()
        self.delimiters: Counter[str] = Counter()
        self.line_terminators: Counter[str] = Counter()
        self.raw_intervals: Counter[int] = Counter()
        self.sorted_intervals: Counter[int] = Counter()
        self.quarantine_reasons: Counter[str] = Counter()
        self.rejection_reasons: Counter[str] = Counter()
        self.raw_dash_counts: Counter[str] = Counter()
        self.raw_empty_counts: Counter[str] = Counter()
        self.null_counts: Counter[str] = Counter()
        self.valid_counts: Counter[str] = Counter()
        self.rejections_by_metric: dict[str, Counter[str]] = defaultdict(Counter)
        self.fill_values: dict[str, Counter[str]] = defaultdict(Counter)
        self.outside_range: Counter[str] = Counter()
        self.outside_extremes: dict[str, list[float]] = {}
        self.observed_extremes: dict[str, list[float]] = {}
        self.raw_states: dict[str, Counter[str]] = defaultdict(Counter)
        self.mapped_states: dict[str, Counter[str]] = defaultdict(Counter)
        self.identity_by_mode: dict[str, Counter[str]] = defaultdict(Counter)
        self.valid_by_mode: dict[str, Counter[str]] = defaultdict(Counter)
        self.rows_by_mode: Counter[str] = Counter()
        self.cohorts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        self.cell_local_to_raw: dict[str, set[str]] = defaultdict(set)
        self.raw_cell_ids: set[str] = set()
        self.static_speed_rows = 0
        self.static_moving_rows = 0
        self.static_maximum_speed: float | None = None
        self.usable_sessions: list[str] = []
        self.download_sessions: list[str] = []
        self.totals: Counter[str] = Counter()

    def add(self, session: NormalizedSession) -> None:
        raw = session.raw
        application = raw.application_cohort
        mobility = raw.mobility_cohort
        self.encodings[raw.encoding] += 1
        self.delimiters[raw.delimiter] += 1
        self.line_terminators[raw.line_terminator] += 1
        self.raw_intervals.update(session.raw_interval_counts_ms)
        for row in raw.rows:
            if len(row.values) != len(RACA_HEADER):
                continue
            for field_name, raw_value in zip(RACA_HEADER, row.values, strict=True):
                if raw_value == "-":
                    self.raw_dash_counts[field_name] += 1
                elif raw_value == "":
                    self.raw_empty_counts[field_name] += 1

        quarantine = Counter(row.reason for row in session.quarantined)
        self.quarantine_reasons.update(quarantine)
        for rejection in session.rejections:
            self.rejection_reasons[rejection.reason] += 1
            self.rejections_by_metric[rejection.raw_field][rejection.reason] += 1
            if rejection.reason == "suspected_fill_value":
                self.fill_values[rejection.raw_field][rejection.raw_value] += 1

        raw_states: Counter[str] = Counter()
        mapped_states: Counter[str] = Counter()
        sorted_intervals: Counter[int] = Counter()
        valid_downlink = 0
        previous_elapsed: int | None = None
        for sample in session.samples:
            raw_states[_code_key(sample.raw_state)] += 1
            mapped_states[sample.download_state] += 1
            if previous_elapsed is not None:
                sorted_intervals[sample.elapsed_ms - previous_elapsed] += 1
            previous_elapsed = sample.elapsed_ms
            mode = _code_key(sample.network_mode)
            self.rows_by_mode[mode] += 1
            identity = sample.cell_identity_basis if sample.serving_cell_id else "null_identity"
            self.identity_by_mode[mode][identity] += 1
            if sample.raw_cell_local_id is not None and sample.serving_cell_id is not None:
                self.cell_local_to_raw[sample.raw_cell_local_id].add(sample.serving_cell_id)
            if sample.serving_cell_id is not None:
                self.raw_cell_ids.add(sample.serving_cell_id)
            for metric, metric_value in zip(METRIC_COLUMNS, sample.metric_values, strict=True):
                if metric_value is None:
                    self.null_counts[metric.value] += 1
                    continue
                self.valid_counts[metric.value] += 1
                self.valid_by_mode[mode][metric.value] += 1
                _extend_extremes(self.observed_extremes, metric.value, metric_value)
            for flag in sample.quality_flags:
                if flag.startswith("outside_documented_range:"):
                    metric_name = flag.split(":", 1)[1]
                    self.outside_range[metric_name] += 1
                    flagged = sample.metric_values[METRIC_COLUMNS.index(MetricName(metric_name))]
                    if flagged is not None:
                        _extend_extremes(self.outside_extremes, metric_name, flagged)
            downlink = sample.metric_values[METRIC_COLUMNS.index(MetricName.downlink_mbps)]
            if downlink is not None:
                valid_downlink += 1
            speed = sample.metric_values[METRIC_COLUMNS.index(MetricName.speed_kph)]
            if mobility == "static" and speed is not None:
                self.static_speed_rows += 1
                if speed > 0:
                    self.static_moving_rows += 1
                if self.static_maximum_speed is None or speed > self.static_maximum_speed:
                    self.static_maximum_speed = speed

        self.sorted_intervals.update(sorted_intervals)
        self.raw_states[application].update(raw_states)
        self.mapped_states[application].update(mapped_states)
        cohort = self.cohorts[application][mobility]
        cohort["sessions"] += 1
        cohort["raw_rows"] += len(raw.rows)
        cohort["accepted_rows"] += len(session.samples)
        cohort["duration_ms"] += session.duration_ms

        accepted = len(session.samples)
        known_state = mapped_states["active"] + mapped_states["idle"]
        longest_segment = max(
            (segment.end_ms - segment.start_ms for segment in session.segments), default=0
        )
        if application == "file_download":
            self.download_sessions.append(raw.session_id)
            if (
                accepted > 0
                and valid_downlink / accepted >= USABLE_MINIMUM_VALID_FRACTION
                and known_state / accepted >= USABLE_MINIMUM_VALID_FRACTION
                and longest_segment >= self.minimum_window_ms
            ):
                self.usable_sessions.append(raw.session_id)

        self.totals["raw_rows"] += len(raw.rows)
        self.totals["accepted_rows"] += accepted
        self.totals["quarantined_rows"] += len(session.quarantined)
        self.totals["value_rejections"] += len(session.rejections)
        self.totals["segments"] += len(session.segments)
        self.totals["sessions_with_several_segments"] += int(len(session.segments) > 1)
        self.totals["same_second_rows"] += session.same_second_row_count
        self.totals["zero_intervals_source_order"] += session.raw_interval_counts_ms.get(0, 0)
        self.totals["reordered_rows"] += session.reordered_row_count
        self.totals["regression_rows"] += len(session.regression_rows)
        self.totals["backward_steps_source_order"] += sum(
            count for interval, count in session.raw_interval_counts_ms.items() if interval < 0
        )

        self.files.append(
            {
                "member_path": raw.member_path,
                "session_id": raw.session_id,
                "application_cohort": application,
                "mobility_cohort": mobility,
                "encoding": raw.encoding,
                "delimiter": raw.delimiter,
                "line_terminator": raw.line_terminator,
                "header_matches_expected": raw.header == RACA_HEADER,
                "raw_row_count": len(raw.rows),
                "accepted_row_count": accepted,
                "quarantined_row_count": len(session.quarantined),
                "quarantine_reasons": dict(sorted(quarantine.items())),
                "value_rejection_count": len(session.rejections),
                "duration_ms": session.duration_ms,
                "segment_count": len(session.segments),
                "raw_interval_counts_ms": _interval_dict(session.raw_interval_counts_ms),
                "zero_interval_count": session.raw_interval_counts_ms.get(0, 0),
                "same_second_row_count": session.same_second_row_count,
                "reordered_row_count": session.reordered_row_count,
                "reordered_rows": [
                    {"source_row_number": row, "backward_ms": step}
                    for row, step in session.reordered_rows[:MAXIMUM_LISTED_LOCATIONS]
                ],
                "regression_row_count": len(session.regression_rows),
                "regression_rows": [
                    {"source_row_number": row, "backward_ms": step}
                    for row, step in session.regression_rows[:MAXIMUM_LISTED_LOCATIONS]
                ],
                "raw_state_counts": _capped(raw_states),
                "download_state_counts": dict(sorted(mapped_states.items())),
            }
        )

    def result(self) -> dict[str, Any]:
        cell_locals_with_many = sum(
            1 for identities in self.cell_local_to_raw.values() if len(identities) > 1
        )
        cohort_counts = {
            application: {
                mobility: dict(sorted(counter.items())) for mobility, counter in sorted(by.items())
            }
            for application, by in sorted(self.cohorts.items())
        }
        missing_by_metric = {
            metric.value: {
                "raw_field": METRIC_DICTIONARY[metric].raw_field,
                "null_count": self.null_counts[metric.value],
                "valid_count": self.valid_counts[metric.value],
                "raw_dash_count": self.raw_dash_counts[METRIC_DICTIONARY[metric].raw_field],
                "nulled_by_reason": dict(
                    sorted(self.rejections_by_metric[METRIC_DICTIONARY[metric].raw_field].items())
                ),
            }
            for metric in METRIC_COLUMNS
        }
        valid_fraction_by_mode = {
            mode: {
                metric.value: round(self.valid_by_mode[mode][metric.value] / rows, 6)
                for metric in METRIC_COLUMNS
            }
            for mode, rows in sorted(self.rows_by_mode.items())
        }
        return {
            "generated_by": "celltrace.data.audit",
            "mapping_version": MAPPING_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "dictionary_version": DICTIONARY_VERSION,
            "dictionary_hash": dictionary_hash(),
            "segment_gap_threshold_ms": SEGMENT_GAP_THRESHOLD_MS,
            "file_count": len(self.files),
            "totals": dict(sorted(self.totals.items())),
            "header": {
                "expected": list(RACA_HEADER),
                "files_matching": sum(1 for item in self.files if item["header_matches_expected"]),
                "rule": "an unexpected header raises an error, columns are never remapped by "
                "position",
            },
            "encodings": dict(self.encodings),
            "delimiters": dict(self.delimiters),
            "line_terminators": dict(self.line_terminators),
            "interval_counts_ms": {
                "source_order": _interval_dict(self.raw_intervals),
                "sorted_accepted": _interval_dict(self.sorted_intervals),
                "note": "Intervals between consecutive parseable rows of a file. Timestamps "
                "have whole second resolution, so a 0 ms interval means two distinct rows "
                "were truncated to the same second.",
            },
            "quarantine_reasons": dict(sorted(self.quarantine_reasons.items())),
            "value_rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "raw_dash_counts_by_field": {name: self.raw_dash_counts[name] for name in RACA_HEADER},
            "raw_empty_counts_by_field": {
                name: self.raw_empty_counts[name] for name in RACA_HEADER
            },
            "missing_by_metric": missing_by_metric,
            "suspected_fill_values_by_field": {
                field_name: dict(sorted(counter.items()))
                for field_name, counter in sorted(self.fill_values.items())
            },
            "outside_documented_range": {
                metric: {
                    "count": count,
                    "documented_range": list(
                        METRIC_DICTIONARY[MetricName(metric)].documented_range or ()
                    ),
                    "observed_minimum": self.outside_extremes.get(metric, [None, None])[0],
                    "observed_maximum": self.outside_extremes.get(metric, [None, None])[1],
                    "handling": "kept and flagged outside_documented_range",
                }
                for metric, count in sorted(self.outside_range.items())
            },
            "observed_metric_extremes": {
                metric: {"minimum": extremes[0], "maximum": extremes[1]}
                for metric, extremes in sorted(self.observed_extremes.items())
            },
            "valid_fraction_by_network_mode": valid_fraction_by_mode,
            "state_distribution": {
                "raw_codes_by_application": {
                    application: _capped(counter)
                    for application, counter in sorted(self.raw_states.items())
                },
                "mapped_by_application": {
                    application: dict(sorted(counter.items()))
                    for application, counter in sorted(self.mapped_states.items())
                },
            },
            "identity_basis_by_network_mode": {
                mode: dict(sorted(counter.items()))
                for mode, counter in sorted(self.identity_by_mode.items())
            },
            "cell_identity_uniqueness": {
                "distinct_cell_local_ids": len(self.cell_local_to_raw),
                "distinct_serving_cell_ids": len(self.raw_cell_ids),
                "cell_local_ids_mapping_to_several_serving_ids": cell_locals_with_many,
                "note": "CellID alone does not identify a cell. serving_cell_id uses RAWCELLID.",
            },
            "speed_in_static_cohort": {
                "rows_with_speed": self.static_speed_rows,
                "rows_with_nonzero_speed": self.static_moving_rows,
                "maximum_speed": self.static_maximum_speed,
            },
            "cohort_counts": cohort_counts,
            "file_download_sessions": len(self.download_sessions),
            "usable_download_sessions": {
                "count": len(self.usable_sessions),
                "session_ids": sorted(self.usable_sessions),
                "definition": (
                    f"file download session with at least {USABLE_MINIMUM_VALID_FRACTION:.0%} "
                    "of accepted samples carrying a downlink value, at least "
                    f"{USABLE_MINIMUM_VALID_FRACTION:.0%} with a known D or I state, and one "
                    f"segment spanning at least {self.minimum_window_ms} ms (reference plus "
                    "event window of the default detector)"
                ),
            },
            "files": sorted(self.files, key=lambda item: item["member_path"]),
        }


def build_data_manifest(sources: list[SourceArchive]) -> dict[str, Any]:
    archives: list[dict[str, Any]] = []
    for source in sources:
        record = source.source_record or {}
        archive_name = source.archive_path.name
        recorded_files = record.get("files") or {}
        recorded = recorded_files.get(archive_name) or {}
        companions: list[dict[str, Any]] = []
        for name, details in sorted(recorded_files.items()):
            if name == archive_name:
                continue
            companion_path = source.archive_path.parent / name
            present = companion_path.is_file()
            actual = _sha256_file(companion_path) if present else None
            companions.append(
                {
                    "name": name,
                    "present": present,
                    "bytes": companion_path.stat().st_size if present else None,
                    "sha256": actual,
                    "recorded_sha256": details.get("sha256"),
                    "sha256_matches_record": actual == details.get("sha256"),
                }
            )
        inventory = source.inventory
        archives.append(
            {
                "archive_relative_path": source.relative_path,
                "source_id": source.source_id,
                "source_record_present": source.source_record is not None,
                "synthetic": bool(record.get("synthetic", False)),
                "repository": record.get("repository"),
                "repository_commit": record.get("repository_commit"),
                "commit_date": record.get("commit_date"),
                "download_url": (record.get("download_urls") or {}).get(archive_name),
                "access_date": record.get("access_date"),
                "license_evidence": record.get("license_evidence"),
                "citation_request": record.get("citation_request"),
                "bytes": inventory.archive_bytes,
                "sha256": inventory.archive_sha256,
                "recorded_sha256": recorded.get("sha256"),
                "sha256_matches_record": (
                    inventory.archive_sha256 == recorded.get("sha256") if recorded else None
                ),
                "companion_files": companions,
                "entry_count": inventory.entry_count,
                "total_uncompressed_bytes": inventory.total_uncompressed_bytes,
                "maximum_uncompressed_bytes": MAXIMUM_ARCHIVE_UNCOMPRESSED_BYTES,
                "member_name_checks": "no absolute paths, drive letters, backslashes or parent "
                "references, no encrypted entries, nothing extracted or executed",
                "skipped_entries": dict(
                    sorted(Counter(entry.reason for entry in inventory.skipped).items())
                ),
                "csv_member_count": len(inventory.members),
                "members": [
                    {
                        **{
                            key: value
                            for key, value in asdict(member).items()
                            if key != "archive_path"
                        },
                        "session_id": make_session_id(source.source_id, member.member_path),
                    }
                    for member in inventory.members
                ],
            }
        )
    return {
        "generated_by": "celltrace.data.audit",
        "input_relative_root": ".",
        "archive_count": len(archives),
        "archives": archives,
        "redistribution": "Raw rows are not copied into public outputs. Only hashes, sizes, "
        "member names and aggregate statistics are published.",
    }


def build_schema_mapping() -> dict[str, Any]:
    metric_by_field = {
        definition.raw_field: definition for definition in METRIC_DICTIONARY.values()
    }
    identity_fields = {
        "RAWCELLID": "serving_cell_id (string, null for '-', empty, 0 or non digits)",
        "CellID": "raw_cell_local_id (verbatim string)",
        "NODEHEX": "raw_node_hex (verbatim string)",
        "CELLHEX": "raw_cell_hex (verbatim string)",
        "LACHEX": "raw_area_hex (verbatim string)",
    }
    fields: list[dict[str, Any]] = []
    for name in RACA_HEADER:
        entry: dict[str, Any] = {"raw_field": name}
        if name in metric_by_field:
            definition = metric_by_field[name]
            entry.update(
                {
                    "canonical_field": definition.metric,
                    "raw_unit": definition.raw_unit,
                    "canonical_unit": definition.canonical_unit,
                    "conversion": definition.conversion,
                    "divisor": definition.divisor,
                    "missing_sentinels": list(definition.missing_sentinels),
                    "suspected_fill_values": list(definition.suspected_fill_values),
                    "documented_range": (
                        list(definition.documented_range) if definition.documented_range else None
                    ),
                    "out_of_range_handling": "kept and flagged outside_documented_range",
                    "audit_status": definition.audit_status,
                    "definition_source": definition.definition_source,
                    "limitations": definition.limitations,
                }
            )
        elif name == "Timestamp":
            entry.update(
                {
                    "canonical_field": "raw_timestamp, local_timestamp, elapsed_ms",
                    "conversion": f"parse {TIMESTAMP_FORMAT} as naive local time, elapsed_ms is "
                    "milliseconds since the earliest accepted timestamp of the session",
                    "audit_status": "observed_format_timezone_unknown",
                }
            )
        elif name == "State":
            entry.update(
                {
                    "canonical_field": "raw_state, download_state",
                    "conversion": "D to active, I to idle, any other code to unknown",
                    "audit_status": "inferred_unverified",
                }
            )
        elif name == "NetworkMode":
            entry.update(
                {
                    "canonical_field": "network_mode",
                    "conversion": "verbatim string, null for '-' or empty",
                    "audit_status": "observed_values",
                }
            )
        elif name in identity_fields:
            entry.update(
                {
                    "canonical_field": identity_fields[name],
                    "conversion": "verbatim string",
                    "audit_status": "layout_checked_scope_unverified",
                }
            )
        elif name in RAW_FIELDS_NOT_INGESTED:
            entry.update({"canonical_field": None, "status": NOT_INGESTED_NOTE})
        fields.append(entry)
    return {
        "generated_by": "celltrace.data.audit",
        "mapping_version": MAPPING_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "dictionary_version": DICTIONARY_VERSION,
        "dictionary_hash": dictionary_hash(),
        "raw_header": list(RACA_HEADER),
        "fields": fields,
        "not_ingested": {name: NOT_INGESTED_NOTE for name in RAW_FIELDS_NOT_INGESTED},
        "missing_values": {
            "dash": "'-' becomes null and is counted, no rejection row",
            "dash_source": DASH_SENTINEL_SOURCE,
            "rejection_reasons": VALUE_REJECTION_REASONS,
            "zero": "a valid numeric zero is kept and is never treated as missing",
        },
        "quarantine_reasons": QUARANTINE_REASONS,
        "state_mapping": {
            "mapping": STATE_MAPPING,
            "unknown_codes": "V, VD and any other code map to unknown, raw_state keeps the "
            "verbatim text",
            "note": STATE_MAPPING_NOTE,
        },
        "cohort_mapping": {
            "application_folders": APPLICATION_FOLDERS,
            "mobility_folders": MOBILITY_FOLDERS,
            "unknown_folders": "map to unknown",
            "note": COHORT_MAPPING_NOTE,
        },
        "time": {
            "timestamp_format": TIMESTAMP_FORMAT,
            "timestamp_basis": TIMESTAMP_BASIS,
            "elapsed_origin": "earliest accepted timestamp of the session",
            "same_second": "rows sharing one second are distinct samples, all are kept in "
            "source row order and flagged same_second",
            "reorder": "rows earlier than the latest accepted timestamp by at most "
            f"{SEGMENT_GAP_THRESHOLD_MS} ms are kept, flagged reordered and placed by a stable "
            "sort on (timestamp, source_row_number)",
            "regression": f"rows more than {SEGMENT_GAP_THRESHOLD_MS} ms behind are quarantined "
            "as timestamp_regression",
        },
        "segmentation": {
            "gap_threshold_ms": SEGMENT_GAP_THRESHOLD_MS,
            "rule": "an interval strictly greater than the threshold starts a new segment_id, "
            "numbered from 0 within the session",
            "segment_span": "[first sample bin start, last sample bin end), end exclusive",
        },
        "bins": {
            "width_ms": BIN_MS,
            "rule": "one row per bin index inside each segment span, empty bins included",
            "statistic": "arithmetic mean of non null sample values, dBm and dB values are "
            "averaged as logged numbers, not as linear power",
            "available_at_ms": "bin_end_ms",
            "is_complete": "false only for the final bin of each session",
        },
        "cell_identity": {
            "serving_cell_id": "RAWCELLID verbatim when it is a positive decimal integer",
            "eutran_cell_identity_layout": "RAWCELLID equals NODEHEX times 256 plus CELLHEX",
            "utran_cell_identity_layout": "RAWCELLID equals NODEHEX times 65536 plus CELLHEX",
            "unknown": "no layout matches, hex fields do not parse, or NODEHEX is zero",
            "scope": "uniqueness scope of the identifier is undocumented",
        },
    }


def run_audit(input_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Audit every archive under input_dir and write the three public audit files."""
    sources = find_source_archives(input_dir)
    accumulator = _AuditAccumulator()
    for source in sources:
        for session in iter_sessions(source):
            accumulator.add(session)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = build_data_manifest(sources)
    schema_mapping = build_schema_mapping()
    audit = accumulator.result()
    audit["archives"] = [
        {
            "archive_relative_path": source.relative_path,
            "sha256": source.inventory.archive_sha256,
            "source_id": source.source_id,
        }
        for source in sources
    ]
    paths = {
        "data_manifest": out_dir / DATA_MANIFEST_NAME,
        "schema_mapping": out_dir / SCHEMA_MAPPING_NAME,
        "audit": out_dir / AUDIT_NAME,
    }
    _write_json(paths["data_manifest"], data_manifest)
    _write_json(paths["schema_mapping"], schema_mapping)
    _write_json(paths["audit"], audit)
    return {
        "outputs": {name: str(path) for name, path in paths.items()},
        "archive_count": len(sources),
        "csv_member_count": audit["file_count"],
        "raw_row_count": audit["totals"].get("raw_rows", 0),
        "accepted_row_count": audit["totals"].get("accepted_rows", 0),
        "quarantined_row_count": audit["totals"].get("quarantined_rows", 0),
        "file_download_sessions": audit["file_download_sessions"],
        "usable_download_sessions": audit["usable_download_sessions"]["count"],
        "archive_hashes_match_records": [
            archive["sha256_matches_record"] for archive in data_manifest["archives"]
        ],
    }
