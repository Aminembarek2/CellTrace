"""Immutable DuckDB measurement snapshot creation and read only access.

Ingestion writes every table into a staging directory, closes the database,
runs SQL integrity checks on a read only connection, computes a logical hash
that does not depend on DuckDB file bytes, and then renames the staging
directory to ``<out_dir>/<snapshot_id>`` in one atomic step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from celltrace.canonical import canonical_json, content_hash, file_sha256
from celltrace.contracts.common import CapabilityStatus, DownloadState
from celltrace.data.dictionary import (
    DICTIONARY_VERSION,
    METRIC_DICTIONARY,
    TIMESTAMP_BASIS,
    dictionary_hash,
)
from celltrace.data.raca import (
    BIN_MS,
    NormalizedSample,
    NormalizedSession,
    SourceArchive,
    find_source_archives,
    iter_sessions,
)
from celltrace.data.schema import (
    CAPABILITY_NAMES,
    MAPPING_VERSION,
    METRIC_COLUMNS,
    NORMALIZATION_VERSION,
    SEGMENT_GAP_THRESHOLD_MS,
    SNAPSHOT_DDL,
    SNAPSHOT_FILE_NAME,
    SNAPSHOT_FORMAT_VERSION,
    SNAPSHOT_MANIFEST_NAME,
    TABLE_ORDER_KEYS,
)

BIN_STATISTIC = (
    "arithmetic mean of non null logged sample values per bin. dBm and dB values are averaged "
    "as logged numbers, not as linear power. No sample duration weighting is applied because "
    "per sample durations are not recorded."
)
FORECASTING_MINIMUM_SESSIONS = 5
FORECASTING_MINIMUM_COMPLETE_VALID_BINS = 60
_STAGING_PREFIX = ".snapshot_staging_"
_LIST_SEPARATOR = "|"

SAMPLE_COLUMNS: tuple[str, ...] = (
    "session_id",
    "source_row_number",
    "elapsed_ms",
    "raw_timestamp",
    "local_timestamp",
    "segment_id",
    "download_state",
    "raw_state",
    *(metric.value for metric in METRIC_COLUMNS),
    "network_mode",
    "serving_cell_id",
    "cell_identity_basis",
    "raw_cell_local_id",
    "raw_node_hex",
    "raw_cell_hex",
    "raw_area_hex",
    "quality_flags",
)
BIN_COLUMNS: tuple[str, ...] = (
    "session_id",
    "segment_id",
    "bin_start_ms",
    "bin_end_ms",
    "available_at_ms",
    "sample_count",
    "active_count",
    "idle_count",
    "unknown_count",
    "active_fraction",
    "is_complete",
    *(
        column
        for metric in METRIC_COLUMNS
        for column in (metric.value, f"{metric.value}_valid_count")
    ),
)
SESSION_COLUMNS: tuple[str, ...] = (
    "session_id",
    "source_id",
    "source_relative_path",
    "source_sha256",
    "application_cohort",
    "mobility_cohort",
    "timestamp_basis",
    "origin_raw_timestamp",
    "raw_row_count",
    "accepted_row_count",
    "quarantined_row_count",
    "reordered_row_count",
    "same_second_row_count",
    "segment_count",
    "duration_ms",
    "mapping_version",
)


type SessionIterator = Callable[[SourceArchive], Iterator[NormalizedSession]]


class SnapshotIntegrityError(RuntimeError):
    """The snapshot file or manifest does not match its recorded identity."""


class SnapshotCheckError(RuntimeError):
    """A SQL integrity check failed while building a snapshot."""


@dataclass(frozen=True)
class SqlCheck:
    name: str
    description: str
    query: str


def _insert_rows(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
    rows: list[tuple[Any, ...]],
    list_columns: frozenset[str] = frozenset(),
) -> None:
    """Insert Python rows through a registered pandas frame. Table and columns are constants."""
    if not rows:
        return
    frame = pd.DataFrame.from_records(rows, columns=list(columns))
    selected = [
        f"CASE WHEN {column} = '' THEN []::VARCHAR[] "
        f"ELSE string_split({column}, '{_LIST_SEPARATOR}') END"
        if column in list_columns
        else column
        for column in columns
    ]
    connection.register("incoming_rows", frame)
    try:
        connection.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) "
            f"SELECT {', '.join(selected)} FROM incoming_rows"
        )
    finally:
        connection.unregister("incoming_rows")


def _sample_row(session_id: str, sample: NormalizedSample) -> tuple[Any, ...]:
    return (
        session_id,
        sample.source_row_number,
        sample.elapsed_ms,
        sample.raw_timestamp,
        sample.local_timestamp,
        sample.segment_id,
        sample.download_state,
        sample.raw_state,
        *sample.metric_values,
        sample.network_mode,
        sample.serving_cell_id,
        sample.cell_identity_basis,
        sample.raw_cell_local_id,
        sample.raw_node_hex,
        sample.raw_cell_hex,
        sample.raw_area_hex,
        _LIST_SEPARATOR.join(sample.quality_flags),
    )


def build_bins(session: NormalizedSession) -> list[tuple[Any, ...]]:
    """One row per one second bin inside every segment span, empty bins included.

    Rows follow BIN_COLUMNS. Means use math.fsum so the result does not depend
    on summation order. Only the final bin of the session is incomplete.
    """
    samples_by_bin: dict[int, list[NormalizedSample]] = {}
    for sample in session.samples:
        samples_by_bin.setdefault((sample.elapsed_ms // BIN_MS) * BIN_MS, []).append(sample)
    if not session.segments:
        return []
    final_bin_start = session.segments[-1].end_ms - BIN_MS
    rows: list[tuple[Any, ...]] = []
    for segment in session.segments:
        for bin_start in range(segment.start_ms, segment.end_ms, BIN_MS):
            members = samples_by_bin.get(bin_start, [])
            if any(sample.segment_id != segment.segment_id for sample in members):
                raise ValueError(f"{session.raw.session_id}: bin {bin_start} spans two segments")
            states = [sample.download_state for sample in members]
            active = states.count(DownloadState.active.value)
            idle = states.count(DownloadState.idle.value)
            metric_columns: list[Any] = []
            for position in range(len(METRIC_COLUMNS)):
                values = [
                    value
                    for sample in members
                    if (value := sample.metric_values[position]) is not None
                ]
                mean = math.fsum(values) / len(values) if values else None
                metric_columns.extend((mean, len(values)))
            rows.append(
                (
                    session.raw.session_id,
                    segment.segment_id,
                    bin_start,
                    bin_start + BIN_MS,
                    bin_start + BIN_MS,
                    len(members),
                    active,
                    idle,
                    len(members) - active - idle,
                    active / len(members) if members else None,
                    bin_start != final_bin_start,
                    *metric_columns,
                )
            )
    return rows


def _insert_session(connection: duckdb.DuckDBPyConnection, session: NormalizedSession) -> None:
    raw = session.raw
    session_id = raw.session_id
    _insert_rows(
        connection,
        "sessions",
        SESSION_COLUMNS,
        [
            (
                session_id,
                raw.source_id,
                raw.member_path,
                raw.source_sha256,
                raw.application_cohort,
                raw.mobility_cohort,
                TIMESTAMP_BASIS,
                session.origin_raw_timestamp,
                len(raw.rows),
                len(session.samples),
                len(session.quarantined),
                session.reordered_row_count,
                session.same_second_row_count,
                len(session.segments),
                session.duration_ms,
                MAPPING_VERSION,
            )
        ],
    )
    _insert_rows(
        connection,
        "segments",
        ("session_id", "segment_id", "start_ms", "end_ms", "sample_count"),
        [
            (session_id, segment.segment_id, segment.start_ms, segment.end_ms, segment.sample_count)
            for segment in session.segments
        ],
    )
    _insert_rows(
        connection,
        "samples",
        SAMPLE_COLUMNS,
        [_sample_row(session_id, sample) for sample in session.samples],
        list_columns=frozenset({"quality_flags"}),
    )
    _insert_rows(connection, "bins_1s", BIN_COLUMNS, build_bins(session))
    _insert_rows(
        connection,
        "quarantined_rows",
        ("session_id", "source_row_number", "raw_line", "reason"),
        [
            (session_id, row.source_row_number, row.raw_line, row.reason)
            for row in session.quarantined
        ],
    )
    _insert_rows(
        connection,
        "value_rejections",
        ("session_id", "source_row_number", "raw_field", "raw_value", "reason"),
        [
            (session_id, item.source_row_number, item.raw_field, item.raw_value, item.reason)
            for item in session.rejections
        ],
    )


def metric_provenance_rows(source_id: str) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for metric in METRIC_COLUMNS:
        definition = METRIC_DICTIONARY[metric]
        sentinels = canonical_json(
            {
                "missing_sentinels": list(definition.missing_sentinels),
                "suspected_fill_values": list(definition.suspected_fill_values),
                "documented_range": (
                    list(definition.documented_range) if definition.documented_range else None
                ),
            }
        )
        rows.append(
            (
                source_id,
                metric.value,
                definition.raw_field,
                definition.raw_unit,
                definition.canonical_unit,
                definition.conversion,
                definition.technology_applicability,
                sentinels,
                definition.definition_source,
                definition.audit_status,
                definition.limitations,
            )
        )
    return rows


def _percent(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "0 percent"
    return f"{100.0 * numerator / denominator:.1f} percent"


def _fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _tiered_status(fraction: float, supported_at: float, partial_at: float) -> CapabilityStatus:
    if fraction >= supported_at:
        return CapabilityStatus.supported
    if fraction >= partial_at:
        return CapabilityStatus.partial
    return CapabilityStatus.unavailable


def derive_capabilities(connection: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    """Capability rows derived from audited coverage of file download samples."""
    row = connection.execute(
        """
        SELECT
            count(*),
            count(*) FILTER (WHERE download_state <> 'unknown'),
            count(downlink_mbps),
            count(reported_rsrp_dbm),
            count(reported_rsrq_db),
            count(reported_snr_db),
            count(reported_cqi),
            count(*) FILTER (WHERE serving_cell_id IS NOT NULL
                             AND cell_identity_basis <> 'unknown'),
            count(speed_kph),
            count(reported_nrx_rsrp_dbm),
            count(reported_nrx_rsrq_db)
        FROM samples JOIN sessions USING (session_id)
        WHERE sessions.application_cohort = 'file_download'
        """
    ).fetchone()
    assert row is not None
    (
        total,
        known_state,
        downlink,
        rsrp,
        rsrq,
        snr,
        cqi,
        known_identity,
        speed,
        nrx_rsrp,
        nrx_rsrq,
    ) = (int(value) for value in row)
    static_row = connection.execute(
        """
        SELECT count(speed_kph), count(*) FILTER (WHERE speed_kph > 0)
        FROM samples JOIN sessions USING (session_id)
        WHERE sessions.mobility_cohort = 'static'
        """
    ).fetchone()
    assert static_row is not None
    static_speed, static_moving = int(static_row[0]), int(static_row[1])
    forecast_row = connection.execute(
        """
        SELECT count(*) FROM (
            SELECT bins_1s.session_id
            FROM bins_1s JOIN sessions USING (session_id)
            WHERE sessions.application_cohort = 'file_download'
              AND bins_1s.is_complete AND bins_1s.downlink_mbps_valid_count > 0
            GROUP BY bins_1s.session_id
            HAVING count(*) >= ?
        )
        """,
        [FORECASTING_MINIMUM_COMPLETE_VALID_BINS],
    ).fetchone()
    assert forecast_row is not None
    forecast_sessions = int(forecast_row[0])

    capabilities: dict[str, tuple[CapabilityStatus, str]] = {}
    if total == 0:
        reason = "No file download samples in this snapshot."
        for name in CAPABILITY_NAMES:
            capabilities[name] = (CapabilityStatus.unavailable, reason)
        return [(name, status.value, text) for name, (status, text) in capabilities.items()]

    capabilities["activity_filtering"] = (
        _tiered_status(_fraction(known_state, total), 0.95, 0.5),
        f"{_percent(known_state, total)} of {total} file download samples carry state D (active) "
        "or I (idle). V, VD and any other code map to unknown with raw_state preserved. The "
        "state mapping is inferred from traffic, not from a primary definition, and active "
        "state does not prove offered load exceeded capacity.",
    )
    capabilities["rate_summaries"] = (
        _tiered_status(_fraction(downlink, total), 0.95, 0.5),
        f"{_percent(downlink, total)} of file download samples have a downlink rate. The kbps "
        "unit is inferred from magnitude and unverified, so absolute Mbps values carry that "
        "assumption while relative changes do not. Rates are delivered application traffic, "
        "not available capacity.",
    )
    rsrp_fraction = _fraction(rsrp, total)
    capabilities["radio_comparison"] = (
        CapabilityStatus.partial if rsrp_fraction >= 0.5 else CapabilityStatus.unavailable,
        f"Valid coverage in file download samples: RSRP {_percent(rsrp, total)}, "
        f"RSRQ {_percent(rsrq, total)}, SNR {_percent(snr, total)}, CQI "
        f"{_percent(cqi, total)}. Units are conventional but the measuring "
        "technology for 5G mode rows (LTE anchor or NR) is undocumented, so comparisons are "
        "limited to reported values.",
    )
    capabilities["serving_cell_transitions"] = (
        CapabilityStatus.partial if known_identity > 0 else CapabilityStatus.unavailable,
        f"{_percent(known_identity, total)} of file download samples carry a serving cell "
        "identifier consistent with a known identity layout. A change is a reported "
        "transition proxy, not a confirmed handover, and the uniqueness scope of the "
        "identifier is undocumented.",
    )
    capabilities["mobility"] = (
        CapabilityStatus.partial
        if _fraction(speed, total) >= 0.5
        else CapabilityStatus.unavailable,
        f"Speed is present in {_percent(speed, total)} of file download samples with an "
        f"inferred km/h unit. {_percent(static_moving, static_speed)} of static cohort samples "
        "with a speed value report a nonzero speed"
        + (", so folder mobility labels and speed disagree." if static_moving else "."),
    )
    capabilities["neighbour_metrics"] = (
        CapabilityStatus.unavailable,
        f"NRxRSRP is present in {_percent(nrx_rsrp, total)} and NRxRSRQ in "
        f"{_percent(nrx_rsrq, total)} of file download samples, but their meaning is "
        "undocumented (see the dictionary limitations), so they are not exposed as neighbour "
        "cell metrics.",
    )
    capabilities["forecasting"] = (
        CapabilityStatus.supported
        if forecast_sessions >= FORECASTING_MINIMUM_SESSIONS
        else (CapabilityStatus.partial if forecast_sessions > 0 else CapabilityStatus.unavailable),
        f"{forecast_sessions} file download sessions have at least "
        f"{FORECASTING_MINIMUM_COMPLETE_VALID_BINS} complete bins with a valid downlink mean, "
        f"against a threshold of {FORECASTING_MINIMUM_SESSIONS} sessions for whole session "
        "splits. This describes data support only, no forecasting model is part of the "
        "snapshot.",
    )
    return [(name, capabilities[name][0].value, capabilities[name][1]) for name in CAPABILITY_NAMES]


def _metadata_rows(sources: list[SourceArchive]) -> list[tuple[Any, ...]]:
    source_ids = sorted({source.source_id for source in sources})
    archive_hashes = sorted(source.inventory.archive_sha256 for source in sources)
    metadata = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "source_id": ",".join(source_ids),
        "mapping_version": MAPPING_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "dictionary_version": DICTIONARY_VERSION,
        "dictionary_hash": dictionary_hash(),
        "segment_gap_threshold_ms": str(SEGMENT_GAP_THRESHOLD_MS),
        "timestamp_basis": TIMESTAMP_BASIS,
        "elapsed_origin": "earliest accepted timestamp of each session after the order audit",
        "bin_ms": str(BIN_MS),
        "bin_statistic": BIN_STATISTIC,
        "bin_availability": "available_at_ms equals bin_end_ms, the final bin of each session "
        "is marked incomplete",
        "source_archive_sha256": ",".join(archive_hashes),
    }
    return sorted(metadata.items())


def _write_database(
    database_path: Path,
    sources: list[SourceArchive],
    session_iterator: SessionIterator = iter_sessions,
) -> dict[str, Any]:
    """Create and fill every table, then close the connection. Returns source evidence."""
    seen_sessions: set[str] = set()
    source_evidence: list[dict[str, Any]] = []
    connection = duckdb.connect(str(database_path))
    connection.execute("SET enable_progress_bar = false")
    try:
        for statement in SNAPSHOT_DDL:
            connection.execute(statement)
        for source in sources:
            members: list[dict[str, Any]] = []
            for session in session_iterator(source):
                session_id = session.raw.session_id
                if session_id in seen_sessions:
                    raise ValueError(f"duplicate session identity {session_id}")
                seen_sessions.add(session_id)
                _insert_session(connection, session)
                members.append(
                    {
                        "member_path": session.raw.member_path,
                        "sha256": session.raw.source_sha256,
                        "session_id": session_id,
                    }
                )
            source_evidence.append(
                {
                    "archive_relative_path": source.relative_path,
                    "archive_sha256": source.inventory.archive_sha256,
                    "archive_bytes": source.inventory.archive_bytes,
                    "source_id": source.source_id,
                    "members": members,
                }
            )
        for source_id in sorted({source.source_id for source in sources}):
            _insert_rows(
                connection,
                "metric_provenance",
                (
                    "source_id",
                    "canonical_metric",
                    "raw_field",
                    "raw_unit",
                    "canonical_unit",
                    "conversion",
                    "technology_applicability",
                    "missing_sentinels",
                    "definition_source",
                    "audit_status",
                    "limitations",
                ),
                metric_provenance_rows(source_id),
            )
        _insert_rows(
            connection,
            "capabilities",
            ("capability", "status", "reason"),
            [tuple(row) for row in derive_capabilities(connection)],
        )
        _insert_rows(connection, "snapshot_metadata", ("key", "value"), _metadata_rows(sources))
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    return {"sources": source_evidence}


def _sql_checks() -> list[SqlCheck]:
    checks = [
        SqlCheck(
            "nonnegative_rates",
            "No negative downlink or uplink rate in samples or bins",
            """
            SELECT (SELECT count(*) FROM samples WHERE downlink_mbps < 0 OR uplink_mbps < 0)
                 + (SELECT count(*) FROM bins_1s WHERE downlink_mbps < 0 OR uplink_mbps < 0)
            """,
        ),
        SqlCheck(
            "nonnegative_elapsed_time",
            "No negative elapsed time and every session starts at elapsed zero",
            """
            SELECT (SELECT count(*) FROM samples WHERE elapsed_ms < 0)
                 + (SELECT count(*) FROM bins_1s WHERE bin_start_ms < 0)
                 + (SELECT count(*) FROM (
                        SELECT session_id FROM samples GROUP BY session_id
                        HAVING min(elapsed_ms) <> 0))
            """,
        ),
        SqlCheck(
            "elapsed_order_and_segments",
            "In (elapsed_ms, source_row_number) order elapsed time never decreases, segment "
            "ids increase by one exactly at gaps above the threshold",
            f"""
            SELECT count(*) FROM (
                SELECT segment_id, elapsed_ms,
                    lag(segment_id) OVER ordered AS previous_segment,
                    lag(elapsed_ms) OVER ordered AS previous_elapsed
                FROM samples
                WINDOW ordered AS (PARTITION BY session_id ORDER BY elapsed_ms, source_row_number)
            )
            WHERE previous_elapsed IS NOT NULL AND (
                elapsed_ms < previous_elapsed
                OR (segment_id = previous_segment
                    AND elapsed_ms - previous_elapsed > {SEGMENT_GAP_THRESHOLD_MS})
                OR (segment_id <> previous_segment
                    AND (segment_id <> previous_segment + 1
                         OR elapsed_ms - previous_elapsed <= {SEGMENT_GAP_THRESHOLD_MS}))
            )
            """,
        ),
        SqlCheck(
            "source_row_reconciliation",
            "Per session raw rows equal accepted plus quarantined rows and both match table "
            "row counts",
            """
            SELECT count(*) FROM sessions
            LEFT JOIN (SELECT session_id, count(*) AS sample_rows FROM samples
                       GROUP BY session_id) AS sample_counts USING (session_id)
            LEFT JOIN (SELECT session_id, count(*) AS quarantine_rows FROM quarantined_rows
                       GROUP BY session_id) AS quarantine_counts USING (session_id)
            WHERE raw_row_count <> accepted_row_count + quarantined_row_count
               OR accepted_row_count <> coalesce(sample_rows, 0)
               OR quarantined_row_count <> coalesce(quarantine_rows, 0)
            """,
        ),
        SqlCheck(
            "bin_sample_totals",
            "Per session the bin sample counts sum to the accepted sample count",
            """
            SELECT count(*) FROM sessions
            LEFT JOIN (SELECT session_id, sum(sample_count) AS binned FROM bins_1s
                       GROUP BY session_id) AS bin_totals USING (session_id)
            WHERE accepted_row_count <> coalesce(binned, 0)
            """,
        ),
        SqlCheck(
            "segment_totals",
            "Segment rows match segment_count and their sample counts sum to accepted rows",
            """
            SELECT count(*) FROM sessions
            LEFT JOIN (SELECT session_id, count(*) AS segment_rows,
                              sum(sample_count) AS segment_samples
                       FROM segments GROUP BY session_id) AS totals USING (session_id)
            WHERE segment_count <> coalesce(segment_rows, 0)
               OR accepted_row_count <> coalesce(segment_samples, 0)
            """,
        ),
        SqlCheck(
            "bin_availability",
            "Bins are one second wide, available at their end, and only the final bin of each "
            "session is incomplete",
            f"""
            SELECT (SELECT count(*) FROM bins_1s
                    WHERE bin_end_ms <> bin_start_ms + {BIN_MS}
                       OR available_at_ms <> bin_end_ms
                       OR bin_start_ms % {BIN_MS} <> 0)
                 + (SELECT count(*) FROM (
                        SELECT session_id,
                               count(*) FILTER (WHERE NOT is_complete) AS incomplete,
                               max(bin_start_ms) AS last_start,
                               max(bin_start_ms) FILTER (WHERE NOT is_complete) AS incomplete_start
                        FROM bins_1s GROUP BY session_id)
                    WHERE incomplete <> 1 OR incomplete_start <> last_start)
            """,
        ),
    ]
    for table, keys in TABLE_ORDER_KEYS.items():
        key_list = ", ".join(keys)
        checks.append(
            SqlCheck(
                f"unique_key_{table}",
                f"No duplicate ({key_list}) in {table}",
                f"SELECT count(*) FROM (SELECT {key_list} FROM {table} "
                f"GROUP BY {key_list} HAVING count(*) > 1)",
            )
        )
    return checks


def run_sql_checks(connection: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for check in _sql_checks():
        row = connection.execute(check.query).fetchone()
        violations = int(row[0]) if row is not None and row[0] is not None else 0
        results.append(
            {
                "name": check.name,
                "description": check.description,
                "violations": violations,
                "passed": violations == 0,
            }
        )
    return results


def _canonical_cell(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, list | tuple):
        return [_canonical_cell(item) for item in value]
    return value


def table_logical_hash(connection: duckdb.DuckDBPyConnection, table: str) -> tuple[str, int]:
    """Hash of column names, types and every row in primary key order."""
    columns = [
        (str(row[0]), str(row[1])) for row in connection.execute(f"DESCRIBE {table}").fetchall()
    ]
    digest = hashlib.sha256()
    digest.update(canonical_json({"table": table, "columns": columns}).encode("utf-8"))
    digest.update(b"\n")
    order = ", ".join(TABLE_ORDER_KEYS[table])
    cursor = connection.execute(f"SELECT * FROM {table} ORDER BY {order}")
    row_count = 0
    while batch := cursor.fetchmany(10_000):
        for row in batch:
            digest.update(canonical_json([_canonical_cell(value) for value in row]).encode("utf-8"))
            digest.update(b"\n")
            row_count += 1
    return digest.hexdigest(), row_count


def logical_hashes(connection: duckdb.DuckDBPyConnection) -> tuple[str, dict[str, dict[str, Any]]]:
    tables: dict[str, dict[str, Any]] = {}
    for table in TABLE_ORDER_KEYS:
        table_hash, row_count = table_logical_hash(connection, table)
        tables[table] = {"logical_hash": table_hash, "row_count": row_count}
    overall = content_hash(
        {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "tables": {table: entry["logical_hash"] for table, entry in tables.items()},
        }
    )
    return overall, tables


def _make_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _connect_read_only(database_path: Path) -> duckdb.DuckDBPyConnection:
    """Seal settings at connection creation, including when readers share a database instance."""
    return duckdb.connect(
        str(database_path),
        read_only=True,
        config={"enable_external_access": False, "lock_configuration": True},
    )


def run_ingest(
    input_dir: Path,
    out_dir: Path,
    clock: Callable[[], datetime] | None = None,
    sources: list[SourceArchive] | None = None,
    session_iterator: SessionIterator = iter_sessions,
) -> Path:
    """Build an immutable snapshot from every archive under input_dir and return its directory.

    Another source adapter passes its own archives and session iterator. By default the Raca
    5G archives under input_dir are read with the Raca adapter.
    """
    now = clock or (lambda: datetime.now(UTC))
    if sources is None:
        sources = find_source_archives(input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=out_dir))
    try:
        database_path = staging / SNAPSHOT_FILE_NAME
        source_evidence = _write_database(database_path, sources, session_iterator)
        wal_path = database_path.with_name(database_path.name + ".wal")
        if wal_path.exists():
            raise SnapshotCheckError("DuckDB left a write ahead log after closing")
        connection = _connect_read_only(database_path)
        try:
            checks = run_sql_checks(connection)
            logical_hash, tables = logical_hashes(connection)
        finally:
            connection.close()
        failed = [check["name"] for check in checks if not check["passed"]]
        if failed:
            raise SnapshotCheckError(f"snapshot SQL checks failed: {failed}")
        snapshot_id = f"snap_{logical_hash[:16]}"
        manifest = {
            "snapshot_id": snapshot_id,
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "file_name": SNAPSHOT_FILE_NAME,
            "file_sha256": file_sha256(database_path),
            "file_bytes": database_path.stat().st_size,
            "logical_hash": logical_hash,
            "tables": tables,
            "source_ids": sorted({source.source_id for source in sources}),
            "sources": source_evidence["sources"],
            "mapping_version": MAPPING_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "dictionary_version": DICTIONARY_VERSION,
            "dictionary_hash": dictionary_hash(),
            "segment_gap_threshold_ms": SEGMENT_GAP_THRESHOLD_MS,
            "timestamp_basis": TIMESTAMP_BASIS,
            "created_at": now().isoformat(),
            "sql_checks": checks,
        }
        manifest_path = staging / SNAPSHOT_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _make_read_only(database_path)
        _make_read_only(manifest_path)
        target = out_dir / snapshot_id
        if target.exists():
            existing = load_manifest(target)
            if existing["logical_hash"] != logical_hash:
                raise SnapshotIntegrityError(
                    f"{target} exists with a different logical hash {existing['logical_hash']}"
                )
            shutil.rmtree(staging)
            return target
        os.rename(staging, target)
        return target
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_manifest(snapshot_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_dir / SNAPSHOT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise SnapshotIntegrityError(f"no {SNAPSHOT_MANIFEST_NAME} in {snapshot_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SnapshotIntegrityError(f"{manifest_path} is not a JSON object")
    for key in ("snapshot_id", "file_sha256", "logical_hash", "tables"):
        if key not in manifest:
            raise SnapshotIntegrityError(f"{manifest_path} lacks {key}")
    if manifest["snapshot_id"] != f"snap_{str(manifest['logical_hash'])[:16]}":
        raise SnapshotIntegrityError(f"{manifest_path}: snapshot_id does not match logical_hash")
    return manifest


def open_snapshot(
    snapshot_dir: Path, verify_file_hash: bool = True
) -> tuple[duckdb.DuckDBPyConnection, dict[str, Any]]:
    """Read only connection with external file access disabled, plus the manifest."""
    manifest = load_manifest(snapshot_dir)
    database_path = snapshot_dir / SNAPSHOT_FILE_NAME
    if not database_path.is_file():
        raise SnapshotIntegrityError(f"no {SNAPSHOT_FILE_NAME} in {snapshot_dir}")
    if verify_file_hash:
        actual = file_sha256(database_path)
        if actual != manifest["file_sha256"]:
            raise SnapshotIntegrityError(
                f"{database_path} sha256 {actual} differs from manifest {manifest['file_sha256']}"
            )
    return _connect_read_only(database_path), manifest


def verify_logical_hash(snapshot_dir: Path) -> bool:
    """Recompute the logical hash from table contents and compare it with the manifest."""
    connection, manifest = open_snapshot(snapshot_dir, verify_file_hash=False)
    try:
        logical_hash, _ = logical_hashes(connection)
    finally:
        connection.close()
    return bool(logical_hash == manifest["logical_hash"])
