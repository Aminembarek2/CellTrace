"""DuckDB measurement snapshot schema.

These are application table names, not raw CSV headers. The raw to canonical
mapping lives in ``celltrace.data.dictionary``. A snapshot is written once by
ingestion, closed, hashed and then only ever opened read only.
"""

from __future__ import annotations

from celltrace.contracts.common import MetricName

SNAPSHOT_FORMAT_VERSION = "1"
MAPPING_VERSION = "raca-5g-mapping-1"
NORMALIZATION_VERSION = "normalization-1"
SNAPSHOT_FILE_NAME = "measurements.duckdb"
SNAPSHOT_MANIFEST_NAME = "snapshot_manifest.json"

# Intervals longer than this start a new segment. Derived from the cadence audit:
# timestamps are truncated to whole seconds, so 0, 1 and 2 second steps are normal
# jitter and a 3 second step is at most one absent sample.
SEGMENT_GAP_THRESHOLD_MS = 3_000

METRIC_COLUMNS: tuple[MetricName, ...] = tuple(MetricName)

_metric_sample_columns = ",\n    ".join(f"{metric.value} DOUBLE" for metric in METRIC_COLUMNS)
_metric_bin_columns = ",\n    ".join(
    f"{metric.value} DOUBLE,\n    {metric.value}_valid_count INTEGER NOT NULL"
    for metric in METRIC_COLUMNS
)

SNAPSHOT_DDL: tuple[str, ...] = (
    """
    CREATE TABLE snapshot_metadata (
        key VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE sessions (
        session_id VARCHAR PRIMARY KEY,
        source_id VARCHAR NOT NULL,
        source_relative_path VARCHAR NOT NULL,
        source_sha256 VARCHAR NOT NULL,
        application_cohort VARCHAR NOT NULL,
        mobility_cohort VARCHAR NOT NULL,
        timestamp_basis VARCHAR NOT NULL,
        origin_raw_timestamp VARCHAR NOT NULL,
        raw_row_count INTEGER NOT NULL,
        accepted_row_count INTEGER NOT NULL,
        quarantined_row_count INTEGER NOT NULL,
        reordered_row_count INTEGER NOT NULL,
        same_second_row_count INTEGER NOT NULL,
        segment_count INTEGER NOT NULL,
        duration_ms BIGINT NOT NULL,
        mapping_version VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE segments (
        session_id VARCHAR NOT NULL,
        segment_id INTEGER NOT NULL,
        start_ms BIGINT NOT NULL,
        end_ms BIGINT NOT NULL,
        sample_count INTEGER NOT NULL,
        PRIMARY KEY (session_id, segment_id)
    )
    """,
    f"""
    CREATE TABLE samples (
        session_id VARCHAR NOT NULL,
        source_row_number INTEGER NOT NULL,
        elapsed_ms BIGINT NOT NULL,
        raw_timestamp VARCHAR NOT NULL,
        local_timestamp TIMESTAMP,
        segment_id INTEGER NOT NULL,
        download_state VARCHAR NOT NULL,
        raw_state VARCHAR,
        {_metric_sample_columns},
        network_mode VARCHAR,
        serving_cell_id VARCHAR,
        cell_identity_basis VARCHAR NOT NULL,
        raw_cell_local_id VARCHAR,
        raw_node_hex VARCHAR,
        raw_cell_hex VARCHAR,
        raw_area_hex VARCHAR,
        quality_flags VARCHAR[] NOT NULL,
        PRIMARY KEY (session_id, source_row_number)
    )
    """,
    f"""
    CREATE TABLE bins_1s (
        session_id VARCHAR NOT NULL,
        segment_id INTEGER NOT NULL,
        bin_start_ms BIGINT NOT NULL,
        bin_end_ms BIGINT NOT NULL,
        available_at_ms BIGINT NOT NULL,
        sample_count INTEGER NOT NULL,
        active_count INTEGER NOT NULL,
        idle_count INTEGER NOT NULL,
        unknown_count INTEGER NOT NULL,
        active_fraction DOUBLE,
        is_complete BOOLEAN NOT NULL,
        {_metric_bin_columns},
        PRIMARY KEY (session_id, bin_start_ms)
    )
    """,
    """
    CREATE TABLE quarantined_rows (
        session_id VARCHAR NOT NULL,
        source_row_number INTEGER NOT NULL,
        raw_line VARCHAR NOT NULL,
        reason VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE value_rejections (
        session_id VARCHAR NOT NULL,
        source_row_number INTEGER NOT NULL,
        raw_field VARCHAR NOT NULL,
        raw_value VARCHAR NOT NULL,
        reason VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE metric_provenance (
        source_id VARCHAR NOT NULL,
        canonical_metric VARCHAR NOT NULL,
        raw_field VARCHAR NOT NULL,
        raw_unit VARCHAR NOT NULL,
        canonical_unit VARCHAR NOT NULL,
        conversion VARCHAR NOT NULL,
        technology_applicability VARCHAR NOT NULL,
        missing_sentinels VARCHAR NOT NULL,
        definition_source VARCHAR NOT NULL,
        audit_status VARCHAR NOT NULL,
        limitations VARCHAR NOT NULL,
        PRIMARY KEY (source_id, canonical_metric)
    )
    """,
    """
    CREATE TABLE capabilities (
        capability VARCHAR PRIMARY KEY,
        status VARCHAR NOT NULL,
        reason VARCHAR NOT NULL
    )
    """,
)

# Primary key order used for the logical content hash of each table.
TABLE_ORDER_KEYS: dict[str, tuple[str, ...]] = {
    "snapshot_metadata": ("key",),
    "sessions": ("session_id",),
    "segments": ("session_id", "segment_id"),
    "samples": ("session_id", "source_row_number"),
    "bins_1s": ("session_id", "bin_start_ms"),
    "quarantined_rows": ("session_id", "source_row_number", "reason"),
    "value_rejections": ("session_id", "source_row_number", "raw_field"),
    "metric_provenance": ("source_id", "canonical_metric"),
    "capabilities": ("capability",),
}

CAPABILITY_NAMES: tuple[str, ...] = (
    "activity_filtering",
    "rate_summaries",
    "radio_comparison",
    "serving_cell_transitions",
    "mobility",
    "neighbour_metrics",
    "forecasting",
)
