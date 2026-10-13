"""Source adapter for the Raca Irish 5G production archive.

The adapter reads CSV members directly from the zip archive without extracting
them, checks the exact observed header, and normalizes each session into
canonical samples, quarantined rows, value rejections and segments. Nothing in
the archive is ever executed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from celltrace.contracts.common import DownloadState, IdentityBasis, MetricName
from celltrace.data.dictionary import (
    ANDROID_UNAVAILABLE,
    APPLICATION_FOLDERS,
    METRIC_DICTIONARY,
    MOBILITY_FOLDERS,
    SOURCE_ID,
    STATE_MAPPING,
    TIMESTAMP_FORMAT,
)
from celltrace.data.schema import METRIC_COLUMNS, SEGMENT_GAP_THRESHOLD_MS

RACA_HEADER: tuple[str, ...] = (
    "Timestamp",
    "Longitude",
    "Latitude",
    "Speed",
    "Operatorname",
    "CellID",
    "NetworkMode",
    "RSRP",
    "RSRQ",
    "SNR",
    "CQI",
    "RSSI",
    "DL_bitrate",
    "UL_bitrate",
    "State",
    "PINGAVG",
    "PINGMIN",
    "PINGMAX",
    "PINGSTDEV",
    "PINGLOSS",
    "CELLHEX",
    "NODEHEX",
    "LACHEX",
    "RAWCELLID",
    "NRxRSRP",
    "NRxRSRQ",
)
FIELD_INDEX: dict[str, int] = {name: position for position, name in enumerate(RACA_HEADER)}

RAW_FIELDS_NOT_INGESTED: tuple[str, ...] = (
    "Longitude",
    "Latitude",
    "Operatorname",
    "PINGMIN",
    "PINGMAX",
    "PINGSTDEV",
    "PINGLOSS",
)
NOT_INGESTED_NOTE = "not ingested, preserved in raw archive"

DATASET_ROOT = "5G-production-dataset"
MAXIMUM_ARCHIVE_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
SOURCE_RECORD_NAME = "SOURCE.json"
DASH_MISSING = "-"
BIN_MS = 1000
RATE_METRICS = frozenset({MetricName.downlink_mbps.value, MetricName.uplink_mbps.value})

TECHNOLOGY_FAMILIES: dict[str, str] = {
    "5G": "lte_nr",
    "LTE": "lte_nr",
    "HSPA+": "umts",
    "HSUPA": "umts",
    "HSDPA": "umts",
    "UMTS": "umts",
    "GPRS": "gsm",
    "EDGE": "gsm",
}

_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_DECIMAL_DIGITS_PATTERN = re.compile(r"\d+")
_HEX_PATTERN = re.compile(r"[0-9A-Fa-f]+")
_SOURCE_ID_PATTERN = re.compile(r"[a-z0-9_]+")


class ArchiveSafetyError(ValueError):
    """The archive has an unsafe member name, an oversized payload or encryption."""


class UnexpectedHeaderError(ValueError):
    """A CSV member does not carry the exact audited header. Columns are never remapped."""

    def __init__(self, member_path: str, observed: tuple[str, ...]) -> None:
        self.member_path = member_path
        self.observed = observed
        super().__init__(
            f"{member_path}: header does not match the audited Raca header. "
            f"Observed {len(observed)} columns: {list(observed)}"
        )


@dataclass(frozen=True)
class ArchiveMember:
    archive_path: Path
    member_path: str
    uncompressed_bytes: int
    compressed_bytes: int
    sha256: str
    application_cohort: str
    mobility_cohort: str


@dataclass(frozen=True)
class SkippedEntry:
    entry_name: str
    reason: str


@dataclass(frozen=True)
class ArchiveInventory:
    archive_path: Path
    archive_sha256: str
    archive_bytes: int
    entry_count: int
    total_uncompressed_bytes: int
    members: tuple[ArchiveMember, ...]
    skipped: tuple[SkippedEntry, ...]


@dataclass(frozen=True)
class SourceArchive:
    archive_path: Path
    relative_path: str
    source_id: str
    source_record: dict[str, Any] | None
    inventory: ArchiveInventory


@dataclass(frozen=True)
class RawRow:
    source_row_number: int
    line: str
    values: tuple[str, ...]


@dataclass
class RawSession:
    session_id: str
    source_id: str
    member_path: str
    source_sha256: str
    application_cohort: str
    mobility_cohort: str
    encoding: str
    delimiter: str
    line_terminator: str
    header: tuple[str, ...]
    rows: list[RawRow]


@dataclass(frozen=True)
class NormalizedSample:
    source_row_number: int
    elapsed_ms: int
    raw_timestamp: str
    local_timestamp: datetime
    segment_id: int
    download_state: str
    raw_state: str | None
    metric_values: tuple[float | None, ...]
    network_mode: str | None
    serving_cell_id: str | None
    cell_identity_basis: str
    raw_cell_local_id: str | None
    raw_node_hex: str | None
    raw_cell_hex: str | None
    raw_area_hex: str | None
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class QuarantinedRow:
    source_row_number: int
    raw_line: str
    reason: str


@dataclass(frozen=True)
class ValueRejection:
    source_row_number: int
    raw_field: str
    raw_value: str
    reason: str


@dataclass(frozen=True)
class Segment:
    segment_id: int
    start_ms: int
    end_ms: int
    sample_count: int


@dataclass
class NormalizedSession:
    raw: RawSession
    origin_raw_timestamp: str
    samples: list[NormalizedSample]
    quarantined: list[QuarantinedRow]
    rejections: list[ValueRejection]
    segments: list[Segment]
    reordered_row_count: int
    same_second_row_count: int
    raw_interval_counts_ms: Counter[int] = field(default_factory=Counter)
    reordered_rows: list[tuple[int, int]] = field(default_factory=list)
    regression_rows: list[tuple[int, int]] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        if not self.samples:
            return 0
        return self.samples[-1].elapsed_ms - self.samples[0].elapsed_ms + BIN_MS


def make_session_id(source_id: str, member_path: str) -> str:
    """Opaque identifier. It never contains the file name or the recording date."""
    digest = hashlib.sha256(f"{source_id}:{member_path}".encode()).hexdigest()
    return f"ses_{digest[:12]}"


def technology_family(network_mode: str | None) -> str:
    if network_mode is None:
        return "unknown"
    return TECHNOLOGY_FAMILIES.get(network_mode, "unknown")


def _check_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if name.startswith(("/", "\\")) or "\\" in name or re.match(r"[A-Za-z]:", name):
        raise ArchiveSafetyError(f"absolute or non portable archive member name: {name!r}")
    if ".." in path.parts:
        raise ArchiveSafetyError(f"path traversal in archive member name: {name!r}")


def _skip_reason(name: str) -> str | None:
    path = PurePosixPath(name)
    if name.endswith("/"):
        return "directory"
    if path.parts and path.parts[0] == "__MACOSX":
        return "macos_resource_metadata"
    if path.name == ".DS_Store":
        return "finder_metadata"
    if path.suffix.lower() != ".csv":
        return "not_csv"
    if len(path.parts) < 4 or path.parts[0] != DATASET_ROOT:
        return "outside_dataset_layout"
    return None


def _cohorts(member_path: str) -> tuple[str, str]:
    parts = PurePosixPath(member_path).parts
    application = APPLICATION_FOLDERS.get(parts[1], "unknown")
    mobility = MOBILITY_FOLDERS.get(parts[2], "unknown")
    return application, mobility


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _read_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    with archive.open(info) as handle:
        content = handle.read(info.file_size + 1)
    if len(content) != info.file_size:
        raise ArchiveSafetyError(
            f"{info.filename}: declared {info.file_size} bytes but read {len(content)}"
        )
    return content


def inspect_archive(
    archive_path: Path, maximum_uncompressed_bytes: int = MAXIMUM_ARCHIVE_UNCOMPRESSED_BYTES
) -> ArchiveInventory:
    """List members, validate names and sizes, and hash every selected CSV member."""
    members: list[ArchiveMember] = []
    skipped: list[SkippedEntry] = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        for info in infos:
            _check_member_name(info.filename)
            if info.flag_bits & 0x1:
                raise ArchiveSafetyError(f"encrypted archive member: {info.filename!r}")
        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > maximum_uncompressed_bytes:
            raise ArchiveSafetyError(
                f"{archive_path.name}: {total_uncompressed} uncompressed bytes exceed the "
                f"ceiling of {maximum_uncompressed_bytes}"
            )
        for info in sorted(infos, key=lambda entry: entry.filename):
            reason = _skip_reason(info.filename)
            if reason is not None:
                skipped.append(SkippedEntry(info.filename, reason))
                continue
            content = _read_entry(archive, info)
            application, mobility = _cohorts(info.filename)
            members.append(
                ArchiveMember(
                    archive_path=archive_path,
                    member_path=info.filename,
                    uncompressed_bytes=info.file_size,
                    compressed_bytes=info.compress_size,
                    sha256=hashlib.sha256(content).hexdigest(),
                    application_cohort=application,
                    mobility_cohort=mobility,
                )
            )
    return ArchiveInventory(
        archive_path=archive_path,
        archive_sha256=_sha256_file(archive_path),
        archive_bytes=archive_path.stat().st_size,
        entry_count=len(infos),
        total_uncompressed_bytes=total_uncompressed,
        members=tuple(members),
        skipped=tuple(skipped),
    )


def discover_members(archive_path: Path) -> list[ArchiveMember]:
    """CSV measurement members of one archive, sorted by member path."""
    return list(inspect_archive(archive_path).members)


def read_member(member: ArchiveMember) -> bytes:
    """Read one member and confirm it still matches the inventory hash."""
    with zipfile.ZipFile(member.archive_path) as archive:
        content = _read_entry(archive, archive.getinfo(member.member_path))
    if hashlib.sha256(content).hexdigest() != member.sha256:
        raise ArchiveSafetyError(f"{member.member_path}: content changed since inventory")
    return content


def load_source_record(archive_path: Path) -> dict[str, Any] | None:
    record_path = archive_path.parent / SOURCE_RECORD_NAME
    if not record_path.is_file():
        return None
    loaded = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{record_path} must contain a JSON object")
    return loaded


def find_source_archives(input_dir: Path) -> list[SourceArchive]:
    """Every zip archive under input_dir, with its provenance record when present."""
    archive_paths = sorted(path for path in input_dir.rglob("*.zip") if path.is_file())
    if not archive_paths:
        raise FileNotFoundError(f"no zip archive found under {input_dir}")
    archives: list[SourceArchive] = []
    for archive_path in archive_paths:
        record = load_source_record(archive_path)
        source_id = str(record.get("source_id", SOURCE_ID)) if record else SOURCE_ID
        if not _SOURCE_ID_PATTERN.fullmatch(source_id):
            raise ValueError(f"{archive_path}: invalid source_id {source_id!r}")
        archives.append(
            SourceArchive(
                archive_path=archive_path,
                relative_path=archive_path.relative_to(input_dir).as_posix(),
                source_id=source_id,
                source_record=record,
                inventory=inspect_archive(archive_path),
            )
        )
    return archives


def parse_member(member: ArchiveMember, content: bytes, source_id: str) -> RawSession:
    """Split one CSV member into physical data lines. Each line is one source row."""
    encoding = "utf-8"
    if content.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    try:
        text = content.decode(encoding)
    except UnicodeDecodeError as error:
        raise ValueError(f"{member.member_path}: not valid UTF-8 ({error})") from error
    line_terminator = "CRLF" if "\r\n" in text else "LF"
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [line[:-1] if line.endswith("\r") else line for line in lines]
    if not lines:
        raise UnexpectedHeaderError(member.member_path, ())
    header = tuple(next(csv.reader([lines[0]])))
    if header != RACA_HEADER:
        raise UnexpectedHeaderError(member.member_path, header)
    rows = [
        RawRow(
            source_row_number=row_number,
            line=line,
            values=tuple(next(csv.reader([line]), [])),
        )
        for row_number, line in enumerate(lines[1:], start=1)
    ]
    return RawSession(
        session_id=make_session_id(source_id, member.member_path),
        source_id=source_id,
        member_path=member.member_path,
        source_sha256=member.sha256,
        application_cohort=member.application_cohort,
        mobility_cohort=member.mobility_cohort,
        encoding=encoding,
        delimiter=",",
        line_terminator=line_terminator,
        header=header,
        rows=rows,
    )


def parse_timestamp(raw_timestamp: str) -> datetime | None:
    """Naive local wall clock time. The timezone is unknown and never assumed."""
    try:
        return datetime.strptime(raw_timestamp, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def convert_metric(metric: str, raw_value: str) -> tuple[float | None, str | None, bool]:
    """Return (canonical value, rejection reason, outside documented range)."""
    definition = METRIC_DICTIONARY[MetricName(metric)]
    if raw_value == DASH_MISSING:
        return None, None, False
    if raw_value == "":
        return None, "empty_value", False
    if not _NUMBER_PATTERN.fullmatch(raw_value):
        return None, "unparseable_number", False
    number = float(raw_value)
    if not math.isfinite(number):
        return None, "nonfinite_number", False
    if number == ANDROID_UNAVAILABLE:
        return None, "android_unavailable_sentinel", False
    if number in definition.suspected_fill_values:
        return None, "suspected_fill_value", False
    value = number / definition.divisor
    if metric in RATE_METRICS and value < 0:
        return None, "negative_rate", False
    outside = False
    if definition.documented_range is not None:
        lower, upper = definition.documented_range
        outside = not lower <= value <= upper
    return value, None, outside


def _parse_hex(raw_value: str) -> int | None:
    if _HEX_PATTERN.fullmatch(raw_value):
        return int(raw_value, 16)
    return None


def cell_identity(raw_cell_id: str, node_hex: str, cell_hex: str) -> tuple[str | None, str]:
    """Serving cell identifier string and the identity layout it is consistent with."""
    if not _DECIMAL_DIGITS_PATTERN.fullmatch(raw_cell_id) or int(raw_cell_id) == 0:
        return None, IdentityBasis.unknown.value
    number = int(raw_cell_id)
    node = _parse_hex(node_hex)
    cell = _parse_hex(cell_hex)
    if node is None or cell is None or node == 0:
        return raw_cell_id, IdentityBasis.unknown.value
    if number == node * 256 + cell:
        return raw_cell_id, IdentityBasis.eutran_cell_identity_layout.value
    if number == node * 65536 + cell:
        return raw_cell_id, IdentityBasis.utran_cell_identity_layout.value
    return raw_cell_id, IdentityBasis.unknown.value


def map_state(raw_state: str | None) -> str:
    if raw_state is None:
        return DownloadState.unknown.value
    return STATE_MAPPING.get(raw_state, DownloadState.unknown.value)


def _optional_text(raw_value: str) -> str | None:
    return raw_value if raw_value != "" else None


def _milliseconds(delta: timedelta) -> int:
    return delta // timedelta(milliseconds=1)


@dataclass(frozen=True)
class _TimedRow:
    timestamp: datetime
    row: RawRow
    reordered: bool


@dataclass
class _OrderAudit:
    accepted: list[_TimedRow] = field(default_factory=list)
    quarantined: list[QuarantinedRow] = field(default_factory=list)
    raw_intervals: Counter[int] = field(default_factory=Counter)
    reordered_rows: list[tuple[int, int]] = field(default_factory=list)
    regression_rows: list[tuple[int, int]] = field(default_factory=list)


def _order_rows(raw: RawSession) -> _OrderAudit:
    """Audit temporal order in source order, then stable sort accepted rows by time."""
    audit = _OrderAudit()
    seen_rows: set[tuple[str, ...]] = set()
    previous_timestamp: datetime | None = None
    running_maximum: datetime | None = None
    timestamp_cache: dict[str, datetime | None] = {}
    for row in raw.rows:
        if len(row.values) != len(RACA_HEADER):
            audit.quarantined.append(
                QuarantinedRow(row.source_row_number, row.line, "malformed_field_count")
            )
            continue
        raw_timestamp = row.values[FIELD_INDEX["Timestamp"]]
        if raw_timestamp not in timestamp_cache:
            timestamp_cache[raw_timestamp] = parse_timestamp(raw_timestamp)
        timestamp = timestamp_cache[raw_timestamp]
        if timestamp is None:
            audit.quarantined.append(
                QuarantinedRow(row.source_row_number, row.line, "unparseable_timestamp")
            )
            continue
        if previous_timestamp is not None:
            audit.raw_intervals[_milliseconds(timestamp - previous_timestamp)] += 1
        previous_timestamp = timestamp
        if row.values in seen_rows:
            audit.quarantined.append(
                QuarantinedRow(row.source_row_number, row.line, "duplicate_row")
            )
            continue
        seen_rows.add(row.values)
        reordered = False
        if running_maximum is not None and timestamp < running_maximum:
            backward_ms = _milliseconds(running_maximum - timestamp)
            if backward_ms > SEGMENT_GAP_THRESHOLD_MS:
                audit.regression_rows.append((row.source_row_number, backward_ms))
                audit.quarantined.append(
                    QuarantinedRow(row.source_row_number, row.line, "timestamp_regression")
                )
                continue
            reordered = True
            audit.reordered_rows.append((row.source_row_number, backward_ms))
        if running_maximum is None or timestamp > running_maximum:
            running_maximum = timestamp
        audit.accepted.append(_TimedRow(timestamp, row, reordered))
    audit.accepted.sort(key=lambda timed: (timed.timestamp, timed.row.source_row_number))
    return audit


def normalize_session(raw: RawSession) -> NormalizedSession:
    """Map one parsed member to canonical samples with explicit flags and rejections."""
    order = _order_rows(raw)
    accepted = order.accepted
    if not accepted:
        return NormalizedSession(
            raw=raw,
            origin_raw_timestamp="",
            samples=[],
            quarantined=order.quarantined,
            rejections=[],
            segments=[],
            reordered_row_count=0,
            same_second_row_count=0,
            raw_interval_counts_ms=order.raw_intervals,
            reordered_rows=order.reordered_rows,
            regression_rows=order.regression_rows,
        )
    origin = accepted[0].timestamp
    elapsed_values = [_milliseconds(timed.timestamp - origin) for timed in accepted]
    elapsed_counts = Counter(elapsed_values)

    segment_ids: list[int] = []
    current_segment = 0
    for position, elapsed in enumerate(elapsed_values):
        if position > 0 and elapsed - elapsed_values[position - 1] > SEGMENT_GAP_THRESHOLD_MS:
            current_segment += 1
        segment_ids.append(current_segment)

    samples: list[NormalizedSample] = []
    rejections: list[ValueRejection] = []
    for timed, elapsed, segment_id in zip(accepted, elapsed_values, segment_ids, strict=True):
        values = timed.row.values
        flags: list[str] = []
        if timed.reordered:
            flags.append("reordered")
        if elapsed_counts[elapsed] > 1:
            flags.append("same_second")
        metric_values: list[float | None] = []
        for metric in METRIC_COLUMNS:
            raw_field = METRIC_DICTIONARY[metric].raw_field
            raw_value = values[FIELD_INDEX[raw_field]]
            value, reason, outside = convert_metric(metric.value, raw_value)
            if reason is not None:
                rejections.append(
                    ValueRejection(timed.row.source_row_number, raw_field, raw_value, reason)
                )
            if outside:
                flags.append(f"outside_documented_range:{metric.value}")
            metric_values.append(value)
        raw_state = _optional_text(values[FIELD_INDEX["State"]])
        network_mode = values[FIELD_INDEX["NetworkMode"]]
        serving_cell_id, basis = cell_identity(
            values[FIELD_INDEX["RAWCELLID"]],
            values[FIELD_INDEX["NODEHEX"]],
            values[FIELD_INDEX["CELLHEX"]],
        )
        samples.append(
            NormalizedSample(
                source_row_number=timed.row.source_row_number,
                elapsed_ms=elapsed,
                raw_timestamp=values[FIELD_INDEX["Timestamp"]],
                local_timestamp=timed.timestamp,
                segment_id=segment_id,
                download_state=map_state(raw_state),
                raw_state=raw_state,
                metric_values=tuple(metric_values),
                network_mode=None if network_mode in ("", DASH_MISSING) else network_mode,
                serving_cell_id=serving_cell_id,
                cell_identity_basis=basis,
                raw_cell_local_id=_optional_text(values[FIELD_INDEX["CellID"]]),
                raw_node_hex=_optional_text(values[FIELD_INDEX["NODEHEX"]]),
                raw_cell_hex=_optional_text(values[FIELD_INDEX["CELLHEX"]]),
                raw_area_hex=_optional_text(values[FIELD_INDEX["LACHEX"]]),
                quality_flags=tuple(flags),
            )
        )
    return NormalizedSession(
        raw=raw,
        origin_raw_timestamp=accepted[0].row.values[FIELD_INDEX["Timestamp"]],
        samples=samples,
        quarantined=order.quarantined,
        rejections=rejections,
        segments=_segments(samples),
        reordered_row_count=len(order.reordered_rows),
        same_second_row_count=sum(1 for sample in samples if "same_second" in sample.quality_flags),
        raw_interval_counts_ms=order.raw_intervals,
        reordered_rows=order.reordered_rows,
        regression_rows=order.regression_rows,
    )


def _segments(samples: list[NormalizedSample]) -> list[Segment]:
    """Segment spans in bin units: first bin start to last bin end, end exclusive."""
    grouped: dict[int, list[int]] = {}
    for sample in samples:
        grouped.setdefault(sample.segment_id, []).append(sample.elapsed_ms)
    segments: list[Segment] = []
    for segment_id in sorted(grouped):
        elapsed_values = grouped[segment_id]
        start = (min(elapsed_values) // BIN_MS) * BIN_MS
        end = (max(elapsed_values) // BIN_MS) * BIN_MS + BIN_MS
        segments.append(Segment(segment_id, start, end, len(elapsed_values)))
    return segments


def iter_sessions(source: SourceArchive) -> Iterator[NormalizedSession]:
    """Parse and normalize members one at a time, in member path order."""
    for member in source.inventory.members:
        raw = parse_member(member, read_member(member), source.source_id)
        yield normalize_session(raw)
