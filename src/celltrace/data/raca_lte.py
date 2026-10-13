"""Source adapter for the Raca Irish 4G LTE dataset of 2017 and 2018 (MMSys 2018).

Same logger and nearly the same header as the 5G archive. Rows are placed into the 5G
field layout, fields this dataset lacks become the dash missing sentinel, and the unchanged
normalization runs on them. Two rules belong to this dataset only, both recorded before any
rate was computed (docs/TECHNICAL_REPORT.md):

- The operator comes from the file name, A_ or B_, and each operator gets its own snapshot.
- CellID is the cell number within a site, so it does not identify a cell alone. The serving
  cell is technology family, the serving eNodeB coordinates the authors looked up in
  OpenCellID, and the local number, kept only when the row reports the file's operator. The
  identity basis stays unknown, because no layout can be verified.
"""

from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from celltrace.contracts.common import IdentityBasis
from celltrace.data import raca
from celltrace.data.raca import (
    ArchiveInventory,
    ArchiveMember,
    NormalizedSession,
    RawRow,
    RawSession,
    SkippedEntry,
    SourceArchive,
)

LTE_HEADER: tuple[str, ...] = (
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
    "NRxRSRP",
    "NRxRSRQ",
    "ServingCell_Lon",
    "ServingCell_Lat",
    "ServingCell_Distance",
)
POSITION = {name: index for index, name in enumerate(LTE_HEADER)}
OPERATORS = ("A", "B")
SOURCE_ID_PREFIX = "raca_irish_lte_2018"
APPLICATION_COHORT = "file_download"
MEMBER_PATTERN = re.compile(
    r"Dataset/(bus|car|pedestrian|static|train)/([AB])_\d{4}\.\d{2}\.\d{2}_\d{2}\.\d{2}\.\d{2}\.csv"
)
TECHNOLOGY_FAMILIES = {
    "LTE": "LTE",
    "HSPA+": "3G",
    "HSUPA": "3G",
    "HSDPA": "3G",
    "UMTS": "3G",
    "EDGE": "2G",
    "GPRS": "2G",
}


def source_id(operator: str) -> str:
    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of {OPERATORS}")
    return f"{SOURCE_ID_PREFIX}_operator_{operator.lower()}"


def inspect_archive(archive_path: Path, operator: str) -> ArchiveInventory:
    """Members of one operator, with names checked and every selected member hashed."""
    members: list[ArchiveMember] = []
    skipped: list[SkippedEntry] = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        for info in infos:
            raca._check_member_name(info.filename)
            if info.flag_bits & 0x1:
                raise raca.ArchiveSafetyError(f"encrypted archive member: {info.filename!r}")
        total = sum(info.file_size for info in infos)
        if total > raca.MAXIMUM_ARCHIVE_UNCOMPRESSED_BYTES:
            raise raca.ArchiveSafetyError(f"{archive_path.name}: {total} bytes exceed the ceiling")
        for info in sorted(infos, key=lambda entry: entry.filename):
            match = MEMBER_PATTERN.fullmatch(info.filename)
            if info.filename.endswith("/"):
                skipped.append(SkippedEntry(info.filename, "directory"))
            elif match is None:
                skipped.append(SkippedEntry(info.filename, "outside_dataset_layout"))
            elif match.group(2) != operator:
                skipped.append(SkippedEntry(info.filename, "other_operator"))
            else:
                content = raca._read_entry(archive, info)
                members.append(
                    ArchiveMember(
                        archive_path=archive_path,
                        member_path=info.filename,
                        uncompressed_bytes=info.file_size,
                        compressed_bytes=info.compress_size,
                        sha256=hashlib.sha256(content).hexdigest(),
                        application_cohort=APPLICATION_COHORT,
                        mobility_cohort=match.group(1),
                    )
                )
    return ArchiveInventory(
        archive_path=archive_path,
        archive_sha256=raca._sha256_file(archive_path),
        archive_bytes=archive_path.stat().st_size,
        entry_count=len(infos),
        total_uncompressed_bytes=total,
        members=tuple(members),
        skipped=tuple(skipped),
    )


def source_archive(input_dir: Path, operator: str) -> SourceArchive:
    archives = sorted(path for path in input_dir.rglob("*.zip") if path.is_file())
    if len(archives) != 1:
        raise FileNotFoundError(f"expected one LTE dataset archive under {input_dir}")
    return SourceArchive(
        archive_path=archives[0],
        relative_path=archives[0].relative_to(input_dir).as_posix(),
        source_id=source_id(operator),
        source_record=raca.load_source_record(archives[0]),
        inventory=inspect_archive(archives[0], operator),
    )


def to_archive_layout(values: tuple[str, ...]) -> tuple[str, ...]:
    """Place a well formed row into the 5G header order. Malformed rows stay as they are."""
    if len(values) != len(LTE_HEADER):
        return values
    mapped = []
    for name in raca.RACA_HEADER:
        if name == "RAWCELLID":
            mapped.append(values[POSITION["CellID"]])
        elif name in POSITION:
            mapped.append(values[POSITION[name]])
        else:
            mapped.append(raca.DASH_MISSING)
    return tuple(mapped)


def parse_member(member: ArchiveMember, content: bytes, source_identifier: str) -> RawSession:
    encoding = "utf-8-sig" if content.startswith(b"\xef\xbb\xbf") else "utf-8"
    text = content.decode(encoding)
    line_terminator = "CRLF" if "\r\n" in text else "LF"
    lines = [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]
    if lines and lines[-1] == "":
        lines.pop()
    header = tuple(next(csv.reader([lines[0]]))) if lines else ()
    if header != LTE_HEADER:
        raise raca.UnexpectedHeaderError(member.member_path, header)
    rows = [
        RawRow(number, line, to_archive_layout(tuple(next(csv.reader([line]), []))))
        for number, line in enumerate(lines[1:], start=1)
    ]
    return RawSession(
        session_id=raca.make_session_id(source_identifier, member.member_path),
        source_id=source_identifier,
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


def proxy_cell(
    local_number: str | None,
    network_mode: str | None,
    row_operator: str,
    longitude: str,
    latitude: str,
    file_operator: str,
) -> str | None:
    """Technology family, site coordinates and local number, or None when unusable."""
    if row_operator != file_operator or network_mode is None:
        return None
    if network_mode not in TECHNOLOGY_FAMILIES:
        return None
    if local_number is None or not local_number.isdigit() or int(local_number) == 0:
        return None
    if longitude in ("", raca.DASH_MISSING) or latitude in ("", raca.DASH_MISSING):
        return None
    return f"{TECHNOLOGY_FAMILIES[network_mode]}:{longitude},{latitude}:{int(local_number)}"


def apply_identity_rule(session: NormalizedSession, file_operator: str) -> NormalizedSession:
    rows = {row.source_row_number: row for row in session.raw.rows}
    samples = []
    for sample in session.samples:
        # The serving cell coordinates are not part of the 5G layout, so read the source line.
        original = next(csv.reader([rows[sample.source_row_number].line]))
        cell = proxy_cell(
            sample.raw_cell_local_id,
            sample.network_mode,
            original[POSITION["Operatorname"]],
            original[POSITION["ServingCell_Lon"]],
            original[POSITION["ServingCell_Lat"]],
            file_operator,
        )
        samples.append(
            replace(sample, serving_cell_id=cell, cell_identity_basis=IdentityBasis.unknown.value)
        )
    return replace(session, samples=samples)


def iter_sessions(source: SourceArchive) -> Iterator[NormalizedSession]:
    operator = source.source_id.rsplit("_", 1)[-1].upper()
    for member in source.inventory.members:
        raw = parse_member(member, raca.read_member(member), source.source_id)
        yield apply_identity_rule(raca.normalize_session(raw), operator)
