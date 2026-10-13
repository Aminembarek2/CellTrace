"""Shared fixture helpers: the synthetic archive, its snapshot and session ids."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from celltrace.data.fixture import FIXTURE_SESSIONS, FIXTURE_SOURCE_ID, write_fixture_archive
from celltrace.data.raca import make_session_id
from celltrace.data.snapshot import open_snapshot, run_ingest


@pytest.fixture(scope="session")
def fixture_archive_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory holding the synthetic zip and its SOURCE.json record."""
    directory = tmp_path_factory.mktemp("fixture_raw")
    write_fixture_archive(directory)
    return directory


@pytest.fixture(scope="session")
def fixture_snapshot_dir(
    fixture_archive_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Snapshot directory built once per test session from the synthetic archive."""
    return run_ingest(fixture_archive_dir, tmp_path_factory.mktemp("fixture_processed"))


@pytest.fixture
def fixture_connection(fixture_snapshot_dir: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Read only connection to the fixture snapshot, closed after the test."""
    connection, _ = open_snapshot(fixture_snapshot_dir)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def fixture_session_ids() -> dict[str, str]:
    """Fixture member path to opaque session id."""
    return {member: make_session_id(FIXTURE_SOURCE_ID, member) for member in FIXTURE_SESSIONS}
