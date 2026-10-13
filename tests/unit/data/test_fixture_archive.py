"""The synthetic fixture archive is deterministic and documents every member."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from celltrace.data import fixture
from celltrace.data.raca import RACA_HEADER, discover_members


def test_fixture_bytes_are_stable(tmp_path: Path) -> None:
    first = fixture.write_fixture_archive(tmp_path / "first")
    second = fixture.write_fixture_archive(tmp_path / "second")
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
    record = json.loads((tmp_path / "first" / "SOURCE.json").read_text())
    assert record["synthetic"] is True
    assert (
        record["files"][fixture.FIXTURE_ARCHIVE_NAME]["sha256"]
        == hashlib.sha256(first.read_bytes()).hexdigest()
    )


def test_every_member_is_documented(tmp_path: Path) -> None:
    archive = fixture.write_fixture_archive(tmp_path)
    members = [member.member_path for member in discover_members(archive)]
    assert members == sorted(fixture.FIXTURE_SESSIONS)
    assert all(description for description in fixture.FIXTURE_SESSIONS.values())


def test_members_use_the_raca_header() -> None:
    for content in fixture.fixture_member_bytes().values():
        assert content.decode().splitlines()[0] == ",".join(RACA_HEADER)
