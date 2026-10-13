"""Shared fixture snapshot and candidates for integration tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from celltrace.analysis.detector import run_detect
from celltrace.clock import ManualClock
from celltrace.config import AppConfig, load_config
from celltrace.contracts.common import Arm
from celltrace.contracts.evidence import CandidateEvent
from celltrace.data.fixture import (
    RADIO_DROP_WITH_CELL_CHANGE_MEMBER,
    STABLE_RADIO_DROP_MEMBER,
    write_fixture_archive,
)
from celltrace.data.snapshot import open_snapshot, run_ingest
from celltrace.storage.run_store import RunStore
from celltrace.tools.context import build_run_context
from celltrace.tools.dispatch import ToolDispatcher
from celltrace.tools.repository import MeasurementRepository


@dataclass(frozen=True)
class FixtureSnapshot:
    snapshot_dir: Path
    snapshot_hash: str
    session_by_member: dict[str, str]
    candidates: list[CandidateEvent]

    def first_candidate(self, member: str) -> CandidateEvent:
        session_id = self.session_by_member[member]
        matches = [candidate for candidate in self.candidates if candidate.session_id == session_id]
        return min(matches, key=lambda candidate: candidate.first_confirmation_ms)


@pytest.fixture(scope="session")
def fixture_snapshot(tmp_path_factory: pytest.TempPathFactory) -> FixtureSnapshot:
    base = tmp_path_factory.mktemp("fixture_snapshot")
    write_fixture_archive(base / "raw")
    snapshot_dir = run_ingest(base / "raw", base / "processed")
    candidates_path = run_detect(
        snapshot_dir, base / "candidates", load_config(environment={}).analysis.detector
    )
    candidates = [
        CandidateEvent.model_validate(entry)
        for entry in json.loads(candidates_path.read_text())["candidates"]
    ]
    connection, manifest = open_snapshot(snapshot_dir)
    try:
        sessions = connection.execute(
            "SELECT session_id, source_relative_path FROM sessions"
        ).fetchall()
    finally:
        connection.close()
    return FixtureSnapshot(
        snapshot_dir=snapshot_dir,
        snapshot_hash=manifest["logical_hash"],
        session_by_member={str(row[1]): str(row[0]) for row in sessions},
        candidates=candidates,
    )


def make_config(tmp_path: Path, overrides: dict[tuple[str, ...], Any] | None = None) -> AppConfig:
    base = {
        ("runtime", "run_store"): str(tmp_path / "runs.sqlite3"),
        ("runtime", "checkpoint_store"): str(tmp_path / "checkpoints.sqlite3"),
    }
    return load_config(cli_overrides={**base, **(overrides or {})}, environment={})


@dataclass
class ToolHarness:
    dispatcher: ToolDispatcher
    store: RunStore
    repository: MeasurementRepository
    context: Any
    connection: Any


@pytest.fixture
def stable_radio_harness(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path
) -> Iterator[ToolHarness]:
    yield from _harness(fixture_snapshot, tmp_path, STABLE_RADIO_DROP_MEMBER, "run_tools")


@pytest.fixture
def radio_drop_harness(fixture_snapshot: FixtureSnapshot, tmp_path: Path) -> Iterator[ToolHarness]:
    yield from _harness(fixture_snapshot, tmp_path, RADIO_DROP_WITH_CELL_CHANGE_MEMBER, "run_radio")


def _harness(
    fixture_snapshot: FixtureSnapshot, tmp_path: Path, member: str, run_id: str
) -> Iterator[ToolHarness]:
    from celltrace.storage.run_store import RunLimits

    config = make_config(tmp_path)
    connection, manifest = open_snapshot(fixture_snapshot.snapshot_dir)
    repository = MeasurementRepository(connection)
    candidate = fixture_snapshot.first_candidate(member)
    context = build_run_context(
        run_id,
        "case_test",
        candidate,
        manifest["logical_hash"],
        str(fixture_snapshot.snapshot_dir),
        repository,
        config.runtime,
    )
    clock = ManualClock()
    store = RunStore(config.runtime.run_store, clock)
    store.create_run(
        run_id,
        "case_test",
        candidate.event_id,
        Arm.investigator_gated.value,
        "scripted_fixture",
        None,
        context.snapshot_hash,
        {"test": True},
        context.model_dump(mode="json"),
        RunLimits(12, 12, 80_000, 12_000, None, 300_000, 2, 1),
    )
    try:
        yield ToolHarness(
            ToolDispatcher(context, repository, store, clock),
            store,
            repository,
            context,
            connection,
        )
    finally:
        store.close()
        connection.close()
