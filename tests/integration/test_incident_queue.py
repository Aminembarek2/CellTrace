"""Queue scope, input freezes and crash recovery use the actual runner and stores."""

import json
from pathlib import Path
from typing import Any

import pytest

from celltrace.cli import build_adapter
from celltrace.config import load_config
from celltrace.contracts.common import Arm
from celltrace.data.snapshot import load_manifest
from celltrace.workflow import queue as queue_module

ARM = Arm.investigator_reviewer_gated


@pytest.fixture
def queued(fixture_snapshot: Any, tmp_path: Path) -> tuple[Path, Path]:
    candidates = fixture_snapshot.candidates[:2]
    path = tmp_path / "queue.json"
    path.write_text(
        json.dumps(
            {
                "generated_by": "fixture",
                "snapshot_id": load_manifest(fixture_snapshot.snapshot_dir)["snapshot_id"],
                "detector_config_hash": candidates[0].detector_config_hash,
                "selection": "private fixture selection; must not enter model context",
                "incident_assignments": [
                    {"event_id": candidate.event_id, "session_id": candidate.session_id}
                    for candidate in candidates
                ],
                "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            }
        )
    )
    return path, fixture_snapshot.snapshot_dir


def _run(queued: tuple[Path, Path], directory: Path) -> dict[str, Any]:
    config = load_config(environment={})
    adapter, _ = build_adapter(config, False)
    path = queue_module.run_queue(*queued, directory, config, ARM, adapter)
    return json.loads(path.read_text())


def test_complete_queue_rerun_does_not_repeat_calls_or_leak_selection(
    queued: tuple[Path, Path], tmp_path: Path
) -> None:
    first = _run(queued, tmp_path / "run")
    second = _run(queued, tmp_path / "run")
    assert first == second
    assert first["selected_candidates"] == first["attempted_candidates"] == 2
    assert first["nonempty_fact_checked_reports"] == 2
    assert first["independently_checked_reports"] == 2
    assert first["accuracy"] is None and first["mode"] == "scripted_fixture"
    import sqlite3

    with sqlite3.connect(tmp_path / "run" / "runs.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 2
        contexts = connection.execute("SELECT context_json FROM runs").fetchall()
        messages = connection.execute("SELECT message_json FROM agent_messages").fetchall()
    assert "private fixture selection" not in json.dumps(contexts + messages)
    assert "incident_assignments" not in json.dumps(contexts + messages)


@pytest.mark.parametrize(
    "mutation", ["snapshot", "candidate", "detector", "duplicate", "assignment"]
)
def test_queue_tampering_refused_before_adapter_use(
    queued: tuple[Path, Path], tmp_path: Path, mutation: str
) -> None:
    path, snapshot = queued
    payload = json.loads(path.read_text())
    if mutation == "snapshot":
        payload["snapshot_id"] = "snap_other"
    elif mutation == "candidate":
        payload["candidates"][0]["reference_median_mbps"] += 1
    elif mutation == "detector":
        payload["detector_config_hash"] = "changed"
    elif mutation == "duplicate":
        payload["candidates"].append(payload["candidates"][0])
    else:
        payload["incident_assignments"][0]["session_id"] = "another_session"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        queue_module.validate_queue(path, snapshot, load_config(environment={}))
    assert not (tmp_path / "run").exists()


def test_changed_queue_refuses_resume_without_new_ledger(
    queued: tuple[Path, Path], tmp_path: Path
) -> None:
    _run(queued, tmp_path / "run")
    path, _ = queued
    payload = json.loads(path.read_text())
    payload["selection"] = "changed selection"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inputs changed"):
        _run(queued, tmp_path / "run")


def test_crash_after_ledger_report_recovers_without_repeating_provider_attempts(
    queued: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = queue_module.write_report_files

    def interrupted(*args: Any) -> None:
        raise OSError("simulated report filesystem interruption")

    monkeypatch.setattr(queue_module, "write_report_files", interrupted)
    first = _run(queued, tmp_path / "run")
    index = json.loads((tmp_path / "run" / "index.json").read_text())
    assert first["dispatch_status"] == "blocked_crash"
    attempted_before = index["runs"][0]["usage"]["provider_attempts"]
    monkeypatch.setattr(queue_module, "write_report_files", original)
    second = _run(queued, tmp_path / "run")
    index = json.loads((tmp_path / "run" / "index.json").read_text())
    assert second["dispatch_status"] == "finished"
    assert index["runs"][0]["usage"]["provider_attempts"] == attempted_before
    assert len(index["dispatch_failures"]) == 1


def test_unknown_usage_in_ledger_blocks_further_dispatch(
    queued: tuple[Path, Path], tmp_path: Path
) -> None:
    from celltrace.providers.base import FailureKind

    config = load_config(environment={})
    adapter, _ = build_adapter(config, False)
    adapter.failures = {1: FailureKind.uncertain}
    path = queue_module.run_queue(*queued, tmp_path / "run", config, ARM, adapter)
    first = json.loads(path.read_text())
    assert first["unknown_usage_attempts"] == 1
    # Same adapter identity, but no new dispatch on resume.
    summary = json.loads(
        queue_module.run_queue(*queued, tmp_path / "run", config, ARM, adapter).read_text()
    )
    assert summary["dispatch_status"] == "blocked_unknown_usage"
    assert summary["recorded_candidates"] == 1
    assert adapter.attempts == 1


def test_queue_lock_prevents_concurrent_dispatch(queued: tuple[Path, Path], tmp_path: Path) -> None:
    from celltrace.storage.lock import RunLock, RunLockedError

    with RunLock(tmp_path / "run" / "locks", "queue"), pytest.raises(RunLockedError):
        _run(queued, tmp_path / "run")


def test_shared_monetary_allowance_prevents_remaining_queue_dispatch(
    queued: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from celltrace.storage.run_store import ActionKind, BudgetExhausted, RunStore

    # Exercise queue routing when the existing ledger rejects admission before
    # dispatch. Transactional monetary calculations are covered by store tests.
    original = RunStore.reserve_action

    def reject(self: RunStore, *args: Any, **kwargs: Any) -> Any:
        if args[4] == ActionKind.provider_attempt:
            raise BudgetExhausted("experiment_estimated_usd", 1, 1, 0.005)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RunStore, "reserve_action", reject)
    summary = _run(queued, tmp_path / "run")
    assert summary["provider_attempts"] == 0
    assert summary["selected_candidates"] == 2
    assert summary["outcomes"] == {"exhausted": 1, "budget_prevented": 1}
    assert summary["known_or_reserved_usd"] == 0
