"""Investigate an existing detector queue without changing or labelling its rows.

Selection metadata stays outside the runtime tool surface. The ordinary runner owns
all provider calls, verification and accounting. This module owns only serial dispatch,
input freezing, crash reconciliation and descriptive aggregation without truth labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from celltrace.analysis.detector import detect_candidates, detector_config_hash
from celltrace.canonical import content_hash, file_sha256
from celltrace.config import AppConfig
from celltrace.contracts.common import Arm
from celltrace.contracts.evidence import CandidateEvent
from celltrace.contracts.report import InvestigationReport
from celltrace.data.snapshot import open_snapshot
from celltrace.providers.base import ModelAdapter
from celltrace.providers.rate_card import RateCard
from celltrace.release import source_tree_hash
from celltrace.storage.lock import RunLock
from celltrace.storage.run_store import RunNotFound
from celltrace.workflow.runner import InvestigationRunner, RunOptions, write_report_files

QUEUE_VERSION = "incident_queue_1"


class InvestigationQueue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generated_by: str
    snapshot_id: str
    detector_config_hash: str
    selection: str
    incident_assignments: list[dict[str, Any]]
    candidates: list[CandidateEvent] = Field(min_length=1)


def validate_queue(
    queue_path: Path, snapshot: Path, config: AppConfig
) -> tuple[InvestigationQueue, dict[str, Any]]:
    queue = InvestigationQueue.model_validate_json(queue_path.read_text())
    connection, manifest = open_snapshot(snapshot)
    try:
        if queue.snapshot_id != manifest["snapshot_id"]:
            raise ValueError("queue belongs to a different snapshot")
        if queue.detector_config_hash != detector_config_hash(config.analysis.detector):
            raise ValueError("queue detector configuration differs from this configuration")
        ids = [candidate.event_id for candidate in queue.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("queue has duplicate event IDs")
        original = {
            candidate.event_id: candidate
            for candidate in detect_candidates(connection, config.analysis.detector).candidates
        }
        for candidate in queue.candidates:
            if candidate != original.get(candidate.event_id):
                raise ValueError("queued candidate differs from the snapshot's detector result")
        assignments = queue.incident_assignments
        if {item["event_id"] for item in assignments} != set(ids):
            raise ValueError("incident assignments and queue candidates differ")
        sessions = {candidate.event_id: candidate.session_id for candidate in queue.candidates}
        if any(item["session_id"] != sessions[item["event_id"]] for item in assignments):
            raise ValueError("incident assignment belongs to another session")
        return queue, manifest
    finally:
        connection.close()


def queue_identity(
    queue_path: Path,
    snapshot: Path,
    directory: Path,
    config: AppConfig,
    arm: Arm,
    live: bool,
) -> tuple[InvestigationQueue, dict[str, Any], AppConfig]:
    queue, manifest = validate_queue(queue_path, snapshot, config)
    resolved = config.model_dump(mode="json")
    resolved["runtime"]["run_store"] = str(directory / "runs.sqlite3")
    resolved["runtime"]["checkpoint_store"] = str(directory / "checkpoints.sqlite3")
    resolved["runtime"]["enable_live_calls"] = live
    run_config = AppConfig.model_validate(resolved)
    provider_inputs = {
        name: None if path is None else file_sha256(path)
        for name in ("verified_rate_card", "verified_endpoint_record", "verified_capability_record")
        if (path := getattr(run_config.runtime, name)) is not None or live
    }
    identity = {
        "version": QUEUE_VERSION,
        "source_sha256": source_tree_hash(),
        "lock_sha256": file_sha256(Path("uv.lock")),
        "queue_sha256": file_sha256(queue_path),
        "snapshot_logical_hash": manifest["logical_hash"],
        "snapshot_file_sha256": manifest["file_sha256"],
        "snapshot_manifest_sha256": file_sha256(snapshot / "snapshot_manifest.json"),
        "snapshot_path": str(snapshot.resolve()),
        "config": run_config.model_dump(mode="json"),
        "provider_inputs": provider_inputs,
        "arm": arm.value,
        "mode": "live" if live else "scripted_fixture",
        "order": [candidate.event_id for candidate in queue.candidates],
        "grading": "descriptive execution and code checks; no causal or semantic truth labels",
    }
    return queue, identity, run_config


def _save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _entry(candidate: CandidateEvent, report: InvestigationReport) -> dict[str, Any]:
    return {
        "event_id": candidate.event_id,
        "run_id": report.run_id,
        "status": "ran",
        "workflow_outcome": report.workflow_outcome.value,
        "fact_check_status": report.fact_check_status.value,
        "review_status": report.review_status.value,
        "cause_status": report.cause_status.value,
        "verified_claims": len(report.verified_claims),
        "verified_observation_patterns": report.verified_observation_patterns,
        "terminal_reason": report.terminal_reason,
        "usage": report.usage.model_dump(mode="json"),
    }


def queue_summary(index: dict[str, Any], used_usd: float, maximum_usd: float) -> dict[str, Any]:
    entries = index["runs"]
    attempted = [entry for entry in entries if entry["status"] != "budget_prevented"]
    return {
        "mode": index["identity"]["mode"],
        "selected_candidates": len(index["identity"]["order"]),
        "recorded_candidates": len(entries),
        "attempted_candidates": len(attempted),
        "dispatch_status": index["status"],
        "outcomes": dict(Counter(row.get("workflow_outcome", row["status"]) for row in entries)),
        "nonempty_fact_checked_reports": sum(
            row.get("fact_check_status") == "passed" and row.get("verified_claims", 0) > 0
            for row in entries
        ),
        "independently_checked_reports": sum(
            row.get("review_status") == "independently_checked" for row in entries
        ),
        "checked_drop_reports": sum(
            "active_rate_drop" in row.get("verified_observation_patterns", []) for row in entries
        ),
        "verified_claims": sum(row.get("verified_claims", 0) for row in entries),
        "provider_attempts": sum(
            row.get("usage", {}).get("provider_attempts", 0) for row in entries
        ),
        "unknown_usage_attempts": sum(
            row.get("usage", {}).get("unknown_usage_attempts", 0) for row in entries
        ),
        "active_seconds": sum(row.get("usage", {}).get("active_ms", 0) for row in entries) / 1000,
        "known_or_reserved_usd": used_usd,
        "maximum_estimated_usd": maximum_usd,
        "accuracy": None,
        "limitation": "No incident truth labels; verified facts do not establish physical cause "
        "or semantic accuracy. Scripted results measure orchestration only.",
    }


def run_queue(
    queue_path: Path,
    snapshot: Path,
    directory: Path,
    config: AppConfig,
    arm: Arm,
    adapter: ModelAdapter,
    rate_card: RateCard | None = None,
) -> Path:
    live = adapter.execution_mode.value == "live"
    queue, identity, run_config = queue_identity(queue_path, snapshot, directory, config, arm, live)
    identity["adapter"] = adapter.describe()
    if live != (rate_card is not None):
        raise ValueError("live queues require a rate card; scripted queues must not have one")
    with RunLock(directory / "locks", "queue"):
        path = directory / "index.json"
        experiment_id = f"exp_queue_{content_hash(identity)[:16]}"
        if path.exists():
            index = json.loads(path.read_text())
            if index["identity"] != identity:
                raise ValueError("queue inputs changed; existing runs and allowance are preserved")
        else:
            # Lost index recovery must not create an experiment with new counters.
            if (directory / "runs.sqlite3").exists():
                raise ValueError(
                    "queue index missing beside an existing ledger; restore its freeze"
                )
            index = {
                "experiment_id": experiment_id,
                "identity": identity,
                "runs": [],
                "dispatch_failures": [],
                "status": "running",
            }
            _save(path, index)
        runner = InvestigationRunner(run_config, adapter, rate_card=rate_card)
        try:
            runner.store.create_experiment(
                experiment_id, run_config.runtime.maximum_estimated_usd_per_experiment, identity
            )
            entries = {row["event_id"]: row for row in index["runs"]}
            budget_stopped = any(
                row.get("terminal_reason", "").startswith("budget_exhausted:experiment")
                for row in entries.values()
            )
            index["status"] = "running"
            _save(path, index)
            for candidate in queue.candidates:
                previous = entries.get(candidate.event_id)
                if previous is not None:
                    # Terminal uncertain results are preserved, and stop further dispatch.
                    if (
                        previous["status"] != "budget_prevented"
                        and runner.store.usage(previous["run_id"]).unknown_usage_attempts
                    ):
                        index["status"] = "blocked_unknown_usage"
                        break
                    if previous["status"] in {"ran", "budget_prevented"}:
                        continue
                if (directory / "STOP").exists():
                    index["status"] = "stopped_by_operator"
                    break
                run_id = f"run_queue_{content_hash([experiment_id, candidate.event_id])[:20]}"
                try:
                    try:
                        runner.store.get_run(run_id)
                    except RunNotFound:
                        used, maximum = runner.store.experiment_usd(experiment_id)
                        if budget_stopped or (live and used >= maximum):
                            entries[candidate.event_id] = {
                                "event_id": candidate.event_id,
                                "run_id": run_id,
                                "status": "budget_prevented",
                            }
                            index["runs"] = list(entries.values())
                            _save(path, index)
                            continue
                        report = runner.start(
                            snapshot,
                            candidate,
                            RunOptions(arm=arm, run_id=run_id, experiment_id=experiment_id),
                        )
                    else:
                        report = runner.resume(run_id)
                    write_report_files(report, directory / "reports")
                    entries[candidate.event_id] = _entry(candidate, report)
                    if report.terminal_reason.startswith("budget_exhausted:experiment"):
                        budget_stopped = True
                    print(json.dumps(entries[candidate.event_id]), flush=True)
                except Exception as failure:
                    entries[candidate.event_id] = {
                        "event_id": candidate.event_id,
                        "run_id": run_id,
                        "status": "crashed",
                        "error": f"{type(failure).__name__}: {failure}",
                    }
                    index["status"] = "blocked_crash"
                    index["dispatch_failures"].append(dict(entries[candidate.event_id]))
                    with suppress(RunNotFound):
                        entries[candidate.event_id]["usage"] = asdict(runner.store.usage(run_id))
                index["runs"] = list(entries.values())
                _save(path, index)
                if index["status"] == "blocked_crash":
                    break
                if report.usage.unknown_usage_attempts:
                    index["status"] = "blocked_unknown_usage"
                    break
            else:
                index["status"] = "finished"
            _save(path, index)
            used, maximum = runner.store.experiment_usd(experiment_id)
            summary = queue_summary(index, used, maximum)
            _save(directory / "summary.json", summary)
        finally:
            runner.close()
    return directory / "summary.json"


def command_queue(arguments: argparse.Namespace) -> int:
    from celltrace.cli import (
        EXIT_BLOCKED,
        EXIT_OK,
        CommandError,
        _config,
        _print_json,
        build_adapter,
    )

    config = _config(arguments, live=arguments.live)
    snapshot, queue_path, directory = (
        Path(arguments.snapshot),
        Path(arguments.queue),
        Path(arguments.out),
    )
    queue, identity, _ = queue_identity(
        queue_path, snapshot, directory, config, Arm(arguments.variant), arguments.live
    )
    if arguments.preflight:
        _print_json(
            {"status": "ready", "distinct_candidates": len(queue.candidates), "identity": identity}
        )
        return EXIT_OK
    adapter, rate_card = build_adapter(config, arguments.live)
    result = run_queue(
        queue_path, snapshot, directory, config, Arm(arguments.variant), adapter, rate_card
    )
    summary = json.loads(result.read_text())
    _print_json({"summary": str(result), **summary})
    if summary["dispatch_status"] != "finished":
        raise CommandError("queue stopped; see its preserved index and ledger", EXIT_BLOCKED)
    return EXIT_OK


def register(subparsers: Any) -> None:
    from celltrace.cli import _add_common

    command = subparsers.add_parser(
        "investigate-queue", help="investigate an existing incident queue"
    )
    _add_common(command)
    command.add_argument("--queue", required=True)
    command.add_argument("--snapshot", required=True)
    command.add_argument("--out", required=True)
    command.add_argument(
        "--variant",
        choices=[arm.value for arm in Arm],
        default=Arm.investigator_reviewer_gated.value,
    )
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--live", action="store_true")
    command.add_argument(
        "--preflight", action="store_true", help="validate inputs without provider dispatch"
    )
    command.set_defaults(handler=command_queue)
