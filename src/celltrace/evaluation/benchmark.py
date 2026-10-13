"""Run every case under every arm with one model, one rubric and shared limits.

The runner gives each run only the case snapshot and its fixed candidate. The
private manifest is read here, outside the runtime, to find case directories
and environment faults. Every attempted run is recorded, including crashes
and runs the shared experiment budget prevented.
"""

from __future__ import annotations

import json
import random
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from celltrace.agents.roles import role_identity
from celltrace.agents.scripted_policies import ScriptedInvestigator, ScriptedReviewer
from celltrace.canonical import content_hash, file_sha256
from celltrace.clock import SystemClock
from celltrace.config import AppConfig, load_config
from celltrace.contracts.common import Arm, ErrorCode
from celltrace.contracts.evidence import CandidateEvent
from celltrace.providers.base import ModelAdapter
from celltrace.providers.rate_card import RateCard
from celltrace.providers.scripted import ScriptedModelAdapter
from celltrace.storage.run_store import RunStore
from celltrace.tools.dispatch import TOOL_VERSION
from celltrace.tools.faults import failing_dispatcher_factory
from celltrace.workflow.runner import InvestigationRunner, RunOptions

AdapterFactory = Callable[[], tuple[ModelAdapter, RateCard | None]]


def offline_adapter_factory() -> tuple[ModelAdapter, RateCard | None]:
    """Deterministic template policy. A harness check and a no model baseline."""
    adapter = ScriptedModelAdapter(
        {"investigator": ScriptedInvestigator(), "reviewer": ScriptedReviewer()}
    )
    return adapter, None


def experiment_directory(manifest_path: Path, experiment_id: str) -> Path:
    return manifest_path.parent.parent / "experiments" / experiment_id


def freeze_record(
    manifest_path: Path, config: AppConfig, arms: list[Arm], mode: str
) -> dict[str, Any]:
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "roles": role_identity(),
        "tool_version": TOOL_VERSION,
        "runtime_limits": config.runtime.model_dump(mode="json"),
        "detector_config_hash": content_hash(config.analysis.detector.model_dump(mode="json")),
        "arms": [arm.value for arm in arms],
        "mode": mode,
    }


class FreezeMismatch(RuntimeError):
    pass


def verify_freeze(frozen: dict[str, Any], config: AppConfig | None = None) -> None:
    """Refuse to evaluate when anything frozen with the held out cases has changed."""
    from celltrace.evaluation.commands import freeze_hashes

    current = freeze_hashes(config)
    changed = sorted(
        key for key in current if key != "frozen_at" and current[key] != frozen.get(key)
    )
    if changed:
        raise FreezeMismatch(f"frozen evaluation inputs changed since the freeze: {changed}")


def run_benchmark(
    manifest_path: Path,
    mode: str,
    arms: list[Arm],
    adapter_factory: AdapterFactory,
    config: AppConfig | None = None,
    experiment_id: str | None = None,
    case_limit: int | None = None,
) -> Path:
    config = config or load_config()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("frozen") and manifest["split"] == "heldout":
        verify_freeze(manifest["freeze"], config)
    record = freeze_record(manifest_path, config, arms, mode)
    record["case_limit"] = case_limit
    experiment_id = experiment_id or f"exp_{mode}_{content_hash(record)[:12]}"
    directory = experiment_directory(manifest_path, experiment_id)
    directory.mkdir(parents=True, exist_ok=True)
    resolved = config.model_dump(mode="json")
    resolved["runtime"]["run_store"] = str(directory / "runs.sqlite3")
    resolved["runtime"]["checkpoint_store"] = str(directory / "checkpoints.sqlite3")
    run_config = AppConfig.model_validate(resolved)
    store = RunStore(run_config.runtime.run_store, SystemClock())
    store.create_experiment(
        experiment_id, config.runtime.maximum_estimated_usd_per_experiment, record
    )
    store.close()

    cases = manifest["cases"][: case_limit or None]
    pairs = [(case, arm) for case in cases for arm in arms]
    random.Random(f"{config.project.seed}:{experiment_id}").shuffle(pairs)
    cases_dir = manifest_path.parent.parent / "cases"
    index_path = directory / "index.json"
    existing_index = json.loads(index_path.read_text()) if index_path.exists() else None
    if existing_index is not None and content_hash(existing_index["freeze"]) != content_hash(
        record
    ):
        raise FreezeMismatch("experiment inputs changed, choose a new experiment id")
    entries = existing_index["runs"] if existing_index is not None else []
    done = {(entry["case_id"], entry["arm"]) for entry in entries}
    budget_stopped = False
    # An operator creates this file to stop between runs, so no provider call is cut off.
    stop_file = directory / "STOP"
    stopped_by_operator = False
    for order, (case, arm) in enumerate(pairs):
        if (case["case_id"], arm.value) in done:
            continue
        if stop_file.exists():
            stopped_by_operator = True
            break
        run_id = f"run_{experiment_id[4:]}_{case['case_id'][5:13]}_{arm.value}"
        entry: dict[str, Any] = {
            "order": order,
            "case_id": case["case_id"],
            "arm": arm.value,
            "run_id": run_id,
        }
        if budget_stopped:
            entry.update(status="budget_prevented", error="experiment allowance reached earlier")
            entries.append(entry)
            continue
        case_dir = cases_dir / case["case_id"]
        candidate = CandidateEvent.model_validate_json((case_dir / "candidate.json").read_text())
        faults = case.get("runtime_faults")
        dispatcher_factory = (
            failing_dispatcher_factory(set(faults["failing_tools"]), ErrorCode(faults["code"]))
            if faults
            else None
        )
        adapter, rate_card = adapter_factory()
        runner = InvestigationRunner(
            run_config,
            adapter,
            rate_card=rate_card,
            dispatcher_factory=dispatcher_factory,
        )
        try:
            report = runner.start(
                case_dir / "snapshot",
                candidate,
                RunOptions(
                    arm=arm,
                    case_id=case["case_id"],
                    run_id=run_id,
                    experiment_id=experiment_id,
                    report_directory=directory / "reports",
                ),
            )
            entry.update(
                status="ran",
                workflow_outcome=report.workflow_outcome.value,
                terminal_reason=report.terminal_reason,
            )
            if report.terminal_reason.startswith("budget_exhausted:experiment"):
                budget_stopped = True
        except Exception as failure:  # keep crashed runs in the denominator
            entry.update(
                status="crashed",
                error=f"{type(failure).__name__}: {failure}",
                traceback=traceback.format_exc()[-2000:],
            )
        finally:
            runner.close()
        entries.append(entry)
        index_path.write_text(
            json.dumps(
                {"experiment_id": experiment_id, "mode": mode, "freeze": record, "runs": entries},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    index_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "mode": mode,
                "freeze": record,
                "runs": entries,
                "stopped_by_operator": stopped_by_operator,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return index_path
