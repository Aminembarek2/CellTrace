"""A frozen development schedule with one durable allowance across model profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from celltrace.canonical import content_hash, file_sha256
from celltrace.clock import SystemClock
from celltrace.config import AppConfig, api_key_present, load_config
from celltrace.contracts.common import Arm, ErrorCode
from celltrace.contracts.evidence import CandidateEvent
from celltrace.evaluation.cases import CaseFamily
from celltrace.providers.profiles import MODEL_PROFILES
from celltrace.storage.run_store import RunStore
from celltrace.tools.faults import failing_dispatcher_factory
from celltrace.workflow.runner import InvestigationRunner, RunOptions

PILOT_FAMILIES = (
    CaseFamily.clean_control,
    CaseFamily.stable_radio_rate_drop,
    CaseFamily.cell_change_without_rate_drop,
    CaseFamily.application_idle,
    CaseFamily.missing_metrics,
    CaseFamily.tool_failure,
)
REVIEW_FAMILIES = PILOT_FAMILIES[:3]


def prepare_pilot(
    manifest_path: Path, configuration_directory: Path, output: Path
) -> dict[str, Any]:
    from celltrace.evaluation.commands import freeze_hashes

    if output.exists():
        raise FileExistsError(f"preserve existing pilot plan at {output}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"] != "dev" or not manifest.get("frozen"):
        raise ValueError("pilot requires frozen development cases")
    selected = {}
    for family in PILOT_FAMILIES:
        matching = sorted(
            (case for case in manifest["cases"] if case["family"] == family.value),
            key=lambda case: case["case_id"],
        )
        if not matching:
            raise ValueError(f"pilot needs a development case for {family.value}")
        selected[family.value] = matching[0]["case_id"]
    configurations = {}
    for profile in MODEL_PROFILES.values():
        path = configuration_directory / f"{profile.model_id.replace('/', '_')}.yaml"
        configuration = load_config(path, environment={})
        if configuration.runtime.model_id != profile.model_id:
            raise ValueError("pilot configuration model mismatch")
        configurations[profile.model_id] = {"path": str(path), "sha256": file_sha256(path)}
    schedule = [
        {"model_id": model_id, "case_id": case_id, "family": family, "arm": arm.value}
        for model_id in configurations
        for family, case_id in selected.items()
        for arm in (
            (Arm.investigator_gated, Arm.investigator_reviewer_gated)
            if CaseFamily(family) in REVIEW_FAMILIES
            else (Arm.investigator_gated,)
        )
    ]
    payload = {
        "version": "development_pilot_1",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "configurations": configurations,
        "freeze": freeze_hashes(),
        "schedule": schedule,
        "repeats": 0,
        "promotion": "manual decision required",
    }
    payload["plan_hash"] = content_hash(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run_pilot(plan_path: Path, output_directory: Path, live: bool) -> dict[str, Any]:
    from celltrace.cli import CommandError, build_adapter
    from celltrace.evaluation.commands import freeze_hashes

    plan = json.loads(plan_path.read_text())
    recorded_hash = plan.pop("plan_hash")
    if recorded_hash != content_hash(plan):
        raise ValueError("pilot plan changed after freezing")
    current_freeze = freeze_hashes()
    for key, expected in plan["freeze"].items():
        if key != "frozen_at" and current_freeze.get(key) != expected:
            raise ValueError(f"pilot execution input changed: {key}")
    manifest_path = Path(plan["manifest"])
    if file_sha256(manifest_path) != plan["manifest_sha256"]:
        raise ValueError("pilot development cases changed")
    configurations = {}
    for model_id, record in plan["configurations"].items():
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError("pilot profile configuration changed")
        configurations[model_id] = load_config(path, environment={})
    manifest = json.loads(manifest_path.read_text())
    cases = {case["case_id"]: case for case in manifest["cases"]}
    experiment_id = f"exp_pilot_{recorded_hash[:16]}"
    output_directory = output_directory / experiment_id
    output_directory.mkdir(parents=True, exist_ok=True)
    index_path = output_directory / "index.json"
    existing = json.loads(index_path.read_text()) if index_path.exists() else None
    if existing is not None and existing["plan_hash"] != recorded_hash:
        raise ValueError("existing pilot has another plan identity")
    entries = (
        [entry for entry in existing["runs"] if entry["status"] != "unavailable"]
        if existing
        else []
    )
    availability_history = existing.get("availability_history", []) if existing else []
    if existing and any(entry["status"] == "unavailable" for entry in existing["runs"]):
        availability_history.append(
            {
                "observed_at": existing.get("observed_at"),
                "runs": [entry for entry in existing["runs"] if entry["status"] == "unavailable"],
            }
        )
    completed = {(entry["model_id"], entry["case_id"], entry["arm"]) for entry in entries}
    allowance = min(
        configuration.runtime.maximum_estimated_usd_per_experiment
        for configuration in configurations.values()
    )
    store = RunStore(output_directory / "runs.sqlite3", SystemClock())
    store.create_experiment(experiment_id, allowance, {"plan_hash": recorded_hash})
    store.close()
    readiness = {}
    for model_id, configuration in configurations.items():
        if not live or not api_key_present():
            readiness[model_id] = "explicit live activation and OPENROUTER_API_KEY required"
            continue
        try:
            build_adapter(configuration, True)
        except (CommandError, ValueError, OSError, KeyError) as failure:
            readiness[model_id] = str(failure)

    def write_index(stopped_by_operator: bool = False) -> None:
        ledger = RunStore(output_directory / "runs.sqlite3", SystemClock())
        try:
            experiment_usage, experiment_ceiling = ledger.experiment_usd(experiment_id)
        finally:
            ledger.close()
        index = {
            "experiment_id": experiment_id,
            "mode": "live",
            "plan_hash": recorded_hash,
            "freeze": plan["freeze"],
            "runs": entries,
            "maximum_estimated_usd": allowance,
            "used_or_reserved_usd": experiment_usage,
            "durable_experiment_ceiling_usd": experiment_ceiling,
            "observed_at": SystemClock().utc_now_iso(),
            "availability_history": availability_history,
            "scheduled": len(plan["schedule"]),
            "attempted": sum(entry.get("provider_attempts", 0) > 0 for entry in entries),
            "finished": sum(entry.get("workflow_outcome") == "completed" for entry in entries),
            "reports_written": sum(entry["status"] == "ran" for entry in entries),
            "unavailable": sum(entry["status"] == "unavailable" for entry in entries),
            "budget_prevented": sum(entry["status"] == "budget_prevented" for entry in entries),
            "stopped_by_operator": stopped_by_operator,
            "recommendation": "retain the primary profile until a manual decision changes it",
        }
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    # An operator creates this file to stop between runs, so no provider call is cut off.
    stop_file = output_directory / "STOP"
    stopped = False
    for position, scheduled in enumerate(plan["schedule"]):
        key = (scheduled["model_id"], scheduled["case_id"], scheduled["arm"])
        if key in completed:
            continue
        if stop_file.exists():
            stopped = True
            break
        entry = {**scheduled, "run_id": f"run_pilot_{recorded_hash[:12]}_{position:02d}"}
        model_id = scheduled["model_id"]
        if model_id in readiness:
            entry.update(status="unavailable", reason=readiness[model_id])
        else:
            resolved = configurations[model_id].model_dump(mode="json")
            resolved["runtime"]["enable_live_calls"] = True
            resolved["runtime"]["run_store"] = str(output_directory / "runs.sqlite3")
            resolved["runtime"]["checkpoint_store"] = str(output_directory / "checkpoints.sqlite3")
            configuration = AppConfig.model_validate(resolved)
            case = cases[scheduled["case_id"]]
            case_directory = manifest_path.parent.parent / "cases" / case["case_id"]
            faults = case.get("runtime_faults")
            dispatcher_factory = (
                failing_dispatcher_factory(set(faults["failing_tools"]), ErrorCode(faults["code"]))
                if faults
                else None
            )
            runner = None
            try:
                adapter, rates = build_adapter(configuration, True)
                runner = InvestigationRunner(
                    configuration, adapter, rate_card=rates, dispatcher_factory=dispatcher_factory
                )
                report = runner.start(
                    case_directory / "snapshot",
                    CandidateEvent.model_validate_json(
                        (case_directory / "candidate.json").read_text()
                    ),
                    RunOptions(
                        arm=Arm(scheduled["arm"]),
                        case_id=case["case_id"],
                        run_id=entry["run_id"],
                        experiment_id=experiment_id,
                        report_directory=output_directory / "reports",
                    ),
                )
                entry.update(
                    status="ran",
                    workflow_outcome=report.workflow_outcome.value,
                    terminal_reason=report.terminal_reason,
                    provider_attempts=report.usage.provider_attempts,
                    estimated_usd=report.usage.estimated_usd,
                )
                if report.terminal_reason.startswith("budget_exhausted:experiment"):
                    entry["status"] = "budget_prevented"
            except Exception as failure:
                entry.update(status="crashed", reason=f"{type(failure).__name__}: {failure}")
            finally:
                if runner is not None:
                    runner.close()
        ledger = RunStore(output_directory / "runs.sqlite3", SystemClock())
        try:
            usage = ledger.usage(entry["run_id"])
            entry.update(
                provider_attempts=usage.provider_attempts,
                estimated_usd=usage.estimated_usd,
                unknown_usage_attempts=usage.unknown_usage_attempts,
            )
        finally:
            ledger.close()
        entries.append(entry)
        write_index()
    if stopped or not index_path.exists():
        write_index(stopped_by_operator=stopped)
    result: dict[str, Any] = json.loads(index_path.read_text())
    return result


def command_pilot(arguments: argparse.Namespace) -> int:
    if arguments.action == "plan":
        payload = prepare_pilot(
            Path(arguments.manifest), Path(arguments.configurations), Path(arguments.out)
        )
        print(json.dumps({"plan": arguments.out, "scheduled_runs": len(payload["schedule"])}))
        return 0
    result = run_pilot(Path(arguments.plan), Path(arguments.out), arguments.live)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "experiment_id",
                    "attempted",
                    "finished",
                    "unavailable",
                    "budget_prevented",
                )
            }
        )
    )
    return 3 if result["unavailable"] else 0


def register(subparsers: Any) -> None:
    pilot = subparsers.add_parser("pilot", help="bounded development model comparison")
    commands = pilot.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--configurations", default="configs")
    plan.add_argument("--out", required=True)
    plan.set_defaults(handler=command_pilot)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--live", action="store_true")
    run.add_argument("--out", default="artifacts/private/pilot")
    run.set_defaults(handler=command_pilot)
