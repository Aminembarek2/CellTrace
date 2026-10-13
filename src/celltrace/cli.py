"""Command line composition root.

Exit codes: 0 success, 1 runtime failure, 2 usage or input error,
3 blocked (missing credentials, model, rate card or data for a live or real
data gate). A blocked gate is reported, never treated as a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from celltrace.config import AppConfig, api_key_present, load_config
from celltrace.contracts.common import Arm
from celltrace.contracts.evidence import CandidateEvent

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_BLOCKED = 3

DEFAULT_CANDIDATES = Path("artifacts/public/candidates/candidates.json")
KEY_PACKAGES = (
    "httpx",
    "duckdb",
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "matplotlib",
    "numpy",
    "pandas",
    "pydantic",
    "pyyaml",
    "scikit-learn",
)


class CommandError(Exception):
    def __init__(self, message: str, exit_code: int = EXIT_USAGE):
        super().__init__(message)
        self.exit_code = exit_code


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _config(arguments: argparse.Namespace, live: bool = False) -> AppConfig:
    overrides: dict[tuple[str, ...], Any] = {}
    if live:
        overrides[("runtime", "enable_live_calls")] = True
    if getattr(arguments, "model", None):
        overrides[("runtime", "model_id")] = arguments.model
    if getattr(arguments, "run_store", None):
        overrides[("runtime", "run_store")] = arguments.run_store
        overrides[("runtime", "checkpoint_store")] = str(
            Path(arguments.run_store).with_name("checkpoints.sqlite3")
        )
    try:
        return load_config(getattr(arguments, "config", None), overrides)
    except ValueError as error:
        exit_code = EXIT_BLOCKED if live else EXIT_USAGE
        raise CommandError(f"invalid configuration: {error}", exit_code) from error


# doctor


def command_doctor(arguments: argparse.Namespace) -> int:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {},
    }
    for package in KEY_PACKAGES:
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report["packages"][package] = "missing"
    try:
        config = _config(arguments)
        report["config"] = {"valid": True, "resolved": config.redacted_summary()}
    except ValueError as error:
        report["config"] = {"valid": False, "error": str(error)}
        _print_json(report)
        return EXIT_USAGE
    report["data"] = _data_status(config)
    report["provider"] = {
        "live_enabled": config.runtime.enable_live_calls,
        "model_id": config.runtime.model_id or os.environ.get("CELLTRACE_MODEL") or None,
        "openrouter_api_key_present": api_key_present(),
        "rate_card": None
        if config.runtime.verified_rate_card is None
        else str(config.runtime.verified_rate_card),
    }
    status = EXIT_OK
    if arguments.live:
        probe = _live_probe(config, arguments)
        report["live_probe"] = probe
        status = EXIT_OK if probe.get("status") == "passed" else EXIT_BLOCKED
    _print_json(report)
    return status


def _data_status(config: AppConfig) -> dict[str, Any]:
    from celltrace.canonical import file_sha256

    raw = Path(config.data.raw_dir)
    archives = sorted(raw.rglob("*.zip")) if raw.exists() else []
    entries = []
    for archive in archives:
        source = archive.with_name("SOURCE.json")
        expected = None
        if source.exists():
            expected = json.loads(source.read_text())["files"].get(archive.name, {}).get("sha256")
        actual = file_sha256(archive)
        entries.append(
            {"path": str(archive), "sha256": actual, "matches_source_record": expected == actual}
        )
    processed = Path(config.data.processed_dir)
    snapshots = sorted(path.parent.name for path in processed.glob("*/snapshot_manifest.json"))
    return {"raw_archives": entries, "snapshots": snapshots}


def _live_probe(config: AppConfig, arguments: argparse.Namespace) -> dict[str, Any]:
    from celltrace.workflow.live_probe import run_live_probe

    if not api_key_present():
        return {"status": "blocked", "reason": "OPENROUTER_API_KEY is not set"}
    try:
        probe_config = _config(arguments, live=True)
        return run_live_probe(probe_config.runtime.model_id, probe_config, Path(arguments.out))
    except (CommandError, ValueError, OSError) as failure:
        return {"status": "blocked", "reason": str(failure)}


# data


def command_audit(arguments: argparse.Namespace) -> int:
    from celltrace.data.audit import run_audit

    input_dir = Path(arguments.input)
    if not any(input_dir.rglob("*.zip")):
        raise CommandError(f"no archive found under {input_dir}", EXIT_BLOCKED)
    summary = run_audit(input_dir, Path(arguments.out))
    _print_json({"out": arguments.out, "summary_keys": sorted(summary)})
    return EXIT_OK


def command_ingest(arguments: argparse.Namespace) -> int:
    from celltrace.data.snapshot import run_ingest

    input_dir = Path(arguments.input)
    if not any(input_dir.rglob("*.zip")):
        raise CommandError(f"no archive found under {input_dir}", EXIT_BLOCKED)
    if arguments.source == "raca_lte_2018":
        from celltrace.data import raca_lte

        if arguments.operator is None:
            raise CommandError("the 2018 LTE dataset needs --operator A or B")
        snapshot_dir = run_ingest(
            input_dir,
            Path(arguments.out),
            sources=[raca_lte.source_archive(input_dir, arguments.operator)],
            session_iterator=raca_lte.iter_sessions,
        )
    else:
        snapshot_dir = run_ingest(input_dir, Path(arguments.out))
    _print_json({"snapshot": str(snapshot_dir)})
    return EXIT_OK


def command_explore(arguments: argparse.Namespace) -> int:
    from celltrace.analysis.explore import run_explore

    summary = run_explore(Path(arguments.snapshot), Path(arguments.out))
    _print_json({"out": arguments.out, "summary_keys": sorted(summary)})
    return EXIT_OK


def command_cells(arguments: argparse.Namespace) -> int:
    from celltrace.analysis.cell_study import REPORT_NAME, run_cell_study

    config = _config(arguments)
    try:
        summary = run_cell_study(
            Path(arguments.snapshot),
            Path(arguments.out),
            arguments.label,
            arguments.cohort,
            arguments.mobility or ["driving"],
            config.project.seed,
            accept_proxy_identity=arguments.accept_proxy_identity,
        )
    except (FileExistsError, ValueError) as error:
        raise CommandError(str(error)) from error
    replication = summary["replication"]
    result = summary["shrinkage"]
    _print_json(
        {
            "report": str(Path(arguments.out) / REPORT_NAME),
            "visits": replication["visits"],
            "scored_visits": replication["primary"]["scored_visits"],
            "skill_against_technology": replication["primary"]["skill_against_technology"],
            "p_value": replication["primary"]["p_value"],
            "decision": replication["decision"],
            "selected_slow_cells": None if result is None else result["selected"],
        }
    )
    return EXIT_OK


def command_population(arguments: argparse.Namespace) -> int:
    from celltrace.analysis.population import run_population

    config = _config(arguments)
    try:
        summary = run_population(
            Path(arguments.snapshot),
            Path(arguments.out),
            Path(arguments.private_out),
            config.analysis.detector,
            config.analysis.observation,
            config.project.seed,
        )
    except (FileExistsError, ValueError) as error:
        raise CommandError(str(error)) from error
    triage = summary["q3_triage"]
    _print_json(
        {
            "report": str(Path(arguments.out) / "population_report.md"),
            "sessions": summary["sessions"],
            "drop_events": triage["events"],
            "unexplained_drops": triage["classes"]["unexplained"]["events"],
            "low_rate_unexplained_drops": summary["queue"]["low_rate_unexplained"],
            "low_rate_episodes_queued": summary["queue"]["incidents_queued"],
            "investigation_queue_candidates": summary["queue"]["unique_candidates_queued"],
        }
    )
    return EXIT_OK


def command_detect(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    if arguments.detector == "prediction_residual":
        from celltrace.prediction.residual import run_residual_detect

        if not arguments.model_artifact:
            raise CommandError("--model is required for the prediction_residual detector")
        path = run_residual_detect(
            Path(arguments.snapshot), Path(arguments.model_artifact), Path(arguments.out)
        )
    else:
        from celltrace.analysis.detector import run_detect

        path = run_detect(Path(arguments.snapshot), Path(arguments.out), config.analysis.detector)
    _print_json({"candidates": str(path)})
    return EXIT_OK


def load_candidate(event_id: str, candidates_path: Path) -> CandidateEvent:
    if not candidates_path.exists():
        raise CommandError(f"candidate file not found: {candidates_path}. Run detect first.")
    payload = json.loads(candidates_path.read_text())
    for entry in payload["candidates"]:
        if entry["event_id"] == event_id:
            return CandidateEvent.model_validate(entry)
    raise CommandError(f"event {event_id} is not in {candidates_path}")


def command_inspect_event(arguments: argparse.Namespace) -> int:
    from celltrace.contracts.common import AggregationBasis, MetricName, TimeWindow
    from celltrace.data.snapshot import open_snapshot
    from celltrace.tools.dispatch import transitions_in_window
    from celltrace.tools.repository import MeasurementRepository

    candidate = load_candidate(arguments.event, Path(arguments.candidates))
    connection, manifest = open_snapshot(Path(arguments.snapshot))
    try:
        repository = MeasurementRepository(connection)
        reference = candidate.representative_reference_window
        event = candidate.representative_event_window
        span = TimeWindow(start_ms=reference.start_ms, end_ms=event.end_ms)
        output = {
            "candidate": candidate.model_dump(mode="json"),
            "snapshot_id": manifest["snapshot_id"],
            "reference_downlink_bin_1s": repository.summarize(
                candidate.session_id, reference, MetricName.downlink_mbps, AggregationBasis.bin_1s
            ).as_dict(),
            "event_downlink_bin_1s": repository.summarize(
                candidate.session_id, event, MetricName.downlink_mbps, AggregationBasis.bin_1s
            ).as_dict(),
            "activity_native": repository.activity(
                candidate.session_id, span, AggregationBasis.native
            ),
            "transitions": transitions_in_window(repository, candidate.session_id, span, span)[
                "transitions"
            ],
            "detection_delay_ms": candidate.first_confirmation_ms - candidate.event_window.start_ms,
        }
    finally:
        connection.close()
    _print_json(output)
    return EXIT_OK


# runs


def build_adapter(config: AppConfig, live: bool) -> tuple[Any, Any]:
    if not live:
        from celltrace.agents.scripted_policies import ScriptedInvestigator, ScriptedReviewer
        from celltrace.providers.scripted import ScriptedModelAdapter

        adapter = ScriptedModelAdapter(
            {"investigator": ScriptedInvestigator(), "reviewer": ScriptedReviewer()}
        )
        return adapter, None
    if not api_key_present():
        raise CommandError("blocked: OPENROUTER_API_KEY is not set", EXIT_BLOCKED)
    if not config.runtime.model_id or config.runtime.verified_rate_card is None:
        raise CommandError(
            "blocked: live runs need a model id and a verified rate card", EXIT_BLOCKED
        )
    from celltrace.providers.openrouter_adapter import OpenRouterAdapter
    from celltrace.providers.readiness import require_capability_probe, verified_rate_card

    try:
        rate_card = verified_rate_card(config.runtime)
        require_capability_probe(config.runtime)
    except (ValueError, OSError, KeyError) as failure:
        raise CommandError(f"blocked provider readiness: {failure}", EXIT_BLOCKED) from failure
    return OpenRouterAdapter(config.runtime.model_id), rate_card


def command_investigate(arguments: argparse.Namespace) -> int:
    from celltrace.workflow.runner import InvestigationRunner, RunOptions

    if arguments.live == arguments.offline:
        raise CommandError("choose exactly one of --offline or --live")
    config = _config(arguments, live=arguments.live)
    adapter, rate_card = build_adapter(config, arguments.live)
    candidate = load_candidate(arguments.event, Path(arguments.candidates))
    runner = InvestigationRunner(config, adapter, rate_card=rate_card)
    try:
        report = runner.start(
            Path(arguments.snapshot),
            candidate,
            RunOptions(arm=Arm(arguments.variant), report_directory=Path(arguments.out)),
        )
    finally:
        runner.close()
    _print_json(_report_summary(report, Path(arguments.out)))
    return EXIT_OK


def _report_summary(report: Any, directory: Path) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "execution_mode": report.execution_mode.value,
        "workflow_outcome": report.workflow_outcome.value,
        "fact_check_status": report.fact_check_status.value,
        "review_status": report.review_status.value,
        "cause_status": report.cause_status.value,
        "verified_claims": len(report.verified_claims),
        "rejected_claims": len(report.rejected_claims),
        "report_json": str(directory / f"{report.run_id}.json"),
        "report_markdown": str(directory / f"{report.run_id}.md"),
    }


def command_resume(arguments: argparse.Namespace) -> int:
    from celltrace.clock import SystemClock
    from celltrace.storage.run_store import RunStore
    from celltrace.workflow.runner import InvestigationRunner, ResumeRefused

    config = _config(arguments)
    probe_store = RunStore(config.runtime.run_store, SystemClock())
    try:
        run = probe_store.get_run(arguments.run)
    finally:
        probe_store.close()
    live = run["execution_mode"] == "live"
    if live:
        config = _config(arguments, live=True)
    adapter, rate_card = build_adapter(config, live)
    runner = InvestigationRunner(config, adapter, rate_card=rate_card)
    try:
        report = runner.resume(arguments.run, retry_uncertain=arguments.retry_uncertain)
    except ResumeRefused as refused:
        raise CommandError(str(refused), EXIT_FAILURE) from refused
    finally:
        runner.close()
    _print_json(_report_summary(report, Path(arguments.out)))
    return EXIT_OK


def command_export_run(arguments: argparse.Namespace) -> int:
    from celltrace.clock import SystemClock
    from celltrace.storage.run_store import RunStore
    from celltrace.workflow.runner import export_run

    config = _config(arguments)
    store = RunStore(config.runtime.run_store, SystemClock())
    try:
        path = export_run(store, arguments.run, Path(arguments.out))
    finally:
        store.close()
    _print_json({"export": str(path)})
    return EXIT_OK


def command_demo(arguments: argparse.Namespace) -> int:
    from celltrace.workflow.demo import run_offline_demo

    if not arguments.offline:
        raise CommandError("the demo runs offline only. Use investigate --live for a live run.")
    summary = run_offline_demo(Path(arguments.out))
    _print_json(summary)
    return EXIT_OK if summary["status"] == "passed" else EXIT_FAILURE


# evaluation and prediction commands are registered by their modules


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="configuration YAML")
    parser.add_argument("--run-store", default=None, help="override runtime.run_store")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="celltrace", description=__doc__)
    parser.add_argument("--env-file", type=Path, help="explicit file containing the OpenRouter key")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="environment, config, data and provider status")
    _add_common(doctor)
    doctor.add_argument("--live", action="store_true", help="run a charged live capability probe")
    doctor.add_argument("--model", default=None)
    doctor.add_argument("--out", default="artifacts/private/probes")
    doctor.set_defaults(handler=command_doctor)

    for name, handler, help_text in (
        ("audit", command_audit, "audit raw archive headers, values and cadence"),
        ("ingest", command_ingest, "build an immutable DuckDB measurement snapshot"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sources = ["raca", "raca_lte_2018"] if name == "ingest" else ["raca"]
        sub.add_argument("--source", choices=sources, default="raca")
        sub.add_argument("--input", required=True)
        sub.add_argument("--out", required=True)
        if name == "ingest":
            sub.add_argument("--operator", choices=["A", "B"], help="2018 LTE dataset only")
        sub.set_defaults(handler=handler)

    explore = commands.add_parser("explore", help="SQL summaries and temporal plots")
    explore.add_argument("--snapshot", required=True)
    explore.add_argument("--out", required=True)
    explore.set_defaults(handler=command_explore)

    population = commands.add_parser(
        "population", help="drop rates, cell change association, triage and missing data"
    )
    _add_common(population)
    population.add_argument("--snapshot", required=True)
    population.add_argument("--out", required=True, help="aggregate report and figures")
    population.add_argument("--private-out", required=True, help="per event triage and queue")
    population.set_defaults(handler=command_population)

    cell_study = commands.add_parser(
        "cells", help="cell rate replication across days and reliably slow cells"
    )
    _add_common(cell_study)
    cell_study.add_argument("--snapshot", required=True)
    cell_study.add_argument("--out", required=True, help="report, summary and figures")
    cell_study.add_argument("--label", required=True, help="dataset name shown in the report")
    cell_study.add_argument("--cohort", default="file_download")
    cell_study.add_argument(
        "--mobility", action="append", help="mobility cohort to include, repeatable"
    )
    cell_study.add_argument(
        "--accept-proxy-identity",
        action="store_true",
        help="2018 LTE dataset: accept the site and local number proxy as cell identity",
    )
    cell_study.set_defaults(handler=command_cells)

    detect = commands.add_parser("detect", help="flag candidate throughput drops")
    _add_common(detect)
    detect.add_argument("--snapshot", required=True)
    detect.add_argument("--out", required=True)
    detect.add_argument(
        "--detector",
        choices=["sustained_relative_drop", "prediction_residual"],
        default="sustained_relative_drop",
    )
    detect.add_argument("--model", dest="model_artifact", default=None)
    detect.set_defaults(handler=command_detect)

    inspect_event = commands.add_parser("inspect-event", help="deterministic view of one event")
    inspect_event.add_argument("--event", required=True)
    inspect_event.add_argument("--snapshot", required=True)
    inspect_event.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    inspect_event.set_defaults(handler=command_inspect_event)

    investigate = commands.add_parser("investigate", help="run one investigation")
    _add_common(investigate)
    investigate.add_argument("--event", required=True)
    investigate.add_argument("--snapshot", required=True)
    investigate.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    investigate.add_argument("--variant", choices=[arm.value for arm in Arm], required=True)
    investigate.add_argument("--offline", action="store_true")
    investigate.add_argument("--live", action="store_true")
    investigate.add_argument("--model", default=None)
    investigate.add_argument("--out", default="artifacts/private/reports")
    investigate.set_defaults(handler=command_investigate)

    resume = commands.add_parser("resume", help="resume an interrupted run")
    _add_common(resume)
    resume.add_argument("--run", required=True)
    resume.add_argument(
        "--retry-uncertain",
        action="store_true",
        help="accept unknown prior usage and issue a newly charged provider attempt",
    )
    resume.add_argument("--out", default="artifacts/private/reports")
    resume.set_defaults(handler=command_resume)

    export = commands.add_parser("export-run", help="export one run as JSONL")
    _add_common(export)
    export.add_argument("--run", required=True)
    export.add_argument("--out", required=True)
    export.set_defaults(handler=command_export_run)

    demo = commands.add_parser("demo", help="offline scripted end to end demonstration")
    demo.add_argument("--offline", action="store_true")
    demo.add_argument("--out", default="artifacts/public/demo")
    demo.set_defaults(handler=command_demo)

    for register in _extension_registrars():
        register(commands)
    return parser


def _extension_registrars() -> list[Callable[[Any], None]]:
    registrars: list[Callable[[Any], None]] = []
    for module_name in (
        "celltrace.evaluation.commands",
        "celltrace.evaluation.allowance_commands",
        "celltrace.evaluation.joint_resource_commands",
        "celltrace.evaluation.planted_commands",
        "celltrace.data.independent_sessions",
        "celltrace.evaluation.pilot",
        "celltrace.evaluation.review.commands",
        "celltrace.prediction.commands",
        "celltrace.providers.commands",
        "celltrace.workflow.queue",
        "celltrace.reporting.commands",
        "celltrace.release",
    ):
        try:
            module = __import__(module_name, fromlist=["register"])
        except ModuleNotFoundError as missing:
            if missing.name != module_name:
                raise
            continue
        registrars.append(module.register)
    return registrars


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.env_file is not None:
            from celltrace.secrets import load_openrouter_key

            load_openrouter_key(arguments.env_file, os.environ)
        return int(arguments.handler(arguments))
    except CommandError as error:
        print(f"celltrace: {error}", file=sys.stderr)
        return error.exit_code
    except (ValueError, OSError) as error:
        print(f"celltrace: {error}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
