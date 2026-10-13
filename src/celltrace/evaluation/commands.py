"""CLI commands for cases, benchmarks, detector evaluation and scoring."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from celltrace.agents.roles import role_identity
from celltrace.canonical import content_hash
from celltrace.config import api_key_present, load_config
from celltrace.contracts.common import Arm
from celltrace.tools.dispatch import TOOL_VERSION

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


# Provider identity, store paths and the live switch are chosen at run time and
# recorded per experiment. Everything else that shapes a run is frozen.
NOT_FROZEN_RUNTIME_FIELDS = frozenset(
    {
        "enable_live_calls",
        "run_store",
        "checkpoint_store",
        "verified_rate_card",
        "verified_endpoint_record",
        "verified_capability_record",
        "require_live_capability_probe",
    }
)


def frozen_runtime_settings(config: Any) -> dict[str, Any]:
    settings = config.runtime.model_dump(mode="json")
    return {key: value for key, value in settings.items() if key not in NOT_FROZEN_RUNTIME_FIELDS}


def freeze_hashes(config: Any = None) -> dict[str, Any]:
    from celltrace.evaluation.cases import (
        CASE_BUILDER_VERSION,
        EXPECTED_PATTERNS,
        required_fact_kinds,
    )
    from celltrace.release import source_tree_hash

    config = config or load_config()
    rubric = {
        family.value: {"patterns": rule, "required_facts": required_fact_kinds(family)}
        for family, rule in EXPECTED_PATTERNS.items()
    }
    return {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "case_builder_version": CASE_BUILDER_VERSION,
        "source_tree_sha256": source_tree_hash(),
        "roles": role_identity(),
        "tool_version": TOOL_VERSION,
        "rubric_sha256": content_hash(rubric),
        "detector_config_sha256": content_hash(config.analysis.detector.model_dump(mode="json")),
        "observation_config_sha256": content_hash(
            config.analysis.observation.model_dump(mode="json")
        ),
        "runtime_limits_sha256": content_hash(frozen_runtime_settings(config)),
    }


def command_cases(arguments: argparse.Namespace) -> int:
    from celltrace.evaluation.cases import build_cases

    out = Path(arguments.out)
    manifest_path = out / arguments.split / "labels" / "manifest.json"
    if arguments.action == "freeze" and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("frozen"):
            print(f"celltrace: {manifest_path} is already frozen and will not be rebuilt")
            return EXIT_USAGE
    # A held out freeze must record the exact runtime settings the benchmark will use.
    config = load_config(getattr(arguments, "config", None))
    path = build_cases(
        Path(arguments.snapshot),
        Path(arguments.raw),
        out,
        arguments.split,
        config,
        maximum_base_windows=arguments.base_windows,
        variants_per_base=arguments.variants,
        base_windows_per_session=getattr(arguments, "windows_per_session", 2),
        session_selection_path=getattr(arguments, "session_selection", None),
    )
    manifest = json.loads(path.read_text())
    if arguments.action == "freeze":
        manifest["frozen"] = True
        manifest["freeze"] = freeze_hashes(config)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    families: dict[str, int] = {}
    for case in manifest["cases"]:
        families[case["family"]] = families.get(case["family"], 0) + 1
    _print(
        {
            "manifest": str(path),
            "split": arguments.split,
            "frozen": manifest["frozen"],
            "cases": len(manifest["cases"]),
            "source_sessions": len({case["source_session_id"] for case in manifest["cases"]}),
            "families": families,
        }
    )
    return EXIT_OK


def command_benchmark(arguments: argparse.Namespace) -> int:
    from celltrace.cli import _config, build_adapter
    from celltrace.evaluation.benchmark import offline_adapter_factory, run_benchmark

    manifest_path = Path(arguments.manifest)
    arms = [Arm(arm) for arm in arguments.arms]
    if arguments.mode == "offline":
        config = _config(arguments)
        index = run_benchmark(
            manifest_path,
            "offline",
            arms,
            offline_adapter_factory,
            config=config,
            experiment_id=arguments.experiment_id,
            case_limit=arguments.case_limit,
        )
    else:
        if not api_key_present():
            print("celltrace: blocked: OPENROUTER_API_KEY is not set, live benchmark not run")
            return EXIT_BLOCKED
        try:
            config = _config(arguments, live=True)
        except ValueError as error:
            print(f"celltrace: blocked: {error}")
            return EXIT_BLOCKED
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("frozen"):
            print("celltrace: live benchmarks need a frozen manifest (cases freeze)")
            return EXIT_USAGE

        def live_factory() -> Any:
            return build_adapter(config, True)

        index = run_benchmark(
            manifest_path,
            "live",
            arms,
            live_factory,
            config=config,
            experiment_id=arguments.experiment_id,
            case_limit=arguments.case_limit,
        )
    payload = json.loads(index.read_text())
    statuses: dict[str, int] = {}
    for entry in payload["runs"]:
        statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
    _print({"index": str(index), "experiment_id": payload["experiment_id"], "statuses": statuses})
    return EXIT_OK


def command_score(arguments: argparse.Namespace) -> int:
    from celltrace.evaluation.benchmark import experiment_directory
    from celltrace.evaluation.scoring import score_experiment

    manifest_path = Path(arguments.manifest)
    # Pilot experiments live outside the case directory and pass their index explicitly.
    index = (
        Path(arguments.index)
        if getattr(arguments, "index", None)
        else experiment_directory(manifest_path, arguments.experiment) / "index.json"
    )
    if json.loads(index.read_text())["experiment_id"] != arguments.experiment:
        raise ValueError("index belongs to another experiment")
    summary = score_experiment(index, manifest_path, Path(arguments.out) / arguments.experiment)
    _print(
        {
            "out": str(Path(arguments.out) / arguments.experiment),
            "label": summary["label"],
            "by_arm": summary["by_arm"],
        }
    )
    return EXIT_OK


def command_detector_eval(arguments: argparse.Namespace) -> int:
    from celltrace.evaluation.detector_eval import evaluate_detector

    summary = evaluate_detector(
        Path(arguments.manifest), load_config().analysis.detector, Path(arguments.out)
    )
    _print({key: value for key, value in summary.items() if key != "rows"})
    return EXIT_OK


def register(subparsers: Any) -> None:
    cases = subparsers.add_parser("cases", help="build or freeze private evaluation cases")
    cases.add_argument("action", choices=["build", "freeze"])
    cases.add_argument("--split", choices=["dev", "heldout"], required=True)
    cases.add_argument("--snapshot", required=True)
    cases.add_argument("--raw", default="data/raw")
    cases.add_argument("--out", default="data/eval_private")
    cases.add_argument("--base-windows", type=int, default=12)
    cases.add_argument("--variants", type=int, default=2)
    cases.add_argument("--windows-per-session", type=int, default=2)
    cases.add_argument(
        "--session-selection", type=Path, help="private reserved whole-session selection"
    )
    cases.add_argument("--config", type=Path, default=None)
    cases.set_defaults(handler=command_cases)

    benchmark = subparsers.add_parser("benchmark", help="run arms over a case manifest")
    benchmark.add_argument("--manifest", required=True)
    benchmark.add_argument("--mode", choices=["offline", "live"], required=True)
    benchmark.add_argument(
        "--arms", nargs="+", default=[arm.value for arm in Arm], choices=[arm.value for arm in Arm]
    )
    benchmark.add_argument("--experiment-id", default=None)
    benchmark.add_argument("--case-limit", type=int, default=None)
    benchmark.add_argument("--model", default=None)
    benchmark.add_argument("--config", type=Path, default=None)
    benchmark.set_defaults(handler=command_benchmark)

    score = subparsers.add_parser("score", help="score an experiment against private labels")
    score.add_argument("--experiment", required=True)
    score.add_argument("--manifest", required=True)
    score.add_argument("--out", default="artifacts/public/evaluation")
    score.add_argument("--index", default=None, help="explicit experiment index, for pilots")
    score.set_defaults(handler=command_score)

    detector = subparsers.add_parser("detector-eval", help="core detector on constructed cases")
    detector.add_argument("--manifest", required=True)
    detector.add_argument("--out", default="artifacts/public/evaluation/detector")
    detector.set_defaults(handler=command_detector_eval)
