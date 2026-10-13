"""Content identified release inputs and reproduction through existing modules."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import uuid
import zipfile
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from celltrace.canonical import content_hash, file_sha256

RELEASE_VERSION = "release_v1"
SOURCE_SUFFIXES = frozenset((".py", ".md", ".html"))


def source_tree_hash() -> str:
    source_directory = Path(__file__).resolve().parent
    return content_hash(
        {
            path.relative_to(source_directory).as_posix(): file_sha256(path)
            for path in sorted(source_directory.rglob("*"))
            if path.is_file() and path.suffix in SOURCE_SUFFIXES
        }
    )


def wheel_source_tree_hash(wheel_path: Path) -> str:
    """Verify archived experimental code using the same identity as installed code."""
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("source wheel contains duplicate members")
        hashes = {
            name.removeprefix("celltrace/"): hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(names)
            if name.startswith("celltrace/") and Path(name).suffix in SOURCE_SUFFIXES
        }
    if not hashes:
        raise ValueError("source wheel has no CellTrace source")
    return content_hash(hashes)


def experiment_release_inputs(
    index_path: Path, manifest_path: Path, source_wheel: Path | None
) -> tuple[list[Path], dict[str, Any]]:
    """Require a finished schedule and record the settings that actually produced it."""
    index = json.loads(index_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    identity = index["freeze"]
    if identity["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("experiment does not match the private case manifest")
    cases = manifest["cases"][: identity.get("case_limit") or None]
    expected_pairs = {(case["case_id"], arm) for case in cases for arm in identity["arms"]}
    recorded_pairs = [(entry["case_id"], entry["arm"]) for entry in index["runs"]]
    if len(set(recorded_pairs)) != len(recorded_pairs) or set(recorded_pairs) != expected_pairs:
        raise ValueError("experiment schedule is incomplete or contains duplicate entries")
    database_path = index_path.parent / "runs.sqlite3"
    connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        allowance = connection.execute(
            "SELECT maximum_estimated_usd FROM experiments WHERE experiment_id = ?",
            (index["experiment_id"],),
        ).fetchone()
        if allowance is None:
            raise ValueError("experiment allowance is absent from the durable ledger")
        provider_rows = connection.execute(
            """SELECT actions.state, actions.usage_known, actions.actual_usd,
            actions.reserved_usd, actions.response_json FROM actions JOIN runs USING (run_id)
            WHERE runs.experiment_id = ? AND actions.kind = 'provider_attempt'""",
            (index["experiment_id"],),
        ).fetchall()
        if any(row[0] in ("reserved", "started") for row in provider_rows):
            raise ValueError("experiment has active provider attempts")
    finally:
        connection.close()
    frozen_source = manifest.get("freeze", {}).get("source_tree_sha256")
    if frozen_source is None:
        raise ValueError("experiment has no recorded source identity")
    if source_wheel is None and frozen_source != source_tree_hash():
        raise ValueError("changed experiment source requires its archived source wheel")
    inputs = [index_path, manifest_path, database_path]
    checkpoint_path = index_path.parent / "checkpoints.sqlite3"
    if checkpoint_path.exists():
        inputs.append(checkpoint_path)
    if source_wheel is not None:
        if wheel_source_tree_hash(source_wheel) != frozen_source:
            raise ValueError("source wheel differs from the frozen experiment source")
        inputs.append(source_wheel)
    case_directory = manifest_path.parent.parent / "cases"
    from celltrace.data.snapshot import load_manifest, verify_logical_hash

    for case in cases:
        directory = case_directory / case["case_id"]
        candidate_path = directory / "candidate.json"
        if content_hash(json.loads(candidate_path.read_text())) != case["candidate_sha256"]:
            raise ValueError("case candidate changed since the experiment freeze")
        snapshot_directory = directory / "snapshot"
        snapshot_manifest = load_manifest(snapshot_directory)
        if (
            snapshot_manifest["snapshot_id"] != case["case_snapshot_id"]
            or snapshot_manifest["logical_hash"] != case["case_snapshot_logical_hash"]
            or not verify_logical_hash(snapshot_directory)
        ):
            raise ValueError("case snapshot changed since the experiment freeze")
        inputs.append(candidate_path)
        inputs.extend(path for path in snapshot_directory.iterdir() if path.is_file())
    settings = dict(identity["runtime_limits"])
    settings["run_store"] = str(database_path)
    settings["checkpoint_store"] = str(checkpoint_path)
    for name in ("verified_rate_card", "verified_endpoint_record", "verified_capability_record"):
        if settings.get(name):
            inputs.append(Path(settings[name]))
    provider_responses = sum(row[0] == "completed" and row[4] is not None for row in provider_rows)
    live_validation = "no live inference in this release"
    if index["mode"] == "live":
        live_validation = (
            "recorded provider inference"
            if provider_responses
            else "live requests attempted without a recorded completed response"
        )
    metadata = {
        "experiment_id": index["experiment_id"],
        "mode": index["mode"],
        "source_tree_sha256": frozen_source,
        "archived_source_wheel": None if source_wheel is None else str(source_wheel),
        "runtime_settings": settings,
        "evaluation_freeze": manifest["freeze"],
        "experiment_identity": identity,
        "recorded_runs": len(recorded_pairs),
        "outcomes": dict(
            Counter(entry.get("workflow_outcome", entry["status"]) for entry in index["runs"])
        ),
        "provider_attempts": len(provider_rows),
        "experiment_ceiling_usd": float(allowance[0]),
        "recorded_provider_responses": provider_responses,
        "known_usage_usd": sum(float(row[2] or 0) for row in provider_rows if row[1]),
        "unknown_usage_reserved_usd": sum(
            float(row[3] or 0) for row in provider_rows if not row[1]
        ),
        "live_validation": live_validation,
        "manual_annotation_status": "pending",
    }
    return inputs, metadata


def make_release_manifest(
    snapshot_dir: Path,
    model_path: Path,
    splits_path: Path,
    raw_dir: Path,
    output_path: Path,
    experiment_index: Path | None = None,
    case_manifest: Path | None = None,
    experiment_source_wheel: Path | None = None,
) -> dict[str, Any]:
    from celltrace.agents.roles import role_identity
    from celltrace.config import load_config
    from celltrace.evaluation.commands import freeze_hashes
    from celltrace.tools.dispatch import TOOL_VERSION

    if output_path.exists():
        raise FileExistsError(f"preserve existing release manifest at {output_path}")
    inputs = [Path("uv.lock"), Path("configs/project-defaults.yaml"), splits_path]
    inputs.extend(path for path in snapshot_dir.iterdir() if path.is_file())
    inputs.extend(sorted(raw_dir.rglob("*.zip")))
    inputs.extend(sorted(raw_dir.rglob("SOURCE.json")))
    inputs.extend(sorted(model_path.parent.glob("hist_gradient_boosting*.pkl")))
    inputs.extend(sorted(model_path.parent.glob("hist_gradient_boosting*.json")))
    if (experiment_index is None) != (case_manifest is None):
        raise ValueError("experiment index and private case manifest must be supplied together")
    experiment = None
    if experiment_source_wheel is not None and experiment_index is None:
        raise ValueError("experiment source wheel requires an experiment index")
    if experiment_index is not None and case_manifest is not None:
        experiment_inputs, experiment = experiment_release_inputs(
            experiment_index, case_manifest, experiment_source_wheel
        )
        inputs.extend(experiment_inputs)
    defaults = load_config(environment={})
    payload = {
        "release_version": RELEASE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_tree_hash": source_tree_hash(),
        "inputs": {str(path): file_sha256(path) for path in sorted(set(inputs))},
        "snapshot": str(snapshot_dir),
        "model": str(model_path),
        "splits": str(splits_path),
        "raw": str(raw_dir),
        "experiment_index": None if experiment_index is None else str(experiment_index),
        "case_manifest": None if case_manifest is None else str(case_manifest),
        "live_validation": "no live inference included"
        if experiment is None
        else experiment["live_validation"],
        "execution": "local reproduction from saved results, no provider inference",
        "claim_scope": {
            "semantic_accuracy": "excluded",
            "human_agreement": "not_measured",
            "physical_root_cause": "not_established",
            "planted_errors": "localized challenge behavior; free-text correctness ungraded",
        },
        "experiment": experiment,
        "roles": role_identity(),
        "tool_version": TOOL_VERSION,
        "evaluation_freeze": freeze_hashes(defaults)
        if experiment is None
        else experiment["evaluation_freeze"],
        "runtime_settings": defaults.runtime.model_dump(mode="json")
        if experiment is None
        else experiment["runtime_settings"],
        "installed_versions": {
            package: version(package)
            for package in (
                "celltrace",
                "duckdb",
                "httpx",
                "pydantic",
                "pandas",
                "numpy",
                "scikit-learn",
                "matplotlib",
                "langgraph",
                "langgraph-checkpoint-sqlite",
            )
        },
    }
    payload["manifest_hash"] = content_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def reproduce(manifest_path: Path, output_directory: Path) -> dict[str, Any]:
    from celltrace.analysis.detector import run_detect
    from celltrace.analysis.explore import run_explore
    from celltrace.config import load_config
    from celltrace.data.audit import run_audit
    from celltrace.prediction.evaluate import run_evaluate
    from celltrace.workflow.demo import run_offline_demo

    manifest = json.loads(manifest_path.read_text())
    recorded_hash = manifest.pop("manifest_hash", None)
    if (
        recorded_hash != content_hash(manifest)
        or manifest.get("release_version") != RELEASE_VERSION
    ):
        raise ValueError("release manifest content or version changed")
    if manifest["source_tree_hash"] != source_tree_hash():
        raise ValueError("release source changed, record a new release manifest")
    for file_name, expected_hash in manifest["inputs"].items():
        if file_sha256(Path(file_name)) != expected_hash:
            raise ValueError(f"release input changed: {file_name}")
    output_directory = output_directory / f"execution_{uuid.uuid4().hex[:16]}"
    output_directory.mkdir(parents=True, exist_ok=False)
    snapshot_dir = Path(manifest["snapshot"])
    results: dict[str, Any] = {
        "manifest_hash": recorded_hash,
        "output_directory": str(output_directory),
    }
    results["audit"] = run_audit(Path(manifest["raw"]), output_directory / "audit")
    results["explore"] = run_explore(snapshot_dir, output_directory / "explore")
    results["candidates"] = str(
        run_detect(
            snapshot_dir,
            output_directory / "candidates",
            load_config(environment={}).analysis.detector,
        )
    )
    prediction = run_evaluate(
        Path(manifest["model"]),
        Path(manifest["splits"]),
        snapshot_dir,
        output_directory / "prediction",
    )
    results["prediction_metrics"] = str(output_directory / "prediction" / "metrics.json")
    results["private_prediction_manifest"] = prediction["private_prediction_manifest"]
    results["demo"] = run_offline_demo(output_directory / "demo")
    if manifest["experiment_index"] is not None:
        from celltrace.evaluation.scoring import score_experiment

        results["scoring"] = score_experiment(
            Path(manifest["experiment_index"]),
            Path(manifest["case_manifest"]),
            output_directory / "evaluation",
        )
    results["status"] = "passed" if results["demo"]["status"] == "passed" else "failed"
    (output_directory / "reproduction.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    return results


def command_manifest(arguments: argparse.Namespace) -> int:
    payload = make_release_manifest(
        Path(arguments.snapshot),
        Path(arguments.model),
        Path(arguments.splits),
        Path(arguments.raw),
        Path(arguments.out),
        Path(arguments.experiment_index) if arguments.experiment_index else None,
        Path(arguments.case_manifest) if arguments.case_manifest else None,
        Path(arguments.experiment_source_wheel) if arguments.experiment_source_wheel else None,
    )
    print(json.dumps({"manifest": arguments.out, "manifest_hash": payload["manifest_hash"]}))
    return 0


def command_reproduce(arguments: argparse.Namespace) -> int:
    from celltrace.cli import CommandError

    try:
        results = reproduce(Path(arguments.manifest), Path(arguments.out))
    except (ValueError, OSError) as failure:
        raise CommandError(f"release reproduction refused: {failure}") from failure
    print(
        json.dumps(
            {
                "status": results["status"],
                "output_directory": results["output_directory"],
                "manifest_hash": results["manifest_hash"],
            }
        )
    )
    return 0 if results["status"] == "passed" else 1


def register(subparsers: Any) -> None:
    manifest = subparsers.add_parser(
        "release-manifest", help="hash release inputs without overwriting"
    )
    manifest.add_argument("--snapshot", required=True)
    manifest.add_argument("--model", required=True)
    manifest.add_argument("--splits", required=True)
    manifest.add_argument("--raw", default="data/raw")
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--experiment-index", default=None)
    manifest.add_argument("--case-manifest", default=None)
    manifest.add_argument("--experiment-source-wheel", default=None)
    manifest.set_defaults(handler=command_manifest)
    command = subparsers.add_parser(
        "reproduce", help="verify release inputs and regenerate artifacts"
    )
    command.add_argument("--manifest", required=True)
    command.add_argument("--out", default="artifacts/public/reproduction")
    command.set_defaults(handler=command_reproduce)
