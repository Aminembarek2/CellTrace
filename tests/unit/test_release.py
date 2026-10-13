"""Release provenance must describe saved inference and refuse changed inputs."""

from __future__ import annotations

import json
import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest

from celltrace import release as release_module
from celltrace.canonical import content_hash, file_sha256
from celltrace.config import load_config
from celltrace.data.snapshot import load_manifest
from celltrace.release import (
    experiment_release_inputs,
    make_release_manifest,
    source_tree_hash,
    wheel_source_tree_hash,
)


def saved_experiment(
    directory: Path, snapshot_directory: Path, response_available: bool = True
) -> tuple[Path, Path]:
    case_directory = directory / "cases" / "case_fixture"
    case_directory.mkdir(parents=True)
    shutil.copytree(snapshot_directory, case_directory / "snapshot")
    candidate = {"event_id": "evt_fixture"}
    (case_directory / "candidate.json").write_text(json.dumps(candidate))
    snapshot = load_manifest(snapshot_directory)
    manifest = {
        "freeze": {"source_tree_sha256": source_tree_hash()},
        "cases": [
            {
                "case_id": "case_fixture",
                "candidate_sha256": content_hash(candidate),
                "case_snapshot_id": snapshot["snapshot_id"],
                "case_snapshot_logical_hash": snapshot["logical_hash"],
            }
        ],
    }
    manifest_path = directory / "labels" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest))
    settings = load_config(environment={}).runtime.model_dump(mode="json")
    settings["maximum_input_tokens"] = 240000
    settings["model_id"] = "deepseek/deepseek-v4.1-flash"
    for name in ("verified_rate_card", "verified_endpoint_record", "verified_capability_record"):
        settings[name] = None
    index = {
        "experiment_id": "exp_fixture",
        "mode": "live",
        "freeze": {
            "manifest_sha256": file_sha256(manifest_path),
            "runtime_limits": settings,
            "arms": ["investigator_gated"],
        },
        "runs": [
            {
                "run_id": "run_fixture",
                "case_id": "case_fixture",
                "arm": "investigator_gated",
                "status": "ran",
                "workflow_outcome": "partial",
            }
        ],
    }
    index_path = directory / "experiment" / "index.json"
    index_path.parent.mkdir()
    index_path.write_text(json.dumps(index))
    connection = sqlite3.connect(index_path.parent / "runs.sqlite3")
    try:
        connection.execute("CREATE TABLE runs(run_id TEXT, experiment_id TEXT)")
        connection.execute(
            "CREATE TABLE experiments(experiment_id TEXT, maximum_estimated_usd REAL)"
        )
        connection.execute("INSERT INTO experiments VALUES (?, ?)", ("exp_fixture", 0.25))
        connection.execute(
            """CREATE TABLE actions(run_id TEXT, kind TEXT, state TEXT, usage_known INTEGER,
            actual_usd REAL, reserved_usd REAL, response_json TEXT)"""
        )
        connection.execute("INSERT INTO runs VALUES (?, ?)", ("run_fixture", "exp_fixture"))
        connection.execute(
            "INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "run_fixture",
                "provider_attempt",
                "completed" if response_available else "failed",
                1,
                0.001 if response_available else 0,
                0,
                "{}" if response_available else None,
            ),
        )
        connection.execute(
            "INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("run_fixture", "provider_attempt", "uncertain", 0, None, 0.004, None),
        )
        connection.commit()
    finally:
        connection.close()
    return index_path, manifest_path


def rewrite_manifest(index_path: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(manifest))
    index = json.loads(index_path.read_text())
    index["freeze"]["manifest_sha256"] = file_sha256(manifest_path)
    index_path.write_text(json.dumps(index))


def test_release_records_actual_experiment_settings_and_preserves_unknown_cost(
    tmp_path: Path, fixture_snapshot_dir: Path
) -> None:
    index_path, manifest_path = saved_experiment(tmp_path, fixture_snapshot_dir)
    model_path = tmp_path / "hist_gradient_boosting_fixture.pkl"
    model_path.write_bytes(b"fixture model identity only")
    splits_path = tmp_path / "splits.json"
    splits_path.write_text("{}")
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    output_path = tmp_path / "release_manifest.json"
    release = make_release_manifest(
        fixture_snapshot_dir,
        model_path,
        splits_path,
        raw_directory,
        output_path,
        index_path,
        manifest_path,
    )
    assert release["live_validation"] == "recorded provider inference"
    assert release["runtime_settings"]["maximum_input_tokens"] == 240000
    assert release["runtime_settings"]["run_store"] == str(index_path.parent / "runs.sqlite3")
    assert release["runtime_settings"]["checkpoint_store"] == str(
        index_path.parent / "checkpoints.sqlite3"
    )
    assert release["evaluation_freeze"] == json.loads(manifest_path.read_text())["freeze"]
    assert release["experiment"]["known_usage_usd"] == pytest.approx(0.001)
    assert release["experiment"]["unknown_usage_reserved_usd"] == pytest.approx(0.004)
    assert release["experiment"]["experiment_ceiling_usd"] == pytest.approx(0.25)
    assert release["experiment"]["manual_annotation_status"] == "pending"
    case_snapshot = manifest_path.parent.parent / "cases/case_fixture/snapshot/measurements.duckdb"
    assert release["inputs"][str(case_snapshot)] == file_sha256(case_snapshot)
    with pytest.raises(FileExistsError):
        make_release_manifest(
            fixture_snapshot_dir, model_path, splits_path, raw_directory, output_path
        )


def test_refused_provider_request_is_not_recorded_as_live_inference(
    tmp_path: Path, fixture_snapshot_dir: Path
) -> None:
    index_path, manifest_path = saved_experiment(tmp_path, fixture_snapshot_dir, False)
    _, experiment = experiment_release_inputs(index_path, manifest_path, None)
    assert experiment["recorded_provider_responses"] == 0
    assert (
        experiment["live_validation"]
        == "live requests attempted without a recorded completed response"
    )


def test_release_refuses_incomplete_schedule_active_dispatch_and_changed_case(
    tmp_path: Path, fixture_snapshot_dir: Path
) -> None:
    index_path, manifest_path = saved_experiment(tmp_path, fixture_snapshot_dir)
    original_index = index_path.read_text()
    index = json.loads(original_index)
    index["runs"] = []
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="incomplete"):
        experiment_release_inputs(index_path, manifest_path, None)
    index_path.write_text(original_index)
    connection = sqlite3.connect(index_path.parent / "runs.sqlite3")
    connection.execute("UPDATE actions SET state = 'started' WHERE usage_known = 0")
    connection.commit()
    with pytest.raises(ValueError, match="active provider"):
        experiment_release_inputs(index_path, manifest_path, None)
    connection.execute("UPDATE actions SET state = 'uncertain' WHERE usage_known = 0")
    connection.commit()
    connection.close()
    candidate_path = manifest_path.parent.parent / "cases/case_fixture/candidate.json"
    candidate_path.write_text(json.dumps({"event_id": "changed_event"}))
    with pytest.raises(ValueError, match="candidate changed"):
        experiment_release_inputs(index_path, manifest_path, None)


def test_changed_source_requires_a_matching_archived_experimental_wheel(
    tmp_path: Path, fixture_snapshot_dir: Path
) -> None:
    index_path, manifest_path = saved_experiment(tmp_path, fixture_snapshot_dir)
    frozen_wheel = tmp_path / "frozen.whl"
    with zipfile.ZipFile(frozen_wheel, "w") as archive:
        archive.writestr("celltrace/__init__.py", '"""Archived fixture implementation."""\n')
    manifest = json.loads(manifest_path.read_text())
    manifest["freeze"]["source_tree_sha256"] = wheel_source_tree_hash(frozen_wheel)
    rewrite_manifest(index_path, manifest_path, manifest)
    with pytest.raises(ValueError, match="archived source wheel"):
        experiment_release_inputs(index_path, manifest_path, None)
    inputs, experiment = experiment_release_inputs(index_path, manifest_path, frozen_wheel)
    assert frozen_wheel in inputs
    assert experiment["source_tree_sha256"] != source_tree_hash()
    wrong_wheel = tmp_path / "wrong.whl"
    with zipfile.ZipFile(wrong_wheel, "w") as archive:
        archive.writestr("celltrace/__init__.py", '"""Different implementation."""\n')
    with pytest.raises(ValueError, match="differs from the frozen"):
        experiment_release_inputs(index_path, manifest_path, wrong_wheel)


def test_viewer_template_changes_package_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "celltrace"
    package.mkdir()
    module = package / "release.py"
    module.write_text("# fixture source\n")
    template = package / "viewer.html"
    template.write_text("<p>Original</p>")
    monkeypatch.setattr(release_module, "__file__", str(module))
    original = source_tree_hash()
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(module, "celltrace/release.py")
        archive.write(template, "celltrace/viewer.html")
    assert wheel_source_tree_hash(wheel) == original
    template.write_text("<p>Changed</p>")
    assert source_tree_hash() != original
