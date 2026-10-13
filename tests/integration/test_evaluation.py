"""Case construction, label isolation, benchmark and scoring on a synthetic archive."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import duckdb
import pytest

from celltrace.config import load_config
from celltrace.contracts.common import Arm
from celltrace.contracts.evidence import RunContext
from celltrace.data import raca_lte
from celltrace.data.raca import DATASET_ROOT, RACA_HEADER
from celltrace.data.snapshot import run_ingest
from celltrace.evaluation.benchmark import (
    experiment_directory,
    offline_adapter_factory,
    run_benchmark,
)
from celltrace.evaluation.cases import CaseFamily, build_cases
from celltrace.evaluation.commands import command_cases
from celltrace.evaluation.scoring import score_experiment

SESSION_SECONDS = 300
FAMILY_NAMES = [family.value for family in CaseFamily]


def _row(day: int, second: int, mobility: str) -> list[str]:
    values = {
        "Timestamp": f"2031.05.{day:02d}_10.{second // 60:02d}.{second % 60:02d}",
        "Longitude": "-8.4",
        "Latitude": "51.9",
        "Speed": "0" if mobility == "Static" else "30",
        "Operatorname": "B",
        "CellID": "11",
        "NetworkMode": "5G",
        "RSRP": str(-85 - (second % 2)),
        "RSRQ": "-10",
        "SNR": "10.0",
        "CQI": "12",
        "RSSI": "-70",
        "DL_bitrate": str(50000 + 1000 * (second % 3)),
        "UL_bitrate": "1000",
        "State": "D",
        "PINGAVG": "-",
        "PINGMIN": "-",
        "PINGMAX": "-",
        "PINGSTDEV": "-",
        "PINGLOSS": "-",
        "CELLHEX": "B",
        "NODEHEX": "A4DF",
        "LACHEX": "9CBA",
        "RAWCELLID": str(0xA4DF * 256 + 11),
        "NRxRSRP": "-100.0",
        "NRxRSRQ": "-12.0",
    }
    return [values[name] for name in RACA_HEADER]


def write_long_archive(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "long.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index, mobility in enumerate(["Static"] * 3 + ["Driving"] * 3):
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow(RACA_HEADER)
            for second in range(SESSION_SECONDS):
                writer.writerow(_row(index + 1, second, mobility))
            member = f"{DATASET_ROOT}/Download/{mobility}/L_{index}.csv"
            archive.writestr(
                zipfile.ZipInfo(member, date_time=(2031, 1, 1, 0, 0, 0)), buffer.getvalue()
            )
    (directory / "SOURCE.json").write_text(json.dumps({"source_id": "synthetic_long", "files": {}}))
    return archive_path


@pytest.fixture(scope="module")
def built_cases(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("evaluation")
    write_long_archive(base / "raw")
    snapshot = run_ingest(base / "raw", base / "processed")
    config = load_config(environment={})
    manifests = {
        split: build_cases(
            snapshot, base / "raw", base / "eval_private", split, config, maximum_base_windows=3
        )
        for split in ("dev", "heldout")
    }
    return {"base": base, "snapshot": snapshot, "manifests": manifests}


def test_splits_are_disjoint_and_cases_cover_families(built_cases: dict[str, Any]) -> None:
    dev = json.loads(built_cases["manifests"]["dev"].read_text())
    heldout = json.loads(built_cases["manifests"]["heldout"].read_text())
    dev_sources = {case["source_session_id"] for case in dev["cases"]}
    heldout_sources = {case["source_session_id"] for case in heldout["cases"]}
    assert dev_sources and heldout_sources and not (dev_sources & heldout_sources)
    families = [case["family"] for case in heldout["cases"]]
    assert families.count("clean_control") == len(heldout["cases"]) // 3


def test_constructed_numbers_match_the_construction(built_cases: dict[str, Any]) -> None:
    cases = json.loads(built_cases["manifests"]["dev"].read_text())["cases"]
    cases += json.loads(built_cases["manifests"]["heldout"].read_text())["cases"]
    for case in cases:
        facts = case["expected_facts"]
        if case["family"] in {
            "stable_radio_rate_drop",
            "rate_drop_with_radio_change",
            "misleading_string",
        }:
            assert facts["downlink_relative_decrease"] == pytest.approx(0.85, abs=0.01)
        if case["family"] == "rate_drop_with_radio_change":
            assert facts["rsrp_delta_db"] == pytest.approx(-15, abs=1)
        if case["family"] == "stable_radio_rate_drop":
            assert abs(facts["rsrp_delta_db"]) <= 1
        if case["family"] == "application_idle":
            assert facts["active_fraction_reference_to_event"] < 1
        if case["family"] == "missing_metrics":
            assert facts["rsrp_delta_db"] is None


@pytest.mark.parametrize("mode", ["HSPA+", "mixed"])
def test_case_builder_never_requires_radio_labels_on_unsupported_windows(
    tmp_path: Path, mode: str
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "long.zip", "w") as archive:
        for index in range(3):
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow(RACA_HEADER)
            for second in range(SESSION_SECONDS):
                row = _row(index + 1, second, "Static")
                # Even one unsupported native sample in a bin invalidates the
                # radio interpretation, although bin means look unchanged.
                row[RACA_HEADER.index("NetworkMode")] = (
                    "HSPA+" if mode == "HSPA+" or second % 2 == 0 else "5G"
                )
                writer.writerow(row)
            archive.writestr(f"{DATASET_ROOT}/Download/Static/L_{index}.csv", buffer.getvalue())
    snapshot = run_ingest(raw, tmp_path / "processed")
    path = build_cases(
        snapshot,
        raw,
        tmp_path / "cases",
        "dev",
        load_config(environment={}),
        maximum_base_windows=3,
        variants_per_base=8,
    )
    manifest = json.loads(path.read_text())
    assert manifest["case_builder_version"] == "case_builder_5"
    assert manifest["cases"]  # 3G remains useful for non-radio tests.
    radio = {"rate_drop_with_radio_change", "stable_radio_rate_drop"}
    assert not any(case["family"] in radio for case in manifest["cases"])
    for case in manifest["cases"]:
        assert radio <= set(case["omitted_inapplicable_families"])
        assert not case["base_window"]["radio_supported_in_windows"]
    groups: dict[tuple[str, int], list[str]] = {}
    for case in manifest["cases"]:
        key = (case["source_session_id"], case["source_confirmation_ms"])
        groups.setdefault(key, []).append(case["family"])
    assert all(len(families) == len(set(families)) for families in groups.values())


def _lte_row(second: int, operator: str = "A") -> str:
    start = 10 * 3600
    clock = start + second
    values = {
        "Timestamp": f"2018.01.16_{clock // 3600:02d}.{clock % 3600 // 60:02d}.{clock % 60:02d}",
        "Longitude": "-8.47",
        "Latitude": "51.89",
        "Speed": "30",
        "Operatorname": operator,
        "CellID": "2",
        "NetworkMode": "LTE",
        "RSRP": str(-90 - second % 2),
        "RSRQ": "-10",
        "SNR": "5",
        "CQI": "9",
        "RSSI": "-60",
        "DL_bitrate": str(8000 + 400 * (second % 3)),
        "UL_bitrate": "50",
        "State": "D",
        "NRxRSRP": "-",
        "NRxRSRQ": "-",
        "ServingCell_Lon": "-8.46",
        "ServingCell_Lat": "51.90",
        "ServingCell_Distance": "300",
    }
    return ",".join(values[name] for name in raca_lte.LTE_HEADER)


def test_cases_from_the_2018_layout_keep_car_sessions_and_omit_cell_changes(
    tmp_path: Path,
) -> None:
    # Cases from sessions no agent has seen.
    raw = tmp_path / "raw"
    raw.mkdir()
    header = ",".join(raca_lte.LTE_HEADER)
    rows = [header] + [_lte_row(second) for second in range(SESSION_SECONDS)]
    with zipfile.ZipFile(raw / "LTE_Dataset.zip", "w") as archive:
        archive.writestr("Dataset/car/A_2018.01.16_10.00.00.csv", "\n".join(rows) + "\n")
        archive.writestr("Dataset/bus/A_2018.01.16_11.00.00.csv", "\n".join(rows) + "\n")
    snapshot = run_ingest(
        raw,
        tmp_path / "processed",
        sources=[raca_lte.source_archive(raw, "A")],
        session_iterator=raca_lte.iter_sessions,
    )
    path = build_cases(
        snapshot,
        raw,
        tmp_path / "cases",
        "heldout",
        load_config(environment={}),
        base_windows_per_session=1,
        variants_per_base=8,
    )
    manifest = json.loads(path.read_text())
    assert manifest["case_builder_version"] == "case_builder_5"
    assert all(case["base_window"]["mobility_cohort"] == "car" for case in manifest["cases"])
    assert sum(len(sessions) for sessions in manifest["session_split"].values()) == 1
    families = {case["family"] for case in manifest["cases"]}
    assert {"clean_control", "stable_radio_rate_drop", "rate_drop_with_radio_change"} <= families
    assert not families & {"cell_change_with_rate_drop", "cell_change_without_rate_drop"}
    case_dir = path.parent.parent / "cases" / manifest["cases"][0]["case_id"]
    connection = duckdb.connect(str(case_dir / "snapshot" / "measurements.duckdb"), read_only=True)
    try:
        cohorts = connection.execute(
            "SELECT application_cohort, mobility_cohort FROM sessions"
        ).fetchall()
    finally:
        connection.close()
    assert cohorts == [("file_download", "driving")]


def test_runtime_case_material_carries_no_labels_or_source_identity(
    built_cases: dict[str, Any],
) -> None:
    for split, manifest_path in built_cases["manifests"].items():
        manifest = json.loads(manifest_path.read_text())
        cases_dir = manifest_path.parent.parent / "cases"
        for case in manifest["cases"]:
            case_dir = cases_dir / case["case_id"]
            assert sorted(path.name for path in case_dir.iterdir()) == [
                "candidate.json",
                "snapshot",
            ]
            runtime_text = (case_dir / "candidate.json").read_text()
            runtime_text += (case_dir / "snapshot" / "snapshot_manifest.json").read_text()
            connection = duckdb.connect(
                str(case_dir / "snapshot" / "measurements.duckdb"), read_only=True
            )
            try:
                runtime_text += json.dumps(
                    connection.execute("SELECT * FROM sessions").fetchall(), default=str
                )
                runtime_text += json.dumps(
                    connection.execute("SELECT * FROM snapshot_metadata").fetchall(), default=str
                )
            finally:
                connection.close()
            for label in FAMILY_NAMES:
                assert label not in runtime_text, (split, label)
            assert case["source_session_id"] not in runtime_text
            assert "2031.05." not in runtime_text  # the source recording date
            assert "L_" not in runtime_text  # the source file name


def test_freeze_refuses_to_rebuild(built_cases: dict[str, Any], tmp_path: Path) -> None:
    arguments = argparse.Namespace(
        action="freeze",
        split="heldout",
        snapshot=str(built_cases["snapshot"]),
        raw=str(built_cases["base"] / "raw"),
        out=str(tmp_path / "eval_private"),
        base_windows=2,
        variants=2,
    )
    assert command_cases(arguments) == 0
    manifest = json.loads(
        (tmp_path / "eval_private" / "heldout" / "labels" / "manifest.json").read_text()
    )
    assert manifest["frozen"] and manifest["freeze"]["rubric_sha256"]
    assert command_cases(arguments) == 2


def test_freeze_records_the_configuration_the_benchmark_will_use(
    built_cases: dict[str, Any], tmp_path: Path
) -> None:
    from celltrace.config import load_config
    from celltrace.evaluation.benchmark import FreezeMismatch, verify_freeze

    live_config = tmp_path / "live.yaml"
    live_config.write_text("runtime:\n  maximum_input_tokens: 160000\n")
    arguments = argparse.Namespace(
        action="freeze",
        split="heldout",
        snapshot=str(built_cases["snapshot"]),
        raw=str(built_cases["base"] / "raw"),
        out=str(tmp_path / "eval_private"),
        base_windows=2,
        variants=2,
        config=live_config,
    )
    assert command_cases(arguments) == 0
    manifest = json.loads(
        (tmp_path / "eval_private" / "heldout" / "labels" / "manifest.json").read_text()
    )
    verify_freeze(manifest["freeze"], load_config(live_config))
    with pytest.raises(FreezeMismatch, match="runtime_limits_sha256"):
        verify_freeze(manifest["freeze"], load_config())


def test_benchmark_stop_file_halts_between_runs_and_resumes(
    built_cases: dict[str, Any], tmp_path: Path
) -> None:
    from celltrace.evaluation.benchmark import experiment_directory

    manifest_path = built_cases["manifests"]["heldout"]
    directory = experiment_directory(manifest_path, "exp_stop_test")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "STOP").touch()
    arguments: dict[str, Any] = {
        "config": load_config(environment={}),
        "experiment_id": "exp_stop_test",
        "case_limit": 1,
    }
    index_path = run_benchmark(
        manifest_path, "offline", list(Arm), offline_adapter_factory, **arguments
    )
    stopped = json.loads(index_path.read_text())
    assert stopped["stopped_by_operator"] is True and stopped["runs"] == []
    (directory / "STOP").unlink()
    index_path = run_benchmark(
        manifest_path, "offline", list(Arm), offline_adapter_factory, **arguments
    )
    resumed = json.loads(index_path.read_text())
    assert resumed["stopped_by_operator"] is False
    assert len(resumed["runs"]) == 3


def test_offline_benchmark_and_scoring_keep_every_attempt(
    built_cases: dict[str, Any], tmp_path: Path
) -> None:
    manifest_path = built_cases["manifests"]["heldout"]
    index_path = run_benchmark(
        manifest_path,
        "offline",
        list(Arm),
        offline_adapter_factory,
        config=load_config(environment={}),
        experiment_id="exp_test",
        case_limit=4,
    )
    index = json.loads(index_path.read_text())
    assert len(index["runs"]) == 4 * 3
    assert all(entry["status"] == "ran" for entry in index["runs"])
    summary = score_experiment(index_path, manifest_path, tmp_path / "score")
    assert summary["attempted_runs"] == 12
    assert summary["scheduled_runs"] == 12
    assert len(summary["paired_differences_by_model"]) == 1
    assert "OFFLINE HARNESS CHECK" in summary["label"]
    rows = list(csv.DictReader((tmp_path / "score" / "runs.csv").open()))
    raw_rows = [row for row in rows if row["arm"] == "investigator_raw"]
    assert raw_rows and all(row["fact_check_status"] == "not_run" for row in raw_rows)
    # Scorer side checks run for arm A, so support is measured, not assumed.
    assert all(row["factual_support"] not in ("", None) for row in raw_rows)
    index["runs"].append(
        {
            "run_id": "unavailable_comparison",
            "case_id": index["runs"][0]["case_id"],
            "model_id": "unavailable_model",
            "arm": "investigator_gated",
            "status": "unavailable",
        }
    )
    index_path.write_text(json.dumps(index))
    mixed_summary = score_experiment(index_path, manifest_path, tmp_path / "mixed_score")
    assert mixed_summary["scheduled_runs"] == 13
    assert mixed_summary["attempted_runs"] == 12
    assert mixed_summary["paired_differences"] == {}
    unavailable_group = next(
        group for group in mixed_summary["by_arm"] if group["model_id"] == "unavailable_model"
    )
    assert unavailable_group["unavailable"] == 1
    assert unavailable_group["failed"] == 0
    assert unavailable_group["dispatched_runs"] == 0
    assert unavailable_group["total_estimated_usd"] == 0
    for name in (
        "by_arm.csv",
        "by_family.csv",
        "tool_usage.csv",
        "manual_annotation_template.csv",
        "summary.md",
    ):
        assert (tmp_path / "score" / name).exists()


def test_runtime_context_cannot_reach_labels_or_other_cases(built_cases: dict[str, Any]) -> None:
    from celltrace.clock import SystemClock
    from celltrace.data.snapshot import open_snapshot
    from celltrace.storage.run_store import RunStore
    from celltrace.tools.dispatch import ToolDispatcher
    from celltrace.tools.repository import MeasurementRepository

    manifest_path = built_cases["manifests"]["heldout"]
    directory = experiment_directory(manifest_path, "exp_test")
    store = RunStore(directory / "runs.sqlite3", SystemClock())
    try:
        runs = store.list_runs("exp_test")
        first, second = runs[0], next(run for run in runs if run["case_id"] != runs[0]["case_id"])
        context = RunContext.model_validate(first["context"])
        context_text = json.dumps(first["context"])
        assert "manifest" not in context_text and "labels" not in context_text
        for label in FAMILY_NAMES:
            assert label not in context_text
        other_evidence = store.run_evidence(second["run_id"])[0]["evidence_id"]
        connection, _ = open_snapshot(Path(context.snapshot_path))
        try:
            dispatcher = ToolDispatcher(
                context, MeasurementRepository(connection), store, SystemClock()
            )
            stolen = dispatcher.dispatch("get_evidence", {"evidence_id": other_evidence})
            assert stolen.error is not None and stolen.error.code.value == "OUT_OF_SCOPE"
            other_session = RunContext.model_validate(second["context"]).session_id
            denied = dispatcher.dispatch("get_session_context", {"session_id": other_session})
            assert denied.error is not None and denied.error.code.value == "OUT_OF_SCOPE"
        finally:
            connection.close()
    finally:
        store.close()
