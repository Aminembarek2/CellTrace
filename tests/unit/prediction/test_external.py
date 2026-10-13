"""PAM adapter follows the Raca rules, and external validation is frozen and run once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from celltrace.canonical import file_sha256
from celltrace.prediction import cross_validation as validation
from celltrace.prediction import external
from celltrace.prediction.features import default_feature_specs, feature_groups

SMALL_CONFIGURATIONS = (
    {"max_iter": 2, "min_samples_leaf": 2},
    {"max_iter": 8, "min_samples_leaf": 2},
)


def write_drive(path: Path, segments: int = 3, seconds: int = 80, seed: int = 0) -> None:
    generator = np.random.default_rng(seed)
    times, rates = [], []
    start = 1_659_975_385.0
    for segment in range(segments):
        offset = start + segment * (seconds + 20)
        for step in range(seconds * 2):
            times.append(offset + step / 2)
            rates.append(float(generator.gamma(2.0, 4.0)) if step % 40 else 0.4)
    pd.DataFrame({"TIME_STAMP": times, external.RATE_COLUMN: rates, "Lat": 34.0}).to_csv(
        path, index=False
    )


def write_pam(directory: Path) -> None:
    directory.mkdir(parents=True)
    entries = []
    for index in range(3):
        path = directory / f"drive_{index}.csv"
        write_drive(path, seed=index)
        entries.append({"file": f"data/raw/pam2025/{path.name}", "sha256": file_sha256(path)})
    (directory / "SOURCE.json").write_text(json.dumps({"files": entries}))


def test_drive_bins_follow_the_raca_ingestion_rules(tmp_path: Path) -> None:
    path = tmp_path / "drive.csv"
    pd.DataFrame(
        {
            "TIME_STAMP": [100.0, 100.5, 101.0, 101.5, None, 106.0, 106.5],
            external.RATE_COLUMN: [2.0, 4.0, 6.0, -1.0, 9.0, 8.0, None],
        }
    ).to_csv(path, index=False)
    samples, counts = external.drive_samples(path)
    assert counts == {"rows": 7, "rows_with_rate": 5, "rejected_negative_or_nonfinite": 1}
    assert list(samples["elapsed_ms"]) == [0, 500, 1000, 6000]
    bins = external.drive_bins(samples)
    # 5000 ms between 1000 and 6000 exceeds the 3000 ms gap rule, so a new segment starts.
    assert list(bins["segment_id"]) == [0, 0, 1]
    assert list(bins["downlink_mbps"]) == [3.0, 6.0, 8.0]
    assert list(bins["available_at_ms"]) == [1000, 2000, 7000]
    assert list(bins["is_complete"]) == [True, True, False]


def test_external_features_are_the_training_history_features() -> None:
    expected = feature_groups(default_feature_specs())["history_only"]
    assert feature_groups(external.history_specs())["history_only"] == expected


def test_external_missing_context_is_not_reported_as_an_observation(tmp_path: Path) -> None:
    path = tmp_path / "drive.csv"
    write_drive(path)
    frame, _ = external.build_external_frame([{"path": str(path), "drive": "fixture"}])
    assert not frame.empty
    assert frame["transition_in_window"].isna().all()
    assert frame["target_active_fraction"].isna().all()


def test_changed_admitted_file_is_refused(tmp_path: Path) -> None:
    write_pam(tmp_path / "pam")
    with (tmp_path / "pam" / "drive_1.csv").open("a") as handle:
        handle.write("1659999999.0,1.0,34.0\n")
    with pytest.raises(ValueError, match="admitted hash"):
        external.verified_files(tmp_path / "pam")


def training_build(names: tuple[str, ...]) -> SimpleNamespace:
    generator = np.random.default_rng(3)
    parts = []
    for session in range(6):
        rows = 60
        history = generator.gamma(2.0, 3.0 + session, rows)
        part = pd.DataFrame({name: history + generator.normal(0, 0.1, rows) for name in names})
        part["session_id"] = f"raca_{session}"
        part["segment_id"] = 0
        part["as_of_ms"] = np.arange(rows) * 1000
        part["target_mbps"] = np.clip(history + generator.normal(0, 1, rows), 0, None)
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    return SimpleNamespace(frame=frame, feature_groups={"history_only": names})


def frozen_plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plan.md"
    plan.write_text("frozen protocol")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "plan": str(plan),
                "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                "frozen_at_utc": "2026-10-11T14:06:19+00:00",
            }
        )
    )
    return freeze


def patch_training(monkeypatch: pytest.MonkeyPatch, seen: list[set[str]]) -> None:
    names = feature_groups(external.history_specs())["history_only"]
    build = training_build(names)
    connection = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(external, "open_snapshot", lambda _: (connection, {"snapshot_id": "s"}))
    monkeypatch.setattr(external, "build_feature_frame", lambda _: build)
    monkeypatch.setattr(validation, "CONFIGURATIONS", SMALL_CONFIGURATIONS)
    original = validation.fit_fold

    def recording_fit(train: pd.DataFrame, test: pd.DataFrame, columns: Any) -> Any:
        seen.append(set(train["session_id"]))
        return original(train, test, columns)

    monkeypatch.setattr(validation, "fit_fold", recording_fit)


def test_plan_changed_after_freeze_refuses_the_run(tmp_path: Path) -> None:
    freeze = frozen_plan(tmp_path)
    (tmp_path / "plan.md").write_text("edited after the freeze")
    with pytest.raises(ValueError, match="frozen plan changed"):
        external.run_external_validation(
            tmp_path / "snapshot", tmp_path / "pam", freeze, tmp_path / "out", tmp_path / "private"
        )


def test_external_validation_trains_on_raca_only_and_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pam(tmp_path / "pam")
    seen: list[set[str]] = []
    patch_training(monkeypatch, seen)
    freeze = frozen_plan(tmp_path)
    arguments = (tmp_path / "snapshot", tmp_path / "pam", freeze, tmp_path / "out", tmp_path / "p")
    summary = external.run_external_validation(*arguments)
    assert seen == [{f"raca_{index}" for index in range(6)}]
    assert summary["training_sessions"] == 6 and summary["rows"] > 0
    assert set(summary["scores"]["drives"]) == {"pam_drive_0", "pam_drive_1", "pam_drive_2"}
    assert set(summary["scores"]["decisions"]) >= {
        "point_forecast_transfers",
        "global_cqr_transfers",
        "mondrian_cqr_transfers",
    }
    record = json.loads((tmp_path / "p" / external.RECORD_NAME).read_text())
    assert record["predictions_sha256"] == file_sha256(tmp_path / "p" / external.PREDICTIONS_NAME)
    assert (
        "The protocol was frozen (sha256" in (tmp_path / "out" / external.REPORT_NAME).read_text()
    )
    with pytest.raises(FileExistsError, match="runs once"):
        external.run_external_validation(*arguments)
