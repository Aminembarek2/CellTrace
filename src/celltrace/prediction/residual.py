"""Residual candidate detector built on the session split forecaster.

residual = actual minus predicted mean downlink over [as_of, as_of + 5 s).
An unusually negative residual means the observed rate fell well below what
the forecaster expected from past data. The threshold is chosen on dev
sessions only, frozen with the model hash, and then applied to test sessions.
A residual flag is a candidate for investigation, never a network fault, and
its comparison with the core detector is agreement between two heuristics,
not precision against real fault labels.

A residual is confirmed only when its target window has been observed, so
the confirmation time is target_end_ms, five seconds after issuance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltrace.canonical import content_hash, file_sha256
from celltrace.prediction.models import (
    BOOSTING,
    PERSISTENCE,
    TRAILING_MEAN,
    ForecastModel,
    load_model,
    mean_absolute_error,
    persistence_forecast,
    trailing_mean_forecast,
)

RESIDUAL_VERSION = "prediction_residual_v1"
THRESHOLD_SUFFIX = ".residual_threshold.json"
THRESHOLD_FILE_NAME = "residual_threshold.json"
RESIDUAL_CANDIDATES_FILE = "residual_candidates.json"
QUANTILE_LEVELS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
MINIMUM_PAST_ACTIVE_FRACTION = 0.8
MINIMUM_TARGET_ACTIVE_FRACTION = 0.8
MINIMUM_HISTORY_COVERAGE = 0.8
FLAG_BUDGET_SECONDS = 600
MERGE_GAP_MS = 1_000
MATCH_TOLERANCE_MS = 5_000
NOT_A_FAULT = (
    "A residual flag means the observed downlink was well below the forecast. It is a candidate "
    "for investigation, not evidence of a network fault or of any cause."
)


class ThresholdError(ValueError):
    pass


def forecast(frame: pd.DataFrame, forecaster: str, model: ForecastModel) -> np.ndarray:
    if forecaster == PERSISTENCE:
        return persistence_forecast(frame)
    if forecaster == TRAILING_MEAN:
        return trailing_mean_forecast(frame)
    return model.predict(frame)


def eligible_mask(frame: pd.DataFrame, history_bins: int = 30) -> np.ndarray:
    """Activity in the past and in the target window, plus past downlink coverage."""
    valid_history = frame["history_bins_available_30s"] - frame["downlink_missing_bins_30s"]
    coverage = valid_history / history_bins
    past_active = frame["active_fraction_30s"].fillna(0.0)
    target_active = frame["target_active_fraction"].fillna(0.0)
    return np.asarray(
        (past_active >= MINIMUM_PAST_ACTIVE_FRACTION)
        & (target_active >= MINIMUM_TARGET_ACTIVE_FRACTION)
        & (coverage >= MINIMUM_HISTORY_COVERAGE),
        dtype=bool,
    )


def group_flags(rows: pd.DataFrame) -> list[pd.DataFrame]:
    """Flagged rows of one session and segment whose issuance times are within MERGE_GAP_MS."""
    groups: list[pd.DataFrame] = []
    ordered = rows.sort_values(["session_id", "segment_id", "as_of_ms"])
    for _, part in ordered.groupby(["session_id", "segment_id"], sort=True):
        breaks = part["as_of_ms"].diff().fillna(MERGE_GAP_MS + 1) > MERGE_GAP_MS
        labels = breaks.cumsum()
        groups.extend(group for _, group in part.groupby(labels, sort=True))
    return groups


def choose_threshold(residuals: np.ndarray, rows: pd.DataFrame) -> dict[str, Any]:
    """Least negative quantile threshold whose grouped dev flags stay within the budget.

    rows must be the eligible dev rows aligned with residuals. Scanning stops
    at the first level that exceeds the budget or is not negative.
    """
    eligible_seconds = len(rows)
    budget = eligible_seconds / FLAG_BUDGET_SECONDS
    scan: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    if eligible_seconds == 0:
        return {"threshold_mbps": None, "scan": scan, "budget_flags": 0.0, "eligible_seconds": 0}
    for level in QUANTILE_LEVELS:
        threshold = float(np.quantile(residuals, level))
        if threshold >= 0:
            scan.append({"quantile": level, "threshold_mbps": threshold, "stop": "not negative"})
            break
        flagged = rows[residuals <= threshold]
        events = len(group_flags(flagged))
        entry = {
            "quantile": level,
            "threshold_mbps": threshold,
            "flagged_rows": len(flagged),
            "grouped_flags": events,
            "within_budget": events <= budget,
        }
        scan.append(entry)
        if events > budget:
            break
        chosen = entry
    return {
        "threshold_mbps": None if chosen is None else chosen["threshold_mbps"],
        "quantile": None if chosen is None else chosen["quantile"],
        "dev_grouped_flags": None if chosen is None else chosen["grouped_flags"],
        "budget_flags": budget,
        "eligible_seconds": eligible_seconds,
        "scan": scan,
    }


def calibrate_rows(frame: pd.DataFrame, model: ForecastModel) -> dict[str, Any]:
    """Pick the forecaster and threshold from dev session rows only.

    frame may hold every split. Rows outside session_group == dev are dropped
    before any prediction or statistic is computed.
    """
    dev = frame[frame["session_group"] == "dev"].reset_index(drop=True)
    if dev.empty:
        raise ThresholdError("no dev rows to calibrate on")
    actual = dev["target_mbps"].to_numpy(dtype=float)
    dev_errors = {
        name: mean_absolute_error(actual, forecast(dev, name, model))
        for name in (BOOSTING, PERSISTENCE, TRAILING_MEAN)
    }
    best_baseline = min((PERSISTENCE, TRAILING_MEAN), key=lambda name: dev_errors[name])
    forecaster = BOOSTING if dev_errors[BOOSTING] <= dev_errors[best_baseline] else best_baseline
    residuals = actual - forecast(dev, forecaster, model)
    eligible = eligible_mask(dev)
    choice = choose_threshold(residuals[eligible], dev[eligible])
    return {
        "forecaster": forecaster,
        "forecaster_reason": "boosting dev MAE is not above the best baseline"
        if forecaster == BOOSTING
        else f"boosting dev MAE is above {best_baseline}, so {best_baseline} supplies residuals",
        "dev_mae_mbps": dev_errors,
        "dev_rows": len(dev),
        "dev_sessions": sorted(dev["session_id"].unique().tolist()),
        "dev_eligible_rows": int(eligible.sum()),
        **choice,
    }


def threshold_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.stem + THRESHOLD_SUFFIX)


def run_calibrate(
    model_path: Path, snapshot_dir: Path, splits_path: Path, out_dir: Path
) -> dict[str, Any]:
    from celltrace.prediction.splits import labelled_rows, load_split_manifest

    manifest = load_split_manifest(splits_path)
    model, sidecar = load_model(model_path)
    if sidecar["split_manifest_hash"] != manifest["manifest_hash"]:
        raise ThresholdError("model was trained on a different split manifest")
    if sidecar.get("split_kind") != "session":
        raise ThresholdError("calibrate a session split model so dev residuals are out of sample")
    _, frame, snapshot_manifest = labelled_rows(snapshot_dir, manifest)
    calibration = calibrate_rows(frame, model)
    frozen: dict[str, Any] = {
        "residual_version": RESIDUAL_VERSION,
        "model_artifact": model_path.name,
        "model_sha256": file_sha256(model_path),
        "split_manifest_hash": manifest["manifest_hash"],
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "residual_definition": "actual minus predicted mean downlink over [as_of, as_of + 5 s), "
        "Mbps",
        "flag_rule": "eligible row with residual <= threshold_mbps",
        "eligibility": {
            "minimum_past_30s_active_fraction": MINIMUM_PAST_ACTIVE_FRACTION,
            "minimum_target_active_fraction": MINIMUM_TARGET_ACTIVE_FRACTION,
            "minimum_past_30s_downlink_coverage": MINIMUM_HISTORY_COVERAGE,
        },
        "budget_rule": f"at most one grouped flag per {FLAG_BUDGET_SECONDS} s of eligible "
        "dev issuance time",
        "quantile_levels": list(QUANTILE_LEVELS),
        "merge_gap_ms": MERGE_GAP_MS,
        "calibration": calibration,
        "statement": NOT_A_FAULT,
    }
    frozen["threshold_hash"] = content_hash(frozen)
    text = json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n"
    threshold_path(model_path).write_text(text, encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / THRESHOLD_FILE_NAME).write_text(text, encoding="utf-8")
    return frozen


def load_threshold(model_path: Path) -> dict[str, Any]:
    path = threshold_path(model_path)
    if not path.is_file():
        raise ThresholdError(f"no frozen threshold at {path}. Run predict calibrate-residual.")
    frozen: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    recorded = frozen.pop("threshold_hash", None)
    if recorded != content_hash(frozen):
        raise ThresholdError(f"{path} content does not match its threshold_hash")
    if frozen["model_sha256"] != file_sha256(model_path):
        raise ThresholdError("model artifact changed after the threshold was frozen")
    frozen["threshold_hash"] = recorded
    return frozen


def _window_gap_ms(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, max(first[0], second[0]) - min(first[1], second[1]))


def compare_with_core(
    residual_events: Sequence[Mapping[str, Any]], core_candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Overlap within MATCH_TOLERANCE_MS between residual target windows and core event windows."""

    def matches(event: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
        if event["session_id"] != candidate["session_id"]:
            return False
        residual_window = (event["target_window"]["start_ms"], event["target_window"]["end_ms"])
        core_window = (candidate["event_window"]["start_ms"], candidate["event_window"]["end_ms"])
        return _window_gap_ms(residual_window, core_window) <= MATCH_TOLERANCE_MS

    residual_matched = [
        event["residual_event_id"]
        for event in residual_events
        if any(matches(event, candidate) for candidate in core_candidates)
    ]
    core_matched = [
        candidate["event_id"]
        for candidate in core_candidates
        if any(matches(event, candidate) for event in residual_events)
    ]
    return {
        "match_rule": f"same session and window gap at most {MATCH_TOLERANCE_MS} ms between the "
        "residual target window and the core event window",
        "residual_flags": len(residual_events),
        "core_candidates": len(core_candidates),
        "residual_flags_matching_core": len(residual_matched),
        "core_candidates_matching_residual": len(core_matched),
        "residual_only": [
            event["residual_event_id"]
            for event in residual_events
            if event["residual_event_id"] not in residual_matched
        ],
        "core_only": [
            candidate["event_id"]
            for candidate in core_candidates
            if candidate["event_id"] not in core_matched
        ],
        "interpretation": "agreement between two heuristics on unlabeled real traces. Without "
        "verified fault labels this is not precision or recall of network faults",
    }


def _event_record(index: int, group: pd.DataFrame, threshold: float) -> dict[str, Any]:
    worst: dict[Any, Any] = group.iloc[int(np.argmin(group["residual_mbps"].to_numpy()))].to_dict()
    first_as_of = int(group["as_of_ms"].min())
    last_as_of = int(group["as_of_ms"].max())
    return {
        "residual_event_id": f"res_{index:04d}",
        "session_id": str(worst["session_id"]),
        "segment_id": int(worst["segment_id"]),
        "first_issuance_ms": first_as_of,
        "last_issuance_ms": last_as_of,
        "target_window": {"start_ms": first_as_of, "end_ms": int(group["target_end_ms"].max())},
        "first_confirmation_ms": int(group["target_end_ms"].min()),
        "flag_count": len(group),
        "threshold_mbps": threshold,
        "most_negative_residual_mbps": float(worst["residual_mbps"]),
        "most_negative_issuance_ms": int(worst["as_of_ms"]),
        "predicted_mbps_at_most_negative": float(worst["predicted_mbps"]),
        "actual_mbps_at_most_negative": float(worst["target_mbps"]),
        "flags": [
            {
                "issuance_ms": int(row["as_of_ms"]),
                "target_end_ms": int(row["target_end_ms"]),
                "predicted_mbps": float(row["predicted_mbps"]),
                "actual_mbps": float(row["target_mbps"]),
                "residual_mbps": float(row["residual_mbps"]),
            }
            for row in group.to_dict(orient="records")
        ],
    }


def run_residual_detect(snapshot_dir: Path, model_artifact: Path, out_dir: Path) -> Path:
    """Apply the frozen model and threshold to test sessions and write residual_candidates.json."""
    from celltrace.analysis.detector import detect_candidates, detector_config_hash
    from celltrace.config import load_config
    from celltrace.data.snapshot import open_snapshot
    from celltrace.prediction.splits import labelled_rows, load_split_manifest

    frozen = load_threshold(model_artifact)
    model, sidecar = load_model(model_artifact)
    manifest = load_split_manifest(Path(sidecar["split_manifest_path"]))
    if manifest["manifest_hash"] != frozen["split_manifest_hash"]:
        raise ThresholdError("split manifest differs from the one used for calibration")
    _, frame, snapshot_manifest = labelled_rows(snapshot_dir, manifest)
    test = frame[frame["session_group"] == "test"].reset_index(drop=True)
    calibration = frozen["calibration"]
    threshold = calibration["threshold_mbps"]
    forecaster = calibration["forecaster"]
    test["predicted_mbps"] = forecast(test, forecaster, model)
    test["residual_mbps"] = test["target_mbps"] - test["predicted_mbps"]
    eligible = eligible_mask(test)
    events: list[dict[str, Any]] = []
    flagged = test.iloc[0:0]
    if threshold is not None:
        flagged = test[eligible & (test["residual_mbps"] <= threshold)]
        events = [
            _event_record(index, group, float(threshold))
            for index, group in enumerate(group_flags(flagged))
        ]

    test_sessions = sorted(manifest["session_split"]["sessions"]["test"])
    detector_config = load_config().analysis.detector
    connection, _ = open_snapshot(snapshot_dir)
    try:
        core = detect_candidates(connection, detector_config, session_ids=test_sessions)
    finally:
        connection.close()
    core_candidates = [candidate.model_dump(mode="json") for candidate in core.candidates]
    comparison = compare_with_core(events, core_candidates)
    eligible_seconds = int(eligible.sum())
    positive_controls = (
        int((eligible & (test["residual_mbps"] >= -threshold)).sum())
        if threshold is not None
        else 0
    )
    payload: dict[str, Any] = {
        "detector_name": "prediction_residual",
        "residual_version": RESIDUAL_VERSION,
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "model_artifact": model_artifact.name,
        "model_sha256": frozen["model_sha256"],
        "model_name": sidecar["name"],
        "threshold_hash": frozen["threshold_hash"],
        "threshold_mbps": threshold,
        "forecaster": forecaster,
        "split_manifest_hash": manifest["manifest_hash"],
        "evaluated_sessions": test_sessions,
        "session_scope": "test sessions of the whole session split only",
        "semantics": {
            "issuance_time": "as_of_ms, all predictors available by then",
            "target_window": "[as_of_ms, as_of_ms + 5000), half open",
            "confirmation_time": "target_end_ms, when the target window has been observed",
            "residual": frozen["residual_definition"],
            "grouping": f"flags of one session and segment within {MERGE_GAP_MS} ms",
            "statement": NOT_A_FAULT,
        },
        "counts": {
            "test_rows": len(test),
            "eligible_rows": eligible_seconds,
            "flagged_rows": len(flagged),
            "grouped_flags": len(events),
            "positive_residual_controls_at_same_magnitude": positive_controls,
            "flags_per_10_minutes_eligible": (
                len(events) / (eligible_seconds / FLAG_BUDGET_SECONDS) if eligible_seconds else None
            ),
            "dev_grouped_flags": calibration.get("dev_grouped_flags"),
            "dev_eligible_rows": calibration.get("dev_eligible_rows"),
        },
        "residual_candidates": events,
        "core_detector": {
            "config_hash": detector_config_hash(detector_config),
            "eligibility_counts": core.eligibility_counts,
            "candidates": [
                {
                    "event_id": candidate["event_id"],
                    "session_id": candidate["session_id"],
                    "event_window": candidate["event_window"],
                    "first_confirmation_ms": candidate["first_confirmation_ms"],
                }
                for candidate in core_candidates
            ],
        },
        "comparison_with_core_detector": comparison,
        "null_result": _null_result_statement(events, comparison, threshold),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RESIDUAL_CANDIDATES_FILE
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def _null_result_statement(
    events: Sequence[Mapping[str, Any]], comparison: Mapping[str, Any], threshold: float | None
) -> str | None:
    if threshold is None:
        return "no negative dev threshold met the flag budget, so the residual detector is unused"
    if not events:
        return "the frozen threshold flagged no test rows, the residual detector added nothing"
    if comparison["residual_flags_matching_core"] == 0 and comparison["core_candidates"]:
        return (
            "no residual flag overlaps a core detector candidate. Without labels this shows the "
            "two heuristics disagree, not which one is useful"
        )
    return None
