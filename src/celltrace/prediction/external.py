"""External validation of the history only forecast on the PAM 2025 AT&T downlink drives.

The protocol was frozen before any PAM outcome was used (docs/TECHNICAL_REPORT.md). PAM
samples go through the same rules as Raca ingestion (3 s gaps start a segment, one second
bins hold the mean of their samples, availability at bin end, only the final bin of a drive
incomplete) and then through the same feature functions, so training and test features
cannot drift apart. The model, interval corrections and probability calibrator are fitted
on the 21 Raca sessions only. PAM has no activity state or comparable cell identities, so
target activity is set from the throughput samples and transitions are not observed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from celltrace.canonical import content_hash, file_sha256
from celltrace.data.raca import BIN_MS
from celltrace.data.schema import SEGMENT_GAP_THRESHOLD_MS
from celltrace.data.snapshot import open_snapshot
from celltrace.prediction import cross_validation as validation
from celltrace.prediction.features import (
    DEFAULT_HORIZON_MS,
    DEFAULT_MAXIMUM_HISTORY_MS,
    HISTORY,
    _segment_grid,
    _segment_rows,
    build_feature_frame,
    default_feature_specs,
    feature_groups,
)
from celltrace.release import source_tree_hash

TIME_COLUMN = "TIME_STAMP"
RATE_COLUMN = "Smart Phone Smart Throughput Mobile Network DL Throughput [Mbps]"
FEATURE_SET = "history_only"
SUMMARY_NAME = "external_validation_summary.json"
REPORT_NAME = "external_validation_report.md"
FIGURE_NAME = "external_warning.png"
PREDICTIONS_NAME = "external_predictions.csv"
RECORD_NAME = "external_validation_record.json"
COVERAGE_RANGE = (0.75, 0.85)


def verified_files(directory: Path) -> list[dict[str, str]]:
    """PAM files with the hashes recorded at admission; any mismatch stops the run."""
    source = json.loads((directory / "SOURCE.json").read_text())
    records = []
    for entry in source["files"]:
        path = Path(entry["file"])
        local = directory / path.name
        digest = file_sha256(local)
        if digest != entry["sha256"]:
            raise ValueError(f"{local} does not match its admitted hash")
        records.append({"drive": local.stem, "path": str(local), "sha256": digest})
    return records


def drive_samples(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Elapsed time from the first throughput sample, and finite non negative rates."""
    frame = pd.read_csv(path, usecols=[TIME_COLUMN, RATE_COLUMN], low_memory=False)
    times = pd.to_numeric(frame[TIME_COLUMN], errors="coerce")
    rates = pd.to_numeric(frame[RATE_COLUMN], errors="coerce")
    present = times.notna() & rates.notna()
    valid = present & np.isfinite(rates) & (rates >= 0)
    counts = {
        "rows": len(frame),
        "rows_with_rate": int(present.sum()),
        "rejected_negative_or_nonfinite": int((present & ~valid).sum()),
    }
    samples = pd.DataFrame({"time_s": times[valid], "downlink_mbps": rates[valid]})
    samples = samples.sort_values("time_s", kind="stable").reset_index(drop=True)
    samples["elapsed_ms"] = np.rint((samples["time_s"] - samples["time_s"].iloc[0]) * 1000)
    samples["elapsed_ms"] = samples["elapsed_ms"].astype(np.int64)
    return samples, counts


def drive_bins(samples: pd.DataFrame) -> pd.DataFrame:
    """One second bins per segment, following the Raca ingestion rules."""
    gaps = samples["elapsed_ms"].diff().fillna(0)
    samples = samples.assign(
        segment_id=(gaps > SEGMENT_GAP_THRESHOLD_MS).cumsum().astype(int),
        bin_start_ms=(samples["elapsed_ms"] // BIN_MS) * BIN_MS,
    )
    bins = (
        samples.groupby(["segment_id", "bin_start_ms"], sort=True)["downlink_mbps"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "downlink_mbps", "size": "sample_count"})
        .reset_index()
    )
    bins["available_at_ms"] = bins["bin_start_ms"] + BIN_MS
    bins["is_complete"] = bins["bin_start_ms"] != bins["bin_start_ms"].max()
    # No activity state exists in PAM. Throughput test samples are counted as active.
    bins["active_count"] = bins["sample_count"]
    return bins


def history_specs() -> tuple[Any, ...]:
    return tuple(spec for spec in default_feature_specs() if spec.group == HISTORY)


def build_external_frame(files: list[dict[str, str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    specs = history_specs()
    horizon_bins = DEFAULT_HORIZON_MS // BIN_MS
    history_bins = DEFAULT_MAXIMUM_HISTORY_MS // BIN_MS
    parts = []
    exclusions: dict[str, Any] = {}
    for record in files:
        samples, counts = drive_samples(Path(record["path"]))
        bins = drive_bins(samples)
        stages: dict[str, int] = {}
        for segment_id, segment in bins.groupby("segment_id", sort=True):
            grid = _segment_grid(segment.drop(columns="segment_id"), {}, {})
            rows, segment_counts = _segment_rows(grid, specs, horizon_bins, history_bins)
            for stage, value in segment_counts.items():
                stages[stage] = stages.get(stage, 0) + value
            rows.insert(0, "segment_id", int(str(segment_id)))
            rows.insert(0, "session_id", f"pam_{record['drive']}")
            parts.append(rows)
        exclusions[record["drive"]] = {
            **counts,
            "segments": int(bins["segment_id"].nunique()),
            "bins": len(bins),
            **stages,
        }
    frame = pd.concat(parts, ignore_index=True)
    frame["mobility_cohort"] = "driving"
    frame["target_start_ms"] = frame["as_of_ms"]
    frame["target_end_ms"] = frame["as_of_ms"] + DEFAULT_HORIZON_MS
    # Schema placeholders must not assert observations missing from this source.
    frame["transition_in_window"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["target_active_fraction"] = np.nan
    for column in ("latest_available_ms", "earliest_support_ms"):
        frame[column] = frame[column].astype(np.int64)
    return frame, exclusions


def score_external(rows: pd.DataFrame) -> dict[str, Any]:
    """Pooled and per drive scores. Bootstrap units are test segments, labeled descriptive."""
    scored = rows.assign(session_id=rows["session_id"] + ":" + rows["segment_id"].astype(str))
    pooled = validation.scheme_scores(scored)
    drives = {
        str(drive): validation.scheme_scores(scored.loc[rows["session_id"] == drive])
        for drive in sorted(rows["session_id"].unique())
    }
    skills = [
        item["point"][f"{FEATURE_SET}_mean"]["skill_vs_persistence"]["value"]
        for item in drives.values()
    ]
    decisions: dict[str, Any] = {
        "point_forecast_transfers": all(value is not None and value > 0 for value in skills),
        "point_rule": "MAE skill against persistence positive in all three drives",
    }
    for kind in ("global_cqr", "mondrian_cqr"):
        coverage = pooled["calibrated_interval"][FEATURE_SET][kind]["row_coverage"]["value"]
        decisions[f"{kind}_transfers"] = bool(
            coverage is not None and COVERAGE_RANGE[0] <= coverage <= COVERAGE_RANGE[1]
        )
    decisions["interval_rule"] = "pooled calibrated coverage between 75 and 85 percent"
    return {"pooled": pooled, "drives": drives, "decisions": decisions}


def _row(name: str, item: dict[str, Any]) -> str:
    point = item["point"]
    interval = item["calibrated_interval"][FEATURE_SET]
    warning = item["warning"]
    onset = warning["future_low_given_current_not_low"]
    return (
        f"| {name} | {item['rows']} | "
        f"{validation._percent(point[f'{FEATURE_SET}_mean']['skill_vs_persistence'])} | "
        f"{validation._percent(point['trailing_mean']['skill_vs_persistence'])} | "
        f"{validation._percent(interval['raw']['row_coverage'])} | "
        f"{validation._percent(interval['global_cqr']['row_coverage'])} | "
        f"{validation._percent(interval['mondrian_cqr']['row_coverage'])} | "
        f"{_three_way(warning)} | {_three_way(onset)} |"
    )


def _three_way(warning: dict[str, Any]) -> str:
    """Model, persistence and trailing mean average precision, in that order."""
    return " / ".join(
        validation._decimal(warning[name]["average_precision"])
        for name in (
            f"{FEATURE_SET}_warning",
            "persistence_warning_score",
            "trailing_mean_warning_score",
        )
    )


def render_report(summary: dict[str, Any]) -> str:
    scores = summary["scores"]
    pooled = scores["pooled"]
    decisions = scores["decisions"]
    interval = pooled["calibrated_interval"][FEATURE_SET]
    lines = [
        "# External validation on PAM 2025",
        "",
        "The history only forecast was trained on all 21 Raca file download sessions from "
        "Ireland, 2019 to 2020, and applied unchanged to AT&T downlink traces from three drives "
        "in the PAM 2025 dataset, recorded in the United States with a different phone and tool. "
        "Configuration, interval corrections and probability calibration came from inner Raca "
        "session folds only. The protocol was frozen "
        f"(sha256 `{summary['plan_sha256'][:16]}…`) before any PAM outcome was used, and the "
        "evaluation ran once.",
        "",
        "| Data | Rows | Skill against persistence | Trailing mean skill against persistence "
        "| Raw 80% coverage | Global CQR coverage | Mondrian CQR coverage | Warning AP, "
        "model / persistence / trailing mean | Same, current rate at least 1.5 Mbps |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
        _row("All three drives", pooled),
        *(_row(name.removeprefix("pam_"), item) for name, item in scores["drives"].items()),
        "",
        "Decisions fixed in advance:",
        "",
        f"- Point forecast transfers ({decisions['point_rule']}): "
        f"**{'yes' if decisions['point_forecast_transfers'] else 'no'}**.",
        f"- Global CQR intervals transfer ({decisions['interval_rule']}): "
        f"**{'yes' if decisions['global_cqr_transfers'] else 'no'}**.",
        f"- Mondrian CQR intervals transfer: "
        f"**{'yes' if decisions['mondrian_cqr_transfers'] else 'no'}**.",
        "",
        "Coverage by predicted level quartile, all drives:",
        "",
        "| Interval | " + " | ".join(f"Quartile {index}" for index in range(4)) + " |",
        "| --- |" + " ---: |" * 4,
    ]
    for kind, label in (
        ("raw", "Raw quantiles"),
        ("global_cqr", "Global CQR"),
        ("mondrian_cqr", "Mondrian CQR"),
    ):
        table = interval[kind]["coverage_by_group"]["predicted_level_quartile"]
        cells = [
            f"{100 * table[str(index)]['coverage']:.1f}%" if str(index) in table else "n/a"
            for index in range(4)
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Bootstrap intervals resample test segments. Segments of one drive share a route and "
        "a day, so they are descriptive, and the per drive rows are the primary view. Units "
        "differ in origin: PAM publishes Mbps, Raca units are inferred as kbps. PAM has no "
        "activity state, and cell transitions were not derived, so neither enters the model.",
        "",
        f"Exclusions per drive: `{json.dumps(summary['exclusions'], sort_keys=True)}`.",
        "",
    ]
    return "\n".join(lines)


def _protect(snapshot_dir: Path, out_dir: Path, private_dir: Path) -> None:
    public, private = out_dir.resolve(), private_dir.resolve()
    if public == private or public in private.parents or private in public.parents:
        raise ValueError("public and private external validation directories must be disjoint")
    if snapshot_dir.resolve() in (public, private):
        raise ValueError("external validation outputs must be separate from the snapshot")
    targets = [
        out_dir / SUMMARY_NAME,
        out_dir / REPORT_NAME,
        out_dir / FIGURE_NAME,
        private_dir / PREDICTIONS_NAME,
        private_dir / RECORD_NAME,
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("external validation runs once; preserve the existing outputs")


def run_external_validation(
    snapshot_dir: Path, pam_dir: Path, plan_freeze: Path, out_dir: Path, private_dir: Path
) -> dict[str, Any]:
    _protect(snapshot_dir, out_dir, private_dir)
    freeze = json.loads(plan_freeze.read_text())
    plan_sha256 = hashlib.sha256(Path(freeze["plan"]).read_bytes()).hexdigest()
    if plan_sha256 != freeze["sha256"]:
        raise ValueError("the frozen plan changed after its freeze; external validation refused")
    files = verified_files(pam_dir)
    source_hash = source_tree_hash()
    connection, manifest = open_snapshot(snapshot_dir)
    try:
        build = build_feature_frame(connection)
    finally:
        connection.close()
    names = list(build.feature_groups[FEATURE_SET])
    if tuple(names) != feature_groups(history_specs())[FEATURE_SET]:
        raise ValueError("external features differ from the training feature set")
    train = build.frame
    test, exclusions = build_external_frame(files)
    identity = [
        "session_id",
        "segment_id",
        "as_of_ms",
        "mobility_cohort",
        "transition_in_window",
        "target_mbps",
        "earliest_support_ms",
        "latest_available_ms",
        "target_start_ms",
        "target_end_ms",
        "target_active_fraction",
    ]
    rows = test[identity].copy()
    rows["training_sessions"] = train["session_id"].nunique()
    for name, values in validation.baseline_fold(train, test).items():
        rows[name] = values
    for name, values in validation.fit_fold(train, test, names).items():
        rows[f"{FEATURE_SET}_{name}"] = values
    if source_tree_hash() != source_hash:
        raise RuntimeError("source changed during external validation; rerun separately")
    summary: dict[str, Any] = {
        "method": "external_validation_v2_partition_transforms_unknown_context",
        "plan": freeze["plan"],
        "plan_sha256": plan_sha256,
        "plan_frozen_at_utc": freeze["frozen_at_utc"],
        "source_sha256": source_hash,
        "training_snapshot_id": manifest["snapshot_id"],
        "training_snapshot_manifest_sha256": content_hash(manifest),
        "training_sessions": int(train["session_id"].nunique()),
        "training_rows": len(train),
        "external_files": files,
        "feature_set": FEATURE_SET,
        "features": names,
        "configuration_index": int(rows[f"{FEATURE_SET}_configuration_index"].iloc[0]),
        "exclusions": exclusions,
        "context_assumptions": {
            "activity": "unobserved; rate-bearing test samples assumed active for eligibility",
            "cell_changes": "unobserved; transition_in_window is missing, not false",
            "radio": "unobserved and excluded from history-only features",
        },
        "rows": len(rows),
        "scores": score_external(rows),
        "figure": FIGURE_NAME,
        "claim_scope": "external validation on three drives of one operator",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    pooled_rows = rows.assign(session_id=rows["session_id"] + ":" + rows["segment_id"].astype(str))
    validation.figure_warning(
        pooled_rows, summary["scores"]["pooled"]["warning"], out_dir / FIGURE_NAME
    )
    (out_dir / REPORT_NAME).write_text(render_report(summary), encoding="utf-8")
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (out_dir / SUMMARY_NAME).write_text(payload, encoding="utf-8")
    rows.to_csv(private_dir / PREDICTIONS_NAME, index=False, na_rep="")
    record = {
        "summary_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "predictions_sha256": file_sha256(private_dir / PREDICTIONS_NAME),
        "source_sha256": source_hash,
        "plan_sha256": plan_sha256,
    }
    (private_dir / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return summary
