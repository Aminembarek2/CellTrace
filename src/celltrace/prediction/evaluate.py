"""Forecast error on held out sessions and on the forward time split.

The two evaluations answer different questions (transfer to unseen sessions
versus later periods of sessions already seen) and are reported side by
side, never averaged. Errors are in Mbps. MAPE is not used because zero and
near zero targets are meaningful observations.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from celltrace.prediction.features import FEATURE_GROUP_NAMES
from celltrace.prediction.models import (
    BOOSTING,
    PERSISTENCE,
    SPLIT_COLUMNS,
    TRAILING_MEAN,
    ForecastModel,
    artifact_name,
    load_model,
    persistence_forecast,
    trailing_mean_forecast,
)

METRICS_JSON = "metrics.json"
METRICS_MARKDOWN = "metrics.md"
FIGURE_SESSION = "forecast_test_session.png"
FIGURE_COHORT = "error_by_cohort.png"
BASELINES = (PERSISTENCE, TRAILING_MEAN)
MAIN_MODELS = (PERSISTENCE, TRAILING_MEAN, f"{BOOSTING}_combined")
UNIT_NOTE = (
    "Rates are in Mbps. The raw bitrate unit was inferred as kbps and divided by 1000, so every "
    "error inherits that unit inference."
)
LIMITATIONS = (
    UNIT_NOTE,
    "The target is the mean observed application downlink over the next five one second bins. "
    "It is measured traffic, not available network capacity.",
    "Rows need five complete bins with a valid downlink value, which excludes most bin "
    "boundaries because logging cadence leaves empty one second bins. The excluded counts are "
    "listed under feature_rows.",
    "Few sessions per cohort are held out, so cohort errors are descriptive and unstable.",
    "Transition rows are those with any qualified reported identity transition in the past "
    "30 s or the target window. A transition is a proxy, not a confirmed handover.",
)


def error_summary(
    actual: np.ndarray, predicted: np.ndarray, sessions: np.ndarray
) -> dict[str, Any]:
    """Sample weighted and session balanced MAE and RMSE in Mbps."""
    finite = np.isfinite(predicted) & np.isfinite(actual)
    summary: dict[str, Any] = {
        "sample_count": int(finite.sum()),
        "missing_prediction_count": int((~finite).sum()),
        "session_count": len(set(sessions[finite].tolist())),
    }
    if not finite.any():
        return {
            **summary,
            "mae_mbps": None,
            "rmse_mbps": None,
            "session_balanced_mae_mbps": None,
            "session_balanced_rmse_mbps": None,
            "mean_error_mbps": None,
        }
    errors = predicted[finite] - actual[finite]
    frame = pd.DataFrame({"session": sessions[finite], "absolute": np.abs(errors)})
    frame["squared"] = errors**2
    per_session = frame.groupby("session")[["absolute", "squared"]].mean()
    return {
        **summary,
        "mae_mbps": float(np.mean(np.abs(errors))),
        "rmse_mbps": float(math.sqrt(float(np.mean(errors**2)))),
        "session_balanced_mae_mbps": float(per_session["absolute"].mean()),
        "session_balanced_rmse_mbps": float(math.sqrt(float(per_session["squared"].mean()))),
        "mean_error_mbps": float(np.mean(errors)),
    }


def relative_difference(model_error: float | None, baseline_error: float | None) -> float | None:
    """(model minus baseline) divided by baseline, defined only for a nonzero baseline."""
    if model_error is None or baseline_error is None or baseline_error == 0:
        return None
    return (model_error - baseline_error) / baseline_error


def add_predictions(
    frame: pd.DataFrame, boosting_models: Mapping[str, ForecastModel]
) -> pd.DataFrame:
    """Columns prediction_<model> for both baselines and each boosting feature group."""
    result = frame.copy()
    result[f"prediction_{PERSISTENCE}"] = persistence_forecast(frame)
    result[f"prediction_{TRAILING_MEAN}"] = trailing_mean_forecast(frame)
    for group, model in boosting_models.items():
        result[f"prediction_{BOOSTING}_{group}"] = model.predict(frame) if len(frame) else []
    return result


def _summaries(rows: pd.DataFrame, model_names: Sequence[str]) -> dict[str, Any]:
    actual = rows["target_mbps"].to_numpy(dtype=float)
    sessions = rows["session_id"].to_numpy(dtype=object)
    return {
        name: error_summary(actual, rows[f"prediction_{name}"].to_numpy(dtype=float), sessions)
        for name in model_names
    }


def evaluate_rows(rows: pd.DataFrame, model_names: Sequence[str]) -> dict[str, Any]:
    overall = _summaries(rows, model_names)
    by_cohort = {
        str(cohort): _summaries(part, model_names)
        for cohort, part in rows.groupby("mobility_cohort", sort=True)
    }
    by_stratum = {
        label: _summaries(rows[rows["transition_in_window"] == flag], model_names)
        for label, flag in (("transition", True), ("stable", False))
    }
    relative: dict[str, Any] = {}
    for name in model_names:
        if name in BASELINES:
            continue
        relative[name] = {
            f"vs_{baseline}": {
                metric: relative_difference(
                    overall[name][f"{metric}_mbps"], overall[baseline][f"{metric}_mbps"]
                )
                for metric in ("mae", "rmse")
            }
            for baseline in BASELINES
        }
    return {
        "row_count": len(rows),
        "session_count": int(rows["session_id"].nunique()),
        "sessions": sorted(rows["session_id"].unique().tolist()),
        "overall": overall,
        "by_mobility_cohort": by_cohort,
        "by_transition_stratum": by_stratum,
        "baseline_relative_difference": relative,
    }


def load_family(model_path: Path) -> tuple[dict[str, dict[str, ForecastModel]], dict[str, Any]]:
    """The main artifact plus its related artifacts, each verified against its sidecar."""
    _, main_sidecar = load_model(model_path)
    family: dict[str, dict[str, ForecastModel]] = {kind: {} for kind in SPLIT_COLUMNS}
    records: dict[str, Any] = {}
    for kind in SPLIT_COLUMNS:
        for group in FEATURE_GROUP_NAMES:
            path = model_path.parent / f"{artifact_name(kind, group)}.pkl"
            if not path.is_file():
                records[path.name] = {"status": "missing"}
                continue
            model, sidecar = load_model(path)
            if sidecar["split_manifest_hash"] != main_sidecar["split_manifest_hash"]:
                raise ValueError(f"{path.name} was trained on a different split manifest")
            family[kind][group] = model
            records[path.name] = {
                "status": "loaded",
                "model_sha256": sidecar["model_sha256"],
                "chosen_configuration": sidecar["chosen_configuration"],
                "training_row_count": sidecar["training_row_count"],
                "dev_scores": sidecar["dev_scores"],
                "failed_fits": sidecar["failed_fits"],
            }
    return family, {"main": main_sidecar, "artifacts": records}


def run_evaluate(
    model_path: Path,
    splits_path: Path,
    snapshot_dir: Path,
    out_dir: Path,
    predictions_dir: Path | None = None,
) -> dict[str, Any]:
    from celltrace.prediction.splits import labelled_rows, load_split_manifest

    manifest = load_split_manifest(splits_path)
    family, records = load_family(model_path)
    if records["main"]["split_manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("model was trained on a different split manifest")
    features, frame, snapshot_manifest = labelled_rows(snapshot_dir, manifest)
    if records["main"]["snapshot_logical_hash"] != snapshot_manifest["logical_hash"]:
        raise ValueError("model snapshot logical hash differs from evaluation data")
    if records["main"]["feature_specification_hash"] != features.specification_hash:
        raise ValueError("model feature specification differs from evaluation features")
    sections: dict[str, Any] = {}
    predicted_rows: dict[str, pd.DataFrame] = {}
    for kind, column in SPLIT_COLUMNS.items():
        rows = add_predictions(frame[frame[column] == "test"], family[kind])
        names = [*BASELINES, *(f"{BOOSTING}_{group}" for group in family[kind])]
        section = evaluate_rows(rows, names)
        section["ablation"] = {
            group: section["overall"].get(f"{BOOSTING}_{group}") for group in FEATURE_GROUP_NAMES
        }
        sections[kind] = section
        predicted_rows[kind] = rows
    failed = [
        failure
        for record in records["artifacts"].values()
        for failure in record.get("failed_fits", [])
    ]
    payload: dict[str, Any] = {
        "task": "forecast mean observed downlink over [as_of, as_of + 5 s) from data available "
        "by as_of",
        "unit": "Mbps",
        "unit_note": UNIT_NOTE,
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "split_manifest_hash": manifest["manifest_hash"],
        "feature_specification_hash": features.specification_hash,
        "feature_rows": features.exclusions,
        "split_row_counts": {
            kind: {str(key): int(value) for key, value in frame[column].value_counts().items()}
            for kind, column in SPLIT_COLUMNS.items()
        },
        "models": records["artifacts"],
        "session_holdout": sections["session"],
        "forward_time": sections["forward_time"],
        "failed_fits": failed,
        "metric_definitions": {
            "mae_mbps": "mean absolute error over rows",
            "rmse_mbps": "square root of the mean squared error over rows",
            "session_balanced_mae_mbps": "mean over sessions of each session MAE",
            "session_balanced_rmse_mbps": "square root of the mean over sessions of each "
            "session mean squared error",
            "mean_error_mbps": "mean of predicted minus actual",
            "baseline_relative_difference": "(model error minus baseline error) divided by "
            "baseline error, null when the baseline error is zero. Negative means the model "
            "error is lower",
        },
        "not_averaged": "session_holdout and forward_time answer different questions and are "
        "never combined into one number",
        "limitations": list(LIMITATIONS),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = predictions_dir or model_path.parent / "predictions" / uuid.uuid4().hex
    predictions_dir.mkdir(parents=True, exist_ok=False)
    from celltrace.canonical import file_sha256

    prediction_files = {}
    error_examples: list[dict[str, Any]] = []
    for split_kind, rows in predicted_rows.items():
        selected_columns = [
            "session_id",
            "segment_id",
            "as_of_ms",
            "earliest_support_ms",
            "latest_available_ms",
            "target_start_ms",
            "target_end_ms",
            "target_mbps",
            "mobility_cohort",
            *(column for column in rows.columns if column.startswith("prediction_")),
        ]
        prediction_path = predictions_dir / f"{split_kind}_predictions.csv"
        rows[selected_columns].to_csv(prediction_path, index=False)
        prediction_files[split_kind] = {
            "path": str(prediction_path),
            "sha256": file_sha256(prediction_path),
        }
        combined_column = f"prediction_{BOOSTING}_combined"
        if combined_column in rows and not rows.empty:
            learned_error = (rows[combined_column] - rows["target_mbps"]).abs()
            baseline_error = (rows[f"prediction_{PERSISTENCE}"] - rows["target_mbps"]).abs()
            sections[split_kind]["persistence_beats_combined_rows"] = int(
                (baseline_error < learned_error).sum()
            )
            worst_positions = learned_error.nlargest(5).index
            error_examples.extend(
                {"split_kind": split_kind, **{str(key): value for key, value in record.items()}}
                for record in rows.loc[worst_positions, selected_columns].to_dict(orient="records")
            )
    prediction_manifest = {
        "snapshot_logical_hash": snapshot_manifest["logical_hash"],
        "split_manifest_hash": manifest["manifest_hash"],
        "model_sha256": records["main"]["model_sha256"],
        "files": prediction_files,
    }
    prediction_manifest_path = predictions_dir / "manifest.json"
    prediction_manifest_path.write_text(
        json.dumps(prediction_manifest, indent=2, sort_keys=True) + "\n"
    )
    (predictions_dir / "error_examples.json").write_text(
        json.dumps(error_examples, indent=2, sort_keys=True) + "\n"
    )
    payload["private_prediction_manifest"] = str(prediction_manifest_path)
    payload["figures"] = {
        "forecast_test_session": str(
            plot_session(predicted_rows["session"], out_dir / FIGURE_SESSION)
        ),
        "error_by_cohort": str(plot_cohort_errors(sections, out_dir / FIGURE_COHORT)),
    }
    (out_dir / METRICS_JSON).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out_dir / METRICS_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")
    return payload


# figures


def _with_gaps(rows: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Reindex to every second so excluded issuance times appear as breaks in the lines."""
    indexed = rows.set_index("as_of_ms")[list(columns)]
    full = np.arange(int(indexed.index.min()), int(indexed.index.max()) + 1000, 1000)
    return indexed.reindex(full)


def plot_session(rows: pd.DataFrame, path: Path) -> Path:
    counts = rows.groupby("session_id").size().sort_values(ascending=False, kind="stable")
    session_id = str(counts.index[0])
    session_rows = rows[rows["session_id"] == session_id].sort_values("as_of_ms")
    series = {
        "actual next 5 s mean": "target_mbps",
        "persistence": f"prediction_{PERSISTENCE}",
        "trailing mean 30 s": f"prediction_{TRAILING_MEAN}",
        "boosting combined": f"prediction_{BOOSTING}_combined",
    }
    present = {label: column for label, column in series.items() if column in session_rows}
    gridded = _with_gaps(session_rows, list(present.values()))
    seconds = gridded.index.to_numpy() / 1000.0
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.5), constrained_layout=True)
    styles = {"actual next 5 s mean": ("black", 1.6)}
    for label, column in present.items():
        color, width = styles.get(label, (None, 1.0))
        axes[0].plot(seconds, gridded[column], ".", markersize=1.5, color=color, label=label)
    start = float(session_rows["as_of_ms"].iloc[len(session_rows) // 2]) / 1000.0
    window = (seconds >= start) & (seconds < start + 300)
    for label, column in present.items():
        color, width = styles.get(label, (None, 1.0))
        axes[1].plot(
            seconds[window],
            gridded[column].to_numpy()[window],
            marker=".",
            markersize=3,
            linewidth=width,
            color=color,
            label=label,
        )
    axes[0].set_title(
        f"Held out test session {session_id}: forecast versus actual, every eligible issuance time"
    )
    axes[1].set_title("300 s excerpt. Breaks are issuance times without five valid target bins")
    for axis in axes:
        axis.set_xlabel("Forecast issuance time since session start (s)")
        axis.set_ylabel("Mean downlink over next 5 s (Mbps)")
        axis.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, markerscale=6)
    axes[1].legend(loc="upper right", fontsize=8)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def plot_cohort_errors(sections: Mapping[str, Any], path: Path) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    titles = {"session": "Held out test sessions", "forward_time": "Forward time test periods"}
    for axis, (kind, section) in zip(axes, sections.items(), strict=True):
        categories: list[tuple[str, dict[str, Any]]] = [("all", section["overall"])]
        categories += [(name, value) for name, value in section["by_mobility_cohort"].items()]
        categories += [(name, value) for name, value in section["by_transition_stratum"].items()]
        positions = np.arange(len(categories))
        width = 0.8 / len(MAIN_MODELS)
        for index, model in enumerate(MAIN_MODELS):
            values = [(entry.get(model) or {}).get("mae_mbps") or np.nan for _, entry in categories]
            axis.bar(positions + index * width, values, width, label=model)
        labels = []
        for name, entry in categories:
            reference = entry.get(PERSISTENCE) or {}
            labels.append(
                f"{name}\n{reference.get('sample_count', 0)} rows\n"
                f"{reference.get('session_count', 0)} sessions"
            )
        axis.set_xticks(positions + width * (len(MAIN_MODELS) - 1) / 2, labels, fontsize=8)
        axis.set_ylabel("MAE of next 5 s mean downlink (Mbps)")
        axis.set_xlabel("Row cohort")
        axis.set_title(titles.get(kind, kind))
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


# markdown


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(section: Mapping[str, Any], heading: str) -> list[str]:
    lines = [
        f"### {heading}",
        "",
        "| Rows | Model | Samples | Sessions | MAE (Mbps) | RMSE (Mbps) | Session balanced MAE "
        "(Mbps) | Session balanced RMSE (Mbps) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    groups: list[tuple[str, Mapping[str, Any]]] = [("all", section["overall"])]
    groups += list(section["by_mobility_cohort"].items())
    groups += list(section["by_transition_stratum"].items())
    for label, models in groups:
        for model, summary in models.items():
            lines.append(
                f"| {label} | {model} | {summary['sample_count']} | {summary['session_count']} | "
                f"{_number(summary['mae_mbps'])} | {_number(summary['rmse_mbps'])} | "
                f"{_number(summary['session_balanced_mae_mbps'])} | "
                f"{_number(summary['session_balanced_rmse_mbps'])} |"
            )
    lines += [
        "",
        "Baseline relative difference (negative means lower error than the baseline):",
        "",
    ]
    for model, comparisons in section["baseline_relative_difference"].items():
        for baseline, values in comparisons.items():
            mae = _number(values["mae"])
            rmse = _number(values["rmse"])
            lines.append(f"- {model} {baseline}: MAE {mae}, RMSE {rmse}")
    lines.append("")
    return lines


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Downlink forecast evaluation",
        "",
        f"Task: {payload['task']}. Snapshot `{payload['snapshot_id']}`, split manifest "
        f"`{payload['split_manifest_hash'][:16]}`.",
        "",
        payload["unit_note"],
        "",
        "## Feature rows and exclusions",
        "",
    ]
    for stage, value in payload["feature_rows"].items():
        lines.append(f"- {stage}: {value}")
    lines += ["", "## Results", "", payload["not_averaged"] + ".", ""]
    lines += _table(payload["session_holdout"], "Held out whole sessions")
    lines += _table(payload["forward_time"], "Forward time split inside sessions")
    lines += ["## Chosen configurations", ""]
    for name, record in payload["models"].items():
        lines.append(f"- {name}: {record.get('chosen_configuration', record.get('status'))}")
    lines += ["", f"Failed fits: {len(payload['failed_fits'])}", "", "## Limitations", ""]
    lines += [f"- {item}" for item in payload["limitations"]]
    return "\n".join(lines) + "\n"
