"""Error analysis of saved forecasts by session and by traffic regime, with resampled intervals.

Only the saved prediction files are read, after checking their recorded hashes, so no model is
refit and no test row informs a choice. Regimes reuse the detector's declared drop rule against
the trailing 30 s mean. Consecutive five second targets overlap, so rows are resampled in whole
time blocks inside each session. Whole sessions are resampled only when there are enough of them
for a cluster bootstrap to be meaningful.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from celltrace.canonical import file_sha256
from celltrace.config import DetectorSection
from celltrace.prediction.evaluate import MAIN_MODELS, UNIT_NOTE
from celltrace.prediction.models import BOOSTING, PERSISTENCE, TRAILING_MEAN

ANALYSIS_VERSION = "forecast_error_analysis_v2"
ANALYSIS_JSON = "error_analysis.json"
ANALYSIS_MARKDOWN = "error_analysis.md"
FIGURE_REGIMES = "error_by_regime.png"
FIGURE_SESSIONS = "relative_error_by_session.png"
BLOCK_MS = 60_000
RESAMPLE_COUNT = 2000
RANDOM_SEED = 42
INTERVAL_LEVEL = 0.95
MINIMUM_CLUSTER_SESSIONS = 10
MINIMUM_INTERVAL_BLOCKS = 10
COMBINED = f"{BOOSTING}_combined"
SPLIT_TITLES = {"session": "Held out whole sessions", "forward_time": "Forward time periods"}
REALIZED_ORDER = ("drop", "steady", "rise", "low_reference")
ISSUANCE_ORDER = ("recent_drop", "level", "recent_rise", "low_reference")
# First three slots of the validated reference palette, light mode, fixed per model.
MODEL_COLORS = {PERSISTENCE: "#2568b8", TRAILING_MEAN: "#ba4b20", COMBINED: "#087e58"}
MODEL_MARKERS = {PERSISTENCE: "o", TRAILING_MEAN: "s", COMBINED: "^"}
MODEL_LABELS = {
    PERSISTENCE: "Last observed rate",
    TRAILING_MEAN: "30-second mean",
    COMBINED: "CPU boosting",
}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7"}
LIMITATIONS = (
    UNIT_NOTE,
    "Realized regimes are conditioned on the actual next five seconds, which is unknown when the "
    "forecast is issued. They describe where errors occur, not a rule a forecaster could apply.",
    "Issuance regimes use only the last observed value and the trailing 30 s mean, both known at "
    "forecast time.",
    "Within session block intervals describe sampling variation over time inside the observed "
    "sessions. They do not describe variation between sessions.",
    "The held out split has four test sessions, too few for a session cluster bootstrap. Its "
    "per session results are shown instead of a session level interval.",
    "The regime rule mirrors the detector thresholds but uses the trailing mean rather than the "
    "detector's binned median reference, so a realized drop row is not a detector candidate.",
    "Intervals are not adjusted for comparing several models and regimes at once.",
    "These are exploratory percentile intervals on aggregate forecast errors, not prediction "
    "intervals for individual forecasts. Fixed 60 s blocks are a declared dependence assumption, "
    "not a guarantee of calibrated coverage or independence between blocks.",
    "Regime resamples retain all observed blocks, including blocks with no rows in the regime. "
    "The regime error is recomputed as an error sum divided by its resampled row count.",
)


class PredictionHashMismatch(ValueError):
    """A saved prediction file no longer matches the hash recorded when it was written."""


@dataclass(frozen=True)
class RegimeRule:
    minimum_reference_mbps: float
    minimum_relative_change: float
    minimum_absolute_change_mbps: float

    @classmethod
    def from_detector(cls, detector: DetectorSection) -> RegimeRule:
        return cls(
            minimum_reference_mbps=detector.minimum_reference_mbps,
            minimum_relative_change=detector.minimum_relative_drop,
            minimum_absolute_change_mbps=detector.minimum_absolute_drop_mbps,
        )

    @property
    def rise_factor(self) -> float:
        """Mirror of the drop rule on a log scale: a 60 percent drop pairs with a 2.5 times rise."""
        return self.minimum_relative_change / (1.0 - self.minimum_relative_change)

    def as_dict(self) -> dict[str, float]:
        return {
            "minimum_reference_mbps": self.minimum_reference_mbps,
            "minimum_relative_change": self.minimum_relative_change,
            "minimum_absolute_change_mbps": self.minimum_absolute_change_mbps,
            "rise_factor": self.rise_factor,
        }


def classify_change(
    reference: np.ndarray, value: np.ndarray, rule: RegimeRule, labels: Sequence[str]
) -> np.ndarray:
    """Label each value as a drop, level or rise against its reference, using the detector rule.

    labels gives the names for drop, level, rise and low reference, in that order.
    """
    drop_label, level_label, rise_label, low_label = labels
    drop_threshold = np.maximum(
        rule.minimum_absolute_change_mbps, rule.minimum_relative_change * reference
    )
    rise_threshold = np.maximum(rule.minimum_absolute_change_mbps, rule.rise_factor * reference)
    result = np.full(reference.shape, level_label, dtype=object)
    result[(reference - value) >= drop_threshold] = drop_label
    result[(value - reference) >= rise_threshold] = rise_label
    result[reference < rule.minimum_reference_mbps] = low_label
    return result


def add_regimes(rows: pd.DataFrame, rule: RegimeRule) -> pd.DataFrame:
    result = rows.copy()
    reference = result[f"prediction_{TRAILING_MEAN}"].to_numpy(dtype=float)
    result["realized_regime"] = classify_change(
        reference, result["target_mbps"].to_numpy(dtype=float), rule, REALIZED_ORDER
    )
    result["issuance_regime"] = classify_change(
        reference,
        result[f"prediction_{PERSISTENCE}"].to_numpy(dtype=float),
        rule,
        ISSUANCE_ORDER,
    )
    result["time_block"] = result["as_of_ms"] // BLOCK_MS
    return result


def load_saved_predictions(manifest_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    frames = {}
    for split_kind, record in sorted(manifest["files"].items()):
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise PredictionHashMismatch(f"{path} differs from its recorded hash")
        frames[split_kind] = pd.read_csv(path)
    return frames, manifest


@dataclass(frozen=True)
class BlockTotals:
    """Absolute error sums per time block and model, so resampling needs no row loops."""

    sessions: np.ndarray
    absolute_error_sums: np.ndarray
    signed_error_sums: np.ndarray
    row_counts: np.ndarray

    @property
    def block_count(self) -> int:
        return len(self.row_counts)


def block_totals(
    rows: pd.DataFrame, models: Sequence[str], universe: pd.DataFrame | None = None
) -> BlockTotals:
    actual = rows["target_mbps"].to_numpy(dtype=float)
    errors = np.column_stack(
        [rows[f"prediction_{model}"].to_numpy(dtype=float) - actual for model in models]
    )
    key_columns = ["session_id"]
    if "segment_id" in rows:
        key_columns.append("segment_id")
    key_columns.append("time_block")
    keys = rows[key_columns].reset_index(drop=True)
    grouped = pd.DataFrame(np.abs(errors), columns=list(models)).groupby(
        [keys[key] for key in key_columns], sort=True
    )
    signed = pd.DataFrame(errors, columns=list(models)).groupby(
        [keys[key] for key in key_columns], sort=True
    )
    absolute_sums = grouped.sum()
    signed_sums = signed.sum()
    counts = grouped.size()
    if universe is not None:
        # Sample the original temporal blocks, then apply the regime filter. Dropping
        # zero-regime blocks first conditions away variation in regime occupancy.
        block_index = universe.groupby(key_columns, sort=True).size().index
        absolute_sums = absolute_sums.reindex(block_index, fill_value=0)
        signed_sums = signed_sums.reindex(block_index, fill_value=0)
        counts = counts.reindex(block_index, fill_value=0)
    return BlockTotals(
        sessions=absolute_sums.index.get_level_values("session_id").to_numpy(dtype=object),
        absolute_error_sums=absolute_sums.to_numpy(dtype=float),
        signed_error_sums=signed_sums.to_numpy(dtype=float),
        row_counts=counts.to_numpy(dtype=float),
    )


def within_session_block_indices(
    sessions: np.ndarray, generator: np.random.Generator, resample_count: int
) -> np.ndarray:
    """Draw each session's own number of blocks, with replacement, from that session only."""
    columns = []
    for session in sorted(set(sessions.tolist())):
        positions = np.flatnonzero(sessions == session)
        draws = generator.integers(0, len(positions), size=(resample_count, len(positions)))
        columns.append(positions[draws])
    return np.concatenate(columns, axis=1)


def session_cluster_indices(
    sessions: np.ndarray, generator: np.random.Generator, resample_count: int
) -> list[np.ndarray]:
    """Draw whole sessions with replacement, keeping every block of each drawn session."""
    session_names = sorted(set(sessions.tolist()))
    positions = [np.flatnonzero(sessions == session) for session in session_names]
    draws = generator.integers(0, len(session_names), size=(resample_count, len(session_names)))
    return [np.concatenate([positions[index] for index in draw]) for draw in draws]


def resampled_mae(totals: BlockTotals, indices: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """MAE per resample and model, weighting rows equally as the headline MAE does."""
    if isinstance(indices, np.ndarray):
        sums = totals.absolute_error_sums[indices].sum(axis=1)
        counts = totals.row_counts[indices].sum(axis=1)
    else:
        sums = np.stack([totals.absolute_error_sums[draw].sum(axis=0) for draw in indices])
        counts = np.array([totals.row_counts[draw].sum() for draw in indices])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(sums / counts[:, None], dtype=float)


def percentile_interval(samples: np.ndarray) -> list[float]:
    tail = (1.0 - INTERVAL_LEVEL) / 2.0 * 100.0
    low, high = np.nanpercentile(samples, [tail, 100.0 - tail])
    return [float(low), float(high)]


def summarize_rows(
    rows: pd.DataFrame, models: Sequence[str], seed: int, universe: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Point MAE, bias and paired difference to persistence, each with its resampled intervals."""
    totals = block_totals(rows, models, universe)
    session_count = int(rows["session_id"].nunique())
    contributing_blocks = int(np.count_nonzero(totals.row_counts))
    row_count = int(totals.row_counts.sum())
    point_mae = totals.absolute_error_sums.sum(axis=0) / row_count
    mean_error = totals.signed_error_sums.sum(axis=0) / row_count
    persistence_index = list(models).index(PERSISTENCE)
    methods: dict[str, Any] = {}
    resampled: dict[str, np.ndarray] = {}
    if contributing_blocks >= MINIMUM_INTERVAL_BLOCKS:
        generator = np.random.default_rng(seed)
        indices = within_session_block_indices(totals.sessions, generator, RESAMPLE_COUNT)
        resampled["within_session_blocks"] = resampled_mae(totals, indices)
        methods["within_session_blocks"] = "reported"
    else:
        methods["within_session_blocks"] = (
            f"not reported: {contributing_blocks} contributing blocks, below "
            f"{MINIMUM_INTERVAL_BLOCKS}"
        )
    if session_count >= MINIMUM_CLUSTER_SESSIONS:
        generator = np.random.default_rng(seed + 1)
        draws = session_cluster_indices(totals.sessions, generator, RESAMPLE_COUNT)
        resampled["session_clusters"] = resampled_mae(totals, draws)
        methods["session_clusters"] = "reported"
    else:
        methods["session_clusters"] = (
            f"not reported: {session_count} sessions, below {MINIMUM_CLUSTER_SESSIONS}"
        )
    valid_resamples = {}
    for method, samples in list(resampled.items()):
        valid = np.all(np.isfinite(samples), axis=1)
        valid_resamples[method] = int(valid.sum())
        if valid.mean() < 0.95:
            methods[method] = "not reported: more than 5% of resamples have no regime rows"
            del resampled[method]
        else:
            resampled[method] = samples[valid]
    model_summaries = {}
    for index, model in enumerate(models):
        summary: dict[str, Any] = {
            "mae_mbps": float(point_mae[index]),
            "mean_error_mbps": float(mean_error[index]),
            "mae_interval_mbps": {
                method: percentile_interval(samples[:, index])
                for method, samples in resampled.items()
            },
        }
        if model != PERSISTENCE:
            summary["difference_to_persistence_mbps"] = float(
                point_mae[index] - point_mae[persistence_index]
            )
            summary["difference_interval_mbps"] = {
                method: percentile_interval(samples[:, index] - samples[:, persistence_index])
                for method, samples in resampled.items()
            }
        model_summaries[model] = summary
    return {
        "row_count": row_count,
        "session_count": session_count,
        "block_count": totals.block_count,
        "contributing_block_count": contributing_blocks,
        "valid_resample_counts": valid_resamples,
        "interval_methods": methods,
        "models": model_summaries,
    }


def per_session_rows(rows: pd.DataFrame, models: Sequence[str]) -> list[dict[str, Any]]:
    records = []
    for session_id, part in rows.groupby("session_id", sort=True):
        actual = part["target_mbps"].to_numpy(dtype=float)
        mae = {
            model: float(
                np.mean(np.abs(part[f"prediction_{model}"].to_numpy(dtype=float) - actual))
            )
            for model in models
        }
        persistence_mae = mae[PERSISTENCE]
        summary = summarize_rows(part, models, RANDOM_SEED)
        records.append(
            {
                "session_id": str(session_id),
                "mobility_cohort": str(part["mobility_cohort"].iloc[0]),
                "row_count": len(part),
                "mean_target_mbps": float(np.mean(actual)),
                "mae_mbps": mae,
                "mae_interval_mbps": {
                    model: record["mae_interval_mbps"]
                    for model, record in summary["models"].items()
                },
                "difference_interval_mbps": {
                    model: record.get("difference_interval_mbps", {})
                    for model, record in summary["models"].items()
                },
                "interval_methods": summary["interval_methods"],
                "relative_mae_to_persistence": {
                    model: None if persistence_mae == 0 else value / persistence_mae
                    for model, value in mae.items()
                    if model != PERSISTENCE
                },
            }
        )
    return records


def analyze_split(
    rows: pd.DataFrame, models: Sequence[str], rule: RegimeRule, seed: int
) -> dict[str, Any]:
    labelled = add_regimes(rows, rule)
    sessions = per_session_rows(labelled, models)
    beats = {
        model: sum(
            1
            for record in sessions
            if record["relative_mae_to_persistence"].get(model) is not None
            and record["relative_mae_to_persistence"][model] < 1.0
        )
        for model in models
        if model != PERSISTENCE
    }
    regimes: dict[str, dict[str, Any]] = {}
    regime_columns = (("realized_regime", REALIZED_ORDER), ("issuance_regime", ISSUANCE_ORDER))
    for column_position, (column, order) in enumerate(regime_columns):
        regimes[column] = {}
        for position, label in enumerate(order):
            part = labelled[labelled[column] == label]
            if part.empty:
                regimes[column][label] = {"row_count": 0}
                continue
            regime_seed = seed + 100 * (column_position + 1) + 10 * (position + 1)
            regimes[column][label] = summarize_rows(part, models, regime_seed, universe=labelled)
    return {
        "row_count": len(labelled),
        "session_count": int(labelled["session_id"].nunique()),
        "overall": summarize_rows(labelled, models, seed),
        "per_session": sessions,
        "sessions_with_lower_mae_than_persistence": {
            model: {"sessions": count, "of": len(sessions)} for model, count in beats.items()
        },
        "regimes": regimes,
        "regime_row_counts": {
            column: {
                str(key): int(value)
                for key, value in labelled[column].value_counts().sort_index().items()
            }
            for column in ("realized_regime", "issuance_regime")
        },
    }


def run_error_analysis(
    manifest_path: Path, out_dir: Path, detector: DetectorSection | None = None
) -> dict[str, Any]:
    frames, manifest = load_saved_predictions(manifest_path)
    rule = RegimeRule.from_detector(detector or DetectorSection())
    splits = {}
    for split_kind, rows in frames.items():
        models = [
            column.removeprefix("prediction_")
            for column in rows.columns
            if column.startswith("prediction_")
        ]
        if PERSISTENCE not in models:
            raise ValueError(f"{split_kind} predictions have no persistence baseline")
        splits[split_kind] = analyze_split(rows, models, rule, RANDOM_SEED)
    payload: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "unit": "Mbps",
        "source_prediction_manifest": str(manifest_path),
        "source_prediction_files": manifest["files"],
        "model_sha256": manifest["model_sha256"],
        "split_manifest_hash": manifest["split_manifest_hash"],
        "snapshot_logical_hash": manifest["snapshot_logical_hash"],
        "regime_rule": rule.as_dict(),
        "regime_definitions": {
            "reference": "trailing 30 s mean downlink, known at forecast time",
            "realized_regime": "actual next 5 s mean against the reference: drop when it falls by "
            "at least the relative and absolute drop thresholds, rise for the mirrored increase, "
            "low_reference below the minimum reference",
            "issuance_regime": "last observed value against the reference with the same rule, "
            "known at forecast time",
        },
        "resampling": {
            "block_ms": BLOCK_MS,
            "resample_count": RESAMPLE_COUNT,
            "random_seed": RANDOM_SEED,
            "interval_level": INTERVAL_LEVEL,
            "interval_type": "percentile",
            "within_session_blocks": "each session's fixed 60 s issuance blocks drawn with "
            "replacement from that session, keeping segments separate and including zero-regime "
            "blocks, so session composition stays fixed",
            "session_clusters": "whole sessions drawn with replacement, reported only with at "
            f"least {MINIMUM_CLUSTER_SESSIONS} sessions",
            "minimum_interval_blocks": MINIMUM_INTERVAL_BLOCKS,
        },
        "splits": splits,
        "not_averaged": "session and forward_time splits answer different questions and are "
        "never combined",
        "limitations": list(LIMITATIONS),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["figures"] = {
        "error_by_regime": str(plot_regimes(splits, out_dir / FIGURE_REGIMES)),
        "relative_error_by_session": str(plot_sessions(splits, out_dir / FIGURE_SESSIONS)),
    }
    (out_dir / ANALYSIS_JSON).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out_dir / ANALYSIS_MARKDOWN).write_text(render_markdown(payload), encoding="utf-8")
    return payload


# figures


def _style_axis(axis: Any) -> None:
    axis.grid(axis="x", color=INK["grid"], linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(INK["axis"])
    axis.tick_params(colors=INK["secondary"], labelsize=8, length=0)


def _regime_rows(split: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Overall, realized and issuance rows. Low reference rows are identical in both groupings."""
    entries: list[tuple[str, Mapping[str, Any]]] = [("all rows", split["overall"])]
    for column, prefix in (("realized_regime", "realized"), ("issuance_regime", "at issuance")):
        for label, summary in split["regimes"][column].items():
            if not summary.get("row_count", 0):
                continue
            if label == "low_reference":
                if column == "realized_regime":
                    entries.append(("low reference, either grouping", summary))
                continue
            entries.append((f"{prefix}: {label.replace('_', ' ')}", summary))
    return entries


def plot_regimes(splits: Mapping[str, Any], path: Path) -> Path:
    """MAE with within session block intervals, one row per regime and one dot per model."""
    figure, axes = plt.subplots(1, len(splits), figsize=(13, 6.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    offsets = {model: (index - 1) * 0.22 for index, model in enumerate(MAIN_MODELS)}
    for axis, (split_kind, split) in zip(axes, splits.items(), strict=True):
        entries = _regime_rows(split)
        positions = np.arange(len(entries))[::-1]
        for model in MAIN_MODELS:
            for position, (_, summary) in zip(positions, entries, strict=True):
                record = summary["models"].get(model)
                if record is None:
                    continue
                interval = record["mae_interval_mbps"].get("within_session_blocks")
                y_value = position + offsets[model]
                if interval:
                    axis.plot(interval, [y_value, y_value], color=MODEL_COLORS[model], lw=2)
                axis.plot(
                    record["mae_mbps"],
                    y_value,
                    MODEL_MARKERS[model],
                    markersize=6,
                    color=MODEL_COLORS[model],
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                )
        axis.set_yticks(
            positions,
            [f"{label}  ({summary['row_count']} rows)" for label, summary in entries],
            fontsize=8,
            color=INK["primary"],
        )
        axis.set_xlabel("MAE of next 5 s mean downlink (Mbps)", color=INK["secondary"])
        axis.set_title(
            f"{SPLIT_TITLES.get(split_kind, split_kind)}, {split['session_count']} sessions",
            color=INK["primary"],
            fontsize=10,
            loc="left",
        )
        axis.set_xlim(left=0)
        _style_axis(axis)
    handles = [
        Line2D(
            [],
            [],
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            lw=2,
            label=MODEL_LABELS[model],
        )
        for model in MAIN_MODELS
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    figure.suptitle(
        "Forecast error by regime. Lines are 95% within session block intervals",
        color=INK["primary"],
        fontsize=11,
    )
    figure.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(figure)
    return path


def plot_sessions(splits: Mapping[str, Any], path: Path) -> Path:
    """Boosting MAE divided by persistence MAE in each session. Left of 1 means boosting helped."""
    figure, axes = plt.subplots(
        1, len(splits), figsize=(13, 6.6), constrained_layout=True, sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, (split_kind, split) in zip(axes, splits.items(), strict=True):
        records = sorted(
            split["per_session"],
            key=lambda record: record["relative_mae_to_persistence"].get(COMBINED) or np.inf,
        )
        values = [record["relative_mae_to_persistence"].get(COMBINED) for record in records]
        positions = np.arange(len(records))[::-1]
        axis.axvline(1.0, color=INK["axis"], linewidth=1.2)
        axis.plot(
            values,
            positions,
            "o",
            markersize=6,
            color=MODEL_COLORS[COMBINED],
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
        axis.set_yticks(
            positions,
            [
                f"{record['session_id']} {record['mobility_cohort']}, {record['row_count']} rows, "
                f"persistence {record['mae_mbps'][PERSISTENCE]:.1f}"
                for record in records
            ],
            fontsize=7,
            color=INK["primary"],
        )
        beats = split["sessions_with_lower_mae_than_persistence"].get(COMBINED, {})
        axis.set_title(
            f"{SPLIT_TITLES.get(split_kind, split_kind)}: better in "
            f"{beats.get('sessions')} of {beats.get('of')}",
            color=INK["primary"],
            fontsize=10,
            loc="left",
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
        axis.set_xlim(0.2, 10)
        axis.set_xlabel("Boosting MAE / persistence MAE (log scale)", color=INK["secondary"])
        _style_axis(axis)
    figure.suptitle(
        "Relative error per session. Left of 1 the learned model has lower error. "
        "Labels give persistence MAE in Mbps",
        color=INK["primary"],
        fontsize=11,
    )
    figure.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(figure)
    return path


# markdown


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _interval(record: Mapping[str, Any], key: str, method: str) -> str:
    interval = record.get(key, {}).get(method)
    return "n/a" if not interval else f"{interval[0]:.2f} to {interval[1]:.2f}"


def _regime_table(split: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Rows | Model | Count | MAE | 95% block interval | Mean error | Difference to "
        "persistence | Block interval | Session interval |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for label, summary in _regime_rows(split):
        for model, record in summary["models"].items():
            if model not in MAIN_MODELS:
                continue
            lines.append(
                f"| {label} | {model} | {summary['row_count']} | {_number(record['mae_mbps'])} | "
                f"{_interval(record, 'mae_interval_mbps', 'within_session_blocks')} | "
                f"{_number(record['mean_error_mbps'])} | "
                f"{_number(record.get('difference_to_persistence_mbps'))} | "
                f"{_interval(record, 'difference_interval_mbps', 'within_session_blocks')} | "
                f"{_interval(record, 'difference_interval_mbps', 'session_clusters')} |"
            )
    return lines


def _session_table(split: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Session | Cohort | Rows | Mean actual Mbps | Persistence MAE | Combined MAE | "
        "Relative MAE | Paired difference 95% block interval |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in split["per_session"]:
        lines.append(
            f"| {record['session_id']} | {record['mobility_cohort']} | {record['row_count']} | "
            f"{_number(record['mean_target_mbps'])} | "
            f"{_number(record['mae_mbps'][PERSISTENCE])} | "
            f"{_number(record['mae_mbps'].get(COMBINED))} | "
            f"{_number(record['relative_mae_to_persistence'].get(COMBINED))} | "
            f"{_interval(record['difference_interval_mbps'], COMBINED, 'within_session_blocks')} |"
        )
    return lines


def render_markdown(payload: Mapping[str, Any]) -> str:
    rule = payload["regime_rule"]
    lines = [
        "# Forecast error analysis",
        "",
        "Saved test predictions only, verified against their recorded hashes. No model was refit. "
        f"Model `{payload['model_sha256'][:16]}`, split manifest "
        f"`{payload['split_manifest_hash'][:16]}`.",
        "",
        payload["regime_definitions"]["reference"].capitalize() + ". A drop means the value is at "
        f"least {rule['minimum_relative_change']:.0%} and {rule['minimum_absolute_change_mbps']} "
        "Mbps below it. These thresholds match the detector; this analysis uses a mean reference "
        f"where the detector uses a median. A rise is at least {rule['rise_factor']:.1f} "
        f"times the reference above it. Below {rule['minimum_reference_mbps']} Mbps the "
        "reference is too low for a ratio.",
        "",
        "Errors are in Mbps. Mean error is predicted minus actual, so a positive value means the "
        "forecast was too high. Difference to persistence is model MAE minus persistence MAE on "
        "the same rows, so a negative value favors the model.",
        "",
    ]
    for split_kind, split in payload["splits"].items():
        lines += [f"## {SPLIT_TITLES.get(split_kind, split_kind)}", ""]
        methods = split["overall"]["interval_methods"]
        lines += [
            f"{split['row_count']} rows, {split['session_count']} sessions. Block intervals: "
            f"{methods['within_session_blocks']}. Session intervals: "
            f"{methods['session_clusters']}.",
            "",
            *_regime_table(split),
            "",
            *_session_table(split),
            "",
        ]
    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in payload["limitations"]]
    return "\n".join(lines) + "\n"
