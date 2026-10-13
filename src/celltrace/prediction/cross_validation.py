"""Nested session cross validation of delivered rate forecasts.

Two outer schemes keep every session whole. Leave one session out measures transfer
to an unseen session. Forward chaining trains only on sessions that ended before the
test session started, which is the rolling origin evaluation of Hyndman and
Athanasopoulos applied to whole sessions. Inside each outer fold, inner grouped
cross validation on the training sessions alone selects one of the six predeclared
configurations (Cawley and Talbot 2010), supplies out of fold quantiles for
conformalized quantile regression with equal session weights (Dunn, Wasserman and
Ramdas, CDF pooling) and fits the sigmoid probability calibrator. Feature design and
the configuration grid were developed on this archive, so results stay exploratory.
"""

from __future__ import annotations

import functools
import hashlib
import json
import pickle
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from celltrace.analysis.bootstrap import session_bootstrap
from celltrace.analysis.impact import SERVICE_LEVELS
from celltrace.canonical import content_hash, file_sha256
from celltrace.data.raca import parse_timestamp
from celltrace.data.snapshot import open_snapshot
from celltrace.prediction.features import CATEGORICAL_FEATURES, build_feature_frame
from celltrace.prediction.models import (
    CONFIGURATION_GRID,
    PERSISTENCE_FEATURE,
    RANDOM_STATE,
    TRAILING_MEAN_FEATURE,
    CategoryEncoder,
    FeatureTransformer,
    design_matrix,
)
from celltrace.release import source_tree_hash

METHOD_VERSION = "nested_session_cross_validation_v4_partition_transforms_bounded_intervals"
# The six predeclared configurations. Selection happens inside each outer fold.
CONFIGURATIONS: tuple[dict[str, Any], ...] = CONFIGURATION_GRID
QUANTILES = (0.1, 0.5, 0.9)
INTERVAL_ALPHA = 0.2
INNER_FOLDS = 5
MINIMUM_FORWARD_TRAINING_SESSIONS = 7
# Fewest nonempty sessions for which session cross validation runs at all.
MINIMUM_CROSS_VALIDATION_SESSIONS = 5
MONDRIAN_MINIMUM_SESSIONS = 5
LEAVE_ONE_OUT = "leave_one_session_out"
FORWARD = "forward_chaining"
WARNING_THRESHOLD_MBPS = dict(SERVICE_LEVELS)["below_basic_use"]
FEATURE_SETS = ("history_only", "combined")
BOOTSTRAP_SEED = 42
SUMMARY_NAME = "cross_validation_summary.json"
REPORT_NAME = "cross_validation_report.md"
PREDICTIONS_NAME = "cross_validation_predictions.csv"
FORWARD_PREDICTIONS_NAME = "forward_chaining_predictions.csv"
FOLDS_NAME = "cross_validation_folds.json"
FIGURES = {
    "skill": "forecast_skill.png",
    "warning": "early_warning.png",
}
# Project palette, light mode, as in the other forecast figures.
SERIES = {"history_only": "#2568b8", "combined": "#ba4b20", "baseline": "#898781"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e"}
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


# Scores


def pinball(actual: np.ndarray, forecast: np.ndarray, quantile: float) -> np.ndarray:
    """Quantile score as defined in Forecasting: Principles and Practice, section 5.9."""
    below = actual < forecast
    scores: np.ndarray = np.where(
        below, 2 * (1 - quantile) * (forecast - actual), 2 * quantile * (actual - forecast)
    )
    return scores


def winkler(lower: np.ndarray, upper: np.ndarray, actual: np.ndarray, alpha: float) -> np.ndarray:
    """Interval width plus 2 / alpha times the distance outside the interval."""
    scores: np.ndarray = (
        (upper - lower)
        + (2 / alpha) * np.clip(lower - actual, 0, None)
        + (2 / alpha) * np.clip(actual - upper, 0, None)
    )
    return scores


# Fold fitting


def _matrices(
    train: pd.DataFrame, test: pd.DataFrame, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Encoders are fitted on the training sessions only."""
    encoders = {
        name: CategoryEncoder.fit(train[name]) for name in names if name in CATEGORICAL_FEATURES
    }
    mask = np.array([name in encoders for name in names], dtype=bool)
    return (
        design_matrix(train, names, encoders),
        design_matrix(test, names, encoders),
        mask if mask.any() else None,
    )


def _regressor(
    categorical: np.ndarray | None, configuration: Mapping[str, Any], **loss: Any
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        early_stopping=False,
        categorical_features=categorical,
        **loss,
        **configuration,
    )


def _classifier(
    categorical: np.ndarray | None, configuration: Mapping[str, Any]
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        early_stopping=False,
        categorical_features=categorical,
        **configuration,
    )


type Splits = list[tuple[np.ndarray, np.ndarray]]


def inner_folds(train: pd.DataFrame) -> Splits:
    """Whole training sessions assigned to inner folds by an outcome blind identity hash.

    Row order, covariates and targets cannot change the assignment, and no session is
    split across folds, so inner validation never mixes rows of one session.
    """
    sessions = sorted(
        train["session_id"].unique(),
        key=lambda identity: hashlib.sha256(f"{RANDOM_STATE}:{identity}".encode()).digest(),
    )
    if len(sessions) < 2:
        raise ValueError("inner cross validation needs at least two training sessions")
    count = min(INNER_FOLDS, len(sessions))
    fold_of_session = {session: index % count for index, session in enumerate(sessions)}
    fold_of_row = train["session_id"].map(fold_of_session).to_numpy()
    return [
        (np.flatnonzero(fold_of_row != fold), np.flatnonzero(fold_of_row == fold))
        for fold in range(count)
    ]


def out_of_fold(
    frame: pd.DataFrame,
    names: Sequence[str],
    target: np.ndarray,
    splits: Splits,
    make: Callable[[np.ndarray | None], HistGradientBoostingRegressor],
) -> np.ndarray:
    """Estimator and transform fits exclude each held session; selection can use its labels."""
    predictions = np.empty(len(target))
    for fit_index, held_index in splits:
        fitting, held, categorical = _matrices(frame.iloc[fit_index], frame.iloc[held_index], names)
        model = make(categorical).fit(fitting, target[fit_index])
        predictions[held_index] = model.predict(held)
    return predictions


def select_configuration(
    frame: pd.DataFrame, names: Sequence[str], target: np.ndarray, splits: Splits
) -> tuple[int, list[float]]:
    """Lowest pooled out of fold MAE among the predeclared configurations.

    Selection uses the training sessions of one outer fold only, so the outer test
    session never influences which configuration it is scored with. Ties keep the
    earlier configuration.
    """
    errors = []
    for configuration in CONFIGURATIONS:
        make = functools.partial(_regressor, configuration=configuration, loss="squared_error")
        predictions = np.clip(out_of_fold(frame, names, target, splits, make), 0, None)
        errors.append(float(np.mean(np.abs(predictions - target))))
    return int(np.argmin(errors)), errors


def pooled_conformal_correction(scores: np.ndarray, sessions: np.ndarray, alpha: float) -> float:
    """Conformalized quantile regression correction with equal weight per session.

    This adapts the CDF weighting idea of Dunn, Wasserman and Ramdas (JASA 2022): every
    calibration session's empirical score distribution gets the same weight, so long
    sessions cannot dominate, and the correction is the smallest score at which the
    pooled distribution reaches 1 - alpha. Selection uses these sessions' labels and
    predictions are cross-fitted then refitted, so the cited theorem does not establish
    coverage for this procedure. Serial rows further limit exchangeability. A
    negative correction narrows intervals, as in Romano, Patterson and Candès.
    """
    if not 0 < alpha < 1:
        raise ValueError("conformal alpha must lie strictly between zero and one")
    if len(scores) == 0 or len(scores) != len(sessions):
        raise ValueError("conformal scores must be nonempty and align with their sessions")
    if not np.isfinite(scores).all():
        raise ValueError("conformal calibration requires finite scores")
    identities = pd.Series(sessions)
    rows_per_session = identities.map(identities.value_counts()).to_numpy(dtype=float)
    weights = 1.0 / (rows_per_session * identities.nunique())
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    # A small tolerance keeps a sum of float weights from missing an exact level.
    position = int(np.searchsorted(cumulative, 1 - alpha - 1e-12, side="left"))
    return float(scores[order][min(position, len(order) - 1)])


def mondrian_corrections(
    training_levels: np.ndarray,
    scores: np.ndarray,
    sessions: np.ndarray,
    test_levels: np.ndarray,
    global_correction: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Group conditional (Mondrian) CQR over quartiles of the predicted level.

    The level is the midpoint of the 0.1 and 0.9 quantile forecasts. Quartile boundaries
    come from out of fold training predictions, and each quartile gets its own equal
    session weighted correction, so coverage is targeted within each predicted rate regime
    rather than only on average. A quartile with fewer than MONDRIAN_MINIMUM_SESSIONS
    calibration sessions uses the global correction. Returns the test quartile, the test
    correction and the number of quartiles that fell back.
    """
    boundaries = np.quantile(training_levels, [0.25, 0.5, 0.75])
    training_bins = np.searchsorted(boundaries, training_levels, side="right")
    test_bins = np.searchsorted(boundaries, test_levels, side="right")
    corrections = np.full(4, global_correction)
    fallback = 0
    for quartile in range(4):
        member = training_bins == quartile
        if len(set(sessions[member])) < MONDRIAN_MINIMUM_SESSIONS:
            fallback += 1
            continue
        corrections[quartile] = pooled_conformal_correction(
            scores[member], sessions[member], INTERVAL_ALPHA
        )
    return test_bins, corrections[test_bins], fallback


type WarningEstimator = Pipeline | CalibratedClassifierCV


def _fit_warning(
    frame: pd.DataFrame,
    names: Sequence[str],
    target: np.ndarray,
    splits: Splits,
    configuration: Mapping[str, Any],
) -> tuple[Pipeline | None, WarningEstimator | None, str, float]:
    """Raw classifier and its sigmoid calibration fitted on inner session folds.

    With ensemble=False, scikit-learn fits the sigmoid on out of fold scores from
    these session folds, then refits one classifier on all training sessions.
    """
    low = target < WARNING_THRESHOLD_MBPS
    prevalence = float(low.mean())
    if low.all() or not low.any():
        return None, None, "constant_training_prevalence_fit_one_class", prevalence
    mask = np.array([name in CATEGORICAL_FEATURES for name in names], dtype=bool)
    classifier = Pipeline(
        [
            ("features", FeatureTransformer(names)),
            ("classifier", _classifier(mask if mask.any() else None, configuration)),
        ]
    )
    if any(low[fit_index].all() or not low[fit_index].any() for fit_index, _ in splits):
        raw = classifier.fit(frame, low)
        return raw, None, "constant_training_prevalence_inner_fold_one_class", prevalence
    calibrated = CalibratedClassifierCV(
        classifier, method="sigmoid", cv=splits, ensemble=False
    ).fit(frame, low)
    refitted: Pipeline = calibrated.calibrated_classifiers_[0].estimator
    return refitted, calibrated, "sigmoid_inner_session_folds", prevalence


def _warning_probability(
    model: WarningEstimator | None, frame: pd.DataFrame, prevalence: float
) -> np.ndarray:
    if model is None:
        return np.full(len(frame), prevalence)
    column = list(model.classes_).index(True)
    result: np.ndarray = model.predict_proba(frame)[:, column]
    return result


def _ordered_quantiles(
    models: Sequence[HistGradientBoostingRegressor], matrix: np.ndarray
) -> np.ndarray:
    # Separately fitted quantiles can cross. Sorting each row keeps intervals valid.
    predictions = np.column_stack([model.predict(matrix) for model in models])
    ordered: np.ndarray = np.clip(np.sort(predictions, axis=1), 0, None)
    return ordered


def calibrated_bounds(
    lower: np.ndarray, upper: np.ndarray, correction: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply CQR correction, capping shrinking at the midpoint to keep valid bounds.

    This is an implementation choice for nonempty reported intervals, not a coverage
    guarantee. Return effective row corrections so recorded values reproduce the bounds.
    """
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("interval bounds must be finite")
    if np.any(lower < 0) or np.any(upper < lower) or not np.isfinite(correction).all():
        raise ValueError("ordered nonnegative bounds and finite corrections required")
    effective = np.maximum(correction, -(upper - lower) / 2)
    return np.clip(lower - effective, 0, None), upper + effective, effective


def fit_fold(
    train: pd.DataFrame, test: pd.DataFrame, names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Select, fit and calibrate on the training sessions; test targets are never read.

    Every inner fitting partition learns its own encoders; outer encoders see only outer
    training sessions. Calibration remains empirical because selection reuses its labels.
    """
    if set(train["session_id"]) & set(test["session_id"]):
        raise ValueError("fit and test sessions must be disjoint")
    splits = inner_folds(train)
    train_matrix, test_matrix, categorical = _matrices(train, test, names)
    target = train["target_mbps"].to_numpy(dtype=float)
    chosen, selection_errors = select_configuration(train, names, target, splits)
    configuration = CONFIGURATIONS[chosen]
    result = {
        "mean": _regressor(categorical, configuration, loss="squared_error")
        .fit(train_matrix, target)
        .predict(test_matrix)
    }
    quantile_models = [
        _regressor(categorical, configuration, loss="quantile", quantile=level).fit(
            train_matrix, target
        )
        for level in QUANTILES
    ]
    ordered = _ordered_quantiles(quantile_models, test_matrix)
    for index, level in enumerate(QUANTILES):
        result[f"q{level}"] = ordered[:, index]
    bounds = [
        out_of_fold(
            train,
            names,
            target,
            splits,
            functools.partial(
                _regressor, configuration=configuration, loss="quantile", quantile=level
            ),
        )
        for level in (QUANTILES[0], QUANTILES[-1])
    ]
    lower = np.clip(np.minimum(*bounds), 0, None)
    upper = np.maximum(*bounds)
    scores = np.maximum(lower - target, target - upper)
    correction = pooled_conformal_correction(scores, train["session_id"].to_numpy(), INTERVAL_ALPHA)
    result["cqr_lower"], result["cqr_upper"], result["interval_correction_mbps"] = (
        calibrated_bounds(ordered[:, 0], ordered[:, -1], correction)
    )
    result["interval_calibration_status"] = np.full(len(test), "pooled_session_cqr")
    result["interval_calibration_sessions"] = np.full(len(test), train["session_id"].nunique())
    level_bins, level_corrections, fallback = mondrian_corrections(
        (lower + upper) / 2,
        scores,
        train["session_id"].to_numpy(),
        (ordered[:, 0] + ordered[:, -1]) / 2,
        correction,
    )
    result["mondrian_lower"], result["mondrian_upper"], result["mondrian_correction_mbps"] = (
        calibrated_bounds(ordered[:, 0], ordered[:, -1], level_corrections)
    )
    result["predicted_level_quartile"] = level_bins
    result["mondrian_fallback_quartiles"] = np.full(len(test), fallback)
    raw, calibrated, status, prevalence = _fit_warning(train, names, target, splits, configuration)
    result["warning_raw"] = _warning_probability(raw, test, prevalence)
    result["warning"] = _warning_probability(calibrated, test, prevalence)
    result["warning_calibration_status"] = np.full(len(test), status)
    result["configuration_index"] = np.full(len(test), chosen)
    for index, error in enumerate(selection_errors):
        result[f"selection_mae_{index}"] = np.full(len(test), error)
    # Delivered rates cannot be negative.
    for key in ("mean", *(f"q{level}" for level in QUANTILES)):
        result[key] = np.clip(result[key], 0, None)
    return result


def baseline_fold(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Naive benchmarks: persistence, trailing mean, persistence plus residual quantiles
    and persistence warning probabilities, all estimated on the training sessions."""
    persistence = test[PERSISTENCE_FEATURE].to_numpy(dtype=float)
    train_persistence = train[PERSISTENCE_FEATURE].to_numpy(dtype=float)
    target = train["target_mbps"].to_numpy(dtype=float)
    residuals = target - train_persistence
    residuals = residuals[np.isfinite(residuals)]
    result = {
        "persistence": persistence,
        "trailing_mean": test[TRAILING_MEAN_FEATURE].to_numpy(dtype=float),
    }
    for level in QUANTILES:
        result[f"naive_q{level}"] = np.clip(persistence + np.quantile(residuals, level), 0, None)
    low = target < WARNING_THRESHOLD_MBPS
    already_low = train_persistence < WARNING_THRESHOLD_MBPS
    rate_if_low = float(low[already_low].mean()) if already_low.any() else 0.0
    rate_if_not = float(low[~already_low].mean()) if (~already_low).any() else 0.0
    result["naive_warning"] = np.where(
        persistence < WARNING_THRESHOLD_MBPS, rate_if_low, rate_if_not
    )
    result["climatology_warning"] = np.full(len(test), float(low.mean()))
    # Continuous rates retain the ranking that a two-valued threshold loses.
    result["persistence_warning_score"] = -persistence
    result["trailing_mean_warning_score"] = -result["trailing_mean"]
    return result


type OuterFolds = list[tuple[str, list[str]]]
type SessionTimes = Mapping[str, tuple[datetime, datetime]]


def leave_one_session_out(frame: pd.DataFrame) -> OuterFolds:
    sessions = sorted(str(session) for session in frame["session_id"].unique())
    return [(session, [other for other in sessions if other != session]) for session in sessions]


def forward_chaining(frame: pd.DataFrame, times: SessionTimes) -> OuterFolds:
    """Train only on sessions that ended before the test session started.

    Test sessions begin once MINIMUM_FORWARD_TRAINING_SESSIONS earlier sessions exist,
    so the first few sessions only ever train. Start and end are naive local wall clock
    times from one archive, comparable with each other but not across time zones.
    """
    sessions = sorted(str(session) for session in frame["session_id"].unique())
    missing = [session for session in sessions if session not in times]
    if missing:
        raise ValueError(f"forward chaining needs start and end times for {missing}")
    folds = []
    for session in sorted(sessions, key=lambda identity: (times[identity][0], identity)):
        start = times[session][0]
        earlier = sorted(other for other in sessions if times[other][1] <= start)
        if len(earlier) >= MINIMUM_FORWARD_TRAINING_SESSIONS:
            folds.append((session, earlier))
    return folds


def cross_validate(
    frame: pd.DataFrame,
    groups: Mapping[str, Sequence[str]],
    folds: OuterFolds | None = None,
) -> pd.DataFrame:
    """Score each outer fold with benchmarks and nested model fits on its training sessions.

    Defaults to leave one session out. Every row of a test session stays in its fold.
    """
    if frame["session_id"].nunique() < 3:
        raise ValueError("session cross validation needs at least three sessions")
    parts = []
    for session_id, training_sessions in folds or leave_one_session_out(frame):
        if session_id in training_sessions:
            raise ValueError("an outer test session cannot also be a training session")
        train = frame[frame["session_id"].isin(training_sessions)]
        test = frame[frame["session_id"] == session_id]
        identity = [
            "session_id",
            "as_of_ms",
            "mobility_cohort",
            "transition_in_window",
            "target_mbps",
            "segment_id",
            "earliest_support_ms",
            "latest_available_ms",
            "target_start_ms",
            "target_end_ms",
            "target_active_fraction",
            "network_mode_last",
        ]
        predictions = test[[name for name in identity if name in test]].copy()
        predictions["training_sessions"] = len(training_sessions)
        for name, values in baseline_fold(train, test).items():
            predictions[name] = values
        for feature_set in FEATURE_SETS:
            for name, values in fit_fold(train, test, groups[feature_set]).items():
                predictions[f"{feature_set}_{name}"] = values
        parts.append(predictions)
    result: pd.DataFrame = pd.concat(parts, ignore_index=True)
    return result


# Aggregation

Columns = Callable[[pd.DataFrame], tuple[float, ...]]
TARGET = "target_mbps"


def _estimate(
    rows: pd.DataFrame, columns: Columns, statistic: Callable[[np.ndarray], np.ndarray]
) -> dict[str, Any]:
    per_session = [columns(part) for _, part in rows.groupby("session_id", sort=True)]
    if not per_session:
        return {"value": None, "interval_95": None, "sessions": 0}
    sums = np.asarray(per_session, dtype=float).sum(axis=0)[np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(statistic(sums)[0])
    return {
        "value": value if np.isfinite(value) else None,
        "interval_95": session_bootstrap(per_session, statistic, BOOTSTRAP_SEED),
        "sessions": len(per_session),
    }


def _skill(sums: np.ndarray) -> np.ndarray:
    skill: np.ndarray = 1 - sums[:, 0] / sums[:, 1]
    return skill


def _share(sums: np.ndarray) -> np.ndarray:
    share: np.ndarray = sums[:, 0] / sums[:, 1]
    return share


def absolute_error_sums(column: str) -> Columns:
    def columns(part: pd.DataFrame) -> tuple[float, ...]:
        return (
            float((part[column] - part[TARGET]).abs().sum()),
            float((part["persistence"] - part[TARGET]).abs().sum()),
        )

    return columns


def quantile_score_sum(part: pd.DataFrame, prefix: str) -> float:
    actual = part[TARGET].to_numpy(dtype=float)
    total = sum(
        float(pinball(actual, part[f"{prefix}q{level}"].to_numpy(dtype=float), level).sum())
        for level in QUANTILES
    )
    return total / len(QUANTILES)


def quantile_score_sums(prefix: str) -> Columns:
    def columns(part: pd.DataFrame) -> tuple[float, ...]:
        return (quantile_score_sum(part, prefix), quantile_score_sum(part, "naive_"))

    return columns


def coverage_counts(prefix: str) -> Columns:
    def columns(part: pd.DataFrame) -> tuple[float, ...]:
        inside = (part[TARGET] >= part[f"{prefix}q{QUANTILES[0]}"]) & (
            part[TARGET] <= part[f"{prefix}q{QUANTILES[-1]}"]
        )
        return (float(inside.sum()), float(len(part)))

    return columns


def missed_counts(prefix: str) -> Columns:
    """Seconds below basic use that the cautious 10th percentile forecast did not foresee."""

    def columns(part: pd.DataFrame) -> tuple[float, ...]:
        low = part[TARGET] < WARNING_THRESHOLD_MBPS
        missed = low & (part[f"{prefix}q{QUANTILES[0]}"] >= WARNING_THRESHOLD_MBPS)
        return (float(missed.sum()), float(low.sum()))

    return columns


def false_warning_counts(prefix: str) -> Columns:
    """Seconds at or above basic use that the cautious forecast still flagged as at risk."""

    def columns(part: pd.DataFrame) -> tuple[float, ...]:
        fine = part[TARGET] >= WARNING_THRESHOLD_MBPS
        warned = fine & (part[f"{prefix}q{QUANTILES[0]}"] < WARNING_THRESHOLD_MBPS)
        return (float(warned.sum()), float(fine.sum()))

    return columns


def brier_sums(column: str, baseline: str = "naive_warning") -> Columns:
    def columns(part: pd.DataFrame) -> tuple[float, ...]:
        low = (part[TARGET] < WARNING_THRESHOLD_MBPS).astype(float)
        return (
            float(((part[column] - low) ** 2).sum()),
            float(((part[baseline] - low) ** 2).sum()),
        )

    return columns


def point_scores(rows: pd.DataFrame) -> dict[str, Any]:
    persistence_mae = (rows["persistence"] - rows[TARGET]).abs().groupby(rows["session_id"]).mean()
    result: dict[str, Any] = {}
    for name in (
        "persistence",
        "trailing_mean",
        *(f"{s}_mean" for s in present_feature_sets(rows)),
    ):
        errors = (rows[name] - rows[TARGET]).abs()
        session_mae = errors.groupby(rows["session_id"]).mean()
        result[name] = {
            "mae_mbps": float(errors.mean()),
            "skill_vs_persistence": _estimate(rows, absolute_error_sums(name), _skill),
            "sessions_better_than_persistence": int((session_mae < persistence_mae).sum()),
            "session_mae_median_mbps": float(session_mae.median()),
        }
    return result


def quantile_scores(rows: pd.DataFrame) -> dict[str, Any]:
    actual = rows[TARGET].to_numpy(dtype=float)
    low = actual < WARNING_THRESHOLD_MBPS
    result: dict[str, Any] = {}
    for prefix in ("naive_", *(f"{s}_" for s in present_feature_sets(rows))):
        lower = rows[f"{prefix}q{QUANTILES[0]}"].to_numpy(dtype=float)
        upper = rows[f"{prefix}q{QUANTILES[-1]}"].to_numpy(dtype=float)
        result[prefix.rstrip("_")] = {
            "pinball_mean": {
                str(level): float(
                    pinball(actual, rows[f"{prefix}q{level}"].to_numpy(dtype=float), level).mean()
                )
                for level in QUANTILES
            },
            "quantile_score_skill_vs_naive": _estimate(rows, quantile_score_sums(prefix), _skill),
            "interval_coverage": _estimate(rows, coverage_counts(prefix), _share),
            "nominal_coverage": 1 - INTERVAL_ALPHA,
            "interval_mean_width_mbps": float(np.mean(upper - lower)),
            "winkler_mean": float(np.mean(winkler(lower, upper, actual, INTERVAL_ALPHA))),
            "missed_by_cautious_forecast": _estimate(rows, missed_counts(prefix), _share),
            "false_warning_by_cautious_forecast": _estimate(
                rows, false_warning_counts(prefix), _share
            ),
            "low_rows": int(low.sum()),
            "missed_low_rows": int((low & (lower >= WARNING_THRESHOLD_MBPS)).sum()),
        }
    return result


INTERVAL_KINDS = {
    "raw": (f"q{QUANTILES[0]}", f"q{QUANTILES[-1]}"),
    "global_cqr": ("cqr_lower", "cqr_upper"),
    "mondrian_cqr": ("mondrian_lower", "mondrian_upper"),
}


def present_feature_sets(rows: pd.DataFrame) -> tuple[str, ...]:
    return tuple(name for name in FEATURE_SETS if f"{name}_mean" in rows)


def interval_coverage_counts(lower_column: str, upper_column: str) -> Columns:
    def columns(part: pd.DataFrame) -> tuple[float, ...]:
        inside = (part[TARGET] >= part[lower_column]) & (part[TARGET] <= part[upper_column])
        return (float(inside.sum()), float(len(part)))

    return columns


def _group_coverage(inside: pd.Series, groups: pd.Series) -> dict[str, Any]:
    table = inside.groupby(groups.astype(str)).agg(["mean", "size"])
    return {
        str(name): {"coverage": float(row["mean"]), "rows": int(row["size"])}
        for name, row in table.iterrows()
    }


def calibrated_interval_scores(rows: pd.DataFrame) -> dict[str, Any]:
    """Coverage overall, per session and per group for raw, global and Mondrian intervals.

    Conformal calibration targets coverage on average over rows. Group coverage, reported
    as the worst group, is the check for conditional miscalibration (Angelopoulos and
    Bates, section 4). Technology is the last reported network mode before forecasting.
    """
    result: dict[str, Any] = {
        "method": (
            "cross conformal CQR on inner session folds with equal session weights (Dunn, "
            "Wasserman and Ramdas, CDF pooling), globally and per predicted level quartile "
            "(Mondrian)"
        ),
        "nominal_row_coverage": 1 - INTERVAL_ALPHA,
        "assumption": "exchangeable sessions; approximate for few calibration sessions",
        "guarantee_in_this_archive": "not established; report empirical coverage only",
    }
    actual = rows[TARGET].to_numpy(dtype=float)
    for feature_set in present_feature_sets(rows):
        prefix = f"{feature_set}_"
        groups = {"predicted_level_quartile": rows[f"{prefix}predicted_level_quartile"]}
        if "network_mode_last" in rows:
            groups["technology"] = rows["network_mode_last"].fillna("unreported")
        kinds: dict[str, Any] = {}
        for kind, (lower_name, upper_name) in INTERVAL_KINDS.items():
            lower = rows[prefix + lower_name].to_numpy(dtype=float)
            upper = rows[prefix + upper_name].to_numpy(dtype=float)
            inside = pd.Series((actual >= lower) & (actual <= upper), index=rows.index)
            session_coverage = inside.groupby(rows["session_id"]).mean()
            by_group = {name: _group_coverage(inside, values) for name, values in groups.items()}
            kinds[kind] = {
                "row_coverage": _estimate(
                    rows, interval_coverage_counts(prefix + lower_name, prefix + upper_name), _share
                ),
                "session_coverage_minimum": float(session_coverage.min()),
                "session_coverage_quartiles": [
                    float(value) for value in np.quantile(session_coverage, [0.25, 0.5, 0.75])
                ],
                "sessions_below_nominal": int((session_coverage < 1 - INTERVAL_ALPHA).sum()),
                "sessions": len(session_coverage),
                "mean_width_mbps": float(np.mean(upper - lower)),
                "winkler_mean": float(np.mean(winkler(lower, upper, actual, INTERVAL_ALPHA))),
                "coverage_by_group": by_group,
                "worst_group_coverage": {
                    name: min(item["coverage"] for item in table.values())
                    for name, table in by_group.items()
                },
            }
        kinds["mondrian_fallback_quartiles_max"] = int(
            rows[f"{prefix}mondrian_fallback_quartiles"].max()
        )
        result[feature_set] = kinds
    return result


def _ap(outcome: np.ndarray, score: np.ndarray) -> float | None:
    return float(average_precision_score(outcome, score)) if outcome.any() else None


def paired_session_ap(rows: pd.DataFrame, column: str) -> dict[str, Any]:
    """Descriptive paired AP changes: exclude and name no-positive sessions.

    This averages sessions rather than bootstrapping correlated forecast rows.
    The interval conditions on already fitted predictions and does not account
    for shared outer training data or earlier hyperparameter selection.
    """
    differences: list[tuple[float, float]] = []
    excluded: list[str] = []
    all_positive: list[str] = []
    for session, part in rows.groupby("session_id", sort=True):
        low = (part[TARGET] < WARNING_THRESHOLD_MBPS).to_numpy()
        model_ap = _ap(low, part[column].to_numpy(dtype=float))
        baseline_ap = _ap(low, part["persistence_warning_score"].to_numpy(dtype=float))
        if model_ap is None or baseline_ap is None:
            excluded.append(str(session))
        elif low.all():
            all_positive.append(str(session))
        else:
            differences.append((model_ap - baseline_ap, 1.0))
    value = float(np.mean([item[0] for item in differences])) if differences else None
    return {
        "value": value,
        "interval_95": session_bootstrap(differences, _share, BOOTSTRAP_SEED),
        "sessions": len(differences),
        "excluded_no_positive_sessions": excluded,
        "excluded_all_positive_sessions": all_positive,
        "interpretation": (
            "descriptive conditional-on-fitted-predictions session mean AP difference"
        ),
    }


def warning_scores(rows: pd.DataFrame, include_onset_subset: bool = True) -> dict[str, Any]:
    low = (rows[TARGET] < WARNING_THRESHOLD_MBPS).to_numpy()
    result: dict[str, Any] = {
        "base_rate": float(low.mean()) if len(low) else None,
        "positive_rows": int(low.sum()),
        "rows": len(rows),
        "sessions": int(rows["session_id"].nunique()),
        "target": (
            "next 5 s mean delivered rate below threshold; not a stall or threshold crossing"
        ),
    }
    probability_names = (
        "naive_warning",
        "climatology_warning",
        *(f"{s}_{suffix}" for s in FEATURE_SETS for suffix in ("warning_raw", "warning")),
    )
    for name in (name for name in probability_names if name in rows):
        probability = rows[name].to_numpy(dtype=float)
        predicted = probability >= 0.5
        result[name] = {
            "average_precision": _ap(low, probability),
            "brier": float(np.mean((probability - low) ** 2)) if len(rows) else None,
            "brier_skill_vs_naive": _estimate(rows, brier_sums(name), _skill),
            "brier_skill_vs_climatology": _estimate(
                rows, brier_sums(name, "climatology_warning"), _skill
            ),
            "precision_at_half": float(low[predicted].mean()) if predicted.any() else None,
            "recall_at_half": float(predicted[low].mean()) if low.any() else None,
            "reliability": reliability(probability, low),
            "paired_session_ap_difference_vs_continuous_persistence": paired_session_ap(rows, name),
        }
    for name in ("persistence_warning_score", "trailing_mean_warning_score"):
        result[name] = {
            "average_precision": _ap(low, rows[name].to_numpy(dtype=float)),
            "score_semantics": "negative past rate; ranking score, not probability",
            "paired_session_ap_difference_vs_continuous_persistence": paired_session_ap(rows, name),
        }
    if include_onset_subset:
        result["future_low_given_current_not_low"] = warning_scores(
            rows.loc[rows["persistence"] >= WARNING_THRESHOLD_MBPS], include_onset_subset=False
        )
    return result


def reliability(
    probability: np.ndarray, outcome: np.ndarray, bins: int = 10
) -> list[dict[str, Any]]:
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    return [
        {
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "rows": int((index == b).sum()),
            "mean_probability": float(probability[index == b].mean())
            if (index == b).any()
            else None,
            "observed_frequency": float(outcome[index == b].mean()) if (index == b).any() else None,
        }
        for b in range(bins)
    ]


def deployability(
    frame: pd.DataFrame, names: Sequence[str], configuration: Mapping[str, Any]
) -> dict[str, Any]:
    """Prediction timing, excluding raw feature construction and interval construction."""
    matrix, single, categorical = _matrices(frame, frame.iloc[:1], names)
    target = frame["target_mbps"].to_numpy(dtype=float)
    _, warning_model, status, _ = _fit_warning(
        frame, names, target, inner_folds(frame), configuration
    )
    models: dict[str, Any] = {
        "cautious_quantile_0.1": _regressor(
            categorical, configuration, loss="quantile", quantile=QUANTILES[0]
        ).fit(matrix, target),
    }
    if warning_model is not None:
        models["warning_sigmoid"] = warning_model
    result = {}
    for name, model in models.items():
        predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        prediction_input = frame.iloc[:1] if name == "warning_sigmoid" else single
        timings = []
        for _ in range(200):
            started = time.perf_counter()
            predict(prediction_input)
            timings.append(time.perf_counter() - started)
        result[name] = {
            "pickle_bytes": len(pickle.dumps(model)),
            "single_prediction_median_ms": float(np.median(timings) * 1000),
            "features": len(names),
            "warning_calibration_status": status,
            "timed_operation": (
                "category transform, classifier and sigmoid on precomputed features"
                if name == "warning_sigmoid"
                else "estimator prediction on precomputed encoded features only"
            ),
            "fit_sessions": int(frame["session_id"].nunique()),
            "configuration": dict(configuration),
        }
    return result


# Figures


def _style(axis: Any, title: str, subtitle: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=12, color=INK["primary"], pad=22)
    axis.text(0, 1.02, subtitle, transform=axis.transAxes, fontsize=9, color=INK["secondary"])
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(AXIS)
    axis.tick_params(colors=INK["secondary"], labelsize=9, length=0)


def figure_skill(summary: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 3.4))
    rows = [
        (
            "Point forecast, MAE skill\nagainst persistence",
            "point",
            "{set}_mean",
            "skill_vs_persistence",
        ),
        (
            "Quantile forecasts, score skill\nagainst persistence plus residuals",
            "quantile",
            "{set}",
            "quantile_score_skill_vs_naive",
        ),
        (
            "Future mean below basic use,\nBrier skill against climatology",
            "warning",
            "{set}_warning",
            "brier_skill_vs_climatology",
        ),
    ]
    height = 0.36
    for position, (_label, section, key, metric) in enumerate(reversed(rows)):
        for offset, feature_set in ((height / 2, "history_only"), (-height / 2, "combined")):
            estimate = summary[section][key.format(set=feature_set)][metric]
            value = 100 * (estimate["value"] or 0.0)
            axis.barh(position + offset, value, height=height - 0.04, color=SERIES[feature_set])
            right = max(value, 0)
            if estimate["interval_95"]:
                low, high = (100 * x for x in estimate["interval_95"])
                axis.plot([low, high], [position + offset] * 2, color=INK["primary"], linewidth=1.2)
                right = max(right, high)
            # Labels sit past the interval end so they never cover it.
            axis.text(
                right + 1,
                position + offset,
                f"{value:.0f}%",
                va="center",
                fontsize=8,
                color=INK["secondary"],
            )
    axis.axvline(0, color=AXIS, linewidth=1)
    axis.set_yticks(range(len(rows)), [row[0] for row in reversed(rows)])
    axis.set_xlabel(
        "Skill score, percent better than the naive benchmark (line: 95% session "
        "bootstrap interval)",
        fontsize=9,
        color=INK["secondary"],
        loc="left",
    )
    axis.legend(
        [Patch(color=SERIES[s]) for s in FEATURE_SETS],
        ["History features only", "History plus radio context"],
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0, -0.2),
    )
    _style(
        axis,
        f"Forecast skill across {summary['sessions']} sessions, leave one session out",
        "Configuration, calibration and benchmarks fitted inside each fold on training sessions",
    )
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


WARNING_CURVES = (
    ("combined_warning", SERIES["combined"], "solid", "History plus radio context"),
    ("history_only_warning", SERIES["history_only"], "solid", "History features only"),
    ("trailing_mean_warning_score", SERIES["baseline"], "dashed", "Trailing mean ranking"),
    ("persistence_warning_score", SERIES["baseline"], "solid", "Persistence ranking"),
)


def _precision_recall_panel(
    axis: Any, rows: pd.DataFrame, warning: dict[str, Any], title: str, subtitle: str
) -> None:
    low = (rows[TARGET] < WARNING_THRESHOLD_MBPS).to_numpy()
    for name, color, style, label in WARNING_CURVES:
        if name not in rows or not low.any():
            continue
        precision, recall, _ = precision_recall_curve(low, rows[name].to_numpy(dtype=float))
        average_precision = _decimal(warning[name]["average_precision"], 2)
        axis.plot(
            recall,
            precision,
            color=color,
            linestyle=style,
            linewidth=1.8,
            label=f"{label}, AP {average_precision}",
        )
    if warning["base_rate"] is not None:
        axis.axhline(warning["base_rate"], color=AXIS, linewidth=1)
    axis.set_xlabel("Recall", fontsize=9, color=INK["secondary"])
    axis.set_ylabel("Precision", fontsize=9, color=INK["secondary"])
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False, fontsize=7, loc="upper right")
    _style(axis, title, subtitle)


def figure_warning(rows: pd.DataFrame, warning: dict[str, Any], path: Path) -> None:
    """All rows, rows whose current rate is not already low, and calibration.

    Both simple rankings are drawn, because the trailing mean is the strongest simple
    baseline and the current not low panel is where a warning has to add information.
    """
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(14.0, 4.0))
    _precision_recall_panel(
        left, rows, warning, "Next 5 s mean below 1.5 Mbps", "All rows, line at base rate"
    )
    onset_rows = rows.loc[rows["persistence"] >= WARNING_THRESHOLD_MBPS]
    _precision_recall_panel(
        middle,
        onset_rows,
        warning["future_low_given_current_not_low"],
        "Only when the current rate is at least 1.5 Mbps",
        "New low outcomes, line at base rate",
    )
    for name, color, _, _ in WARNING_CURVES:
        if name not in warning or "reliability" not in warning[name]:
            continue
        points = [point for point in warning[name]["reliability"] if point["rows"] >= 20]
        right.plot(
            [point["mean_probability"] for point in points],
            [point["observed_frequency"] for point in points],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
        )
    right.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    right.set_xlabel(
        "Forecast probability (bins with 20 or more rows)", fontsize=9, color=INK["secondary"]
    )
    right.set_ylabel("Observed frequency", fontsize=9, color=INK["secondary"])
    _style(right, "Calibration", "On the diagonal, a 30% forecast comes true 30% of the time")
    for axis in (left, middle, right):
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


# Report


def _percent(estimate: dict[str, Any]) -> str:
    value = estimate["value"]
    interval = estimate["interval_95"]
    text = "n/a" if value is None else f"{100 * value:.1f}%"
    if interval is not None:
        text += f" ({100 * interval[0]:.1f} to {100 * interval[1]:.1f})"
    return text


def _decimal(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _point_table(point: dict[str, Any], sessions: int) -> list[str]:
    lines = [
        "| Forecast | MAE (Mbps) | Skill against persistence | Sessions better than persistence "
        "| Median session MAE |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for name, item in point.items():
        lines.append(
            f"| {name} | {item['mae_mbps']:.2f} | {_percent(item['skill_vs_persistence'])} | "
            f"{item['sessions_better_than_persistence']} of {sessions} | "
            f"{item['session_mae_median_mbps']:.2f} |"
        )
    return lines


def _interval_table(scheme: dict[str, Any]) -> list[str]:
    lines = [
        "| Forecast | Interval | Row coverage | Lowest predicted level quartile | Lowest "
        "technology | Lowest session | Width (Mbps) | Winkler |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"raw": "raw quantiles", "global_cqr": "global CQR", "mondrian_cqr": "Mondrian CQR"}
    for name in present_feature_sets_of(scheme):
        for kind, label in labels.items():
            item = scheme["calibrated_interval"][name][kind]
            worst = item["worst_group_coverage"]
            technology = worst.get("technology")
            lines.append(
                f"| {name} | {label} | {_percent(item['row_coverage'])} | "
                f"{100 * worst['predicted_level_quartile']:.1f}% | "
                f"{'n/a' if technology is None else f'{100 * technology:.1f}%'} | "
                f"{100 * item['session_coverage_minimum']:.1f}% | "
                f"{item['mean_width_mbps']:.1f} | {item['winkler_mean']:.1f} |"
            )
    return lines


def present_feature_sets_of(scheme: dict[str, Any]) -> tuple[str, ...]:
    return tuple(name for name in FEATURE_SETS if name in scheme["calibrated_interval"])


def _selection_lines(selection: dict[str, Any]) -> list[str]:
    lines = [
        "| Scheme | Feature set | Folds per configuration index | Mean inner MAE by index |",
        "| --- | --- | --- | --- |",
    ]
    for scheme, by_feature_set in selection.items():
        for feature_set, item in by_feature_set.items():
            counts = ", ".join(f"{key}: {value}" for key, value in item["folds_by_index"].items())
            errors = ", ".join(f"{value:.2f}" for value in item["mean_inner_mae_by_index"])
            lines.append(f"| {scheme} | {feature_set} | {counts} | {errors} |")
    return lines


def render_report(summary: dict[str, Any]) -> str:
    point, quantile, warning = summary["point"], summary["quantile"], summary["warning"]
    forward = summary[FORWARD]
    lines = [
        "# Nested session cross validation of delivered rate forecasts",
        "",
        f"{summary['sessions']} file download sessions and {summary['rows']} forecast rows. "
        "Two outer schemes keep every session whole. Leave one session out tests transfer to "
        "an unseen session. Forward chaining trains only on sessions that ended before each "
        f"test session started ({forward['sessions']} test sessions, each after at least "
        f"{MINIMUM_FORWARD_TRAINING_SESSIONS} earlier sessions), the rolling origin "
        "evaluation of Hyndman and Athanasopoulos applied to whole sessions.",
        "",
        "Inside every outer fold, five inner folds of whole training sessions select one of "
        "the six predeclared configurations by out of fold mean absolute error, supply out "
        "of fold quantiles for interval calibration and fit the sigmoid probability "
        "calibrator. No outer test session influences its own configuration, calibrator or "
        "benchmark. Feature design and the configuration grid were developed earlier on this "
        "archive, so the results remain exploratory, not an independent held out estimate. "
        "Bootstrap intervals resample sessions, conditional on the saved predictions.",
        "",
        "## Configuration selected inside each outer fold",
        "",
        *_selection_lines(summary["configuration_selection"]),
        "",
        "## Point forecasts of the next 5 s mean rate",
        "",
        "Leave one session out:",
        "",
        *_point_table(point, summary["sessions"]),
        "",
        "Forward chaining:",
        "",
        *_point_table(forward["point"], forward["sessions"]),
        "",
        "## Quantile forecasts (0.1, 0.5, 0.9), leave one session out",
        "",
        "Scored with the quantile score, interval coverage and Winkler score. The benchmark "
        "adds the training sessions' residual quantiles to persistence.",
        "",
        "| Forecast | Quantile score skill | 80% interval coverage | Mean width (Mbps) "
        "| Winkler | Missed by the 10th percentile | False warnings by the 10th percentile |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for name, item in quantile.items():
        lines.append(
            f"| {name} | {_percent(item['quantile_score_skill_vs_naive'])} | "
            f"{_percent(item['interval_coverage'])} | {item['interval_mean_width_mbps']:.1f} | "
            f"{item['winkler_mean']:.1f} | {_percent(item['missed_by_cautious_forecast'])} | "
            f"{_percent(item['false_warning_by_cautious_forecast'])} |"
        )
    lines += [
        "",
        '"Missed by the 10th percentile" is the share of forecast rows whose future mean '
        'was below 1.5 Mbps while the cautious forecast was not. "False warnings" is the '
        "share of future means at or above 1.5 Mbps whose cautious forecast was below it. "
        "These are overlapping forecast windows, not independent seconds or incidents.",
        "",
        "## Intervals calibrated with equal session weights",
        "",
        "Conformalized quantile regression (Romano, Patterson and Candès) shifts both "
        "quantile bounds by a correction learned from out of fold predictions on every "
        "training session, with equal weight per session, adapting the CDF weighting of Dunn, "
        "Wasserman and Ramdas for grouped data. Global CQR uses one correction per outer "
        "fold. Mondrian CQR uses one per quartile of the predicted level, the midpoint of "
        "the 0.1 and 0.9 forecasts, because throughput spread grows with its level and one "
        "additive shift over covers low forecasts while under covering high ones. Both "
        "target 80% coverage empirically. Selection reuses calibration-session labels and "
        "models are refitted; the cited theorem supplies no coverage guarantee here. "
        "Shrinking is capped at the raw interval midpoint to keep reported bounds ordered. "
        "Coverage on average can hide groups below target, so the "
        "lowest predicted level quartile, the lowest current technology and the lowest "
        "session are shown. Values are empirical.",
        "",
        "Leave one session out:",
        "",
        *_interval_table(summary),
        "",
        "Forward chaining:",
        "",
        *_interval_table(forward),
        "",
        "## Probability that the next 5 s mean is below 1.5 Mbps, leave one session out",
        "",
        f"Base rate {100 * warning['base_rate']:.1f}% ({warning['positive_rows']} rows). "
        "The sigmoid calibrator is fitted on out of fold scores from the inner session folds. "
        "When an inner fold lacks one outcome, the forecast falls back to the training "
        "prevalence and the fallback is recorded. A reliability diagram measures empirical "
        "calibration; it does not guarantee it.",
        "",
        "| Forecast | Average precision | Brier | Brier skill against climatology "
        "| Precision at 0.5 | Recall at 0.5 |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for name in ("climatology_warning", "naive_warning", *(f"{s}_warning" for s in FEATURE_SETS)):
        item = warning[name]
        precision = item["precision_at_half"]
        recall = item["recall_at_half"]
        lines.append(
            f"| {name} | {_decimal(item['average_precision'])} | {_decimal(item['brier'], 4)} | "
            f"{_percent(item['brier_skill_vs_climatology'])} | "
            f"{'n/a' if precision is None else f'{100 * precision:.1f}%'} | "
            f"{'n/a' if recall is None else f'{100 * recall:.1f}%'} |"
        )
    onset = warning["future_low_given_current_not_low"]
    forward_warning = forward["warning"]
    forward_onset = forward_warning["future_low_given_current_not_low"]
    lines += [
        "",
        "The two-valued naive probability benchmark discards continuous rate rankings. "
        "Average precision is also compared with negative persistence and negative "
        "trailing mean, which are ranking scores rather than probabilities. The subset "
        "whose current rate is at least 1.5 Mbps asks whether a new low future mean is "
        "foreseen, rather than an already low one continuing.",
        "",
        "| Ranking forecast | AP, all rows | AP, current rate at least 1.5 Mbps | "
        "Session mean AP difference against continuous persistence (descriptive) | "
        "Forward chaining AP, all rows | Forward chaining AP, current at least 1.5 Mbps |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for name in (
        "persistence_warning_score",
        "trailing_mean_warning_score",
        *(f"{s}_warning" for s in FEATURE_SETS),
    ):
        difference = warning[name]["paired_session_ap_difference_vs_continuous_persistence"]
        lines.append(
            f"| {name} | {_decimal(warning[name]['average_precision'])} | "
            f"{_decimal(onset[name]['average_precision'])} | "
            f"{_decimal(difference['value'])} | "
            f"{_decimal(forward_warning[name]['average_precision'])} | "
            f"{_decimal(forward_onset[name]['average_precision'])} |"
        )
    deploy = summary["deployability"]
    lines += [
        "",
        f"The current-not-low subset has {onset['rows']} rows and {onset['positive_rows']} "
        "future low means under leave one session out. The target is a 5 s mean, not an "
        "instantaneous crossing, a playback stall or a physical fault.",
        "",
        "## Could it run on a device?",
        "",
        "| Model | Size | Single prediction (median) |",
        "| --- | ---: | ---: |",
        *(
            f"| {name} | {item['pickle_bytes'] / 1024:.0f} KiB | "
            f"{item['single_prediction_median_ms']:.2f} ms |"
            for name, item in deploy.items()
        ),
        "",
        "Fitted on every session with the configuration selected most often under leave one "
        "session out. Measured prediction in Python on the development laptop. Quantile timing "
        "uses encoded features; warning timing includes its category transform and sigmoid. "
        "Both exclude data loading, raw feature computation and interval construction. "
        "This is not end-to-end device latency. Conversion was not attempted.",
        "",
        "## Limits",
        "",
        f"{summary['sessions']} sessions from one 2019 to 2020 archive, recorded on few days "
        "and routes. The target is delivered application rate, not capacity. The kbps unit is "
        "inferred. Rows inside a session overlap in time. Session start times are naive local "
        "wall clock values. The session-paired AP analysis excludes sessions without both "
        "classes, with identities saved; pooled metrics retain every forecast row.",
        "",
    ]
    return "\n".join(lines)


# Entry point


def _protect_output_paths(snapshot_dir: Path, out_dir: Path, private_dir: Path) -> None:
    public, private, snapshot = out_dir.resolve(), private_dir.resolve(), snapshot_dir.resolve()
    if public == private or public in private.parents or private in public.parents:
        raise ValueError("public and private cross validation directories must be disjoint")
    for directory in (public, private):
        if directory == snapshot or directory in snapshot.parents or snapshot in directory.parents:
            raise ValueError("cross validation outputs must be separate from the source snapshot")
    targets = [
        out_dir / SUMMARY_NAME,
        out_dir / REPORT_NAME,
        *(out_dir / filename for filename in FIGURES.values()),
        private_dir / PREDICTIONS_NAME,
        private_dir / FORWARD_PREDICTIONS_NAME,
        private_dir / FOLDS_NAME,
    ]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("preserve the previous cross validation outputs, including figures")


def load_session_times(
    connection: Any, sessions: Sequence[str]
) -> dict[str, tuple[datetime, datetime]]:
    """Start and end of each session from its first raw timestamp and duration."""
    records = connection.execute(
        "SELECT session_id, origin_raw_timestamp, duration_ms FROM sessions "
        "WHERE session_id IN (SELECT unnest(?))",
        [list(sessions)],
    ).fetchall()
    times = {}
    for session_id, raw_timestamp, duration_ms in records:
        start = parse_timestamp(raw_timestamp)
        if start is None:
            raise ValueError(f"session {session_id} has no parseable start time")
        times[str(session_id)] = (start, start + timedelta(milliseconds=int(duration_ms)))
    return times


def _fold_manifest(
    frame: pd.DataFrame, folds: OuterFolds, rows: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for session, training_sessions in folds:
        train = frame.loc[frame["session_id"].isin(training_sessions)]
        predictions = (
            frame.loc[frame["session_id"] == session]
            if rows is None
            else rows.loc[rows["session_id"] == session]
        )
        inner = inner_folds(train)
        fold: dict[str, Any] = {
            "test_sessions": [str(session)],
            "training_sessions": sorted(training_sessions),
            "inner_fold_sessions": [
                sorted(str(item) for item in train.iloc[held]["session_id"].unique())
                for _, held in inner
            ],
            "training_rows": len(train),
            "test_rows": len(predictions),
            "encoder_categories": {
                name: list(CategoryEncoder.fit(train[name]).categories)
                for name in CATEGORICAL_FEATURES
                if name in train
            },
            "selection_and_calibration": {},
        }
        for feature_set in FEATURE_SETS if rows is not None else ():
            prefix = f"{feature_set}_"
            first = predictions.iloc[0]
            fold["selection_and_calibration"][feature_set] = {
                "configuration_index": int(first[f"{prefix}configuration_index"]),
                "inner_mae_by_index": [
                    float(first[f"{prefix}selection_mae_{index}"])
                    for index in range(len(CONFIGURATIONS))
                ],
                "warning_status": str(first[f"{prefix}warning_calibration_status"]),
                "interval_status": str(first[f"{prefix}interval_calibration_status"]),
                "interval_correction_mbps": float(first[f"{prefix}interval_correction_mbps"]),
            }
        manifest.append(fold)
    return manifest


def selection_summary(rows: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for feature_set in FEATURE_SETS:
        per_fold = rows.groupby("session_id").first()
        counts = Counter(int(index) for index in per_fold[f"{feature_set}_configuration_index"])
        result[feature_set] = {
            "folds_by_index": {str(index): counts[index] for index in sorted(counts)},
            "mean_inner_mae_by_index": [
                float(per_fold[f"{feature_set}_selection_mae_{index}"].mean())
                for index in range(len(CONFIGURATIONS))
            ],
        }
    return result


def scheme_scores(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "sessions": int(rows["session_id"].nunique()),
        "rows": len(rows),
        "point": point_scores(rows),
        "quantile": quantile_scores(rows),
        "warning": warning_scores(rows),
        "calibrated_interval": calibrated_interval_scores(rows),
        "warning_calibration_status_rows": {
            feature_set: {
                str(key): int(value)
                for key, value in rows[f"{feature_set}_warning_calibration_status"]
                .value_counts()
                .items()
            }
            for feature_set in present_feature_sets(rows)
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_cross_validation(snapshot_dir: Path, out_dir: Path, private_dir: Path) -> dict[str, Any]:
    _protect_output_paths(snapshot_dir, out_dir, private_dir)
    connection, manifest = open_snapshot(snapshot_dir)
    try:
        build = build_feature_frame(connection)
        frame = build.frame
        times = load_session_times(connection, sorted(frame["session_id"].unique()))
    finally:
        connection.close()
    if frame.empty or frame["session_id"].nunique() < MINIMUM_CROSS_VALIDATION_SESSIONS:
        raise ValueError("session cross validation needs at least five nonempty sessions")
    groups = {name: build.feature_groups[name] for name in FEATURE_SETS}
    folds = {LEAVE_ONE_OUT: leave_one_session_out(frame), FORWARD: forward_chaining(frame, times)}
    if not folds[FORWARD]:
        raise ValueError("forward chaining needs more sessions than its training minimum")
    source_hash = source_tree_hash()
    lock = Path(__file__).resolve().parents[3] / "uv.lock"
    lock_hash = file_sha256(lock) if lock.is_file() else None
    configuration = {
        "method": METHOD_VERSION,
        "configurations": list(CONFIGURATIONS),
        "selection": "inner grouped cross validation, lowest pooled out of fold MAE",
        "random_state": RANDOM_STATE,
        "early_stopping": False,
        "feature_groups": groups,
        "inner_folds": INNER_FOLDS,
        "inner_assignment": "sha256(random_state:session_identity) order, modulo inner folds",
        "warning_calibration": "sigmoid on inner session folds, ensemble=False",
        "warning_fallback": "training prevalence when the fit or an inner fold is one class",
        "interval_calibration": "cross conformal CQR, equal session weighted CDF pooling",
        "interval_alpha": INTERVAL_ALPHA,
        "quantiles": QUANTILES,
        "warning_threshold_mbps": WARNING_THRESHOLD_MBPS,
        "minimum_forward_training_sessions": MINIMUM_FORWARD_TRAINING_SESSIONS,
        "session_times": {
            session: [start.isoformat(), end.isoformat()]
            for session, (start, end) in sorted(times.items())
        },
    }
    fold_artifact: dict[str, Any] = {
        "source_sha256": source_hash,
        "lock_sha256": lock_hash,
        "lock_identity_status": "available" if lock_hash is not None else "unavailable_no_uv_lock",
        "snapshot_manifest_sha256": content_hash(manifest),
        "configuration": configuration,
        "configuration_sha256": content_hash(configuration),
        "feature_specification_hash": build.specification_hash,
        "folds": {scheme: _fold_manifest(frame, outer) for scheme, outer in folds.items()},
        "exclusions": build.exclusions,
        "exclusions_by_session": build.exclusions_by_session,
        "claim_scope": "exploratory_nested_session_cross_validation",
        "status": "started",
    }
    private_dir.mkdir(parents=True, exist_ok=True)
    fold_path = private_dir / FOLDS_NAME
    _write_json(fold_path, fold_artifact)
    try:
        rows = {scheme: cross_validate(frame, groups, outer) for scheme, outer in folds.items()}
        if source_tree_hash() != source_hash:
            raise RuntimeError(
                "source changed during forecast fitting; preserve and rerun separately"
            )
    except Exception as error:
        fold_artifact["status"] = (
            "failed_source_integrity" if source_tree_hash() != source_hash else "failed_fit"
        )
        fold_artifact["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(fold_path, fold_artifact)
        raise
    fold_artifact["folds"] = {
        scheme: _fold_manifest(frame, outer, rows[scheme]) for scheme, outer in folds.items()
    }
    fold_artifact["status"] = "complete"
    primary = rows[LEAVE_ONE_OUT]
    selected = selection_summary(primary)["combined"]["folds_by_index"]
    modal_index = int(max(selected, key=lambda index: (selected[index], -int(index))))
    summary: dict[str, Any] = {
        "method": METHOD_VERSION,
        "plan": "docs/TECHNICAL_REPORT.md",
        "claim_scope": "exploratory_nested_session_cross_validation",
        "source_sha256": source_hash,
        "lock_sha256": lock_hash,
        "lock_identity_status": "available" if lock_hash is not None else "unavailable_no_uv_lock",
        "configuration_sha256": content_hash(configuration),
        "fold_manifest_sha256": content_hash(fold_artifact),
        "snapshot_manifest_sha256": content_hash(manifest),
        "snapshot_id": manifest["snapshot_id"],
        "feature_specification_hash": build.specification_hash,
        "configurations": list(CONFIGURATIONS),
        "configuration_selection": {
            scheme: selection_summary(scheme_rows) for scheme, scheme_rows in rows.items()
        },
        "quantiles": list(QUANTILES),
        "warning_threshold_mbps": WARNING_THRESHOLD_MBPS,
        **scheme_scores(primary),
        FORWARD: {
            **scheme_scores(rows[FORWARD]),
            "test_sessions": [session for session, _ in folds[FORWARD]],
            "minimum_training_sessions": MINIMUM_FORWARD_TRAINING_SESSIONS,
        },
        "exclusions": build.exclusions,
        "exclusions_by_session": build.exclusions_by_session,
        "deployability": deployability(frame, groups["combined"], CONFIGURATIONS[modal_index]),
        "deployability_configuration_index": modal_index,
        "figures": FIGURES,
    }
    if source_tree_hash() != source_hash:
        fold_artifact["status"] = "failed_source_integrity"
        fold_artifact["error"] = {
            "type": "RuntimeError",
            "message": "source changed during scoring",
        }
        _write_json(fold_path, fold_artifact)
        raise RuntimeError("source changed during forecast scoring; preserve and rerun separately")
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_skill(summary, out_dir / FIGURES["skill"])
    figure_warning(primary, summary["warning"], out_dir / FIGURES["warning"])
    (out_dir / REPORT_NAME).write_text(render_report(summary), encoding="utf-8")
    _write_json(out_dir / SUMMARY_NAME, summary)
    primary.to_csv(private_dir / PREDICTIONS_NAME, index=False, na_rep="")
    rows[FORWARD].to_csv(private_dir / FORWARD_PREDICTIONS_NAME, index=False, na_rep="")
    _write_json(fold_path, fold_artifact)
    return summary
