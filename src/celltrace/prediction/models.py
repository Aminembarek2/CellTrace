"""Baselines and the histogram gradient boosting forecaster.

Persistence repeats the most recent valid downlink bin mean within the
history window. The trailing mean averages valid bin means over the past
30 seconds. The boosting model uses a bounded grid of at most six
configurations, a fixed seed and early_stopping=False, so no internal random
validation split can bypass the declared session or forward time scheme.
The configuration is chosen on dev rows only. The network mode encoder is
fitted on training rows only and maps unseen or missing modes to missing.

Model artifacts are pickles. They are loaded only after their sha256 matches
the JSON sidecar written next to them, and only artifacts this project wrote
should ever be loaded.
"""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor

from celltrace.canonical import file_sha256
from celltrace.prediction.features import (
    CATEGORICAL_FEATURES,
    FEATURE_GROUP_NAMES,
    check_predictor_names,
)

MODEL_VERSION = "prediction_models_v1"
RANDOM_STATE = 42
PERSISTENCE = "persistence"
TRAILING_MEAN = "trailing_mean_30s"
BOOSTING = "hist_gradient_boosting"
PERSISTENCE_FEATURE = "downlink_last_valid_mbps"
TRAILING_MEAN_FEATURE = "downlink_mean_30s_mbps"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIGURATION_GRID: tuple[dict[str, Any], ...] = (
    {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 40},
    {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 15, "min_samples_leaf": 40},
    {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 20},
    {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 80},
    {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 7, "min_samples_leaf": 100},
    {
        "learning_rate": 0.03,
        "max_iter": 300,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
)


class ArtifactIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class CategoryEncoder:
    """Integer codes for categories seen in training rows. Anything else becomes missing."""

    categories: tuple[str, ...]

    @classmethod
    def fit(cls, values: pd.Series) -> CategoryEncoder:
        seen = sorted({str(value) for value in values.dropna()})
        return cls(tuple(seen))

    def transform(self, values: pd.Series) -> np.ndarray:
        codes = {category: float(index) for index, category in enumerate(self.categories)}
        return np.array(
            [np.nan if pd.isna(value) else codes.get(str(value), np.nan) for value in values],
            dtype=float,
        )


class FeatureTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Cloneable training-only category transform for nested sklearn pipelines."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = feature_names

    def fit(self, frame: pd.DataFrame, target: Any = None) -> FeatureTransformer:
        self.encoders_ = {
            name: CategoryEncoder.fit(frame[name])
            for name in self.feature_names
            if name in CATEGORICAL_FEATURES
        }
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return design_matrix(frame, self.feature_names, self.encoders_)


@dataclass
class ForecastModel:
    name: str
    feature_group: str
    feature_names: tuple[str, ...]
    encoders: dict[str, CategoryEncoder]
    estimator: HistGradientBoostingRegressor
    configuration: dict[str, Any]
    training_rows: int

    def design_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return design_matrix(frame, self.feature_names, self.encoders)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict(self.design_matrix(frame)), dtype=float)


@dataclass
class SelectionResult:
    model: ForecastModel | None
    scores: list[dict[str, Any]] = field(default_factory=list)
    failed_fits: list[dict[str, Any]] = field(default_factory=list)


def persistence_forecast(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(frame[PERSISTENCE_FEATURE].to_numpy(), dtype=float)


def trailing_mean_forecast(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(frame[TRAILING_MEAN_FEATURE].to_numpy(), dtype=float)


def design_matrix(
    frame: pd.DataFrame, feature_names: Sequence[str], encoders: Mapping[str, CategoryEncoder]
) -> np.ndarray:
    columns = []
    for name in feature_names:
        if name in encoders:
            columns.append(encoders[name].transform(frame[name]))
        else:
            columns.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
    if not columns:
        raise ValueError("no features selected")
    return np.column_stack(columns)


def make_estimator(
    configuration: Mapping[str, Any], categorical_mask: Sequence[bool]
) -> HistGradientBoostingRegressor:
    mask = np.asarray(categorical_mask, dtype=bool)
    return HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=RANDOM_STATE,
        early_stopping=False,
        categorical_features=mask if mask.any() else None,
        **configuration,
    )


def fit_model(
    train: pd.DataFrame,
    feature_names: Sequence[str],
    configuration: Mapping[str, Any],
    feature_group: str = "custom",
    name: str = BOOSTING,
) -> ForecastModel:
    """Fit encoders and the estimator on training rows only."""
    check_predictor_names(feature_names)
    encoders = {
        feature: CategoryEncoder.fit(train[feature])
        for feature in feature_names
        if feature in CATEGORICAL_FEATURES
    }
    mask = [feature in encoders for feature in feature_names]
    estimator = make_estimator(configuration, mask)
    matrix = design_matrix(train, feature_names, encoders)
    estimator.fit(matrix, train["target_mbps"].to_numpy(dtype=float))
    if estimator.get_params()["early_stopping"] is not False:
        raise RuntimeError("early stopping must stay disabled")
    return ForecastModel(
        name=name,
        feature_group=feature_group,
        feature_names=tuple(feature_names),
        encoders=encoders,
        estimator=estimator,
        configuration=dict(configuration),
        training_rows=len(train),
    )


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def select_configuration(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    feature_names: Sequence[str],
    feature_group: str,
    grid: Sequence[Mapping[str, Any]] = CONFIGURATION_GRID,
) -> SelectionResult:
    """Fit each configuration on train, score MAE on dev, keep the lowest. Ties keep order."""
    if len(grid) > 6:
        raise ValueError("the configuration grid is bounded to six points")
    result = SelectionResult(model=None)
    best_error = math.inf
    actual = dev["target_mbps"].to_numpy(dtype=float)
    for index, configuration in enumerate(grid):
        try:
            model = fit_model(train, feature_names, configuration, feature_group)
            error = mean_absolute_error(actual, model.predict(dev))
        except (ValueError, RuntimeError, MemoryError) as failure:
            result.failed_fits.append(
                {
                    "feature_group": feature_group,
                    "grid_index": index,
                    "configuration": dict(configuration),
                    "error": f"{type(failure).__name__}: {failure}",
                }
            )
            continue
        result.scores.append(
            {"grid_index": index, "configuration": dict(configuration), "dev_mae_mbps": error}
        )
        if error < best_error:
            best_error = error
            result.model = model
    return result


def lock_sha256() -> str | None:
    lock = PROJECT_ROOT / "uv.lock"
    return file_sha256(lock) if lock.is_file() else None


def save_model(model: ForecastModel, path: Path, sidecar: Mapping[str, Any]) -> Path:
    """Write the pickle and a JSON sidecar that records its sha256. Returns the pickle path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    params = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in model.estimator.get_params().items()
    }
    record = {
        "model_version": MODEL_VERSION,
        "name": model.name,
        "feature_group": model.feature_group,
        "feature_names": list(model.feature_names),
        "categorical_features": {
            feature: list(encoder.categories) for feature, encoder in model.encoders.items()
        },
        "estimator": type(model.estimator).__name__,
        "estimator_parameters": params,
        "chosen_configuration": model.configuration,
        "training_row_count": model.training_rows,
        "uv_lock_sha256": lock_sha256(),
        "model_sha256": file_sha256(path),
        **sidecar,
    }
    sidecar_path(path).write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


def sidecar_path(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def load_sidecar(model_path: Path) -> dict[str, Any]:
    path = sidecar_path(model_path)
    if not path.is_file():
        raise ArtifactIntegrityError(f"model sidecar not found: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def load_model(model_path: Path) -> tuple[ForecastModel, dict[str, Any]]:
    """Load a pickle written by save_model after checking its sha256 against the sidecar."""
    if not model_path.is_file():
        raise ArtifactIntegrityError(f"model artifact not found: {model_path}")
    sidecar = load_sidecar(model_path)
    actual = file_sha256(model_path)
    if actual != sidecar.get("model_sha256"):
        raise ArtifactIntegrityError(f"{model_path} sha256 {actual} differs from its sidecar")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, ForecastModel):
        raise ArtifactIntegrityError(f"{model_path} does not hold a ForecastModel")
    return model, sidecar


def artifact_name(split_kind: str, feature_group: str) -> str:
    if feature_group not in FEATURE_GROUP_NAMES:
        raise ValueError(f"unknown feature group {feature_group}")
    return f"{BOOSTING}_{split_kind}_{feature_group}"


SPLIT_COLUMNS = {"session": "session_group", "forward_time": "forward_group"}
TRAINING_INDEX_NAME = "training_index.json"
MAIN_ARTIFACT = f"{BOOSTING}_session_combined"


def train_models(snapshot_dir: Path, splits_path: Path, out_dir: Path) -> dict[str, Any]:
    """Select and fit one boosting model per split kind and feature group.

    Each model is fitted on its split's train rows and its configuration is
    chosen by dev MAE. Test rows are never read here. The session split model
    with the combined feature group is the main artifact.
    """
    from celltrace.prediction.splits import labelled_rows, load_split_manifest

    manifest = load_split_manifest(splits_path)
    features, frame, snapshot_manifest = labelled_rows(snapshot_dir, manifest)
    names = [
        artifact_name(split_kind, group)
        for split_kind in SPLIT_COLUMNS
        for group in FEATURE_GROUP_NAMES
    ]
    summary: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "split_manifest_hash": manifest["manifest_hash"],
        "configuration_grid": list(CONFIGURATION_GRID),
        "artifacts": {},
        "failed_fits": [],
        "main_artifact": f"{MAIN_ARTIFACT}.pkl",
    }
    for split_kind, column in SPLIT_COLUMNS.items():
        train = frame[frame[column] == "train"]
        dev = frame[frame[column] == "dev"]
        for group in FEATURE_GROUP_NAMES:
            name = artifact_name(split_kind, group)
            selection = select_configuration(train, dev, features.feature_groups[group], group)
            summary["failed_fits"].extend(
                {**failure, "split_kind": split_kind} for failure in selection.failed_fits
            )
            entry: dict[str, Any] = {
                "split_kind": split_kind,
                "feature_group": group,
                "train_rows": len(train),
                "dev_rows": len(dev),
                "train_sessions": int(train["session_id"].nunique()),
                "dev_sessions": int(dev["session_id"].nunique()),
                "dev_scores": selection.scores,
            }
            if selection.model is None:
                entry["status"] = "all_fits_failed"
                summary["artifacts"][name] = entry
                continue
            selection.model.name = name
            path = save_model(
                selection.model,
                out_dir / f"{name}.pkl",
                {
                    "split_kind": split_kind,
                    "split_manifest_path": str(splits_path),
                    "split_manifest_hash": manifest["manifest_hash"],
                    "snapshot_id": snapshot_manifest["snapshot_id"],
                    "snapshot_logical_hash": snapshot_manifest["logical_hash"],
                    "feature_version": manifest["feature_version"],
                    "feature_specification_hash": features.specification_hash,
                    "selection_metric": "dev rows sample weighted MAE in Mbps",
                    "dev_scores": selection.scores,
                    "failed_fits": selection.failed_fits,
                    "related_artifacts": [f"{other}.pkl" for other in names],
                    "random_state": RANDOM_STATE,
                },
            )
            entry.update(
                {
                    "status": "fitted",
                    "path": str(path),
                    "model_sha256": file_sha256(path),
                    "chosen_configuration": selection.model.configuration,
                }
            )
            summary["artifacts"][name] = entry
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / TRAINING_INDEX_NAME).write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return summary
